import os
import cv2
import imageio
import random
import rospy
import time
import pickle
import numpy as np
np.set_printoptions(precision=2, suppress=True)
import torch as th
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from scipy.spatial.transform import Rotation as R
from datetime import datetime
from argparse import ArgumentParser

from moma_safety.tiago.tiago_gym import TiagoGym
from moma_safety.tiago import RESET_POSES as RP
from moma_safety.utils.rviz_utils import publish_target_pose, rviz_visualize_trajectories
from moma_safety.safety_models.base_collision import BaseCollisionModel
from moma_safety.utils.viz_utils import visualize_trajectories
from moma_safety.tiago.utils.camera_utils import Camera, RecordVideo
from moma_safety.test_scripts.test_slahmr_hamer_nav import navigate_primitive
from moma_safety.scripts.move_base_vel import move_base_vel
from moma_safety.scripts.nav_success_det import NavSuccessDetection
from moma_safety.utils.object_config import object_config as OC

prior = np.array([
    [ 0.3,     0.1,     -0.1,    0.,    0.,     0,   0,  0,  0,    0.   ],
    [ 0.3,     0.0,     -0.2,    0.,    0.,     0,   0,  0,  0,    0.   ],
    [ 0.2,     0.05,     0.,    0.,    0.,     0,   0,  0,  0,    0.   ],
    [ 0.3,     0.0,     0.,    0.,    0.,     0,   0,  0,  0,    0.   ],
])

def config_parser(parser=None):
    if parser is None:
        parser = ArgumentParser("moma_safety_exploration")
    parser.add_argument('--f_name', type=str)
    parser.add_argument("--record", action="store_true", default=False)
    return parser

# Note: for now the norm of the sampled vectors are the same as the original vector
def generate_symmetric_rotated_vectors(V, N=7, k=10): 
    """
    V: tuple of (V_x, V_y) original vector
    N: Number of vectors to generate
    k: Angle of rotation in degrees
    """
    # Convert degrees to radians
    k_radians = np.deg2rad(k)
    V_x, V_y = V
    magnitude = np.sqrt(V_x**2 + V_y**2)
    theta_V = np.arctan2(V_y, V_x)  # Angle of the original vector

    vectors = []
    M = (N - 1) // 2  # Number of steps in each direction
    for i in range(-M, M + 1):
        theta = theta_V + i * k_radians
        # Compute the rotated vector
        rotated_vector = (magnitude * np.cos(theta), magnitude * np.sin(theta))
        if i == 0:
            continue
            # vectors.insert(0, rotated_vector)
        else:
            vectors.append(rotated_vector)
    return vectors

def generate_numbers_around_x(X, d=0.09, range_offset=0.35):
    """
    X: number around which to generate numbers
    d: step size
    range_offset: range around X
    """
    # Define the range from X-range_offset to X+range_offset
    start = X - range_offset
    end = X + range_offset
    # Generate numbers spaced by d
    yaws = [start + i * d for i in range(0, int((end - start) / d) + 1)]
    return yaws

def sample_2d_pose(symmetric_vectors, yaws, num_samples=20):
    combined_vectors = []
    for _ in range(num_samples):
        # Randomly sample a vector (V_x, V_y) from rotated_vectors
        V_x, V_y = random.choice(symmetric_vectors)
        # Randomly sample a number from numbers
        number = random.choice(yaws)
        # Append the combined vector
        combined_vectors.append([V_x, V_y, number])
    return combined_vectors

class NavExploration:
    def __init__(self, env, prior, num_samples, num_top_samples, epochs, success, noise_params, start_idx, object_name=None, use_safe_list=False):
        self.env = env
        self.prior = prior
        self.traj_length = len(prior)
        self.num_samples = num_samples
        self.num_top_samples = num_top_samples
        self.epochs = epochs
        self.success = success
        self.noise_params = noise_params
        self.start_idx = start_idx
        self.final_actions = []
        self.object_name = object_name
        self.check_grasp_reachability_flag = False
        self.use_safe_list = use_safe_list
        if object_name is not None:
            self.check_grasp_reachability_flag = OC[object_name]["check_grasp_reachability_flag"]

        self.nav_success_det = NavSuccessDetection(env, exec_on_robot=False)
        self.base_collision_model = BaseCollisionModel(env)
        # Sample trajectories near the human prior (Note: in current implementtion, I am only sampling for the last base pose)
        self.sample_actions(t=0, actions=prior[:, :3])

    def sample_actions(self, t, actions):
        sampled_actions = []
        sampled_actions = np.repeat(actions[np.newaxis, ...], self.num_samples, axis=0)
        for waypt in range(t, self.traj_length):
            # change later
            if waypt == self.traj_length - 1:
                symmetric_vectors = generate_symmetric_rotated_vectors(actions[waypt][:2])
                yaws = generate_numbers_around_x(actions[waypt][2])
                sampled_2d_poses = sample_2d_pose(symmetric_vectors, yaws, num_samples=self.num_samples-1)
                # keep the 0th sample the same as the original action
                sampled_actions[1:, waypt] = sampled_2d_poses
        self.sampled_actions = np.array(sampled_actions)
        # for i, sampled_action in enumerate(sampled_actions[:, self.traj_length-1]):
        #     print("i, sampled_action: ", i, sampled_action)
        # print("original action: ", actions[self.traj_length-1])
        # return sampled_actions
    
    def safe(self, action):
        # change later
        return True
        # publish_target_pose(self.env, delta_pos=action[3:6], delta_ori=action[6:10])
        # pred_choice, prob = self.collision_model.predict_collisions(action=action)
        # if pred_choice == 1.0:
        #     return False
        # else:
        #     inp = input("Press Y to execute action and N to skip")
        #     if inp == 'Y':
        #         return True
        #     else:
        #         return False

    def undo_action(self, t, action):
        print("Undoing action")
        action_undo = -action[t]
        self.step(action_undo)
        # if undo was successful, remove the action from the final actions
        self.final_actions.pop()
    
    def step(self, action):
        move_base_vel(self.env, action)

    def expl(self, t):
        
        if t == self.traj_length:
            print("Reached end of recursion")
            
            task_success = self.nav_success_det.check_nav_success(check_grasp_reachability_flag=self.check_grasp_reachability_flag, select_object_for_mask=True)
            # task_success = True

            retval = dict()
            if task_success:
                print("Sub-task succeeded!")
                retval["resample"] = False
                retval["stop_expl"] = True
                retval["subtask_success"] = True
                retval["actions"] = self.final_actions
            else:
                retval["resample"] = False
                retval["stop_expl"] = False
                retval["subtask_success"] = False
            return retval
        
        if self.base_collision_model is not None:
            safe_list = []
            self.base_collision_model.get_set_scan()
            for i, action_sample in enumerate(self.sampled_actions): 
                action = action_sample[t]
                pred_choice, prob = self.base_collision_model.predict(action=action)
                if pred_choice == 1.0:
                    safe_list.append(False)
                else:
                    safe_list.append(True)
        
        for action_num, action in enumerate(self.sampled_actions):
            if self.use_safe_list and not safe_list[action_num]:
                continue
            print("--- time step, action: ", t, action[t])
            ee_pose_before = self.env.tiago.arms["right"].arm_pose
            joint_pos_before = self.env.tiago.arms["right"].joint_reader.get_most_recent_msg()
            if self.safe(action[t]):
                # # remove this later
                # if t == self.traj_length - 1:
                breakpoint()
                # Move the robot
                self.step(action[t])
                # if action was executed, add to the final actions
                self.final_actions.append({'right': None, 'left': None, 'base': action[t], 'gripper': 'stay'})
                retval = self.expl(t+1)

                if retval["stop_expl"]:
                    return retval

                # Either subtask success not achieved or no future action is safe => undo the last action and keep exploring from there 
                # Rewind by going back to either exact joint positions or go back to the ee pose before the action
                print("undoing action")
                breakpoint()
                self.undo_action(t, action)        

                # Optional: Resample the future actions
                if retval["resample"]:
                    # sample t+1 actions again
                    # actions = self.sample_actions(t+1, actions)
                    self.sampled_actions = self.sample_from_cone(t=t+1, actions=self.sampled_actions)

        # TODO change this
        all_failed = True
        return all_failed 

        
def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True

def main():
    rospy.init_node('moma_safety_exploration_script')
    set_all_seeds(seed=3)
    args = config_parser().parse_args()

    env = TiagoGym(
        frequency=10,
        right_arm_enabled=True,
        left_arm_enabled=False,
        right_gripper_type='robotiq2F-140',
        left_gripper_type='robotiq2F-85',
        base_enabled=True,
        torso_enabled=False,
    )
    num_samples = 20
    num_top_samples = 3
    epochs = 10
    success = False
    start_idx = 0
    noise_params = dict()

    # Initialize exploration class
    nav_exploration = NavExploration(
        env,
        prior,
        num_samples,
        num_top_samples,
        epochs,
        success,
        noise_params,
        start_idx
    )

    # if args.record:
    #     recorder.start_recording()
    #     print("Start recording")

    # 3. Perform exploration with the sampled trajectories
    nav_exploration.nav_success_det.set_object_name("cup")
    retval = nav_exploration.expl(t=0)

    # move_base_vel(env, -action[:3])
    # breakpoint()

if __name__ == "__main__":
    main()