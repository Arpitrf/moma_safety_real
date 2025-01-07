import os
import cv2
import imageio
import random
import rospy
import time
import pickle
import numpy as np
import torch as th
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
from datetime import datetime
from argparse import ArgumentParser

from moma_safety.tiago.tiago_gym import TiagoGym
from moma_safety.tiago import RESET_POSES as RP
from moma_safety.utils.rviz_utils import publish_target_pose, rviz_visualize_trajectories
from moma_safety.safety_models.arm_collision import ArmCollisionModel
from moma_safety.utils.viz_utils import visualize_trajectories
from moma_safety.tiago.utils.camera_utils import Camera, RecordVideo
from moma_safety.test_scripts.test_slahmr_hamer_nav import navigate_primitive
from moma_safety.grasp import grasp

GRIPPER_OPEN_POS = 0.5
GRIPPER_CLOSED_POS = 0

# prior = np.array([
#     [ 0.   ,  0.,     0.,     0.035, -0.016,  0.105, 0.0, 0.0, 0.0, 0.0, -1.   ],
#     [ 0.   ,  0.,     0.,     0.074, -0.017,  0.092, 0.0, 0.0, 0.0, 0.0, -1.   ],
#     [ 0.   ,  0.,     0.,     0.078, -0.017,  0.054, 0.0, 0.0, 0.0, 0.0, -1.   ],
# ])

# prior = np.array([
#     [ 0.   ,  0.,     0.,     0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.   ],
#     [ 0.   ,  0.,     0.,     0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 1.0, -1.   ],
#     [ 0.   ,  0.,     0.,     0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0, -1.   ],
#     [ 0.   ,  0.,     0.,     0.1, 0.0, -0.1, 0.0, 0.0, 0.0, 1.0, -1.   ],
# ])

# prior = np.array([
#     [ 0.        ,  0.        ,  0.        ,  0.15635495, 0.03853774,
#          -0.04268378,  0.        ,  0.        ,  0.        ,  1.        ,
#         -1.        ],
#     [ 0.        ,  0.        ,  0.        ,  0.14940402, 0.04695197,   #0.14940402
#          0.01762438,  0.        ,  0.        ,  0.        ,  1.        ,
#         -1.        ],
#     [ 0.        ,  0.        ,  0.        ,  0.07072136,  0.02199493,
#          0.05153857,  0.        ,  0.        ,  0.        ,  1.        ,
#         -1.        ]
# ])

# -0.04,-0.03,0,0,0,-0.05
-0.04,-0.03,0,0,0,0

prior = np.array([
    [ 0.,     0.,     0.,    -0.057, -0.04,   0.,     0.,    -0.,     -0.05,    0.   ],
    [ 0.,     0.,     0.,    -0.053, -0.038,  0.,     0.,    -0.,     0.0,    0.   ],
    [ 0.,     0.,     0.,    -0.053, -0.051,  0.,     0.,    -0.,     -0.05,    0.   ],
    [ 0.,     0.,     0.,    -0.031, -0.058,  0.,     0.,     0.,     0.0,    0.   ],
    [ 0.,     0.,     0.,     -0.033, -0.057,  0.,     0.,     0.,     0.0,    0.   ],
])

def config_parser(parser=None):
    if parser is None:
        parser = ArgumentParser("Bimanual Skill Learning")
    parser.add_argument('--f_name', type=str)
    parser.add_argument("--record", action="store_true", default=False)
    return parser

def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True

def get_prior(f_name):
    with open(f_name, "rb") as file:
        prior_dict = pickle.load(file)
        prior = []
        for i in range(len(prior_dict["delta_body_positions"])): 
            # TODO: Change this. Add delta base pose (x, y, yaw)
            delta_base = [0.0, 0.0, 0.0]
            delta_hand_pos = prior_dict["delta_hand_positions"][i].tolist()
            delta_hand_ori = [0.0, 0.0, 0.0, 1.0]
            gripper_action = [-1.0]
            action = delta_base + delta_hand_pos + delta_hand_ori + gripper_action
            prior.append(action)
        prior = np.array(prior)
        breakpoint()
        return prior

class Exploration:
    def __init__(self, env, prior, num_samples, num_top_samples, epochs, success, noise_params, start_idx):
        self.env = env
        self.prior = prior
        self.traj_length = len(prior)
        self.num_samples = num_samples
        self.num_top_samples = num_top_samples
        self.epochs = epochs
        self.success = success
        self.noise_params = noise_params
        self.start_idx = start_idx
        self.collision_model = ArmCollisionModel(env)

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
    
    def check_success(self):
        return False
    
    def undo_action(self, t, action):
        print("Undoing action")
        action_undo = np.concatenate((-action[t][:-1], action[t][-1:])) 
        self.step(action_undo)

    def step(self, action): 
        action_base = action[:3] # (delta_x, delta_y, delta_yaw)
        action_right_ee = action[3:] # (delta_pos, delta_ori in quaternion, binary gripper_action)

        obs, reward, done, info = None, None, None, None
        action_dict = {'right': action_right_ee, 'left': None, 'base': None}
        obs, reward, done, info = self.env.step(action_dict, delay_scale_factor=6.0)
        return obs, reward, done, info

    def expl(self, t, actions, grasp_mode=None):

        while t != self.traj_length:
            user_input = input("Enter the action")        
            a = [float(i.strip()) for i in user_input.split(",")]
            a_quat = R.from_rotvec(a[3:6]).as_quat()
            action = np.concatenate((np.zeros(3), a[0:3], a_quat, np.array([0.0])))
            print("action: ", action)
            breakpoint()
            _, _, _, info = self.step(action)
            t += 1

            # check if F/T sensor is high
            wrench = self.env.tiago.arms["right"].ft_right_sub.get_most_recent_msg()
            force = wrench["force"]
            torque = wrench["torque"]
            force_sum = abs(force.x) + abs(force.y) + abs(force.z)
            torque_sum = abs(torque.x) + abs(torque.y) + abs(torque.z)
            print("force_sum: ", force_sum)
            if force_sum > 100:
                self.undo_action(t, action)
                t -= 1

            # action not reachable
            if info is not None and info["arm_right"]["joint_goal"] is None:
                print("action not reachable")
                t -= 1

            breakpoint()




        # # check if F/T sensor is high
        # wrench = self.env.tiago.arms["right"].ft_right_sub.get_most_recent_msg()
        # f = wrench["force"]
        # t = wrench["torque"]
        # force_sum = abs(f.x) + abs(f.y) + abs(f.z)
        # torque_sum = abs(t.x) + abs(t.y) + abs(t.z)
        # print("force_sum: ", force_sum)
        # if force_sum > 100:
        #     return True
        
        # if t == self.traj_length:
        #     print("Reached end of recursion")
        #     # open gripper and see
            
        #     # TODO: Change this
        #     # all_failed = True
        #     inp = input("Press Y if success else N if failed")
        #     if inp == 'Y':
        #         self.env.tiago.gripper["right"].step(1.0)
        #         all_failed = False
        #     else:
        #         all_failed = True

        #     return all_failed
        
        # for action in actions:
        #     print("--- time step, action: ", t, action[t][3:6])
        #     ee_pose_before = self.env.tiago.arms["right"].arm_pose
        #     joint_pos_before = self.env.tiago.arms["right"].joint_reader.get_most_recent_msg()
        #     if self.safe(action[t]):
        #         # Move the robot
        #         breakpoint()
        #         _, _, _, info = self.step(action[t])

        #         # action not reachable
        #         if info is not None and info["arm_right"]["joint_goal"] is None:
        #             print("action not reachable")
        #             breakpoint()
        #             continue
                
        #         all_failed = self.expl(t+1, actions, grasp_mode)

        #         if not all_failed:
        #             return all_failed

        #         # No future action is safe, undo the last action. Either exact joint positions or go back to the ee pose before the action
        #         self.undo_action(t, action)
        #         user_input = input("Enter the action")        
        #         a = [float(i.strip()) for i in user_input.split(",")]
        #         actions[0, t, 3:9] = a

        #         # # Resample the future actions
        #         # if all_failed:
        #         #     # sample t+1 actions again
        #         #     # actions = self.sample_actions(t+1, actions)
        #         #     actions = self.sample_from_cone(t=t+1, actions=actions)

        # all_failed = True
        # return all_failed 

    def sample_actions(self, t, actions):
        if t == self.traj_length:
            print("No action sampling needed.")
            return actions
        
        x_noise = np.random.multivariate_normal(self.noise_params["mu_x"], self.noise_params["sigma_x"], self.num_samples)
        y_noise = np.random.multivariate_normal(self.noise_params["mu_y"], self.noise_params["sigma_y"], self.num_samples)
        z_noise = np.random.multivariate_normal(self.noise_params["mu_z"], self.noise_params["sigma_z"], self.num_samples)
        episode_pos_noise = np.concatenate((
                                np.expand_dims(x_noise, axis=2), 
                                np.expand_dims(y_noise, axis=2), 
                                np.expand_dims(z_noise, axis=2)), axis=2)
        
        actions_org = self.prior.copy()
        actions_org = actions_org[None, ...]
        actions_org = np.repeat(actions_org, self.num_samples, axis=0)
        actions[:, t:, 3:6] = actions_org[:, t:, 3:6] + episode_pos_noise[:, t:]
        return actions
    
    def sample_from_cone(self, t, actions, max_angle=np.pi/6, norm_variance=0.4):
        if t == self.traj_length:
            print("No action sampling needed.")
            return actions
        
        for waypt in range(t, self.traj_length):
            original_vector = self.prior[waypt, 3:6]
            original_norm = np.linalg.norm(original_vector)

            original_vector = original_vector / np.linalg.norm(original_vector)  # Normalize input vector

            # Compute the rotation matrix to align [0, 0, 1] with the original vector
            z_axis = np.array([0.0, 0.0, 1.0])
            if np.allclose(original_vector, z_axis):
                rotation_matrix = np.eye(3)  # No rotation needed if already aligned
            else:
                rotation_axis = np.cross(z_axis, original_vector)
                rotation_axis /= np.linalg.norm(rotation_axis)
                angle = np.arccos(np.clip(np.dot(z_axis, original_vector), -1.0, 1.0))
                rotation_matrix = R.from_rotvec(angle * rotation_axis).as_matrix()

            for sample_num in range(self.num_samples):
                # Sample a random point in the cone aligned with the z-axis
                z = np.cos(max_angle) + (1 - np.cos(max_angle)) * np.random.rand()
                phi = 2 * np.pi * np.random.rand()
                x = np.sqrt(1 - z**2) * np.cos(phi)
                y = np.sqrt(1 - z**2) * np.sin(phi)
                random_point = np.array([x, y, z], dtype=np.float64)

                # Apply the rotation to align the point with the original vector
                noisy_vector = rotation_matrix @ random_point

                # Vary the norm of the noisy vector
                varied_norm = original_norm * (1 + np.random.uniform(-norm_variance, norm_variance))
                noisy_vector *= varied_norm

                actions[sample_num, waypt, 3:6] = noisy_vector

        return actions


    def get_episode_reward(robot):
        target_pos = [1.1888, -0.1884,  0.8387]
        curr_pos = robot.eef_links["right"].get_position_orientation()[0].numpy()
        dist = np.linalg.norm(target_pos - curr_pos)
        return -dist


def main():
    rospy.init_node('moma_safety_exploration_script')
    set_all_seeds(seed=3)
    args = config_parser().parse_args()
    
    # # for saving videos
    # current_date = datetime.now().strftime("%Y-%m-%d")  # Format: YYYY-MM-DD
    # current_time = datetime.now().strftime("%H-%M-%S")  # Format: HH-MM-SS
    # base_folder = f"{current_date}"
    # time_folder = os.path.join(base_folder, current_time)
    # folder_path = f"outputs_expl/{time_folder}"
    # os.makedirs(folder_path, exist_ok=True)
    # imgio_kargs = {'fps': 10, 'quality': 10, 'macro_block_size': None,  'codec': 'h264',  'ffmpeg_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']}
    # output_path = f'{folder_path}/video.mp4'
    # writer = imageio.get_writer(output_path, **imgio_kargs)

    env = TiagoGym(
        frequency=10,
        right_arm_enabled=True,
        left_arm_enabled=False,
        right_gripper_type='robotiq2F-140',
        left_gripper_type='robotiq2F-85',
        base_enabled=True,
        torso_enabled=False,
    )
    # Initialize exploration class
    num_samples = 10
    num_top_samples = 3
    epochs = 10
    success = False
    start_idx = 0

    # Read the prior
    f_name = "slahmr_hamer_outputs/robot_actions_forward_test1_1.pkl"
    # prior = get_prior(f_name=f_name)

    traj_len = len(prior)
    noise_params = dict()
    noise_params["mu_x"] = np.zeros(traj_len)   # Example: 2-dimensional problem
    noise_params["sigma_x"] = np.eye(traj_len) * 0.001
    noise_params["mu_y"] = np.zeros(traj_len) # Example: 2-dimensional problem
    noise_params["sigma_y"] = np.eye(traj_len) * 0.001
    noise_params["mu_z"] = np.zeros(traj_len)  # Example: 2-dimensional problem
    noise_params["sigma_z"] = np.eye(traj_len) * 0.001
    
    exploration = Exploration(
        env,
        prior,
        num_samples,
        num_top_samples,
        epochs,
        success,
        noise_params,
        start_idx
    )

    # Start recorder
    side_cam, top_down_cam, ego_cam = None, None, None
    if args.record:
        # side_cam = Camera(img_topic="/side_1/color/image_raw", depth_topic="/side_1/aligned_depth_to_color/image_raw")
        # top_down_cam = Camera(img_topic="/top_down/color/image_raw", depth_topic="/top_down/aligned_depth_to_color/image_raw")
        ego_cam = Camera(img_topic="/xtion/rgb/image_raw", depth_topic="/xtion/depth/image_raw")
        recorder = RecordVideo(camera_interface_ego=ego_cam)
        recorder.setup_recording()

    # Move Tiago to reset pose
    # open gripper. 1 is open and 0 is close
    env.tiago.gripper['right'].step(GRIPPER_OPEN_POS)
    time.sleep(2)
    # reset_joint_pos = RP.FORWARD_R_H
    reset_joint_pos = RP.PREGRASP_R_H
    reset_joint_pos["right"][-1] = GRIPPER_OPEN_POS
    env.reset(reset_arms=True, reset_pose=reset_joint_pos, allowed_delay_scale=6.0)

    # Try two modes
    modes = ["horizontal"]

    # # Perform navigation
    # navigate_primitive()

    for mode in modes:
        grasp_mode = mode

        # # Get initial samples
        # # simple x,y,z gaussian noise
        # # actions = exploration.sample_actions(t=0, actions=prior)

        # # cone method
        # actions = prior.copy()
        # actions = actions[None, ...]
        # actions = np.repeat(actions, num_samples, axis=0) 
        # actions = exploration.sample_from_cone(t=0, actions=actions)

        # rviz_visualize_trajectories(actions, start_position=env.tiago.arms["right"].arm_pose[:3])

        # print("Start actions shape: ", actions.shape)
        # breakpoint()

        # perform grasp
        grasp(env, exec_on_robot=True)
        print("Grasping done. Continue with exploration.")
        breakpoint()

        # remove later
        new_prior = []
        for prior_action in prior:
            prior_quat = R.from_rotvec(prior_action[6:9]).as_quat()
            new_prior.append(np.concatenate((prior_action[:6], prior_quat, prior_action[-1:])))
        new_prior = np.array(new_prior)
        new_prior = new_prior[None, ...]
        actions = new_prior.copy()


        if args.record:
            recorder.start_recording()
            print("Start recording")

        all_failed = exploration.expl(t=0, actions=actions, grasp_mode=grasp_mode)

        time.sleep(2)
        if args.record:
            recorder.save_video(save_folder="output")

        if not all_failed:
            break


if __name__ == "__main__":
    main()