import os
import cv2
import imageio
import random
import rospy
import numpy as np
import torch as th
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
from datetime import datetime

from moma_safety.tiago.tiago_gym import TiagoGym
from moma_safety.tiago import RESET_POSES as RP

# prior = np.array([
#     [ 0.   ,  0.,     0.,     0.035, -0.016,  0.105, 0.0, 0.0, 0.0, 0.0, -1.   ],
#     [ 0.   ,  0.,     0.,     0.074, -0.017,  0.092, 0.0, 0.0, 0.0, 0.0, -1.   ],
#     [ 0.   ,  0.,     0.,     0.078, -0.017,  0.054, 0.0, 0.0, 0.0, 0.0, -1.   ],
# ])

prior = np.array([
    [ 0.   ,  0.,     0.,     0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.   ],
    [ 0.   ,  0.,     0.,     0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.   ],
    [ 0.   ,  0.,     0.,     0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, -1.   ],
    [ 0.   ,  0.,     0.,     0.1, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, -1.   ],
])

def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True

class Exploration:
    def __init__(self, env, prior, traj_length, num_samples, num_top_samples, epochs, success, noise_params):
        self.env = env
        self.prior = prior
        self.traj_length = traj_length
        self.num_samples = num_samples
        self.num_top_samples = num_top_samples
        self.epochs = epochs
        self.success = success
        self.noise_params = noise_params

    def safe(self, action):
        return True
    
    def check_success(self):
        return False
    
    def undo_action(self, t, action):
        print("Undoing action")
        action_undo = np.concatenate((-action[t][:-1], action[t][-1:])) 
        self.step(action_undo)

    def step(self, action): 
        action_base = action[:3] # (delta_x, delta_y, delta_yaw)
        action_right_ee = action[3:] # (delta_pos, delta_ori in quaternion, binary gripper_action)

        action_dict = {'right': action_right_ee, 'left': None, 'base': None}
        obs, reward, done, info = self.env.step(action_dict, delay_scale_factor=2.0)

    def expl(self, t, actions, start_idx, grasp_mode=None):
        
        if t == self.traj_length:
            print("Reached end of recursion")
            # open gripper and see
            all_failed = True
            return all_failed
        
        for action in actions:
            print("--- time step, action: ", t, action[t][3:6])
            ee_pose_before = self.env.tiago.arms["right"].arm_pose
            joint_pos_before = self.env.tiago.arms["right"].joint_reader.get_most_recent_msg()
            if self.safe(action[t]):
                # Move the robot
                self.step(action[t])
                
                all_failed = self.expl(t+1, actions, start_idx, grasp_mode)

                # if task success
                if self.check_success():
                    print("Task succeeded!")
                    all_failed = False 
                    return all_failed
                
                # No future action is safe, undo the last action. Either exact joint positions or go back to the ee pose before the action
                self.undo_action(t, action)        

                # Resample the future actions
                # if all_failed:
                #     # sample t+1 actions again
                #     actions = sample_actions(t+1, actions, traj_length, start_idx)

        all_failed = True
        return all_failed 

    def sample_actions(t, actions, traj_length, start_idx):
        if t == traj_length:
            print("No action sampling needed.")
            return actions
        
        x_noise = np.random.multivariate_normal(mu_x, sigma_x, num_samples)
        y_noise = np.random.multivariate_normal(mu_y, sigma_y, num_samples)
        z_noise = np.random.multivariate_normal(mu_z, sigma_z, num_samples)
        episode_pos_noise = np.concatenate((np.expand_dims(x_noise, axis=1), 
                        np.expand_dims(y_noise, axis=1), 
                        np.expand_dims(z_noise, axis=1)), axis=1)
        
        actions_org = temp_prior.clone()
        actions_org = actions_org[None, start_idx:-1]
        actions_org = actions_org.repeat(num_samples, 1, 1)
        actions[:, t:, 3:6] = actions_org[:, t:, 3:6] + episode_pos_noise[:, t:]
        return actions


    def get_episode_reward(robot):
        target_pos = [1.1888, -0.1884,  0.8387]
        curr_pos = robot.eef_links["right"].get_position_orientation()[0].numpy()
        dist = np.linalg.norm(target_pos - curr_pos)
        return -dist


def main():
    rospy.init_node('moma_safety_exploration_script')
    set_all_seeds(seed=0)

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
        left_arm_enabled=True,
        right_gripper_type='robotiq2F-140',
        left_gripper_type='robotiq2F-85',
        base_enabled=False,
        torso_enabled=False,
    )

    # Initialize exploration class
    num_samples = 1
    num_top_samples = 3
    epochs = 10
    success = False
    traj_length = len(prior) # for the place subtask
    start_idx = 0

    traj_len = len(prior)
    noise_params = dict()
    noise_params["mu_x"] = np.zeros(traj_len)   # Example: 2-dimensional problem
    noise_params["sigma_x"] = np.eye(traj_len) * 0.003
    noise_params["mu_y"] = np.zeros(traj_len) # Example: 2-dimensional problem
    noise_params["sigma_y"] = np.eye(traj_len) * 0.003
    noise_params["mu_z"] = np.zeros(traj_len)  # Example: 2-dimensional problem
    noise_params["sigma_z"] = np.eye(traj_len) * 0.003

    exploration = Exploration(
        env,
        prior,
        traj_length,
        num_samples,
        num_top_samples,
        epochs,
        success,
        noise_params
    )

    # # Move Tiago to reset pose
    # reset_joint_pos = RP.VERTICAL_R_H
    # env.reset(reset_arms=True, reset_pose=reset_joint_pos, allowed_delay_scale=6.0)

    # Try two modes
    modes = ["horizontal"]

    for mode in modes:
        grasp_mode = mode

        x_noise = np.random.multivariate_normal(exploration.noise_params["mu_x"], exploration.noise_params["sigma_x"], num_samples)
        y_noise = np.random.multivariate_normal(exploration.noise_params["mu_y"], exploration.noise_params["sigma_y"], num_samples)
        z_noise = np.random.multivariate_normal(exploration.noise_params["mu_z"], exploration.noise_params["sigma_z"], num_samples)
        episode_pos_noise = np.concatenate((
                                np.expand_dims(x_noise, axis=2), 
                                np.expand_dims(y_noise, axis=2), 
                                np.expand_dims(z_noise, axis=2)), axis=2)

        actions = prior.copy()
        actions = actions[None, ...]
        actions = np.repeat(actions, num_samples, axis=0)
        
        # Uncomment to add noise to the actions
        # actions[:, :, 3:6] = actions[:, :, 3:6] + episode_pos_noise
        print("Start actions shape: ", actions.shape)

        all_failed = exploration.expl(t=0, actions=actions, start_idx=start_idx, grasp_mode=grasp_mode)

        # if not all_failed:
        #     break


if __name__ == "__main__":
    main()