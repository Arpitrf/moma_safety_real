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
from moma_safety.safety_models.obj_dropping import ObjDroppingDetector
from moma_safety.safety_models.grasp_loss_ft import GraspLossFTDetector
from moma_safety.utils.viz_utils import visualize_trajectories
from moma_safety.utils.vlm_utils import check_manip_success_using_vlm
from moma_safety.tiago.utils.camera_utils import Camera, RecordVideo
from moma_safety.test_scripts.test_slahmr_hamer_nav import navigate_primitive
from moma_safety.grasp import grasp
from moma_safety.utils.object_config import object_config as OC

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

# prior = np.array([
#     [ 0.,     0.,     0.,    -0.057, -0.04,   0.,     0.,    -0.,     -0.05,    0.   ],
#     [ 0.,     0.,     0.,    -0.053, -0.038,  0.,     0.,    -0.,     0.0,    0.   ],
#     [ 0.,     0.,     0.,    -0.053, -0.051,  0.,     0.,    -0.,     -0.05,    0.   ],
#     [ 0.,     0.,     0.,    -0.031, -0.058,  0.,     0.,     0.,     0.0,    0.   ],
#     [ 0.,     0.,     0.,     -0.033, -0.057,  0.,     0.,     0.,     0.0,    0.   ],
# ])

prior = np.array([
    [ 0.,     0.,     0.,    -0.07, -0.01,  0,   0,  0,  0,    0.   ],
    [ 0.,     0.,     0.,    -0.05, -0.03,  0,   0,  0,  0,    0.   ],
    [ 0.,     0.,     0.,    -0.05, -0.0,  0,   0,  0,  0,    0.   ],
    [ 0.,     0.,     0.,    -0.05, -0.0,  0,   0,  0,  0,    0.   ],
    [ 0.,     0.,     0.,    -0.00, -0.0,  0,   0,  0,  0,    0.   ],
])

def test_ft(env):
    wrench = env.tiago.arms["right"].ft_right_sub.get_most_recent_msg()
    f = wrench["force"]
    t = wrench["torque"]
    force_sum = abs(f.x) + abs(f.y) + abs(f.z)
    if force_sum > 60:
        return True
    return False

def config_parser(parser=None):
    if parser is None:
        parser = ArgumentParser("moma_safety_exploration")
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

# def get_prior(f_name):
#     with open(f_name, "rb") as file:
#         prior_dict = pickle.load(file)
#         prior = []
#         for i in range(len(prior_dict["delta_body_positions"])): 
#             # TODO: Change this. Add delta base pose (x, y, yaw)
#             delta_base = [0.0, 0.0, 0.0]
#             delta_hand_pos = prior_dict["delta_hand_positions"][i].tolist()
#             delta_hand_ori = [0.0, 0.0, 0.0, 1.0]
#             gripper_action = [-1.0]
#             action = delta_base + delta_hand_pos + delta_hand_ori + gripper_action
#             prior.append(action)
#         prior = np.array(prior)
#         breakpoint()
#         return prior

class ManipExploration:
    def __init__(self, env, prior, num_samples, num_top_samples, epochs, success, noise_params, start_idx, object_name, segment_description, check_arm_collision=False, check_obj_dropping=False, check_grasp_loss=False, check_ft=False, arm_collision_th=0.5, grasp_loss_ft_th=0.5, use_impedance_controller=False):
        self.env = env
        # Not doing this for now: adding 1 for the open gripper action
        self.traj_length = len(prior)
        self.num_samples = num_samples
        self.num_top_samples = num_top_samples
        self.epochs = epochs
        self.success = success
        self.noise_params = noise_params
        self.start_idx = start_idx
        self.object_name = object_name
        self.arm_collision_th = arm_collision_th
        self.grasp_loss_ft_th = grasp_loss_ft_th
        self.final_actions = []
        self.collision_model = None
        self.obj_dropping_model = None
        self.grasp_loss_model = None
        self.ft_model = None
        self.grasp_loss_ft_model = None
        self.check_arm_collision = check_arm_collision
        self.check_obj_dropping = check_obj_dropping
        self.check_grasp_loss = check_grasp_loss
        self.check_ft = check_ft
        self.use_impedance_controller = use_impedance_controller
        self.segment_description = segment_description

        if self.use_impedance_controller:
            start_controllers = ['arm_right_impedance_controller', 'arm_left_impedance_controller']
            stop_controllers = ['arm_right_controller', 'arm_left_controller']
            self.env.tiago.arms["right"].switch_controller(start_controllers, stop_controllers)

        if check_arm_collision:
            self.collision_model = ArmCollisionModel(env)
        if check_obj_dropping:
            self.obj_dropping_model = ObjDroppingDetector(env)
        if check_grasp_loss and check_ft:
            self.grasp_loss_ft_model = GraspLossFTDetector(env)

        self.prior = self.process_prior(prior)
        self.sample_actions()

        # set head position 
        self.env.tiago.head.write_head_command(OC[self.object_name]["manip_head_joint_pos"])

    # TODO: convert [pos], [rpy] -> (pos, quat)
    def process_prior(self, prior):
        processed_prior = []
        for waypt in prior:
            delta_pos = waypt[0]
            abs_orn = R.from_euler('xyz', waypt[1:4]).as_quat()
            # keeping delta orn as 0 for now. Later we will use it and probably use euler repr as easier to interpret noise
            delta_orn = np.array([0.0, 0.0, 0.0, 1.0])
            # assuming that gripper action is closed throughout the manipulation
            processed_waypt = np.concatenate((np.zeros(3), delta_pos, delta_orn, np.array([0.0])))
            processed_prior.append(processed_waypt)
        
        # Not doing this for now: add open gripper as the last element (the 1.0 is for the last element of delta quat)
        # processed_waypt = np.concatenate((np.zeros(9), np.array([1.0]), np.array([OC[self.object_name]["gripper_open_pos"]])))
        # processed_prior.append(processed_waypt)
        return np.array(processed_prior)

    
    def sample_from_cube(self, t, actions):
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
        
        # if last action is open gripper, no need to sample for that
        end_idx = self.traj_length
        # if actions[0][-1, 3:6].sum() == 0:
        #     end_idx = self.traj_length - 1
        for waypt in range(t, end_idx):
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

                if OC[self.object_name]["zero_out_z"]:
                    noisy_vector[2] = 0.0
                actions[sample_num, waypt, 3:6] = noisy_vector

        return actions

    def sample_actions(self):
        # cone method
        actions = self.prior.copy()
        actions = actions[None, ...]
        actions = np.repeat(actions, self.num_samples, axis=0) 
        sampled_actions = self.sample_from_cone(t=0, actions=actions)
        self.sampled_actions = np.array(sampled_actions)
        if np.isnan(actions).any():
            print("NaN values detected in sampled actions")
            breakpoint()


    def get_episode_reward(robot):
        target_pos = [1.1888, -0.1884,  0.8387]
        curr_pos = robot.eef_links["right"].get_position_orientation()[0].numpy()
        dist = np.linalg.norm(target_pos - curr_pos)
        return -dist
    
    def safe(self, action=None, check_arm_collision=True, check_object_dropping=False):    
        # if check_arm_collision:
        #     publish_target_pose(self.env, delta_pos=action[3:6], delta_ori=action[6:10])
        #     # TODO: process the action for inputting to the models
        #     pred_choice, prob = self.collision_model.predict_collisions(action=action, threshold=0.5, object_name=self.object_name)
        #     if pred_choice == 1.0:
        #         return False
        #     else:
        #         inp = input("Press Y to execute action and N to skip")
        #         if inp == 'Y':
        #             return True
        #         else:
        #             return False
        
        if check_object_dropping:
            self.obj_dropping_model.get_set_points(self.object_name)
            pred_choice, prob = self.obj_dropping_model.predict(threshold=0.5, object_name=self.object_name)

        inp = input("Press Y to execute action and N to skip")
        if inp == 'Y':
            return True
        else:
            return False
        
    def undo_action(self, t, action):
        print("Undoing action")
        action_undo = np.concatenate((-action[t][:-1], action[t][-1:])) 
        print("undo action: ", action_undo)
        self.step(action_undo)
        self.final_actions.pop()

    def step(self, action): 
        action_base = action[:3] # (delta_x, delta_y, delta_yaw)
        action_right_ee = action[3:] # (delta_pos, delta_ori in quaternion, binary gripper_action)

        obs, reward, done, info = None, None, None, None
        action_dict = {'right': action_right_ee, 'left': None, 'base': None}
        obs, reward, done, info = self.env.step(action_dict, delay_scale_factor=6.0, timeout=5.0)
        return obs, reward, done, info
        
    def check_manip_success(self, select_object_for_mask=False):
        obs = self.env._observation()
        img = obs['tiago_head_image']
        segment_succ = check_manip_success_using_vlm(img, self.object_name, self.segment_description)
        print("segment_succ: ", segment_succ)

        # remove later
        inp = input("Press Y if subtask success else N if failed")
        if inp == 'Y':
            task_success = True
        else:
            task_success = False

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

            # close gripper as well (as part of the rewind)
            gripper_close_action = np.concatenate((np.zeros(9), np.array([1.0]), np.array([OC[self.object_name]["gripper_closed_pos"]])))
            _, _, _, info = self.step(gripper_close_action)

        return retval

    def expl(self, t):
        
        if t == self.traj_length:
            print("Reached end of recursion")

            # Assume that open gripper is always the last waypoint of a manipulation trajectory
            if self.safe(check_arm_collision=False, check_object_dropping=self.check_obj_dropping):
                print("Opening gripper")
                breakpoint()
                gripper_open_action = np.concatenate((np.zeros(9), np.array([1.0]), np.array([OC[self.object_name]["gripper_open_pos"]])))
                _, _, _, info = self.step(gripper_open_action)
            else:
                retval = dict()
                retval["resample"] = True
                retval["stop_expl"] = False
                retval["subtask_success"] = False
                return retval

            retval = self.check_manip_success(select_object_for_mask=True)
            return retval
        
        # check all actions for safety
        if self.collision_model is not None:
            safe_list = []
            self.collision_model.get_set_points(self.object_name)
            for i, action_sample in enumerate(self.sampled_actions): 
                action = action_sample[t] 
            
            # batched prediction
            # inference_batch_size = 5
            # for i in range(0, len(self.sampled_actions), inference_batch_size): 
            #     action_batch = self.sampled_actions[i:i + inference_batch_size, t] 
                              
                # hack. remove later
                if t == self.traj_length - 1:
                    self.arm_collision_th = min(0.9, self.arm_collision_th+0.2)
                
                if self.collision_model.points.shape[1] == 0:
                    print("pcd does not have any points. Skip prediction.")
                    pred_choice, prob = 0.0, 1.0
                else:
                    pred_choice, prob = self.collision_model.predict_collisions(action=action, threshold=self.arm_collision_th, object_name=self.object_name)
                
                # for j in range(pred_choice.shape[0]):
                if pred_choice == 1.0:
                    safe_list.append(False)
                    color = [1.0, 0.0, 0.0, 1.0]
                else:
                    safe_list.append(True)
                    color = [0.0, 1.0, 0.0, 1.0]
                # Only use this if visualizing. This takes time
                # publish_target_pose(self.env, delta_pos=action[3:6], delta_ori=action[6:10], color=color, id=i)

        # check all actions for safety
        if self.grasp_loss_ft_model is not None:
            safe_list = []
            self.grasp_loss_ft_model.get_set_points(self.object_name)
            for i, action_sample in enumerate(self.sampled_actions): 
                action = action_sample[t]               
                if self.grasp_loss_ft_model.points.shape[1] == 0:
                    print("pcd does not have any points. Skip prediction.")
                    pred_choice, prob = 1.0, 1.0
                else:
                    pred_choice, prob = self.grasp_loss_ft_model.predict(action=action, threshold=self.grasp_loss_ft_th, object_name=self.object_name)
                
                # Note pred_choice=1.0 means grasp would be intact and ft would be below threshold
                if pred_choice == 0.0:
                    safe_list.append(False)
                    color = [1.0, 0.0, 0.0, 1.0]
                else:
                    safe_list.append(True)
                    color = [0.0, 1.0, 0.0, 1.0]
                # publish_target_pose(self.env, delta_pos=action[3:6], delta_ori=action[6:10], color=color, id=i)
        
        # if sum(safe_list) < 0.4 * self.num_samples:
        #     print("Seems like an unsafe state. returning all future actions unsafe")
        #     retval = dict()
        #     retval["resample"] = False
        #     retval["stop_expl"] = False
        #     retval["subtask_success"] = False
        #     retval["actions"] = self.final_actions
        #     return retval       
        
        for action_num, action in enumerate(self.sampled_actions):
            if OC[self.object_name]["use_safe_list"] and not safe_list[action_num]:
                continue
            print("--- time step, action: ", t, action[t, 3:6])
            ee_pose_before = self.env.tiago.arms["right"].arm_pose
            joint_pos_before = self.env.tiago.arms["right"].joint_reader.get_most_recent_msg()
            # check by human once. Remove this later
            inp = input("Press Y to execute action and N to skip")
            if inp == 'Y':
                pass
            elif inp == 'B':
                break
            else:
                continue
            # if self.safe(action[t]):
            # Move the robot
            breakpoint()
            _, _, _, info = self.step(action[t])
            self.final_actions.append({'right': action[t][3:10], 'left': None, 'base': None, 'gripper': 'stay'})

            # action not reachable
            if info is not None and info["arm_right"]["joint_goal"] is None:
                print("action not reachable")
                continue

            # TEST: if FT > th, rewind
            if test_ft(self.env):
                retval = dict()
                retval["resample"] = True
                retval["stop_expl"] = False
                retval["subtask_success"] = False
                retval["actions"] = self.final_actions
                return retval 


            # # if grasp is lost, return either to rewind to last grasp or send Failure
            # if not self.env.tiago.gripper["right"].is_grasping():
            #     inp = input("Grasp lost. Press Y to rewind to last grasp or N to send Failure")
            #     if inp == 'Y':
            #         self.env.tiago.gripper["right"].step(OC[self.object_name]["gripper_open_pos"])
            #         retval = dict()
            #         retval["resample"] = False
            #         retval["stop_expl"] = False
            #         retval["subtask_success"] = False
            #         retval["actions"] = self.final_actions
            #         return retval
            #     else:
            #         print("Sending Failure")
            #         retval = dict()
            #         retval["resample"] = False
            #         retval["stop_expl"] = True
            #         retval["subtask_success"] = False
            #         retval["actions"] = self.final_actions
            #         return retval
            
            self.final_actions.append({'right': action[t], 'left': None, 'base': None, 'gripper': 'stay'})
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

        # Optional: If all future samples fail, check if opening gripper here is safe
        if self.check_obj_dropping and self.safe(check_arm_collision=False, check_object_dropping=self.check_obj_dropping):
            print("Opening gripper")
            breakpoint()
            gripper_open_action = np.concatenate((np.zeros(9), np.array([1.0]), np.array([OC[self.object_name]["gripper_open_pos"]])))
            _, _, _, info = self.step(gripper_open_action)
            # check segment success
            retval = self.check_manip_success(select_object_for_mask=True)
            return retval

        
        # If reached here, means all future actions are unsafe, so return to previous waypoint (with rewind)
        # We can change resmaple to false later if we want to resample
        retval = dict()
        retval["resample"] = True
        retval["stop_expl"] = False
        retval["subtask_success"] = False
        retval["actions"] = self.final_actions
        return retval 


def main():
    rospy.init_node('moma_safety_exploration_script')
    set_all_seeds(seed=5)
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
    
    exploration = ManipExploration(
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

    # # hack
    # delta_pos = np.array([0.0, 0.0, -0.10])
    # delta_ori = np.array([0.0, 0.0, 0.0, 1.0])
    # gripper_act = np.array([0.0])
    # delta_act = np.concatenate(
    #     (delta_pos, delta_ori, gripper_act)
    # )
    # action = {'right': None, 'left': None, 'base': None}
    # action["right"] = delta_act
    # env.step(action)

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

        # # perform grasp
        # grasp(env, exec_on_robot=True)
        # print("Grasping done. Continue with exploration.")
        # breakpoint()

        # remove later
        new_prior = []
        for prior_action in prior:
            prior_quat = R.from_rotvec(prior_action[6:9]).as_quat()
            new_prior.append(np.concatenate((prior_action[:6], prior_quat, prior_action[-1:])))
        new_prior = np.array(new_prior)
        new_prior = new_prior[None, ...]
        actions = new_prior.copy()

        # # remove later
        # T_grasp = np.array([
        #     [ 0.954, -0.092,  0.284,  0.808],
        #     [ 0.284, -0.013, -0.959, -0.469],
        #     [ 0.092,  0.996,  0.013,  0.936],
        #     [ 0.   ,  0.,     0.,     1.   ],
        # ])
        # # 2. Move to final pose =========================
        # current_right_ee_pose = env.tiago.arms["right"].arm_pose
        
        # target_right_ee_pos = T_grasp[:3, 3]
        # # Fix the tooltip offset
        # right_tooltip_ee_offset = np.array([-0.28, 0, 0]) #-0.24 for other objects -0.28 for cabinets and drawers
        # right_eef_pose_mat = T_grasp[:3, :3]
        # tooltip_ee_offset_wrt_robot = np.dot(right_eef_pose_mat, right_tooltip_ee_offset)
        # target_right_ee_pos = target_right_ee_pos + tooltip_ee_offset_wrt_robot[:3]        
        # target_right_ee_orn = R.from_matrix(T_grasp[:3, :3]).as_quat()
        # target_right_ee_pose = (target_right_ee_pos, target_right_ee_orn)
        
        # # Obtaining delta pose
        # delta_pos = target_right_ee_pose[0] - current_right_ee_pose[:3]
        # delta_ori = R.from_quat(target_right_ee_pose[1]) * R.from_quat(current_right_ee_pose[3:]).inv()
        # delta_ori = delta_ori.as_quat()
        # gripper_act = np.array([GRIPPER_CLOSED_POS])
        # delta_pose = np.concatenate((delta_pos, delta_ori, gripper_act))
        # print(f"delta_pos: {delta_pos}")
        # print(f"delta_ori: {delta_ori}")
        # action = {'right': delta_pose, 'left': None, 'base': None}

        # breakpoint()
        # obs, reward, done, info = env.step(action, delay_scale_factor=2.0)
        # print("info: ", info["arm_right"])
        # # =======================================================

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