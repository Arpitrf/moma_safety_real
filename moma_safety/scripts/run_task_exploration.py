import os
import json
import rospy
import pickle
import imageio
import numpy as np

from copy import deepcopy
from datetime import datetime
from argparse import ArgumentParser
from scipy.spatial.transform import Rotation as R
from moma_safety.tiago import RESET_POSES as RP
from moma_safety.tiago.tiago_gym import TiagoGym
from moma_safety.scripts.move_base_vel import move_base_vel
from moma_safety.scripts.nav_exploration import NavExploration
from moma_safety.utils.object_config import object_config as OC
from moma_safety.scripts.manip_exploration import ManipExploration
from moma_safety.tiago.utils.camera_utils import Camera, RecordVideo
from moma_safety.grasp import obtain_mask, get_pcd, obtain_grasp_modes, execute_grasp


# TODO: move this somewhere else later
num_samples = 20
num_top_samples = 3
epochs = 10
success = False
start_idx = 0
noise_params = dict()

def move_arm_to_held_pose(env, obj_name):
    final_actions = []
    current_right_ee_pose = env.tiago.arms["right"].arm_pose

    z_pos_lis = [0.8, 0.7, 0.65, 0.60, 0.55]
    info = dict()
    info["arm_right"] = {"joint_goal": None}
    tries = 0
    while info["arm_right"]["joint_goal"] is None and tries < len(z_pos_lis):
        target_right_finger_pos = np.array([0.25, -0.5, z_pos_lis[tries]])
        target_right_ee_orn = current_right_ee_pose[3:]

        right_tooltip_ee_offset = np.array([-0.24, 0, 0])
        right_eef_pose_mat = R.from_quat(current_right_ee_pose[3:]).as_matrix()
        tooltip_ee_offset_wrt_robot = np.dot(right_eef_pose_mat, right_tooltip_ee_offset)

        target_right_ee_pos = target_right_finger_pos + tooltip_ee_offset_wrt_robot[:3]
        target_right_ee_pose = (target_right_ee_pos, target_right_ee_orn)
        # breakpoint()

        delta_pos = target_right_ee_pose[0] - current_right_ee_pose[:3]
        delta_ori = R.from_quat(target_right_ee_pose[1]) * R.from_quat(current_right_ee_pose[3:]).inv()
        delta_ori = delta_ori.as_quat()
        gripper_act = np.array([OC[obj_name]["gripper_closed_pos"]])
        delta_pose = np.concatenate((delta_pos, delta_ori, gripper_act))
        print(f"delta_pos: {delta_pos}")
        print(f"delta_ori: {delta_ori}")
        action = {'right': delta_pose, 'left': None, 'base': None, 'gripper': 'stay'}
        breakpoint()
        obs, reward, done, info = env.step(action, delay_scale_factor=4.0)
        print("info: ", info["arm_right"])
        tries += 1

    final_actions.append(action)
    return final_actions


def move_arm_to_held_pose2(env, obj_name):
    final_actions = []
    current_right_ee_pose = env.tiago.arms["right"].arm_pose

    target_right_ee_pos = RP.PREGRASP2_R_H_EE_POSE["right"][:3]
    target_right_ee_orn = current_right_ee_pose[3:]    
    # target_right_ee_orn = RP.PREGRASP_R_H_EE_POSE["right"][3:]
    target_right_ee_pose = (target_right_ee_pos, target_right_ee_orn)
    original_target_right_ee_pose = deepcopy(target_right_ee_pose)
    # In case pregrasp pose is not reachable, find a pose near it
    info = dict()
    info["arm_right"] = {"joint_goal": None}
    tries = 0
    while info["arm_right"]["joint_goal"] is None and tries < 20:
        delta_pos = target_right_ee_pose[0] - current_right_ee_pose[:3]
        delta_ori = R.from_quat(target_right_ee_pose[1]) * R.from_quat(current_right_ee_pose[3:]).inv()
        delta_ori = delta_ori.as_quat()
        gripper_act = np.array([OC[obj_name]["gripper_closed_pos"]])
        delta_pose = np.concatenate((delta_pos, delta_ori, gripper_act))
        print(f"delta_pos: {delta_pos}")
        print(f"delta_ori: {delta_ori}")
        action = {'right': delta_pose, 'left': None, 'base': None, 'gripper': 'stay'}
        breakpoint()
        obs, reward, done, info = env.step(action, delay_scale_factor=4.0)
        print("info: ", info["arm_right"])
        # add noise to the pregrasp position
        noise = np.random.normal(0, 0.05, 2)
        noise = np.clip(noise, -0.1, 0.1)
        noise = np.concatenate((noise, [0]))
        target_right_ee_pose = (original_target_right_ee_pose[0] + noise, original_target_right_ee_pose[1])
        tries += 1
    print("tries: ", tries)
    final_actions.append(action)

    # # TODO: Move arm to navigate pose
    # print("MOVE ARM TO NAVIGATE POSE")
    # target_right_ee_pos = RP.NAV_R_H_EE_POSE["right"][:3]
    # target_right_ee_orn = current_right_ee_pose[3:]    
    # # target_right_ee_orn = RP.PREGRASP_R_H_EE_POSE["right"][3:]
    # target_right_ee_pose = (target_right_ee_pos, target_right_ee_orn)
    # original_target_right_ee_pose = deepcopy(target_right_ee_pose)
    # # In case pregrasp pose is not reachable, find a pose near it
    # info = dict()
    # info["arm_right"] = {"joint_goal": None}
    # tries = 0
    # while info["arm_right"]["joint_goal"] is None and tries < 20:
    #     delta_pos = target_right_ee_pose[0] - current_right_ee_pose[:3]
    #     delta_ori = R.from_quat(target_right_ee_pose[1]) * R.from_quat(current_right_ee_pose[3:]).inv()
    #     delta_ori = delta_ori.as_quat()
    #     gripper_act = np.array([OC[obj_name]["gripper_closed_pos"]])
    #     delta_pose = np.concatenate((delta_pos, delta_ori, gripper_act))
    #     print(f"delta_pos: {delta_pos}")
    #     print(f"delta_ori: {delta_ori}")
    #     action = {'right': delta_pose, 'left': None, 'base': None}
    #     breakpoint()
    #     obs, reward, done, info = env.step(action, delay_scale_factor=4.0)
    #     print("info: ", info["arm_right"])
    #     # add noise to the pregrasp position
    #     noise = np.random.normal(0, 0.05, 3)
    #     noise = np.clip(noise, -0.1, 0.1)
    #     target_right_ee_pose = (original_target_right_ee_pose[0] + noise, original_target_right_ee_pose[1])
    #     tries += 1
    # print("tries: ", tries)

    return final_actions

def normalize_quaternion(q):
    """
    Normalize a quaternion to ensure it has unit length.
    :param q: A quaternion as a tuple or list (x, y, z, w)
    :return: Normalized quaternion (x, y, z, w)
    """
    norm = np.linalg.norm(q)
    return tuple(q_i / norm for q_i in q)

def opposite_rotation(q):
    """
    Compute the opposite rotation (conjugate) of a quaternion in (x, y, z, w) format.
    :param q: A quaternion as a tuple or list (x, y, z, w)
    :return: Conjugate quaternion (x, y, z, w)
    """
    # Ensure the quaternion is normalized
    q = normalize_quaternion(q)
    
    # Compute the conjugate
    x, y, z, w = q
    return (-x, -y, -z, w)

def get_inverse_action(action):
    action_inv = deepcopy(action)
    if action["right"] is not None:
        action_inv["right"][:3] = -action["right"][:3]
        # TODO: confirm this is correct
        action_inv["right"][3:7] = opposite_rotation(action["right"][3:7])
        if action["gripper"] == "open":
            # close gripper
            if "object_name" in action.keys(): action_inv["right"][-1] = OC[action["object_name"]]["gripper_closed_pos"] 
            else: action_inv["right"][-1] = 0
        elif action["gripper"] == "close":
            # open gripper
            if "object_name" in action.keys(): action_inv["right"][-1] = OC[action["object_name"]]["gripper_open_pos"]
            else: action_inv["right"][-1] = 1
    if action["base"] is not None:
        yaw = action["base"][2]
        T_curr_frame_wrt_prev_frame = np.array([
            [np.cos(yaw), -np.sin(yaw)], 
            [np.sin(yaw), np.cos(yaw)],
        ]) 
        action_inv["base"][:2] = np.dot(np.linalg.inv(T_curr_frame_wrt_prev_frame), -action["base"][:2])
        action_inv["base"][2] = -action["base"][2]
        # action_inv["base"] = -action["base"]

    return action_inv

class TaskExploration:
    def __init__(self, env, recorder=None, video_save_folder=None):
        self.env = env
        self.grasp_modes = None
        self.grasp_mode_idx = 0
        self.recorder = recorder
        self.video_save_folder = video_save_folder
        
        with open('resources/human_tracking_outputs/store_in_shelf_1.json') as f:
            self.human_tracking_actions = json.load(f)

        self.subtask_actions = dict()
        self.subtask_successes = dict()
        self.subtask_to_replay = dict()
        for k in self.human_tracking_actions.keys():
            self.subtask_successes[int(k)] = False
            self.subtask_to_replay[int(k)] = False


    def execute_action(self, action):
        print("action: ", action)
        breakpoint()
        if action["right"] is not None:
            obs, reward, done, info = self.env.step(action, delay_scale_factor=4.0)
            print("info: ", info["arm_right"])
        if action["base"] is not None:
            move_base_vel(self.env, action["base"])
    
    def replay(self, subtask_idx):
        for action in self.subtask_actions[subtask_idx]:
            self.execute_action(action)
    
    def rewind(self, subtask_idx):
        # with open('resources/tmp_outputs/actions_until_last_grasp.pkl', 'wb') as f:
        #     pickle.dump(self.actions_until_last_grasp, f)
        # with open('resources/tmp_outputs/actions_grasp.pkl', 'wb') as f:
        #     pickle.dump(self.actions_grasp, f)
        # reverse the action sequence for the give subtask
        for action in self.subtask_actions[subtask_idx][::-1]:
            action_inv = get_inverse_action(action)
            self.execute_action(action_inv)


    def expl(self, subtask_idx):
        
        # End of recursion
        if subtask_idx == len(self.human_tracking_actions.keys()):
            # Task has been successfully completed
            return {"task_success": True, "rewind": False}
        
        subtask = self.human_tracking_actions[f"{subtask_idx}"]
        print("subtask_idx, subtask: ", subtask_idx, subtask["description"])
        # FIXME: E.g. "place cup in shelf" will give shelf as the object
        object_name = subtask["object_name"]
        print("object_name: ", object_name)
        current_subtask_retval = {"subtask_success": False, "rewind": False}
        self.env.tiago.head.write_head_command(OC[object_name]["head_joint_pos"])
        
        # remove this later
        if int(subtask_idx) < 2:
            next_subtask_retval = self.expl(subtask_idx+1)
            return next_subtask_retval
        breakpoint()
        
        # TODO: ensure GPT gives object name as the last word in the description
        if "grasp" in subtask["description"].lower():

            # TODO: move to pregrasp pose
            reset_joint_pos = RP.PREGRASP_HIGH
            reset_joint_pos["right"][-1] = OC[object_name]["gripper_open_pos"]
            self.env.reset(reset_arms=True, reset_pose=reset_joint_pos, delay_scale_factor=4.0)
            
            self.grasp_mode_idx = 0

            print("obtaining mask")
            mask = obtain_mask(self.env, select_object=True)
            pcd = get_pcd(self.env, mask)
            # Obtain grasp modes
            self.grasp_modes = obtain_grasp_modes(pcd, self.env, object_name, select_mode=True)
            should_explore_grasp = True

            # We explore grasp mode until we have one an until the future subtasks ask for it (if all subtasks succees, they won't ask for it) 
            while self.grasp_mode_idx < len(self.grasp_modes) and should_explore_grasp:

                # 1. Undo grasp (ignore for first grasp)
                if self.grasp_mode_idx > 0:
                    print(f"Asking subtask: {subtask['description']} to rewind.")
                    self.rewind(subtask_idx)

                # 2. Try the next grasp mode
                grasp_mode = self.grasp_modes[self.grasp_mode_idx]
                self.grasp_mode_idx += 1
                
                if self.recorder is not None:
                    self.recorder.start_recording()
                    print("Start recording")
                
                # TODO: Handle if grasp fails
                retval = execute_grasp(self.env, grasp_mode, object_name)
                
                if self.recorder is not None:
                    self.recorder.save_video(save_folder=self.video_save_folder)
                    self.recorder.stop_recording()
                
                self.subtask_actions[subtask_idx] = retval["subtask_actions"] 
                if not retval["grasp_success"]:
                    # Make sure we go back to joint position controller
                    start_controllers = ['arm_right_controller', 'arm_left_controller']
                    stop_controllers = ['arm_right_impedance_controller', 'arm_left_impedance_controller']
                    self.env.tiago.arms["right"].switch_controller(start_controllers, stop_controllers)
                    continue 
                next_subtask_retval = self.expl(subtask_idx+1)
                print("in grasp modes: next_subtask_retval: ", next_subtask_retval)
                should_explore_grasp = next_subtask_retval["rewind"]
        
        else:
            # if there is a pick and the next action is navigate, ignore the pick and just go to a held pose
            if "pick" in subtask["description"].lower() and self.human_tracking_actions[f"{int(subtask_idx)+1}"]["action"] == "navigating":
                if not self.subtask_to_replay[subtask_idx]:
                    if self.recorder is not None:
                        self.recorder.start_recording()
                        print("Start recording")
                    
                    # Go to held pose
                    delta_actions = move_arm_to_held_pose(self.env, object_name)

                    if self.recorder is not None:
                        self.recorder.save_video(save_folder=self.video_save_folder)
                        self.recorder.stop_recording()

                    self.subtask_actions[subtask_idx] = delta_actions
                    # We assume that this will succeed
                    current_subtask_retval["subtask_success"] = True
                else:
                    self.replay(subtask_idx)
                    self.subtask_to_replay[subtask_idx] = False

            elif subtask["action"].lower() == "navigating":
                self.subtask_actions[subtask_idx] = []
                # if there was a grasp and the current action is navigate, go to a held pose
                if f"{int(subtask_idx)-1}" in self.human_tracking_actions and "grasp" in self.human_tracking_actions[f"{int(subtask_idx)-1}"]["description"].lower():
                    if self.recorder is not None:
                        self.recorder.start_recording()
                        print("Start recording")
                    # Go to held pose
                    delta_actions = move_arm_to_held_pose(self.env, object_name)
                    if self.recorder is not None:
                        self.recorder.save_video(save_folder=self.video_save_folder)
                        self.recorder.stop_recording()
                    self.subtask_actions[subtask_idx] += delta_actions
                
                # if not self.subtask_to_replay[subtask_idx]:
                prior = np.array(subtask["robot_actions"])                
                nav_exploration = NavExploration(self.env,
                                                prior,
                                                num_samples=20,
                                                num_top_samples=3,
                                                epochs=10,
                                                success=False,
                                                noise_params=dict(),
                                                start_idx=0,
                                                object_name=object_name
                                            )
                nav_exploration.nav_success_det.set_object_name(object_name)
                
                if self.recorder is not None:
                    self.recorder.start_recording()
                    print("Start recording")


                # Execute the nav exploration
                current_subtask_retval = nav_exploration.expl(t=0)

                if self.recorder is not None:
                    self.recorder.save_video(save_folder=self.video_save_folder)
                    self.recorder.stop_recording()

                print("Navigation Subtask Success: ", current_subtask_retval["subtask_success"])
                self.subtask_actions[subtask_idx] += current_subtask_retval["actions"]
                # Assume nav succeeds on reaply
                # else:
                #     self.replay(subtask_idx)
                #     self.subtask_to_replay[subtask_idx] = False
                #     current_subtask_retval = {"subtask_success": True, "rewind": False}
                # # remove later
                # current_subtask_retval = {"subtask_success": False}

            else:
                # this will be the manipulation exploration
                prior = np.array(subtask["robot_actions"])
                # hack: remove later
                if self.grasp_mode_idx == 1:
                    arm_collision_th = 0.2
                else:
                    arm_collision_th = 0.5
                manip_exploration = ManipExploration(self.env,
                                                prior,
                                                num_samples=OC[object_name]["manip_num_samples"],
                                                num_top_samples=3,
                                                epochs=10,
                                                success=False,
                                                noise_params=dict(),
                                                start_idx=0,
                                                object_name=object_name,
                                                segment_description=subtask["description"],
                                                check_arm_collision=OC[object_name]["check_arm_collision"],
                                                check_obj_dropping=OC[object_name]["check_obj_dropping"],
                                                check_grasp_loss=OC[object_name]["check_grasp_loss"],
                                                check_ft=OC[object_name]["check_ft"],
                                                arm_collision_th=arm_collision_th,
                                                # use_impedance_controller=OC[object_name]["use_impedance_controller"]
                                            )
                
                if self.recorder is not None:
                    self.recorder.start_recording()
                    print("Start recording")

                current_subtask_retval = manip_exploration.expl(t=0)
                
                if self.recorder is not None:
                    self.recorder.save_video(save_folder=self.video_save_folder)
                    self.recorder.stop_recording()
                
                # reset head pose
                self.env.tiago.head.write_head_command(OC[object_name]["head_joint_pos"])
                print("Manipulation Subtask Success: ", current_subtask_retval["subtask_success"])
                self.subtask_actions[subtask_idx] = current_subtask_retval["actions"]
                # # remove later
                # current_subtask_retval = {"subtask_success": False}

            
            # If subtask succeeds, move to next subtask
            if current_subtask_retval["subtask_success"]:
                next_subtask_retval = self.expl(subtask_idx+1)
            # If with the current mode, the subtask fails, rewind and try the next mode
            else:
                print(f"Subtask: {subtask['description']} failed")
                current_subtask_retval["rewind"] = True
                return current_subtask_retval
            
            # If this subtask is asked to rewind 
            if next_subtask_retval["rewind"]:
                print(f"Asking subtask: {subtask['description']} to rewind.")
                breakpoint()

                if self.recorder is not None:
                    self.recorder.start_recording()
                    print("Start recording")

                self.rewind(subtask_idx)

                if self.recorder is not None:
                    self.recorder.save_video(save_folder=self.video_save_folder)
                    self.recorder.stop_recording()

                # don't want to replay pick up to held pose 
                self.subtask_to_replay[subtask_idx] = False
                # if "pick" in subtask["description"].lower() and self.human_tracking_actions[f"{int(subtask_idx)+1}"]["action"] == "navigating":
                #     self.subtask_to_replay[subtask_idx] = False
                # else:
                #     self.subtask_to_replay[subtask_idx] = True
                # current_subtask_retval["rewind"] = True
                return next_subtask_retval

        return next_subtask_retval

def config_parser(parser=None):
    if parser is None:
        parser = ArgumentParser("moma_safety_exploration")
    parser.add_argument('--f_name', type=str)
    parser.add_argument("--record", action="store_true", default=False)
    return parser


def main():
    rospy.init_node('moma_safety_exploration_script')
    args = config_parser().parse_args()
    
    env = TiagoGym(
        frequency=10,
        right_arm_enabled=True,
        left_arm_enabled=False,
        right_gripper_type='robotiq2F-140',
        left_gripper_type=None,
        base_enabled=True,
        torso_enabled=False,
    )

    # For saving videos
    current_time = datetime.now()
    date_folder = current_time.strftime("%Y-%m-%d")
    time_folder = current_time.strftime("%H-%M-%S")
    path = os.path.join(date_folder, time_folder)
    video_save_folder = f"ego_videos/{path}"
    # folder_path = "ego_videos"
    os.makedirs(video_save_folder, exist_ok=True)
    
    # Start recorder
    record = args.record
    ego_cam = None
    recorder = None
    if record:
        ego_cam = Camera(img_topic="/xtion/rgb/image_raw", depth_topic="/xtion/depth/image_raw")
        recorder = RecordVideo(camera_interface_ego=ego_cam)
        recorder.setup_recording()
    # # reset to a start pose
    # obj_name = "pringles"
    # # move_arm_to_held_pose(env, obj_name)
    # # breakpoint()
    # # open gripper. 1 is open and 0 is close
    # env.tiago.gripper['right'].step(OC[obj_name]["gripper_open_pos"])
    # rospy.sleep(2)
    # # reset_joint_pos = RP.PREGRASP_R_H
    # reset_joint_pos = RP.PREGRASP_HIGH
    # reset_joint_pos["right"][-1] = OC[obj_name]["gripper_open_pos"]
    # env.reset(reset_arms=True, reset_pose=reset_joint_pos, allowed_delay_scale=6.0)

    task_exploration = TaskExploration(env, recorder=recorder, video_save_folder=video_save_folder)
    retval = task_exploration.expl(subtask_idx=0)
    print("retval: ", retval)

if __name__ == '__main__':
    main()