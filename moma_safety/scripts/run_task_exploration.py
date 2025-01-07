import json
import rospy
import pickle
import numpy as np

from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from moma_safety.tiago import RESET_POSES as RP
from moma_safety.tiago.tiago_gym import TiagoGym
from moma_safety.scripts.move_base_vel import move_base_vel
from moma_safety.scripts.nav_exploration import NavExploration
from moma_safety.utils.object_config import object_config as OC
from moma_safety.scripts.manip_exploration import ManipExploration
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

    z_pos_lis = [0.8, 0.7, 0.6, 0.55]
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
        action_inv["base"] = -action["base"]

    return action_inv

class TaskExploration:
    def __init__(self, env):
        self.env = env
        self.grasp_modes = None
        self.grasp_mode_idx = 0
        
        with open('resources/human_tracking_outputs/arpit_pick_and_place_modified.json') as f:
            self.human_tracking_actions = json.load(f)

        self.subtask_actions = dict()
        self.subtask_successes = dict()
        self.subtask_to_replay = dict()
        for k in self.human_tracking_actions.keys():
            self.subtask_successes[int(k)] = False
            self.subtask_to_replay[int(k)] = False


    def execute_action(self, action):
        breakpoint()
        print("action: ", action)
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
        object_name = subtask["description"].split(" ")[-1].lower()
        print("object_name: ", object_name)
        current_subtask_retval = {"subtask_success": False, "rewind": False}
        self.env.tiago.head.write_head_command(OC[object_name]["head_joint_pos"])
        
        # remove this later
        if int(subtask_idx) < 1:
            next_subtask_retval = self.expl(subtask_idx+1)
            return next_subtask_retval
        breakpoint()
        
        # TODO: ensure GPT gives object name as the last word in the description
        if "grasp" in subtask["description"].lower():
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
                # TODO: Handle if grasp fails
                subtask_actions = execute_grasp(self.env, grasp_mode, object_name)
                self.subtask_actions[subtask_idx] = subtask_actions 
                next_subtask_retval = self.expl(subtask_idx+1)
                print("in grasp modes: next_subtask_retval: ", next_subtask_retval)
                should_explore_grasp = next_subtask_retval["rewind"]
        
        else:
            # if there is a pick and the next action is navigate, ignore the pick and just go to a held pose
            if "pick" in subtask["description"].lower() and self.human_tracking_actions[f"{int(subtask_idx)+1}"]["action"] == "navigating":
                if not self.subtask_to_replay[subtask_idx]:
                    # Go to held pose
                    delta_actions = move_arm_to_held_pose(self.env, object_name)
                    self.subtask_actions[subtask_idx] = delta_actions
                    # We assume that this will succeed
                    current_subtask_retval["subtask_success"] = True
                else:
                    self.replay(subtask_idx)
                    self.subtask_to_replay[subtask_idx] = False

            elif subtask["action"].lower() == "navigating":
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
                # Execute the nav exploration
                current_subtask_retval = nav_exploration.expl(t=0)
                print("Navigation Subtask Success: ", current_subtask_retval["subtask_success"])
                self.subtask_actions[subtask_idx] = current_subtask_retval["actions"]
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
                                                num_samples=20,
                                                num_top_samples=3,
                                                epochs=10,
                                                success=False,
                                                noise_params=dict(),
                                                start_idx=0,
                                                object_name=object_name,
                                                arm_collision_th=arm_collision_th
                                            )
                current_subtask_retval = manip_exploration.expl(t=0)
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
                self.rewind(subtask_idx)
                # don't want to replay pick up to held pose 
                if "pick" in subtask["description"].lower() and self.human_tracking_actions[f"{int(subtask_idx)+1}"]["action"] == "navigating":
                    self.subtask_to_replay[subtask_idx] = False
                else:
                    self.subtask_to_replay[subtask_idx] = True
                # current_subtask_retval["rewind"] = True
                return next_subtask_retval

        return next_subtask_retval






# 1. navigate to cup
# 2. grasp cup
# if there is a nav in the next step, go to a held pose, else follow prior
# 3. navigate to shelf
# 4. place cup on shelf
# 5. If fails, rewind to last grasp pose and try new mode


def main():
    rospy.init_node('moma_safety_exploration_script')
    env = TiagoGym(
        frequency=10,
        right_arm_enabled=True,
        left_arm_enabled=False,
        right_gripper_type='robotiq2F-140',
        left_gripper_type='robotiq2F-85',
        base_enabled=True,
        torso_enabled=False,
    )

    # reset to a start pose
    obj_name = "cup"
    # move_arm_to_held_pose(env, obj_name)
    # breakpoint()
    # open gripper. 1 is open and 0 is close
    env.tiago.gripper['right'].step(OC[obj_name]["gripper_open_pos"])
    rospy.sleep(2)
    # reset_joint_pos = RP.PREGRASP_R_H
    reset_joint_pos = RP.PREGRASP_HIGH
    reset_joint_pos["right"][-1] = OC[obj_name]["gripper_open_pos"]
    env.reset(reset_arms=True, reset_pose=reset_joint_pos, allowed_delay_scale=6.0)

    task_exploration = TaskExploration(env)
    retval = task_exploration.expl(subtask_idx=0)
    print("retval: ", retval)

if __name__ == '__main__':
    main()