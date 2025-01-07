import os
import cv2
import json
import time
import torch
import random
import rospy
import pickle
import imageio
import datetime
import open3d as o3d
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
import matplotlib
matplotlib.use("agg", force=True)
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn_extra.cluster import KMedoids
# from scipy.linalg import logm, norm
from scipy.spatial.transform import Rotation as R

from telemoma.human_interface.teleop_policy import TeleopPolicy
from importlib.machinery import SourceFileLoader
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, load_image, predict
from moma_safety.tiago.tiago_gym import TiagoGym
from moma_safety.grasping.grasp_selector import GraspSelector
from moma_safety.grasping.grasp_pose_generator import translateFrameNegativeZ
from moma_safety.tiago import RESET_POSES as RP
from moma_safety.tiago.utils.transformations import quat_diff
from moma_safety.tiago.utils.camera_utils import Camera, RecordVideo
from moma_safety.safety_run import start_gravity_compensation, end_gravity_compensation
from moma_safety.utils.env_variables import *
from moma_safety.utils.object_config import object_config as OC 

"""
Hyper parameters
"""
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("grounded_sam_outputs/grounded_sam2_local_demo")
DUMP_JSON_RESULTS = True

# 0.5 for drawer and cabinets
# GRIPPER_OPEN_POS = 1.0
# GRIPPER_CLOSED_POS = 0.0

clicked_points = []

def check_grasp_reachability(env, grasp, obj_name):
    T_grasp = transform_proposed_grasp_to_executable_grasp(grasp)
    # T_grasp is wrt robot base but we need to make it wrt ik_base_link
    ik_base_link = env.tiago.arms["right"].ik_base_link
    pos_R_wrt_ik_base_link, orn_R_wrt_ik_base_link = env.tiago.arms["right"].arm_reader.get_transform(target_link='/base_footprint', base_link=f'/{ik_base_link}')
    T_R_wrt_ik_base_link = np.eye(4)
    T_R_wrt_ik_base_link[:3, :3] = R.from_quat(orn_R_wrt_ik_base_link).as_matrix()
    T_R_wrt_ik_base_link[:3, 3] = pos_R_wrt_ik_base_link
    T_grasp = np.dot(T_R_wrt_ik_base_link, T_grasp)

    target_right_ee_pos = T_grasp[:3, 3]
    # Fix the tooltip offset
    right_tooltip_ee_offset = OC[obj_name]["right_tooltip_ee_offset"] #-0.24 for other objects -0.28 for cabinets and drawers
    right_eef_pose_mat = T_grasp[:3, :3]
    tooltip_ee_offset_wrt_robot = np.dot(right_eef_pose_mat, right_tooltip_ee_offset)
    target_right_ee_pos = target_right_ee_pos + tooltip_ee_offset_wrt_robot[:3]    
    target_right_ee_orn = R.from_matrix(T_grasp[:3, :3]).as_quat()

    joint_goal, _ = env.tiago.arms["right"].find_ik(target_right_ee_pos, target_right_ee_orn)
    if joint_goal is None:
        return False
    return True

def teleop(env):
    # ==================================== Telemoma ====================================
    teleop_config = SourceFileLoader('conf', '/home/pal/arpit/telemoma/telemoma/configs/only_spacemouse.py').load_module().teleop_config
    teleop = TeleopPolicy(teleop_config)
    teleop.start()

    def shutdown_helper():
        teleop.stop()
    rospy.on_shutdown(shutdown_helper)

    i = 0
    obs = env._observation()
    gripper_should_be_closed = True
    while not rospy.is_shutdown():
        action = teleop.get_action(obs) # get_random_action()
        # fix: euler to quat
        for side in ['right', 'left']:
            orn = R.from_euler("xyz", action.right[3:6]).as_quat()
            action[side][:-1] = action[side][:-1] * 0.05
            action[side] = np.concatenate((action[side][:3], orn, action[side][6:]))
            if gripper_should_be_closed:
                if action.right[-1] == 0:
                    gripper_should_be_closed = False
                    action.right[-1] = 1
                else:
                    action.right[-1] = 0 # 0 for close
        buttons = action.extra['buttons'] if 'buttons' in action.extra else {}

        if buttons.get('A', False) or buttons.get('B', False):
            break

        obs, _, _, _ = env.step(action, teleop=True)
        # if i % 4 == 0:
            # print(obs['base'][:3])
        i += 1

    shutdown_helper()
    # ==================================== Telemoma ====================================

# def sample_near_quaternion(base_quaternion, epsilon=0.1, num_samples=10):
#     samples = []
#     for _ in range(num_samples):
#         # Random axis
#         axis = np.random.normal(size=3)
#         axis /= np.linalg.norm(axis)
        
#         # Small random angle
#         angle = np.random.uniform(-epsilon, epsilon)
        
#         # Perturbation quaternion
#         perturbation = R.from_rotvec(angle * axis).as_quat()
        
#         # Combine with base quaternion
#         base_rotation = R.from_quat(base_quaternion)
#         new_rotation = R.from_quat(perturbation) * base_rotation
        
#         samples.append(new_rotation.as_quat())
#     return samples

def rotation_distance(R_a, R_b):
    """
    Calculate the geodesic distance between two rotation matrices.
    
    Parameters:
        R_a (numpy.ndarray): First rotation matrix (3x3).
        R_b (numpy.ndarray): Second rotation matrix (3x3).
    
    Returns:
        float: The rotation distance (angle in radians).
    """
    R_diff = np.dot(R_a, R_b.T)
    trace = np.trace(R_diff)
    theta = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))  # Clip for numerical stability
    return theta

def closest_rotation_matrix(current_rotation, target_rotation):
    """
    Finds the closer rotation matrix (between target and its Z-axis symmetric equivalent) to the current rotation.
    
    Args:
        current_rotation (np.ndarray): Current 3x3 rotation matrix (R_current).
        target_rotation (np.ndarray): Target 3x3 rotation matrix (R_target).
    
    Returns:
        np.ndarray: The closer rotation matrix to the current rotation.
    """
    # Define the 180-degree rotation about the Z-axis
    theta = np.pi
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    
    # Compute the symmetric equivalent of the target rotation
    flipped_rotation = target_rotation @ R_x

    # Compute the Frobenius norm distances
    # distance_to_target = np.linalg.norm(current_rotation - target_rotation, ord='fro')
    # distance_to_flipped = np.linalg.norm(current_rotation - flipped_rotation, ord='fro')

    distance_to_target = rotation_distance(current_rotation, target_rotation)
    distance_to_flipped = rotation_distance(current_rotation, flipped_rotation)

    print("distance_to_target, distance_to_flipped: ", distance_to_target, distance_to_flipped)
    
    # Choose the closer rotation matrix
    if distance_to_target <= distance_to_flipped:
        print("Returning target rotation.")
        return target_rotation
    else:
        print("Returning flipped rotation.")
        return flipped_rotation
    
def transform_proposed_grasp_to_executable_grasp(grasp, obj_name=None):
    # T_grasp = np.array([
    #     [ 9.58720158e-01,  1.61368108e-02,  2.83893048e-01,  1.44742620e+00],
    #     [ 2.83857898e-01,  4.47660806e-03, -9.58855909e-01, -6.43027075e-01],
    #     [-1.67437542e-02,  9.99859772e-01, -2.88746793e-04,  7.39621336e-01],
    #     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
    # ])

    # T_grasp = np.array([
    #     [-0.89625254,  0.35218421,  0.26962505,  1.45187837],
    #     [-0.26385715,  0.06528135, -0.96235012, -0.64318448],
    #     [-0.356526  , -0.93365124,  0.03441775,  0.75985456],
    #     [ 0.        ,  0.,          0.,          1.        ],
    # ])

    # In sim: robotiq_eef_to_grasp_proposals = np.array([
    #     [-1, 0, 0, 0.0],
    #     [0, -1, 0, 0.0],
    #     [0, 0, 1, 0.0],
    #     [0, 0, 0, 1.0],
    # ])
    # Real Tiago has a different frame for ee
    # This was working with can
    robotiq_eef_to_grasp_proposals_1 = np.array([
        [0, 0, 1, 0.0],
        [0, 1, 0, 0.0],
        [-1, 0, 0, 0.0],
        [0, 0, 0, 1.0],
    ])
    # This was working with pringles
    robotiq_eef_to_grasp_proposals_2 = np.array([
        [0, 0, -1, 0.0],
        [0, -1, 0, 0.0],
        [-1, 0, 0, 0.0],
        [0, 0, 0, 1.0],
    ])
    T_grasp = np.array(grasp)
    T_grasp_original = T_grasp.copy()
    
    # Version 1: Give IK solver both target poses and IK solver would discard the one which has 
    # the opposite orientation. Not using this, cuase this might not always work.
    # T_grasp_1 = T_grasp @ robotiq_eef_to_grasp_proposals_1
    # T_grasp_2 = T_grasp @ robotiq_eef_to_grasp_proposals_2
    # T_grasps = [T_grasp_1, T_grasp_2]
    
    # Version 2 (fix for z axis)
    T_grasp = T_grasp @ robotiq_eef_to_grasp_proposals_1
    if abs(T_grasp[2, 0]) > 0.7:
        # print("Checking z axis.")
        T_grasp_1 = T_grasp_original @ robotiq_eef_to_grasp_proposals_1
        T_grasp_2 = T_grasp_original @ robotiq_eef_to_grasp_proposals_2
        # discard the one where the gripper fingers are pointing upwards
        # print("T_grasp_1, T_grasp_2: ", T_grasp_1[2, 0], T_grasp_2[2, 0])
        if T_grasp_1[2, 0] > 0.0:
            T_grasp = T_grasp_1
        else:
            T_grasp = T_grasp_2

    # fix for gello
    if obj_name is not None and obj_name == "jello": 
        # if red component is along +- z axis, rotate by 90 degrees about x axis and add some +z offset
        print("red component on +- z axes: ", abs(T_grasp[2, 0]))
        if abs(T_grasp[2, 2]) > 0.8:
            print("Rotating!!")
            theta = -1.57
            R_x = np.array([
                    [1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]
                ])
            T_grasp[:3, :3] = R_x @ T_grasp[:3, :3]
            breakpoint()
        T_grasp[:3, 3] += np.array([0, 0, 0.04])

    if obj_name is not None and obj_name == "can":
        T_grasp[:3, 3] += np.array([0, 0, 0.02])

    # print("----------- Original T_grasp: ", T_grasp)
    
    # To fix the problem with curvature of object (pringles, etc.) and cabinet etc.
    # What this is doing is that converting "inside-out" grasps into slightly outside-in". But, we never want to do this in case the gripper is pointing downwards
    # Maybe don't use with fridges and drawers (test it)
    # if obj_name != "pot":
    print("22T_grasp: ", T_grasp)
    if T_grasp[1, 0] < -0.2 and abs(T_grasp[2, 0]) < 0.5:
        # # new version
        angle = R.from_matrix(T_grasp[:3, :3]).as_euler("xyz")[2]
        print("angle: ", angle)
        R_z =  np.array([
            [np.cos(-1.2 * angle), -np.sin(-1.2 * angle), 0], # originally it was 2
            [np.sin(-1.2 * angle), np.cos(-1.2 * angle), 0],
            [0, 0, 1]
        ])
        T_grasp[:3, :3] = R_z @ T_grasp[:3, :3]
        
        # # old version
        # # rotate T_grasp[:3, :3] about z axis by 45 degrees
        # # R_z_45 = np.array([
        # #     [0.70710678, -0.70710678, 0],
        # #     [0.70710678, 0.70710678, 0],
        # #     [0, 0, 1]
        # # ])
        # angle = 0.785
        # R_z =  np.array([
        #     [np.cos(angle), -np.sin(angle), 0],
        #     [np.sin(angle), np.cos(angle), 0],
        #     [0, 0, 1]
        # ])
        # T_grasp[:3, :3] = R_z @ T_grasp[:3, :3]

    # print("----------- Final T_grasp: ", T_grasp)
    return T_grasp

def execute_grasp(env, grasp, obj_name, switch_to_impedance_controller=False):

    final_actions = []
    T_grasp = transform_proposed_grasp_to_executable_grasp(grasp, obj_name)

    # Finding closer orn
    current_right_ee_pose = env.tiago.arms["right"].arm_pose
    current_right_ee_orn = current_right_ee_pose[3:]
    R_current = R.from_quat(current_right_ee_orn).as_matrix()
    R_target = T_grasp[:3, :3]
    closer_rotation = closest_rotation_matrix(R_current, R_target)
    T_grasp[:3, :3] = closer_rotation
    
    # For testing: move to target grasp orientaiton at current position =========================
    current_right_ee_pose = env.tiago.arms["right"].arm_pose
    target_right_ee_orn = R.from_matrix(T_grasp[:3, :3]).as_quat()
    # # Find the closer target orientation
    # current_right_ee_orn = current_right_ee_pose[3:]
    # R_current = R.from_quat(current_right_ee_orn).as_matrix()
    # R_target = R.from_quat(target_right_ee_orn).as_matrix()
    # closer_rotation = closest_rotation_matrix(R_current, R_target)
    # target_right_ee_orn = R.from_matrix(closer_rotation).as_quat()
    target_right_ee_pose = (current_right_ee_pose[:3], target_right_ee_orn)
    
    # Obtaining delta pose
    delta_pos = target_right_ee_pose[0] - current_right_ee_pose[:3]
    delta_ori = R.from_quat(target_right_ee_pose[1]) * R.from_quat(current_right_ee_pose[3:]).inv()
    delta_ori = delta_ori.as_quat()
    gripper_act = np.array([OC[obj_name]["gripper_open_pos"]])
    delta_pose = np.concatenate((delta_pos, delta_ori, gripper_act))
    print(f"delta_pos: {delta_pos}")
    print(f"delta_ori: {delta_ori}")
    action = {'right': delta_pose, 'left': None, 'base': None}
    print("Press c to continue...")
    breakpoint()
    obs, reward, done, info = env.step(action)
    print("info: ", info["arm_right"])
    # =======================================================

    # # check if target pose is ik solvable
    # joint_goal, duration_scale = env.tiago.arms["right"].find_ik(T_grasp[:3, 3], R.from_matrix(T_grasp[:3, :3]).as_quat())
    # if joint_goal is None:
    #     print("IK not solvable for the target pose.")
    #     return
    #     breakpoint()
    #     base_quaternion = R.from_matrix(T_grasp[:3, :3]).as_quat()
    #     samples = sample_near_quaternion(base_quaternion, epsilon=0.5, num_samples=1000)
    #     print(samples)
    #     for i, sample in enumerate(samples):
    #         print(f"Trying sample {i+1}...")
    #         try_grasp_pose = (T_grasp[:3, 3], sample)
    #         joint_goal, duration_scale = env.tiago.arms["right"].find_ik(try_grasp_pose[0], try_grasp_pose[1])
    #         if joint_goal is not None:
    #             T_grasp[:3, :3] = R.from_quat(sample).as_matrix()
    #             break
    
    # 1. Move to pregrasp pose =========================
    current_right_ee_pose = env.tiago.arms["right"].arm_pose
    target_right_ee_pos = T_grasp[:3, 3]
    # Fix the tooltip offset
    right_tooltip_ee_offset = OC[obj_name]["right_tooltip_ee_offset_pregrasp"]
    right_eef_pose_mat = T_grasp[:3, :3]
    tooltip_ee_offset_wrt_robot = np.dot(right_eef_pose_mat, right_tooltip_ee_offset)
    target_right_ee_pos = target_right_ee_pos + tooltip_ee_offset_wrt_robot[:3]
    
    target_right_ee_orn = R.from_matrix(T_grasp[:3, :3]).as_quat()
    # # Find the closer target orientation
    # current_right_ee_orn = current_right_ee_pose[3:]
    # R_current = R.from_quat(current_right_ee_orn).as_matrix()
    # R_target = R.from_quat(target_right_ee_orn).as_matrix()
    # closer_rotation = closest_rotation_matrix(R_current, R_target)
    # target_right_ee_orn = R.from_matrix(closer_rotation).as_quat()

    # target_right_ee_pose = (current_right_ee_pose[:3], target_right_ee_orn)
    target_right_ee_pose = (target_right_ee_pos, target_right_ee_orn)
    
    # Obtaining delta pose
    delta_pos = target_right_ee_pose[0] - current_right_ee_pose[:3]
    delta_ori = R.from_quat(target_right_ee_pose[1]) * R.from_quat(current_right_ee_pose[3:]).inv()
    delta_ori = delta_ori.as_quat()
    gripper_act = np.array([OC[obj_name]["gripper_open_pos"]])
    delta_pose = np.concatenate((delta_pos, delta_ori, gripper_act))
    print(f"delta_pos: {delta_pos}")
    print(f"delta_ori: {delta_ori}")
    action = {'right': delta_pose, 'left': None, 'base': None, 'gripper': 'stay', 'object_name': obj_name}

    print("Press c to continue...")
    breakpoint()
    obs, reward, done, info = env.step(action, delay_scale_factor=4.0)
    print("info: ", info["arm_right"])
    if info["arm_right"] is not None:
        final_actions.append(action)
    # =======================================================

    # 2. Move to final pose =========================
    current_right_ee_pose = env.tiago.arms["right"].arm_pose
    
    target_right_ee_pos = T_grasp[:3, 3]
    # Fix the tooltip offset
    right_tooltip_ee_offset = OC[obj_name]["right_tooltip_ee_offset"] #-0.24 for other objects -0.28 for cabinets and drawers
    right_eef_pose_mat = T_grasp[:3, :3]
    tooltip_ee_offset_wrt_robot = np.dot(right_eef_pose_mat, right_tooltip_ee_offset)
    # print("BEFORE target_right_ee_pos: ", target_right_ee_pos)
    target_right_ee_pos = target_right_ee_pos + tooltip_ee_offset_wrt_robot[:3]
    # print("AFTER target_right_ee_pos: ", target_right_ee_pos)
    
    target_right_ee_orn = R.from_matrix(T_grasp[:3, :3]).as_quat()
    # # Find the closer target orientation
    # current_right_ee_orn = current_right_ee_pose[3:]
    # R_current = R.from_quat(current_right_ee_orn).as_matrix()
    # R_target = R.from_quat(target_right_ee_orn).as_matrix()
    # closer_rotation = closest_rotation_matrix(R_current, R_target)
    # target_right_ee_orn = R.from_matrix(closer_rotation).as_quat()

    # target_right_ee_pose = (current_right_ee_pose[:3], target_right_ee_orn)
    target_right_ee_pose = (target_right_ee_pos, target_right_ee_orn)
    
    # Obtaining delta pose
    delta_pos = target_right_ee_pose[0] - current_right_ee_pose[:3]
    delta_ori = R.from_quat(target_right_ee_pose[1]) * R.from_quat(current_right_ee_pose[3:]).inv()
    delta_ori = delta_ori.as_quat()
    gripper_act = np.array([OC[obj_name]["gripper_open_pos"]])
    delta_pose = np.concatenate((delta_pos, delta_ori, gripper_act))
    print(f"delta_pos: {delta_pos}")
    print(f"delta_ori: {delta_ori}")
    action = {'right': delta_pose, 'left': None, 'base': None, 'gripper': 'stay', 'object_name': obj_name}

    # print("Press c to continue...")
    # breakpoint()
    obs, reward, done, info = env.step(action, delay_scale_factor=2.0)
    print("info: ", info["arm_right"])
    if info["arm_right"] is not None:
        final_actions.append(action)
    # rospy.sleep(1)
    # =======================================================

    if switch_to_impedance_controller:
        # switch to impedance controller
        start_controllers = ['arm_right_impedance_controller', 'arm_left_impedance_controller']
        stop_controllers = ['arm_right_controller', 'arm_left_controller']
        env.tiago.arms["right"].switch_controller(start_controllers, stop_controllers)
    
    # 3. close gripper =========================
    delta_pos = np.array([0.0, 0.0, 0.0])
    delta_ori = np.array([0.0, 0.0, 0.0, 1.0])
    gripper_act = np.array([OC[obj_name]["gripper_closed_pos"]])
    delta_pose = np.concatenate((delta_pos, delta_ori, gripper_act))
    action = {'right': delta_pose, 'left': None, 'base': None, 'gripper': 'close', 'object_name': obj_name}
    obs, reward, done, info = env.step(action, delay_scale_factor=4.0)
    print("info: ", info["arm_right"])
    if info["arm_right"] is not None:
        final_actions.append(action)
    # rospy.sleep(2)


    # 4. Move to pregrasp pose again =========================
    if OC[obj_name]["post_grasp_pregrasp_pose_flag"]:
        current_right_ee_pose = env.tiago.arms["right"].arm_pose
        
        target_right_ee_pos = T_grasp[:3, 3]
        # Fix the tooltip offset
        right_tooltip_ee_offset = OC[obj_name]["right_tooltip_ee_offset_pregrasp"]
        right_eef_pose_mat = T_grasp[:3, :3]
        tooltip_ee_offset_wrt_robot = np.dot(right_eef_pose_mat, right_tooltip_ee_offset)
        target_right_ee_pos = target_right_ee_pos + tooltip_ee_offset_wrt_robot[:3]
        
        target_right_ee_orn = R.from_matrix(T_grasp[:3, :3]).as_quat()
        # # Find the closer target orientation
        # current_right_ee_orn = current_right_ee_pose[3:]
        # R_current = R.from_quat(current_right_ee_orn).as_matrix()
        # R_target = R.from_quat(target_right_ee_orn).as_matrix()
        # closer_rotation = closest_rotation_matrix(R_current, R_target)
        # target_right_ee_orn = R.from_matrix(closer_rotation).as_quat()

        target_right_ee_pose = (target_right_ee_pos, target_right_ee_orn)
        delta_pos = target_right_ee_pose[0] - current_right_ee_pose[:3]
        # always add a little bit of +z delta (to move away from surface)
        delta_pos[2] += 0.03
        delta_ori = R.from_quat(target_right_ee_pose[1]) * R.from_quat(current_right_ee_pose[3:]).inv()
        delta_ori = delta_ori.as_quat()
        gripper_act = np.array([OC[obj_name]["gripper_closed_pos"]])
        delta_pose = np.concatenate((delta_pos, delta_ori, gripper_act))
        print(f"delta_pos: {delta_pos}")
        print(f"delta_ori: {delta_ori}")
        action = {'right': delta_pose, 'left': None, 'base': None, 'gripper': 'stay', 'object_name': obj_name}

        # # Need to set explicitly so that first the gripper is opened and only then the arm is taken back to pregrasp pose
        # env.tiago.gripper['right'].step(OC[obj_name]["gripper_open_pos"])
        # time.sleep(2)
        obs, reward, done, info = env.step(action, delay_scale_factor=4.0)
        if info["arm_right"] is not None:
            final_actions.append(action)
        # =======================================================

    # check grasp status
    grasp_success = env.tiago.gripper["right"].is_grasping()
    retval = {"subtask_actions": final_actions, "grasp_success": grasp_success}
    return retval



def obtain_grasp_modes(pcd, env, obj_name, select_mode=True, k=3):
    # TODO: perform grasp clustering until user is satisfied
    num_samples = len(pcd.points)
    object_frame = np.eye(4)
    gs = GraspSelector(object_frame, pcd)
    sampled_poses = gs.getRankedGraspPoses()
    # print("sampled_poses: ", np.array(sampled_poses).shape)
    desired_sampled_poses = sampled_poses[:num_samples]
    desired_sampled_poses = [translateFrameNegativeZ(p, gs.dist_from_point_to_ee_link) for p in desired_sampled_poses]

    desired_sampled_poses = cluster_sampled_grasps(np.array(desired_sampled_poses),
                                                     k=k,
                                                     translation_weight=1.0,
                                                     rotation_weight=1.0)
    for i in range(len(desired_sampled_poses)):
        print("desired_sampled_poses: ", desired_sampled_poses[i])
    
    if select_mode:
        gs.visualizeGraspPoses(desired_sampled_poses)
        # Accept a list of grasp modes from the user input
        user_input = input("Enter a list of grasp modes (separated by commas): ")
        grasp_mode_indices = [int(mode.strip()) for mode in user_input.split(",")]
        print("Selected grasp modes:", grasp_mode_indices)
        grasp_modes = [desired_sampled_poses[index-1] for index in grasp_mode_indices]

        # FOR DRAWER ONLY: add two more grasp modes for drawer:
        if obj_name == "drawer" and OC[obj_name]["add_grasp_modes"]:
            org_grasp = grasp_modes[0]
            # thetas = [0.78, -0.78, -0.12]
            thetas = [-0.78, -1.3]
            # thetas = [-0.52, -1.04]
            # angle = R.from_matrix(org_grasp[:3, :3]).as_euler("xyz")[1]
            # print("angle: ", angle)
            # thetas = [-angle]
            for theta in thetas:
                new_grasp = org_grasp.copy()
                R_y = np.array([
                    [np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]
                ])
                new_grasp[:3, :3] = R_y @ org_grasp[:3, :3]
                grasp_modes.append(new_grasp)

        # FOR FRIDGE ONLY: add two more grasp modes for fridge:
        print("obj_name: ", obj_name)
        if obj_name == "fridge handle" and OC[obj_name]["add_grasp_modes"]:
            org_grasp = grasp_modes[0]
            thetas = [0.78]
            for theta in thetas:
                new_grasp = org_grasp.copy()
                R_z =  np.array([
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ])
                new_grasp[:3, :3] = R_z @ org_grasp[:3, :3]
                grasp_modes.append(new_grasp)

        if obj_name == "pot" and OC[obj_name]["add_grasp_modes"]:
            org_grasp = grasp_modes[0]
            thetas = [-1.1]
            for theta in thetas:
                new_grasp = org_grasp.copy()
                R_y = np.array([
                    [np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]
                ])
                new_grasp[:3, :3] = R_y @ org_grasp[:3, :3]
                grasp_modes.append(new_grasp)

        if obj_name == "faucet handle" and OC[obj_name]["add_grasp_modes"]:
            org_grasp = grasp_modes[0]
            thetas = [0.3]
            for theta in thetas:
                new_grasp = org_grasp.copy()
                R_y = np.array([
                    [np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]
                ])
                new_grasp[:3, :3] = R_y @ org_grasp[:3, :3]
                grasp_modes.append(new_grasp)

        # remove later
        inp = input(f"len(grasp_modes): {len(grasp_modes)}. Give sequence of grasp modes to execute: ")
        sequence = [int(mode.strip()) for mode in inp.split(",")]
        grasp_modes = [grasp_modes[index-1] for index in sequence]
        # grasp = grasp_modes[int(inp)-1]
    else:
        grasp_modes = desired_sampled_poses

    return grasp_modes

# Function to compute geodesic distance between two rotation matrices
def geodesic_distance(R1, R2):
    """Computes geodesic distance between two 3x3 rotation matrices."""
    # Extract the z-axis component of the rotation matrices
    z_axis_component_R1 = R1[:, 2]
    z_axis_component_R2 = R2[:, 2]
    
    # Compute the dot product of the z-axis components to get the cosine of the angle between them
    dot_product = np.dot(z_axis_component_R1, z_axis_component_R2)
    
    # Compute the angle between the z-axis components
    try:
        angle_between_z_axes = np.arccos(dot_product)
    except ValueError:
        return 5.0
    if np.isnan(angle_between_z_axes):
        return 5.0
    # print("angle_between_z_axes: ", angle_between_z_axes)
    
    # Return the angle as the distance along the z-axis
    return angle_between_z_axes
    
    # relative_rotation = R1.T @ R2
    # log_rot = logm(relative_rotation)  # Matrix logarithm
    # return norm(log_rot, 'fro') / np.sqrt(2)

# Custom distance function for pose clustering
def pose_distance(pose1, pose2, translation_weight=1.0, rotation_weight=1.0):
    """Combines translation and rotation distances for two 4x4 pose matrices."""
    # Extract rotation matrices and translations
    R1, t1 = pose1[:3, :3], pose1[:3, 3]
    R2, t2 = pose2[:3, :3], pose2[:3, 3]
    
    # Compute geodesic distance for rotation
    rotation_dist = geodesic_distance(R1, R2)
    
    # Compute Euclidean distance for translation
    translation_dist = np.linalg.norm(t1 - t2)
    # print("translation_dis, rotation_dist: ", translation_dist, rotation_dist)
    
    # Weight the rotation and translation distances (you can adjust the weights)
    # w1 = 1.0
    # w2 = 2.0
    # print("rotation_weight: ", rotation_weight)
    return (rotation_weight * rotation_dist) + (translation_weight * translation_dist)

def cluster_sampled_grasps(grasp_poses, k=3, translation_weight=1.0, rotation_weight=1.0):
    # ============== trial 1 ===================
    # Flatten the poses into a list of matrices (needed for the distance matrix calculation)
    n_poses = len(grasp_poses)
    distance_matrix = np.zeros((n_poses, n_poses))

    # Compute the pairwise distance matrix
    for i in range(n_poses):
        for j in range(i, n_poses):
            dist = pose_distance(grasp_poses[i], grasp_poses[j], translation_weight, rotation_weight)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist  # Symmetric

    # Perform K-medoids clustering with the custom distance matrix
    kmedoids = KMedoids(n_clusters=k, metric='precomputed', random_state=42)
    labels = kmedoids.fit_predict(distance_matrix)

    # Print the resulting cluster labels for each grasp pose
    # print("Cluster labels for grasp poses:", labels)
    medoid_indices = kmedoids.medoid_indices_
    # print("medoid_indices: ", medoid_indices)
    # comment/uncomment this
    return grasp_poses[medoid_indices]
    # ==========================================

def random_point_dropout(point_cloud, fraction_to_keep=0.1):
    indices = random.sample(range(len(point_cloud.points)), int(len(point_cloud.points) * fraction_to_keep))
    downsampled_point_cloud = point_cloud.select_by_index(indices)
    return downsampled_point_cloud

def generate_point_cloud_from_depth(depth_image, intrinsic_matrix, mask, extrinsic_matrix):
    """
    Generate a point cloud from a depth image and intrinsic matrix.
    
    Parameters:
    - depth_image: np.array, HxW depth image (in meters).
    - intrinsic_matrix: np.array, 3x3 intrinsic matrix of the camera.
    
    Returns:
    - point_cloud: Open3D point cloud.
    """
    
    # Get image dimensions
    height, width = depth_image.shape

    # Create a meshgrid of pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Flatten the pixel coordinates and depth values
    u_flat = u.flatten()
    v_flat = v.flatten()
    depth_flat = depth_image.flatten()
    mask_flat = mask.flatten()

    # # Filter points where the mask is 1
    # valid_indices = np.where(mask_flat == 1)
    
    # Filter points where the mask is 1 AND depth is valid (not inf and not 0)
    valid_indices = np.where(
        (mask_flat == 1) & 
        (np.isfinite(depth_flat)) &  # Remove inf values
        (depth_flat > 0)             # Remove 0 or negative values
    )[0]

    # Apply the mask to the pixel coordinates and depth
    u_valid = u_flat[valid_indices]
    v_valid = v_flat[valid_indices]
    depth_valid = depth_flat[valid_indices]

    # Generate normalized pixel coordinates in homogeneous form
    pixel_coords = np.vstack((u_valid, v_valid, np.ones_like(u_valid)))

    # Compute inverse intrinsic matrix
    intrinsic_inv = np.linalg.inv(intrinsic_matrix)

    # Apply the inverse intrinsic matrix to get normalized camera coordinates
    cam_coords = intrinsic_inv @ pixel_coords

    # Multiply by depth to get 3D points in camera space
    cam_coords *= depth_valid
    # breakpoint()

    # # Reshape the 3D coordinates
    # x = cam_coords[0].reshape(height, width)
    # y = cam_coords[1].reshape(height, width)
    # z = depth_image

    # # Stack the coordinates into a single 3D point array
    # points = np.dstack((x, y, z)).reshape(-1, 3)

    # breakpoint()
    points = np.vstack((cam_coords[0], cam_coords[1], depth_valid)).T

    # Transform points to world frame
    points = points / 1000.0    
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    points = extrinsic_matrix @ points.T
    points = points.T
    points = points[:, :3]

    # random dropout of points
    # keep_ratio = 0.7  # Keep % of points randomly
    # num_points = len(points)
    # mask = np.random.choice([True, False], size=num_points, p=[keep_ratio, 1-keep_ratio])
    # points = points[mask]

    # # Create an Open3D point cloud object
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(points)

    return points

def fix_pcd_normals(point_cloud):
     # fixing normal direction
    # Convert point cloud to numpy array for easier manipulation
    normals_np = np.asarray(point_cloud.normals)

    # 1. Original normals (no flipping)
    original_normals = normals_np.copy()
    # print("original_normals: ", original_normals)

    # 2. Flipped normals (flip each normal)
    flipped_normals = -normals_np.copy()
    
    # 3. Define the vector to compute the dot product with (1, 0, 0)
    vector = np.array([1, 0, 0])

    # 4. Function to compute the average dot product with the vector (1, 0, 0)
    def compute_average_dot_product(normals, vector, subset_size):
        random_indices = random.sample(range(len(normals)), subset_size)
        subset_normals = normals[random_indices]
        dot_products = np.dot(subset_normals, vector)  # Dot product with each normal
        return np.mean(dot_products)  # Return the average dot product
    
    # 5. Choose a random subset size (you can adjust the size based on your needs)
    subset_size = len(point_cloud.points) // 10

    # 6. Compute the average dot products for both sets
    avg_dot_original = compute_average_dot_product(original_normals, vector, subset_size)
    avg_dot_flipped = compute_average_dot_product(flipped_normals, vector, subset_size)
    # print("avg_dot_original: ", avg_dot_original)
    # print("avg_dot_flipped: ", avg_dot_flipped)

    # 7. Choose the set with the lesser dot product (which is the more "opposite" direction)
    if avg_dot_original < avg_dot_flipped:
        # print("The set with flipped normals is chosen.")
        chosen_normals = flipped_normals
    else:
        # print("The set with original normals is chosen.")
        chosen_normals = original_normals

    return chosen_normals


def get_pcd(env, mask):
    obs = env._observation()
    depth = obs['tiago_head_depth'].squeeze()
    rgb = obs['tiago_head_image']
    intr = np.asarray(list(env.cameras['tiago_head'].camera_info.K)).reshape(3,3)
    extrinsic_matrix = env.tiago.head.camera_extrinsic
    
    # mask = obtain_mask()
    points = generate_point_cloud_from_depth(depth, intr, mask, extrinsic_matrix)

    # TODO: remove outlier points
    
    # Create an Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    # Estimate normals for the point cloud
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    point_cloud.orient_normals_consistent_tangent_plane(k=30)
   
    # o3d.visualization.draw_geometries([point_cloud],  point_show_normal=True)
    chosen_normals = fix_pcd_normals(point_cloud)
    point_cloud.normals = o3d.utility.Vector3dVector(chosen_normals)

    # print("BEFORE point_cloud.points: ", point_cloud.points)
    point_cloud = random_point_dropout(point_cloud, fraction_to_keep=0.1)
    # print("AFTER point_cloud.points: ", point_cloud.points)

    # without this breakpoint, I'm getting X Error of failed request:  BadWindow (invalid Window parameter) error
    # breakpoint()

    # o3d.visualization.draw_geometries([point_cloud],  point_show_normal=True)
    
    return point_cloud 

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def mouseclick_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global clicked_points
        clicked_points.append([x, y])
        # print("click_point_pix", [x, y])

def obtain_mask(env, select_object=False):
    obs = env._observation()
    depth = obs['tiago_head_depth']
    rgb = obs['tiago_head_image']
    if rgb.dtype != np.uint8:
        rgb = cv2.convertScaleAbs(rgb)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    img = rgb
    
    # build SAM2 image predictor
    sam2_checkpoint = SAM2_CHECKPOINT
    model_cfg = SAM2_MODEL_CONFIG
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # img_path = IMG_PATH
    # image_source, image = load_image(img_path)
    # breakpoint()
    sam2_predictor.set_image(img)

    torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

    # if torch.cuda.get_device_properties(0).major >= 8:
    #     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    #     torch.backends.cuda.matmul.allow_tf32 = True
    #     torch.backends.cudnn.allow_tf32 = True

    # # try ===========
    # # Create a figure and axis
    # def on_click(event):
    #     if event.xdata is not None and event.ydata is not None:
    #         global clicked_points
    #         clicked_points.append((int(event.xdata), int(event.ydata)))
    #         print(f"Point clicked: {clicked_points[-1]}")
    #         ax.imshow(img)
    #         for point in clicked_points:
    #             ax.plot(point[0], point[1], 'ro', markersize=5)
    #         plt.draw()

    # def on_close(event):
    #     print("Figure closed, performing cleanup.")
    #     plt.close()

    # fig, ax = plt.subplots()
    # ax.imshow(img)
    # fig.canvas.mpl_connect('button_press_event', on_click)  # Mouse click event
    # fig.canvas.mpl_connect('close_event', on_close)  # Window close event

    # plt.show()
    # # ==============

    # TODO: Implement when not slecting object (i.e. using text query)
    if select_object:
        # choose a point
        global clicked_points
        cv2.namedWindow('color', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('color', mouseclick_callback)
        # color_im = cv2.imread(img_path)
        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        while True:
            if len(clicked_points) != 0:
                for point in clicked_points:
                    bgr_img = cv2.circle(bgr_img, point, 7, (0, 0, 255), 2)
            cv2.imshow('color', bgr_img)

            if cv2.waitKey(1) == ord('q'):
                break
        cv2.destroyWindow('color')

        clicked_points = np.array(clicked_points)
        print("clicked_points: ", clicked_points)
        input_label = np.ones(len(clicked_points), dtype=int)

        masks, scores, logits = sam2_predictor.predict(
            point_coords=clicked_points,
            point_labels=input_label,
            box=None,
            multimask_output=False,
        )

        # breakpoint()
        plt.figure(figsize=(10,10))
        plt.imshow(img)
        show_mask(masks, plt.gca())
        show_points(clicked_points, input_label, plt.gca())
        plt.axis('off')
        plt.savefig(f"resources/tmp_outputs/gsam_mask.jpg")
        # plt.show() 
        clicked_points = []

    # print("masks.shape:", masks.shape)
    return masks[0]

def grasp(env, obj_name, exec_on_robot=False, mask=None, pcd=None, select_mode=True):
    # 1. Obtaining the mask of the part of the object for which we need grasp proposals
    if mask is None:
        mask = obtain_mask(env)
        # with open("temp_files/gello.pkl", 'wb') as f:
        #     pickle.dump(mask, f)
        # with open("temp_files/gello.pkl", 'rb') as f:
        #     mask = pickle.load(f)

    # 2. Obtain the pcd of the object part 
    if pcd is None:
        pcd = get_pcd(env, mask)

    # # remove later
    # points = np.array(pcd.points)
    # distances = np.linalg.norm(points[:, :2], axis=1)
    # print("min, max: ", min(distances), max(distances))
    # # Sort points by distance
    # distances_sorted = np.sort(distances)[:20]
    # near_point_distance = np.median(distances_sorted, axis=0)
    # print("near_point_distance: ", near_point_distance)

    # 3. Obtain grasp proposals
    grasp_modes = obtain_grasp_modes(pcd, env, obj_name, select_mode=select_mode)
    # print("grasp_modes: ", grasp_modes)
    # Was using this for drawer to choose which one to execute first
    # inp = input("choose grasp mode")
    # grasp = grasp_modes[int(inp)-1]

    # 4. Execute the grasp
    if exec_on_robot:
        for i, grasp in enumerate(grasp_modes):
        #     if i != 2:
        #         continue
            execute_grasp(env, grasp, obj_name)
            # teleop(env)
            breakpoint()
            env.tiago.gripper['right'].step(OC[obj_name]["gripper_open_pos"])
            time.sleep(2)
            reset_joint_pos = RP.PREGRASP_R_H
            reset_joint_pos["right"][-1] = OC[obj_name]["gripper_open_pos"]
            env.reset(reset_arms=True, reset_pose=reset_joint_pos, allowed_delay_scale=6.0)

    return grasp_modes
    


def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if __name__ == "__main__":
    rospy.init_node('tiago_test')
    set_all_seeds(seed=1)
    obj_name = "cup"

    # for saving videos
    # current_date = datetime.now().strftime("%Y-%m-%d")  # Format: YYYY-MM-DD
    # current_time = datetime.now().strftime("%H-%M-%S")  # Format: HH-MM-SS
    # base_folder = f"{current_date}"
    # time_folder = os.path.join(base_folder, current_time)
    # folder_path = f"outputs/{time_folder}"
    folder_path = "output"
    os.makedirs(folder_path, exist_ok=True)
    imgio_kargs = {'fps': 10, 'quality': 10, 'macro_block_size': None,  'codec': 'h264',  'ffmpeg_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']}
    output_path = f'{folder_path}/grasping_video.mp4'
    writer = imageio.get_writer(output_path, **imgio_kargs)

    env = TiagoGym(
        frequency=10,
        right_arm_enabled=True,
        left_arm_enabled=False,
        right_gripper_type='robotiq2F-140',
        left_gripper_type='robotiq2F-85',
        base_enabled=True,
        torso_enabled=False,
    )

    # Start recorder
    record = False
    exec_on_robot = True
    side_cam, top_down_cam, ego_cam = None, None, None
    if record:
        # side_cam = Camera(img_topic="/side_1/color/image_raw", depth_topic="/side_1/aligned_depth_to_color/image_raw")
        # top_down_cam = Camera(img_topic="/top_down/color/image_raw", depth_topic="/top_down/aligned_depth_to_color/image_raw")
        ego_cam = Camera(img_topic="/xtion/rgb/image_raw", depth_topic="/xtion/depth/image_raw")
        recorder = RecordVideo(camera_interface_ego=ego_cam)
        recorder.setup_recording()

    # breakpoint()

    # # remove later
    # current_right_ee_pose = env.tiago.arms["right"].arm_pose
    # target_right_ee_pose = (current_right_ee_pose[:3], np.array([0.0, 0.0, 0.0, 1.0]))
    # delta_pos = target_right_ee_pose[0] - current_right_ee_pose[:3]
    # delta_ori = quat_diff(target_right_ee_pose[1], current_right_ee_pose[3:])
    # gripper_act = np.array([1.0])
    # delta_pose = np.concatenate((delta_pos, delta_ori, gripper_act))
    # print(f"delta_pos: {delta_pos}")
    # print(f"delta_ori: {delta_ori}")
    # action = {'right': None, 'left': None, 'base': None}
    # action["right"] = delta_pose
    # breakpoint()
    # obs, reward, done, info = env.step(action)

    # Move Tiago to reset pose
    if exec_on_robot:
        # open gripper. 1 is open and 0 is close
        env.tiago.gripper['right'].step(OC[obj_name]["gripper_open_pos"])
        time.sleep(2)
        # reset_joint_pos = RP.FORWARD_R_H
        reset_joint_pos = RP.PREGRASP_R_H
        reset_joint_pos["right"][-1] = OC[obj_name]["gripper_open_pos"]
        env.reset(reset_arms=True, reset_pose=reset_joint_pos, allowed_delay_scale=6.0)

    # if record:
    #     recorder.start_recording()
    #     print("Start recording")

    # grasp(env, obj_name=obj_name, exec_on_robot=exec_on_robot)

    # # if record:
    # #     recorder.save_video(save_folder="output")
    # #     recorder.stop_recording()

    # # # Move Tiago to reset pose
    # # if exec_on_robot:
    # #     # open gripper. 1 is open and 0 is close
    # #     env.tiago.gripper['right'].step(OC[obj_name]["gripper_open_pos"])
    # #     time.sleep(2)
    # #     reset_joint_pos = RP.PREGRASP_R_H
    # #     reset_joint_pos["right"][-1] = OC[obj_name]["gripper_open_pos"]
    # #     env.reset(reset_arms=True, reset_pose=reset_joint_pos, allowed_delay_scale=6.0)
