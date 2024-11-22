import os
import sys
import cv2
import pickle
import time
import rospy
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


# ros messages
from control_msgs.msg  import JointTrajectoryControllerState

from moma_safety.tiago.tiago_gym import TiagoGym
from moma_safety.tiago.utils.ros_utils import Listener
from moma_safety.tiago.camera_utils import Camera

rospy.init_node('tiago_test')
env = TiagoGym(
    frequency=10,
    right_arm_enabled=True,
    left_arm_enabled=True,
    right_gripper_type='robotiq2F-140',
    left_gripper_type='robotiq2F-140'
)

# -------------- ---- Print the current pose --------------------
right_eef_pose, left_eef_pose = None, None
right_arm = env._observation()['right']
left_arm = env._observation()['left']
print(f"Right Arm: {right_arm}. Left Arm: {left_arm}")
# print("right_gripper, left_gripper: ", right_gripper_state, left_gripper_state)

right_eef_pos, right_eef_quat = right_arm[:3], right_arm[3:7]
print(f"right_eef_pos: {right_eef_pos}, right_eef_quat: {right_eef_quat}")
right_eef_euler = R.from_quat(right_eef_quat).as_euler('XYZ', degrees=True)
print("right_eef_euler: ", right_eef_euler)
# ----------------------------------------------------------------

obs = env._observation()
print(obs.keys())
depth = obs['tiago_head_depth']
rgb = obs['tiago_head_image']
