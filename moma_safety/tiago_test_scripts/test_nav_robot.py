import os
import sys
import cv2
import pickle
import time
import rospy
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import open3d

from moma_safety.tiago.tiago_gym import TiagoGym
from moma_safety.tiago.utils.ros_utils import Listener,TFTransformListener
from moma_safety.tiago.utils.camera_utils import Camera

import plotly
import plotly.graph_objects as go
import open3d as o3d

from moma_safety.tiago.skills.move_to import MoveToSkill

rospy.init_node('tiago_test')
env = TiagoGym(
    frequency=10,
    right_arm_enabled=True,
    left_arm_enabled=True,
    right_gripper_type='robotiq2F-140',
    left_gripper_type='robotiq2F-140',
    base_enabled=True,
    torso_enabled=False,
)
env.reset()

move_to_skill = MoveToSkill()
move_to_skill.step((np.array([1.0,0.0,0.0]), np.array([0.0,0.0,0.0,1.0])))
# import

# obs = env._observation()
# print(obs.keys())
# depth = obs['tiago_head_depth']
# rgb = obs['tiago_head_image']

# arm_action_left = np.asarray([0.0]*7)
# arm_action_right  = np.asarray([0.0]*7)
# base_action_to_move_front = np.asarray([0.5,0.0,0.0])
# torso_action = np.asarray([0.0])

# # import ipdb; ipdb.set_trace()
# action = {
#     'right': arm_action_right,
#     'left': arm_action_left,
#     'base': base_action_to_move_front,
#     # 'torso': torso_action,
# }

# for _ in range(10):
#     obs, _, _, _ = env.step(action)

