import os
import cv2
import rospy
import numpy as np
import matplotlib.pyplot as plt
from moma_safety.tiago.utils.camera_utils import Camera
from moma_safety.tiago.tiago_gym import TiagoGym
from moma_safety.tiago import RESET_POSES as RP
from moma_safety.tiago.utils.ros_utils import Publisher, Listener, TFTransformListener

from geometry_msgs.msg import Point, Quaternion, Pose, PoseStamped, WrenchStamped

import moma_safety.utils.utils as U
import moma_safety.utils.transform_utils as T # transform_utils
rospy.init_node('tiago_shift_pose')
env = TiagoGym(
    frequency=10,
    right_arm_enabled=True,
    left_arm_enabled=True,
    right_gripper_type='robotiq2F-140',
    left_gripper_type='robotiq2F-85',
    base_enabled=True,
    torso_enabled=True,
)
# ft_reader = Listener(input_topic_name=f'/wrist_right_ft/corrected', input_message_type=WrenchStamped)
# msg = ft_reader.get_most_recent_msg()
# print(msg)
# print(msg.wrench.force.z)
# _exec = U.confirm_user(True, 'Move to reset pose?')
# if not _exec:
#     exit()
# grasp_h_r = RP.HOME_L_HORIZONTAL_R_H
# env.reset(reset_arms=True, reset_pose=grasp_h_r, allowed_delay_scale=6.0)
# _exec = U.confirm_user(True, 'Move to reset pose?')
# if not _exec:
#     exit()
# grasp_h_r = RP.HOME_L_PUSH_R_H
# env.reset(reset_arms=True, reset_pose=grasp_h_r, allowed_delay_scale=6.0)

# _exec = U.reset_env(env, reset_pose=RP.PICKUP_TABLE_L_HOME_R_H, delay_scale_factor=2.0)
# _exec = U.reset_env(env, reset_pose=RP.PICKUP_TABLE_L_HOME_R_H, int_pose=RP.INT_L_H, delay_scale_factor=2.0)
_exec = U.reset_env(env, reset_pose=RP.PUSH_R_H, delay_scale_factor=2.0)
# _exec = U.reset_env(env, reset_pose=RP.OPEN_DOOR_R, int_pose=RP.INT_R_H, delay_scale_factor=2.0)
# _exec = U.reset_env(env, reset_pose=RP.HOME_R, int_pose=RP.INT_R_H, delay_scale_factor=2.0)
# _exec = U.reset_env(env, reset_pose=RP.HOME_L, int_pose=RP.INT_L_H, delay_scale_factor=2.0)
# _exec = U.reset_env(env, reset_pose=RP.OPEN_DOOR_R, int_pose=RP.INT_R_H, delay_scale_factor=2.0)
# _exec = U.reset_env(env, reset_pose=RP.OPEN_DOOR_L, int_pose=RP.INT_L_H, delay_scale_factor=2.0)


# test_pose = {'left': None, 'right': None, 'torso': 0.20}
# print("before reset")
# _exec = U.reset_env(env, reset_pose=test_pose, delay_scale_factor=2.0)
print("DONE")
