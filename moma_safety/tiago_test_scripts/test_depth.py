import os
import cv2
import rospy
import numpy as np
import matplotlib.pyplot as plt
from moma_safety.tiago.utils.camera_utils import Camera
from moma_safety.tiago.tiago_gym import TiagoGym
from moma_safety.tiago import RESET_POSES as RP
from moma_safety.tiago.utils.ros_utils import Publisher, Listener, TFTransformListener

import moma_safety.utils.utils as U
import moma_safety.utils.transform_utils as T # transform_utils
rospy.init_node('tiago_test')
env = TiagoGym(
    frequency=10,
    right_arm_enabled=False,
    left_arm_enabled=True,
    right_gripper_type=None,
    left_gripper_type='robotiq2F-85',
    base_enabled=True,
    torso_enabled=False,
)
obs = env._observation()
depth = obs['tiago_head_depth']
plt.imshow(depth)
plt.show()
