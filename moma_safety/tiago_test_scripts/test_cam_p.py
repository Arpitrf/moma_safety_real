import os
import sys
import cv2
import pickle
import time
import rospy
import numpy as np
import matplotlib.pyplot as plt

from moma_safety.tiago.tiago_gym import TiagoGym
from moma_safety.tiago.utils.ros_utils import TFTransformListener

import moma_safety.utils.utils as U # utils
import moma_safety.utils.transform_utils as T # transform_utils
import moma_safety.utils.vision_utils as VU # vision_utils

rospy.init_node('tiago_test')
env = TiagoGym(
    frequency=10,
    right_arm_enabled=True,
    left_arm_enabled=True,
    right_gripper_type='robotiq2F-140',
    left_gripper_type='robotiq2F-140'
)

obs = env._observation()
print(obs.keys())
depth = obs['tiago_head_depth']
rgb = obs['tiago_head_image']
print("rgb: ", rgb.shape)
print("depth: ", depth.shape)

cam_intr = np.asarray(list(env.cameras['tiago_head'].camera_info.K)).reshape(3,3)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(rgb)
ax[1].imshow(depth)
plt.show()

pcd = VU.pcd_from_depth(
    depth.astype(np.float32),
    intrinsic_matrix=cam_intr
)
# breakpoint()
# pcd_np = np.asarray(pcd.points)
# U.plotly_draw_3d_pcd(pcd_np)
# cam_to_world_tf = T.pose2mat(cam_pose)
# cam_to_world_tf = np.linalg.inv(cam_to_world_tf)
# import ipdb; ipdb.set_trace()
# pcd_g = pcd.transform(cam_to_world_tf)
# pcd_np_g = np.asarray(pcd_g.points)
# U.plotly_draw_3d_pcd(pcd_np_g)

# import ipdb; ipdb.set_trace()

