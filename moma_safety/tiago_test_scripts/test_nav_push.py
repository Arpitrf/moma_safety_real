import rospy
import sys
import numpy as np
import cv2

from moma_safety.tiago.tiago_gym import TiagoGym
import moma_safety.utils.vision_utils as VU # vision_utils
from moma_safety.tiago.skills.push_button import PushButtonSkill
from moma_safety.tiago.utils.ros_utils import Publisher, Listener, TFTransformListener
from moma_safety.tiago.skills.navigate_to import NavigateToSkill

import moma_safety.utils.utils as U
import moma_safety.utils.transform_utils as T # transform_utils

rospy.init_node('move_base_python', anonymous=True)
env = TiagoGym(
    frequency=10,
    right_arm_enabled=True,
    left_arm_enabled=False,
    right_gripper_type='robotiq2F-140',
    left_gripper_type='robotiq2F-140',
    base_enabled=True,
    torso_enabled=False,
)
# env.reset()
import moveit_commander

push_button = PushButtonSkill(oracle_position=True, debug=False)
skill = NavigateToSkill(oracle_action=True)
tf_listener = TFTransformListener('/base_footprint')
if True:
    obs = env._observation()
    rgb = obs['tiago_head_image']
    depth = obs['tiago_head_depth']
    cam_intr = np.asarray(list(env.cameras['tiago_head'].camera_info.K)).reshape(3,3)
    cam_pose = tf_listener.get_transform('/xtion_optical_frame')
    cam_extr = T.pose2mat(cam_pose)
    pos, pcd, normals = VU.pixels2pos(
        np.asarray([(rgb.shape[0]//2, rgb.shape[1]//2)]),
        depth=depth.astype(np.float32),
        cam_intr=cam_intr,
        cam_extr=cam_extr,
        return_normal=True,
    )
    skill.step(env, rgb, depth, pcd, normals)

if True:
    obs = env._observation()
    rgb = obs['tiago_head_image']
    depth = obs['tiago_head_depth']
    cam_intr = np.asarray(list(env.cameras['tiago_head'].camera_info.K)).reshape(3,3)
    cam_pose = tf_listener.get_transform('/xtion_optical_frame')
    cam_extr = T.pose2mat(cam_pose)
    pos, pcd, normals = VU.pixels2pos(
        np.asarray([(rgb.shape[0]//2, rgb.shape[1]//2)]),
        depth=depth.astype(np.float32),
        cam_intr=cam_intr,
        cam_extr=cam_extr,
        return_normal=True,
    )
    push_button.step(env, rgb, depth, pcd=pcd, normals=normals, arm='right')
    rospy.signal_shutdown("Shutdown")
    rospy.spin()
    sys.exit(0)
