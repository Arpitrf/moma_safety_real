import os
import sys
import cv2
import pickle
import time
import rospy
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

import open3d as o3d
import plotly.graph_objects as go
from moma_safety.tiago.utils.ros_utils import TFTransformListener

from moma_safety.tiago.tiago_gym import TiagoGym
import moma_safety.utils.utils as U # utils
import moma_safety.utils.transform_utils as T # transform_utils
from moma_safety.tiago.utils.transformations import quat_diff
import moma_safety.utils.vision_utils as VU # vision_utils

rospy.init_node('tiago_test')

def goto_pose(
        env,
        pose,
        gripper_act,
        adj_gripper=True,
        n_steps=2,
    ):
    '''
        pose = (pos, ori) w.r.t. base_footprint
        gripper_act = 0.0 (close) or 1.0 (open)
        adj_gripper = True accounts for the final pose of the tip of the gripper
        n_steps = number of steps to interpolate between the current and final pose
    '''
    final_pos = pose[0]
    if adj_gripper:
        # reduce the x value by the length of the gripper.
        # Note: this is a rough estimate and wrt to the base_footprint
        final_pos[0] -= 0.2
    final_ori = pose[1]
    cur_arm_pose = env.tiago.arms['right'].arm_pose
    inter_pos = np.linspace(cur_arm_pose[:3], final_pos, n_steps)

    for pos in inter_pos[1:]:
        cur_arm_pose = env.tiago.arms['right'].arm_pose
        delta_pos = pos - cur_arm_pose[:3]
        delta_ori = R.from_quat(quat_diff(final_ori, cur_arm_pose[3:7])).as_euler('xyz')
        delta_act = np.concatenate(
            (delta_pos, delta_ori, np.asarray([gripper_act]))
        )
        print(delta_act)
        env.step({'right': delta_act, 'left': None})
    return

def main():
    debug = False
    env = TiagoGym(
        frequency=10,
        right_arm_enabled=True,
        left_arm_enabled=False,
        right_gripper_type='robotiq2F-140',
        left_gripper_type='robotiq2F-140',
        base_enabled=False,
        torso_enabled=False,
    )

    # reset_arms
    env.reset()

    obs = env._observation()
    depth = obs['tiago_head_depth']
    rgb = obs['tiago_head_image']

    cam_intr = np.asarray(list(env.cameras['tiago_head'].camera_info.K)).reshape(3,3)
    # base_footprint to is located at the base of the robot.
    # for global poses, use odom frame.
    # Note: arm_reader.get_transform returns poses w.r.t. base_footprint
    tf_listener = TFTransformListener('/base_footprint')
    cam_pose = tf_listener.get_transform('/xtion_optical_frame')
    cam_extr = T.pose2mat(cam_pose)

    # get user_pixel from mouse_click
    points = get_user_input(rgb, num_pts=1)
    # points = [(284, 168)]
    print(points)
    pos, pts, normals = VU.pixels2pos(
        # np.asarray([(i, j) for i in range(rgb.shape[0], rgb.shape[1]) for j in range(rgb.shape[0], rgb.shape[1])]),
        np.asarray([(rgb.shape[0]//2, rgb.shape[1]//2)]),
        depth=depth.astype(np.float32),
        cam_intr=cam_intr,
        cam_extr=cam_extr,
        return_normal=True,
    )
    pos = pts[points[0][1], points[0][0]]
    normal = normals[points[0][1], points[0][0]]
    # quat = T.vec2quat(-1*normal)
    # approach_pos = pos + 0.1 * normal
    approach_pos = pos + np.asarray([-0.1, 0.0, 0.0])
    grasp_pos = pos + np.asarray([0.05, 0.0, 0.0])
    after_grasp_pos = pos + np.asarray([0.0, 0.0, 0.1])

    if debug:
        pts_to_plot = np.concatenate((pts.reshape(-1,3), approach_pos.reshape(1,3)), axis=0)
        pts_to_plot = np.concatenate((pts_to_plot.reshape(-1,3), grasp_pos.reshape(1,3)), axis=0)
        rgb_to_plot = np.concatenate((rgb.reshape(-1,3), np.asarray([[255.0, 0.0, 0.0]])), axis=0)
        rgb_to_plot = np.concatenate((rgb_to_plot.reshape(-1,3), np.asarray([[0.0, 255.0, 0.0]])), axis=0)
        U.plotly_draw_3d_pcd(pts_to_plot, rgb_to_plot)

    # testing a dummy pose
    # local_pos = np.asarray([0.0, 0.0, 0.2])
    # import ipdb; ipdb.set_trace()
    # base_pos, base_ori = env.tiago.base.reference_odom['pose'][:3], env.tiago.base.reference_odom['pose'][3:]
    # transform = T.pose2mat((base_pos, base_ori))
    # global_pos = (transform @ np.array([local_pos[0], local_pos[1], local_pos[2], 1]))[:3]
    cur_arm_pose = env.tiago.arms['right'].arm_pose
    quat = R.from_rotvec(np.asarray([np.pi/2, 0.0, 0.0])).as_quat()
    goto_pose(
        env,
        pose=(approach_pos, quat),
        gripper_act=1.0,
    )
    print("Moved to approach pose")
    goto_pose(
        env,
        pose=(grasp_pos, quat),
        gripper_act=1.0,
    )
    print("Moved to grasping pose")
    goto_pose(
        env,
        pose=(grasp_pos, quat),
        gripper_act=0.0,
    )
    print("Grasping")
    goto_pose(
        env,
        pose=(after_grasp_pos, quat),
        gripper_act=0.0,
    )

if __name__ == '__main__':
    main()
