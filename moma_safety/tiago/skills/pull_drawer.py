#!/usr/bin/env python3
import os
import sys
import copy
import numpy as np
from math import pi
from scipy.spatial.transform import Rotation as R

import rospy
import moveit_commander
import moveit_msgs.msg
from control_msgs.msg import JointTrajectoryControllerState
import geometry_msgs.msg
from std_msgs.msg import String

from moma_safety.tiago.skills.base import SkillBase
from moma_safety.tiago.utils.ros_utils import Publisher, Listener, TFTransformListener
import moma_safety.utils.utils as U
import moma_safety.utils.transform_utils as T # transform_utils
from moma_safety.tiago.utils.transformations import quat_diff

import moveit_commander
moveit_commander.roscpp_initialize(sys.argv)

class PullDrawerSkill(SkillBase):
    def __init__(
            self,
            oracle_position: bool = False,
            adjust_gripper: bool = True,
            debug: bool = False,
        ):
        super(PullDrawerSkill).__init__()
        self.oracle_position = oracle_position
        self.setup_listeners()
        self.adjust_gripper = adjust_gripper
        self.adjust_gripper_length = 0.17 # 85
        self.approach_vec_base = np.asarray([-0.05, 0.0, 0.0])
        self.debug = debug
        self.display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )
        self.robot = moveit_commander.RobotCommander()

    def setup_listeners(self):
        def joint_process_func(data):
            return np.array(data.actual.positions)

        self.tf_odom = TFTransformListener('/odom')
        self.tf_base = TFTransformListener('/base_footprint')

    def left_arm_pose(self, frame='odom'):
        pos, quat = None, None
        if frame == 'odom':
            pos, quat = self.tf_odom.get_transform(target_link=f'/arm_left_tool_link')
        if frame == 'base_footprint':
            pos, quat = self.tf_base.get_transform(target_link=f'/arm_left_tool_link')
        if pos is None:
            return None
        return np.concatenate((pos, quat))

    def right_arm_pose(self, frame='odom'):
        pos, quat = None, None
        if frame == 'odom':
            pos, quat = self.tf_odom.get_transform(target_link=f'/arm_right_tool_link')
        if frame == 'base_footprint':
            pos, quat = self.tf_base.get_transform(target_link=f'/arm_right_tool_link')
        if pos is None:
            return None
        return np.concatenate((pos, quat))

    def get_approach_pose(self, pos, normal, frame='odom'):
        '''
            pos, normal: np.ndarray are w.r.t. the base_footprint
        '''
        assert frame in ['odom', 'base_footprint']
        # this must be based on the normal of the drawer
        # approach_pos = pos + np.asarray([0.0, 0.05, 0.0])
        # approach_ori = R.from_rotvec(np.asarray([0.0, 0.0, -np.pi/2])).as_quat()
        approach_pos = pos + self.approach_vec_base
        approach_ori = R.from_rotvec(np.asarray([0.0, 0.0, 0.0])).as_quat()
        if frame == 'odom':
            transform = T.pose2mat(self.tf_odom.get_transform('/base_footprint'))
            approach_pose = T.pose2mat((approach_pos, approach_ori))
            approach_pose = transform @ approach_pose
            approach_pos, approach_ori = T.mat2pose(approach_pose)
        return approach_pos, approach_ori

    def get_group_name(self, name):
        if name == 'left':
            return "arm_left"
        elif name == 'right':
            return "arm_right"
        else:
            raise ValueError(f'Invalid arm name: {name}')


    def close_gripper(self, env, side):
        assert side in ['left', 'right', 'both']
        if side == 'left':
            env.tiago.gripper['left'].step(0.0)
        if side == 'right':
            env.tiago.gripper['right'].step(0.0)
        if side == 'both':
            env.tiago.gripper['left'].step(0.0)
            env.tiago.gripper['right'].step(0.0)
        rospy.sleep(1)
        return

    def open_gripper(self, env, side):
        if side == 'left':
            env.tiago.gripper['left'].step(1.0)
        if side == 'right':
            env.tiago.gripper['right'].step(1.0)
        if side == 'both':
            env.tiago.gripper['left'].step(1.0)
            env.tiago.gripper['right'].step(1.0)
        rospy.sleep(1)
        return

    def goto_pose(
            self,
            env,
            pose,
            arm,
            frame,
            gripper_act=None,
            adj_gripper=True,
            n_steps=2,
        ):
        '''
            THIS FUNCTION DOES NOT USE COLLISION CHECKING
                pose = (pos, ori) w.r.t. base_footprint
                gripper_act = 0.0 (close) or 1.0 (open)
                adj_gripper = True accounts for the final pose of the tip of the gripper
                n_steps = number of steps to interpolate between the current and final pose
        '''
        pose = copy.deepcopy(pose)
        final_pos = pose[0]
        if adj_gripper:
            # TODO: This should be in the orientation of the gripper tip and not the base_footprint
            final_pos[0] -= self.adjust_gripper_length
            final_pos[1] += 0.03 # move it to the left by 5cm

        final_ori = pose[1]
        cur_arm_pose = self.left_arm_pose(frame=frame) if arm == 'left' else self.right_arm_pose(frame=frame)
        inter_pos = np.linspace(cur_arm_pose[:3], final_pos, n_steps)

        if gripper_act is None:
            gripper_act = env.tiago.gripper[arm].get_state()
            gripper_act = 0.0 if gripper_act < 0.5 else 1.0
            gripper_act = np.asarray([gripper_act])

        for pos in inter_pos[1:]:
            cur_arm_pose = self.left_arm_pose(frame=frame) if arm == 'left' else self.right_arm_pose(frame=frame)
            delta_pos = pos - cur_arm_pose[:3]
            delta_ori = R.from_quat(quat_diff(final_ori, cur_arm_pose[3:7])).as_euler('xyz')
            delta_act = np.concatenate(
                (delta_pos, delta_ori, gripper_act)
            )
            print(f'delta_pos: {delta_pos}', f'delta_ori: {delta_ori}')
            action = {'right': None, 'left': None, 'base': None}
            action[arm] = delta_act
            obs, reward, done, info = env.step(action)
        return obs, reward, done, info

    def step(self, env, rgb, depth, pcd, normals, arm, execute=True):
        '''
            This is an open-loop skill to push a button
            if execute is False, then it will only return the success flag
        '''
        pos = None
        if self.oracle_position:
            clicked_points = U.get_user_input(rgb)
            assert len(clicked_points) == 1
            print(f'clicked_points: {clicked_points}')
            pos = pcd[clicked_points[0][1], clicked_points[0][0]]
            normal = normals[clicked_points[0][1], clicked_points[0][0]]

        frame = 'base_footprint'
        # current_arm_pose in base_footprint
        current_arm_pose = self.left_arm_pose(frame=frame) if arm == 'left' else self.right_arm_pose(frame=frame)

        start_arm_pos, start_arm_ori = current_arm_pose[:3], current_arm_pose[3:7]
        approach_pos, approach_ori = self.get_approach_pose(pos, normal, frame=frame)

        # this must be based on the normal of the drawer
        goto_pos_base = pos #+ np.asarray([0.005, 0.0, 0.0])
        transform = None # no transformation from base_footprint is required.

        if self.debug:
            pcd_to_plot = pcd.reshape(-1,3)
            # transform it to the global frame
            if transform is not None:
                # pad it with 1.0 for homogeneous coordinates
                pcd_to_plot = np.concatenate((pcd_to_plot, np.ones((pcd_to_plot.shape[0], 1))), axis=1)
                pcd_to_plot = (transform @ pcd_to_plot.T).T
                pcd_to_plot = pcd_to_plot[:, :3]

            # concatenate the approach pose to the pcd
            pcd_to_plot = np.concatenate((pcd_to_plot, approach_pos.reshape(1,3)), axis=0)
            rgb_to_plot = np.concatenate((rgb.reshape(-1,3), np.asarray([[255.0, 0.0, 0.0]])), axis=0) # approach pose in red

            # concatenate the goto pose to the pcd
            pcd_to_plot = np.concatenate((pcd_to_plot, goto_pos_base.reshape(1,3)), axis=0)
            rgb_to_plot = np.concatenate((rgb_to_plot, np.asarray([[0.0, 0.0, 255.0]])), axis=0) # goto pose in blue

            # concatenate the current arm pose to the pcd
            pcd_to_plot = np.concatenate((pcd_to_plot, current_arm_pose[:3].reshape(1,3)), axis=0)
            rgb_to_plot = np.concatenate((rgb_to_plot, np.asarray([[0.0, 255.0, 0.0]])), axis=0) # current arm pose in green
            U.plotly_draw_3d_pcd(pcd_to_plot, rgb_to_plot)

        goto_args = {
            'env': env,
            'arm': arm,
            'frame': frame,
            'gripper_act': None,
            'adj_gripper': self.adjust_gripper,
        }

        success = True
        if execute:
            print("Moving to the approach pose")
            start_joint_angles = env.tiago.arms[arm].joint_reader.get_most_recent_msg()
            obs, reward, done, info = self.goto_pose(pose=(approach_pos, approach_ori), n_steps=2, **goto_args)
            approach_joint_angles = env.tiago.arms[arm].joint_reader.get_most_recent_msg()
            print("Moving to the goto pose")
            obs, reward, done, info = self.goto_pose(pose=(goto_pos_base, approach_ori), n_steps=2, **goto_args)
            print("Closing the gripper")
            self.close_gripper(env, arm)
            print("Moving back to the approach pose")
            cur_joint_angles = env.tiago.arms[arm].joint_reader.get_most_recent_msg()
            duration_scale = np.linalg.norm(approach_joint_angles-cur_joint_angles)
            env.tiago.arms[arm].write(approach_joint_angles, duration_scale)
            print("Opening the gripper")
            self.open_gripper(env, arm)
            print("Moving back to the different x-position")
            before_start_arm_pos = approach_pos - np.asarray([0.03, 0.0, 0.0])
            obs, reward, done, info = self.goto_pose(pose=(before_start_arm_pos, approach_ori), n_steps=2, **goto_args)
            print("Moving back to the start pose")
            # obs, reward, done, info = self.goto_pose(pose=(start_arm_pos, start_arm_ori), n_steps=3, **goto_args)
            cur_joint_angles = env.tiago.arms[arm].joint_reader.get_most_recent_msg()
            duration_scale = np.linalg.norm(start_joint_angles-cur_joint_angles)
            env.tiago.arms[arm].write(start_joint_angles, duration_scale)

        return success
