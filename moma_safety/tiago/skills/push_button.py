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

class PushButtonSkill(SkillBase):
    def __init__(
            self,
            oracle_position: bool = False,
            adjust_gripper: bool = True,
            debug: bool = False,
        ):
        super().__init__()
        self.oracle_position = oracle_position
        self.setup_listeners()
        self.adjust_gripper = adjust_gripper
        self.adjust_gripper_length = 0.17 # 85
        self.debug = debug
        self.display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )
        self.robot = moveit_commander.RobotCommander()

    def create_move_group_msg(self, pose, move_group):
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.position.x = float(pose[0][0])
        pose_goal.position.y = float(pose[0][1])
        pose_goal.position.z = float(pose[0][2])
        pose_goal.orientation.x = float(pose[1][0])
        pose_goal.orientation.y = float(pose[1][1])
        pose_goal.orientation.z = float(pose[1][2])
        pose_goal.orientation.w = float(pose[1][3])
        return pose_goal

    def send_pose_goal(self, pose_goal, move_group):
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        return success

    def get_approach_pose(self, pos, normal, arm, frame):
        '''
            pos, normal: np.ndarray are w.r.t. the base_footprint
        '''
        assert frame in ['odom', 'base_footprint']
        approach_pos = pos + np.asarray([-0.1, 0.0, 0.0])
        approach_ori = R.from_rotvec(np.asarray([3.0*np.pi/2, 0.0, 0.0])).as_quat()
        if frame == 'odom':
            transform = T.pose2mat(self.tf_odom.get_transform('/base_footprint'))
            approach_pose = T.pose2mat((approach_pos, approach_ori))
            approach_pose = transform @ approach_pose
            approach_pos, approach_ori = T.mat2pose(approach_pose)
        return approach_pos, approach_ori

    def get_goto_pose(self, pos, normal, arm, approach_ori, frame):
        goto_pos = pos - np.asarray([0.01, 0.0, 0.0])
        goto_ori = approach_ori
        return goto_pos, goto_ori

    def check_box_is_in_scene(
            self,
            box_name,
            scene,
            box_is_known=False,
            box_is_attached=False,
            timeout=5
        ):
        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():
            # Test if the box is in attached objects
            attached_objects = scene.get_attached_objects([box_name])

            is_attached = len(attached_objects.keys()) > 0
            is_known = box_name in scene.get_known_object_names()

            # Test if we are in the expected state
            if (box_is_attached == is_attached) and (box_is_known == is_known):
                return True

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            seconds = rospy.get_time()

        # If we exited the while loop without returning then we timed out
        return False

    def add_object_to_scene(self, scene, box_name, box_pos, size=(0.1, 0.1, 0.1)):
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = "odom"
        box_pose.pose.position.x = box_pos[0]
        box_pose.pose.position.y = box_pos[1]
        box_pose.pose.position.z = box_pos[2]
        box_pose.pose.orientation.x = 0.0
        box_pose.pose.orientation.y = 0.0
        box_pose.pose.orientation.z = 0.0
        box_pose.pose.orientation.w = 1.0
        scene.add_box(box_name, box_pose, size=size)
        added = self.check_box_is_in_scene(box_name, scene, box_is_known=True, box_is_attached=False, timeout=5)
        assert added, f'Failed to add object to scene: {box_name}'
        return added

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
            pos = pcd[clicked_points[0][1], clicked_points[0][0]] # pos of pad in the base_footprint frame
            normal = normals[clicked_points[0][1], clicked_points[0][0]]
            orig_pos = copy.deepcopy(pos)
            opp_arm = 'right' if arm == 'left' else 'left'
            right_pad_wrt_base = T.pose2mat(self.tf_base.get_transform(f'/gripper_{arm}_{opp_arm}_inner_finger_pad'))
            right_arm_wrt_base = T.pose2mat(self.tf_base.get_transform(f'/arm_{arm}_tool_link'))
            # calculate the pos of the arm_tool_link in the base_footprint frame
            translation = right_pad_wrt_base[:3, 3] - right_arm_wrt_base[:3, 3] - np.asarray([0.0, 0.03, 0.0]) # this is some small offset observed in the real robot
            pos = pos - translation - np.asarray([0.02, 0.0, 0.0]) # this is offset for avoid collision

        frame = 'base_footprint'
        current_arm_pose = self.left_arm_pose(frame=frame) if arm == 'left' else self.right_arm_pose(frame=frame)

        start_arm_pos, start_arm_ori = current_arm_pose[:3], current_arm_pose[3:7]
        approach_pos, approach_ori = self.get_approach_pose(pos, normal, arm=arm, frame=frame)
        goto_pos_base, goto_ori = self.get_goto_pose(pos, normal, arm=arm, approach_ori=approach_ori, frame=frame)
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

            # # add the orig_pos
            pcd_to_plot = np.concatenate((pcd_to_plot, orig_pos.reshape(1,3)), axis=0)
            rgb_to_plot = np.concatenate((rgb_to_plot, np.asarray([[0.0, 255.0, 0.0]])), axis=0) # orig pos in green
            # # concatenate the goto pose to the pcd
            pcd_to_plot = np.concatenate((pcd_to_plot, pos.reshape(1,3)), axis=0)
            rgb_to_plot = np.concatenate((rgb_to_plot, np.asarray([[0.0, 0.0, 255.0]])), axis=0) # goto pose in blue

            # concatenate the goto pose to the pcd
            # pcd_to_plot = np.concatenate((pcd_to_plot, goto_pos_base.reshape(1,3)), axis=0)
            # rgb_to_plot = np.concatenate((rgb_to_plot, np.asarray([[0.0, 0.0, 255.0]])), axis=0) # goto pose in blue

            # concatenate the current arm pose to the pcd
            # pcd_to_plot = np.concatenate((pcd_to_plot, current_arm_pose[:3].reshape(1,3)), axis=0)
            # rgb_to_plot = np.concatenate((rgb_to_plot, np.asarray([[128.0, 128.0, 128.0]])), axis=0) # current arm pose in grey

            U.plotly_draw_3d_pcd(pcd_to_plot, rgb_to_plot)

        duration_scale_factor = 2.0
        goto_args = {
            'env': env,
            'arm': arm,
            'frame': frame,
            'gripper_act': None,
            'adj_gripper': False, # we do not adjust the gripper here, since the pos is in tool_link frame
            'duration_scale_factor': duration_scale_factor, # go two times slower
        }

        success = True
        if execute:
            print("Moving to the approach pose")
            start_joint_angles = env.tiago.arms[arm].joint_reader.get_most_recent_msg()
            obs, reward, done, info = self.arm_goto_pose(pose=(approach_pos, approach_ori), n_steps=2, **goto_args)
            approach_joint_angles = env.tiago.arms[arm].joint_reader.get_most_recent_msg()
            print("Moving to the goto pose")
            obs, reward, done, info = self.arm_goto_pose(pose=(goto_pos_base, approach_ori), n_steps=2, **goto_args)
            print("Moving back to the approach pose")
            # rospy.sleep(2)
            cur_joint_angles = env.tiago.arms[arm].joint_reader.get_most_recent_msg()
            duration_scale = np.linalg.norm(approach_joint_angles-cur_joint_angles)*duration_scale_factor
            env.tiago.arms[arm].write(approach_joint_angles, duration_scale)
            print("Moving back to the start pose")
            cur_joint_angles = env.tiago.arms[arm].joint_reader.get_most_recent_msg()
            duration_scale = np.linalg.norm(start_joint_angles-cur_joint_angles)*duration_scale_factor
            env.tiago.arms[arm].write(start_joint_angles, duration_scale)

        return success
