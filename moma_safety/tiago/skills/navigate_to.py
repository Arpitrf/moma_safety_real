import os
import sys
import copy
import numpy as np

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

from moma_safety.tiago.utils.ros_utils import Publisher, Listener, TFTransformListener
from moma_safety.tiago.skills.base import SkillBase
import moma_safety.utils.utils as U
import moma_safety.utils.transform_utils as T # transform_utils

class NavigateToSkill(SkillBase):
    def __init__(self, oracle_action=False):
        super().__init__()
        self.oracle_action = oracle_action
        self.setup_listeners()

    def setup_listeners(self):
        self.tf_map = TFTransformListener('/map')

    def create_move_base_goal(self, pose):
        goal_pos = pose[0]
        goal_ori = pose[1]
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = goal_pos[0]
        goal.target_pose.pose.position.y = goal_pos[1]
        goal.target_pose.pose.position.z = goal_pos[2]
        goal.target_pose.pose.orientation.x = goal_ori[0]
        goal.target_pose.pose.orientation.y = goal_ori[1]
        goal.target_pose.pose.orientation.z = goal_ori[2]
        goal.target_pose.pose.orientation.w = goal_ori[3]
        return goal

    def step(self, env, rgb, depth, pcd, normals):
        '''
            action: Position, Quaternion (xyzw) of the goal
        '''
        print("NavigateToSkill: Move to the initial position")
        if self.oracle_action:
            clicked_points = U.get_user_input(rgb)
            assert len(clicked_points) == 1
            print(f'clicked_points: {clicked_points}')
            # w.r.t to the base_footprint since cam_extr is w.r.t. base_footprint
            # better for manipulation
            pos = pcd[clicked_points[0][1], clicked_points[0][0]]
            normal = normals[clicked_points[0][1], clicked_points[0][0]]

            transform = T.pose2mat((self.tf_map.get_transform(target_link=f'/base_footprint')))
            pos_map = np.concatenate((pos, [1.0]))
            pos_map = (transform @ pos_map)[:3] # w.r.t. the map frame

        # get the current position of the robot_base w.r.t. odom
        cur_pos = env.tiago.base.odom_listener.get_most_recent_msg()['pose'][:3]
        cur_ori = env.tiago.base.odom_listener.get_most_recent_msg()['pose'][3:]
        goal_pos = copy.deepcopy(cur_pos)
        goal_ori = copy.deepcopy(cur_ori)

        transform = T.pose2mat((self.tf_map.get_transform(target_link=f'/odom')))
        goal_pose = T.pose2mat((goal_pos, goal_ori))
        goal_pose_map = transform @ goal_pose
        goal_pos_map = goal_pose_map[:3, 3]
        goal_ori_map = T.mat2quat(goal_pose_map[:3, :3])

        goal_pos_map[0] = pos_map[0] # goal in map frame
        goal_pos_map[1] = pos_map[1] # goal in map frame

        goal = self.create_move_base_goal((goal_pos_map, goal_ori_map))
        state = self.send_move_base_goal(goal)
        return state

