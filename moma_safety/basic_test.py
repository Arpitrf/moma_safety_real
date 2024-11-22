import numpy as np
np.set_printoptions(precision=3, suppress=True)
import rospy

from moma_safety.tiago.tiago_gym import TiagoGym
from moma_safety.tiago import RESET_POSES as RP
import moma_safety.utils.transform_utils as T # transform_utils
from moma_safety.tiago.utils.ros_utils import TFTransformListener
from moma_safety.tiago.utils.transformations import quat_diff
from scipy.spatial.transform import Rotation as R


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
cur_pos = env.tiago.base.odom_listener.get_most_recent_msg()['pose'][:3]
cur_ori = env.tiago.base.odom_listener.get_most_recent_msg()['pose'][3:]
print("cur_pose: ", cur_pos, cur_ori)

current_right_arm_joint_angles = env.tiago.arms["right"].joint_reader.get_most_recent_msg()
current_left_arm_joint_angles = env.tiago.arms["left"].joint_reader.get_most_recent_msg()
print("current_right_arm_joint_angles: ", current_right_arm_joint_angles)
print("current_left_arm_joint_angles: ", current_left_arm_joint_angles)

# Get current ee pose
# current_right_ee_pose = env.tiago.arms["right"].arm_pose

# # Joint Controller: Move the right arm to vertical reset position
# reset_joint_pos = RP.VERTICAL_H
# env.reset(reset_arms=True, reset_pose=reset_joint_pos, allowed_delay_scale=6.0)

# # # Joint Controller: Move the right arm to home reset position
# reset_joint_pos = RP.HOME_LEFT
# env.reset(reset_arms=True, reset_pose=reset_joint_pos, allowed_delay_scale=6.0)

# # IK Controller: Move right arm 10 cm forward
# current_right_ee_pose = env.tiago.arms["right"].arm_pose
# print(f"current_right_ee_pose: {current_right_ee_pose}")
# target_right_ee_pose = np.array([0.46, -0.20, 0.76, -0.202, -0.093, 0.614, 0.757])
# delta_pos = target_right_ee_pose[:3] - current_right_ee_pose[:3]
# delta_ori = quat_diff(target_right_ee_pose[3:], current_right_ee_pose[3:])
# delta_pose = np.concatenate((delta_pos, delta_ori))
# print(f"delta_pos: {delta_pos}")
# print(f"delta_ori: {delta_ori}")
# action = {'right': None, 'left': None, 'base': None}
# action["right"] = delta_pose
# # breakpoint()
# obs, reward, done, info = env.step(action)

# target_right_ee_pose = current_right_ee_pose + np.array([0.1, 0, 0, 0, 0, 0, 0])
delta_pos = np.array([0.0, 0.0, 0.1])
delta_ori = np.array([0.0, 0.0, 0.0, 0.0])
gripper_act = np.array([0.0])
delta_act = np.concatenate(
    (delta_pos, delta_ori, gripper_act)
)
print(f'delta_pos: {delta_pos}', f'delta_ori: {delta_ori}')
action = {'right': None, 'left': None, 'base': None}
action["right"] = delta_act
obs, reward, done, info = env.step(action)


# # Move base 10 cm forward
# import actionlib
# from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


# def send_move_base_goal(goal, client):
#     print("sending move base goal")
#     client.send_goal(goal)
#     wait = client.wait_for_result()
#     result = client.get_result()
#     state = client.get_state()
#     print("State from move_base: ", state)
#     rospy.sleep(2) # the robot takes some time to reach the goal
#     return state

# def create_move_base_goal(pose):
#     goal_pos = pose[0]
#     goal_ori = pose[1]
#     goal = MoveBaseGoal()
#     goal.target_pose.header.frame_id = "map"
#     goal.target_pose.header.stamp = rospy.Time.now()
#     goal.target_pose.pose.position.x = goal_pos[0]
#     goal.target_pose.pose.position.y = goal_pos[1]
#     goal.target_pose.pose.position.z = goal_pos[2]
#     goal.target_pose.pose.orientation.x = goal_ori[0]
#     goal.target_pose.pose.orientation.y = goal_ori[1]
#     goal.target_pose.pose.orientation.z = goal_ori[2]
#     goal.target_pose.pose.orientation.w = goal_ori[3]
#     return goal


# print("waiting for move_base server")
# client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
# client.wait_for_server()
# # breakpoint()
# # print("move_base server found")

# debug = False
# execute = True

# tf_map = TFTransformListener('/map')
# pos = [0.3, 0.0, 0.0]
# transform = T.pose2mat((tf_map.get_transform(target_link=f'/base_footprint')))
# pose_map = T.pose2mat((pos, [0.0, 0.0, 0.0, 1.0]))
# # breakpoint()
# pose_map = transform @ pose_map
# pos_map = pose_map[:3, 3]
# ori_map = T.mat2quat(pose_map[:3, :3])
# print(f"Calculated Position in map: {pos_map}")
# print(f"Calculated Orientation in map: {ori_map}")
# goal_pos_map = pos_map
# goal_ori_map = ori_map # goal in map frame

# # goal_pos = [0.5, 0.0, 0.0]
# # goal_ori = [0.0, 0.0, 0.0, 1.0]

# if execute:
#     goal = create_move_base_goal((goal_pos_map, goal_ori_map))
#     state = send_move_base_goal(goal, client)
#     print(f"Move base state: {state}")

# # if debug:
# #     pcd_wrt_map = np.concatenate((pcd.reshape(-1, 3), np.ones((pcd.reshape(-1,3).shape[0], 1))), axis=1)
# #     transform = T.pose2mat((self.tf_map.get_transform(target_link=f'/base_footprint')))
# #     pcd_wrt_map = (transform @ pcd_wrt_map.T).T
# #     pcd_wrt_map = pcd_wrt_map[:, :3]
# #     pcd_to_plot = pcd_wrt_map.reshape(-1,3)
# #     rgb_to_plot = rgb.reshape(-1,3)

# #     pcd_to_plot = np.concatenate((pcd_to_plot, goal_pos_map.reshape(1,3)), axis=0)
# #     rgb_to_plot = np.concatenate((rgb_to_plot.reshape(-1,3), np.asarray([[255.0, 0.0, 0.0]])), axis=0) # goal pos in red

# #     U.plotly_draw_3d_pcd(pcd_to_plot, rgb_to_plot)


