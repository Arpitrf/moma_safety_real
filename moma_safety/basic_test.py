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
    left_arm_enabled=False,
    right_gripper_type='robotiq2F-140',
    left_gripper_type=None,
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
current_right_ee_pose = env.tiago.arms["right"].arm_pose
print("current_right_ee_pose: ", current_right_ee_pose)

# rospy.sleep(2)
# env.tiago.gripper['right'].step(0.1)
# rospy.sleep(2)
# gripper_state = env.tiago.gripper["right"].get_state()
# print("gripper_state: ", gripper_state)

# from moma_safety.utils.object_config import object_config as OC
# object_name = "shelf"
# env.tiago.head.write_head_command(np.array([-0.6, -0.6]))

# # # Joint Controller: Move the right arm to home reset position
# reset_joint_pos = RP.PREGRASP_HIGH_2
# env.reset(reset_arms=True, reset_pose=reset_joint_pos, delay_scale_factor=12.0)

# # move ee by delta pose
# # Obtaining delta pose
# delta_pos = np.array([0.0, 0.0, 0.1])
# delta_ori = R.from_euler('xyz', [0.0, 0.0, 0.0]).as_quat()
# gripper_act = np.array([0.0])
# delta_pose = np.concatenate((delta_pos, delta_ori, gripper_act))
# print(f"delta_pos: {delta_pos}")
# print(f"delta_ori: {delta_ori}")
# action = {'right': delta_pose, 'left': None, 'base': None}
# print("Press c to continue...")
# breakpoint()
# obs, reward, done, info = env.step(action)
# print("info: ", info["arm_right"])



# # move to 0 orn of ee
# current_right_ee_pose = env.tiago.arms["right"].arm_pose
# target_right_ee_orn = np.array([0.0, 0.0, 0.0, 1.0])
# target_right_ee_pose = (current_right_ee_pose[:3], target_right_ee_orn)
# # Obtaining delta pose
# delta_pos = target_right_ee_pose[0] - current_right_ee_pose[:3]
# delta_ori = R.from_quat(target_right_ee_pose[1]) * R.from_quat(current_right_ee_pose[3:]).inv()
# delta_ori = delta_ori.as_quat()
# gripper_act = np.array([1.0])
# delta_pose = np.concatenate((delta_pos, delta_ori, gripper_act))
# print(f"delta_pos: {delta_pos}")
# print(f"delta_ori: {delta_ori}")
# action = {'right': delta_pose, 'left': None, 'base': None}
# print("Press c to continue...")
# breakpoint()
# obs, reward, done, info = env.step(action)
# print("info: ", info["arm_right"])

# # if red component is along +- z axis, rotate by 90 degrees about x axis and add some +z offset
# current_right_ee_pose = env.tiago.arms["right"].arm_pose
# ee_pose = T.pose2mat((current_right_ee_pose[:3], current_right_ee_pose[3:]))
# for theta in [0.6, 0.6, 0.6, 0.6]:
#     print("Rotating!!")
#     R_x = np.array([
#             [1, 0, 0],
#             [0, np.cos(theta), -np.sin(theta)],
#             [0, np.sin(theta), np.cos(theta)]
#         ])
#     ee_pose[:3, :3] = R_x @ ee_pose[:3, :3]
#     target_right_ee_pose = T.mat2pose(ee_pose)

#     delta_pos = target_right_ee_pose[0] - current_right_ee_pose[:3]
#     delta_ori = R.from_quat(target_right_ee_pose[1]) * R.from_quat(current_right_ee_pose[3:]).inv()
#     delta_ori = delta_ori.as_quat()
#     gripper_act = np.array([1.0])
#     delta_pose = np.concatenate((delta_pos, delta_ori, gripper_act))
#     print(f"delta_pos: {delta_pos}")
#     print(f"delta_ori: {delta_ori}")
#     action = {'right': delta_pose, 'left': None, 'base': None}
#     print("Press c to continue...")
#     breakpoint()
#     obs, reward, done, info = env.step(action)
#     print("info: ", info["arm_right"])



# # =============================== MOVE BASE ===============================
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
# transform = T.pose2mat((tf_map.get_transform(target_link=f'/base_footprint')))
# # print("transform: ", transform)
# pos_map = transform[:3, 3]
# # ori_map = R.from_matrix(transform[:3, :3]).as_euler('xyz')
# ori_map = T.mat2quat(transform[:3, :3])
# print(f"Current base pos in map: {pos_map}")
# print(f"Current base orn in map: {ori_map}")


# pos = [-0.3, -0.3, 0.0]
# pose_map = T.pose2mat((pos, [0.0, 0.0, 0.0, 1.0]))
# pose_map = transform @ pose_map

# # # pose_map = np.array([
# # #     [ 1.0,  0.0,  0.0, -4.142], 
# # #     [ 0.0,  1.0,  0.0,  8.949], 
# # #     [ 0.0,  0.0,  1.0,  0.0], 
# # #     [ 0.0,  0.0,  0.0,  1.0]
# # # ])

# pos_map = pose_map[:3, 3]
# ori_map = T.mat2quat(pose_map[:3, :3])
# # ori_map = R.from_matrix(pose_map[:3, :3]).as_euler('xyz')
# print(f"Target base pos in map: {pos_map}")
# print(f"Target base orn in map: {ori_map}")
# goal_pos_map = pos_map
# goal_ori_map = ori_map

# # # goal_pos = [0.5, 0.0, 0.0]
# # # goal_ori = [0.0, 0.0, 0.0, 1.0]

# if execute:
#     goal = create_move_base_goal((goal_pos_map, goal_ori_map))
#     state = send_move_base_goal(goal, client)
#     print(f"Move base state: {state}")


# tf_map = TFTransformListener('/map')
# transform = T.pose2mat((tf_map.get_transform(target_link=f'/base_footprint')))
# print("transform: ", transform)
# pos_map = transform[:3, 3]
# # ori_map = R.from_matrix(transform[:3, :3]).as_euler('xyz')
# ori_map = T.mat2quat(transform[:3, :3])
# print(f"Final base pos in map: {pos_map}")
# print(f"Final base orn in map: {ori_map}")



