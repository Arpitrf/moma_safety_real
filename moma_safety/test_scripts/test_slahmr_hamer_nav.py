import math
import rospy
import actionlib
import numpy as np
from scipy.spatial.transform import Rotation as R

import moma_safety.utils.transform_utils as T # transform_utils
from moma_safety.tiago.utils.ros_utils import TFTransformListener
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from moma_safety.utils.rviz_utils import visualize_pose

np.set_printoptions(precision=3, suppress=True)
# rospy.init_node('test_slahmr_hamer_nav')

def send_move_base_goal(goal, client):
    print("sending move base goal")
    client.send_goal(goal)
    wait = client.wait_for_result()
    result = client.get_result()
    state = client.get_state()
    print("State from move_base: ", state)
    rospy.sleep(2) # the robot takes some time to reach the goal
    return state

def create_move_base_goal(pose):
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

def navigate_primitive():
    video_name = "nav_test8"
    # "/home/arpit/projects/og_prior_npz_files/{data_seq}_prior_results.npz"
    data = np.load(f"slahmr_hamer_outputs/{video_name}_prior_results.npz")
    body_trans = data["body_positions"]
    body_orient = data["body_orientations"]
    hand_positions = data["hand_positions"]
    hand_rotations = data["hand_orientations"]

    print(f"\n\n\nhand_positions shape : {hand_positions.shape}")
    print(f"hand_rotations shape: {hand_rotations.shape}")
    print(f"body_trans shape: {body_trans.shape}")
    print(f"body_orient shape: {body_orient.shape}\n\n\n")

    # --------------------------------------- WORKING -------------------------------------------
    points = [(t[0], t[2]) for t in body_trans]
    root_orient = data["body_orientations"]

    # Getting orientations
    yaw = []
    for i in range(root_orient.shape[0]):
        rotmat = R.from_euler("xyz", root_orient[i]).as_matrix()
        unit_vector = np.array([1., 0., 0.])
        direction_vector = np.matmul(rotmat, unit_vector)
        orientation = math.atan2(direction_vector[2], direction_vector[0])
        # yaw.append(orientation)
        yaw.append(orientation - 3.14/2)

    init_matrix = np.array([
        [np.cos(yaw[0]), np.sin(yaw[0])],
        [-np.sin(yaw[0]), np.cos(yaw[0])]
    ])

    points_init_frame = []
    yaw_init_frame = []

    for i in range(len(points)):
        delta_pos = np.array([points[i][0] - points[0][0], points[i][1] - points[0][1]])
        pos_robot_frame = np.matmul(init_matrix, np.transpose(delta_pos))
        # print("pos_robot_frame: ", pos_robot_frame)

        points_init_frame.append((pos_robot_frame[0], pos_robot_frame[1]))
        yaw_init_frame.append(yaw[i] - yaw[0])
        # print("yaw[i] - yaw[0]: ", yaw[i] - yaw[0])

    points_init_frame = np.array(points_init_frame)
    yaw_init_frame = np.array(yaw_init_frame)
    print("points_init_frame: ", points_init_frame.shape)
    print("yaw_init_frame: ", yaw_init_frame.shape)


    tf_map = TFTransformListener('/map')
    T_R0_to_world = T.pose2mat((tf_map.get_transform(target_link=f'/base_footprint')))
    pos_map = T_R0_to_world[:3, 3]
    ori_map = T.mat2quat(T_R0_to_world[:3, :3])
    print(f"Calculated Position in map for step 0: {pos_map}")
    print(f"Calculated Orientation in map for step 0: {ori_map}")

    # visualize the base pose
    visualize_pose(pos_map, ori_map, ref_frame="map", id=i)
    # breakpoint()

    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    client.wait_for_server()
    step_size = 25
    execute = True
    for i in range(0, len(points_init_frame), step_size):
        print(f"points_init_frame[{i}]: {points_init_frame[i]}")
        print(f"yaw_init_frame[{i}]: {yaw_init_frame[i]}")
        T_Ri_to_R0 = np.eye(4)
        T_Ri_to_R0[:2, 3] = points_init_frame[i]
        T_Ri_to_R0[:3, :3] = R.from_euler("z", yaw_init_frame[i]).as_matrix()
        print("T_Ri_to_R0: ", T_Ri_to_R0)

        # move the robot
        T_Ri_to_world = T_R0_to_world @ T_Ri_to_R0
        pos_map = T_Ri_to_world[:3, 3]
        ori_map = T.mat2quat(T_Ri_to_world[:3, :3])
        print(f"Calculated Position in map for step {i}: {pos_map}")
        print(f"Calculated Orientation in map for step {i}: {ori_map}")
        goal_pos_map = pos_map
        goal_ori_map = ori_map # goal in map frame

        # visualize the base pose
        # T_R_current_to_world = T.pose2mat((tf_map.get_transform(target_link=f'/base_footprint')))
        # T_Ri_to_Rc = np.linalg.inv(T_R_current_to_world) @ T_Ri_to_world
        visualize_pose(pos_map, ori_map, ref_frame="map", id=i)
        inp = input("Press Y to execute action and N to skip")
        if inp == 'Y':
            execute = True
        else:
            execute = False
        if execute:
            goal = create_move_base_goal((goal_pos_map, goal_ori_map))
            state = send_move_base_goal(goal, client)
            print(f"Move base state: {state}")
