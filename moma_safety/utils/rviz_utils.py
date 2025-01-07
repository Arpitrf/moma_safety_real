import rospy
import numpy as np

from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from moma_safety.tiago.utils.transformations import add_quats

def rviz_visualize_trajectories(actions, start_position):
    total_lines = 0
    for i in range(actions.shape[0]):
        trajectory = actions[i, :, 3:6]
        prev_position = start_position

        for j in range(trajectory.shape[0]):
            direction = trajectory[j]  # Direction vector at this waypoint
            magnitude = np.linalg.norm(direction)  # Magnitude of the direction vector
            direction_normalized = direction / magnitude if magnitude != 0 else direction  # Normalize the direction
            step = magnitude * direction_normalized

            next_position = prev_position + step
            visualize_marker(start_position=prev_position, end_position=next_position, id=total_lines)
            prev_position = prev_position + step  # Move to the new position
            total_lines += 1

def visualize_marker(start_position, end_position, id):
    marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
    rospy.sleep(1)
    
    # show line between current pose and target pose
    line_marker = Marker()
    line_marker.header.frame_id = "base_footprint"  # Ensure this matches your robot's frame
    line_marker.header.stamp = rospy.Time.now()
    line_marker.ns = "line"
    line_marker.id = id
    line_marker.type = Marker.LINE_STRIP
    line_marker.action = Marker.ADD
    line_marker.scale.x = 0.01  # Line width

    # Set the color of the line
    line_marker.color.r = 1.0
    line_marker.color.g = 0.0
    line_marker.color.b = 0.0
    line_marker.color.a = 1.0

    # Add points to the line strip
    # start_point = pose_msg.pose.position
    start_point = PoseStamped().pose.position
    start_point.x = start_position[0]
    start_point.y = start_position[1]
    start_point.z = start_position[2]
    end_point = PoseStamped().pose.position
    end_point.x = end_position[0]
    end_point.y = end_position[1]
    end_point.z = end_position[2]

    line_marker.points.append(start_point)
    line_marker.points.append(end_point)

    marker_pub.publish(line_marker)

def visualize_pose(pos, ori, ref_frame="base_footprint", id=0):
    pub = rospy.Publisher('/target_pose', PoseStamped, queue_size=10)
    start_time = rospy.get_time()
    while rospy.get_time() - start_time < 2:  # Run for 2 seconds
    # while not rospy.is_shutdown():

        # show current pose
        current_pose_msg = PoseStamped()
        current_pose_msg.header.stamp = rospy.Time.now()
        current_pose_msg.header.frame_id = ref_frame  # Change to your robot's base frame
        # current_pose_msg.id = id

        current_pose_msg.pose.position.x = pos[0]
        current_pose_msg.pose.position.y = pos[1]
        current_pose_msg.pose.position.z = pos[2]

        current_pose_msg.pose.orientation.x = ori[0]
        current_pose_msg.pose.orientation.y = ori[1]
        current_pose_msg.pose.orientation.z = ori[2]
        current_pose_msg.pose.orientation.w = ori[3]

        pub.publish(current_pose_msg)
        # rate.sleep()

def publish_target_pose(env, delta_pos, delta_ori, color=[1.0, 0.0, 0.0, 1.0], id=0):
    pub = rospy.Publisher('/target_pose', PoseStamped, queue_size=10)
    marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
    rospy.sleep(1)
    rate = rospy.Rate(10)  # 10 Hz

    # Assuming you have the current pose
    current_right_ee_pose = env.tiago.arms["right"].arm_pose
    current_pos = current_right_ee_pose[:3]
    current_ori = current_right_ee_pose[3:]
    # current_pos = np.array([0.5, 0.0, 0.5])  # Example current position
    # current_ori = np.array([0.0, 0.0, 0.0, 0.1])  # Example current orientation (quaternion)

    # Calculate new pose
    new_pos = current_pos + delta_pos
    new_ori = add_quats(delta=delta_ori, source=current_ori)

    if id == 0:
        delete_all_marker = Marker()
        delete_all_marker.action = Marker.DELETEALL
        marker_pub.publish(delete_all_marker)
        rospy.sleep(1)

    # show line between current pose and target pose
    line_marker = Marker()
    line_marker.header.frame_id = "base_footprint"  # Ensure this matches your robot's frame
    line_marker.header.stamp = rospy.Time.now()
    line_marker.ns = "line"
    line_marker.id = id
    line_marker.type = Marker.LINE_STRIP
    line_marker.action = Marker.ADD
    line_marker.scale.x = 0.01  # Line width

    # Set the color of the line
    line_marker.color.r = color[0]
    line_marker.color.g = color[1]
    line_marker.color.b = color[2]
    line_marker.color.a = color[3]

    # Add points to the line strip
    # start_point = pose_msg.pose.position
    start_point = PoseStamped().pose.position
    start_point.x = current_pos[0]
    start_point.y = current_pos[1]
    start_point.z = current_pos[2]
    end_point = PoseStamped().pose.position
    end_point.x = new_pos[0]
    end_point.y = new_pos[1]
    end_point.z = new_pos[2]

    line_marker.points.append(start_point)
    line_marker.points.append(end_point)

    marker_pub.publish(line_marker)


    # start_time = rospy.get_time()
    # while rospy.get_time() - start_time < 2:  # Run for 2 seconds
    # # while not rospy.is_shutdown():

    #     # show current pose
    #     current_pose_msg = PoseStamped()
    #     current_pose_msg.header.stamp = rospy.Time.now()
    #     current_pose_msg.header.frame_id = "base_footprint"  # Change to your robot's base frame

    #     current_pose_msg.pose.position.x = current_pos[0]
    #     current_pose_msg.pose.position.y = current_pos[1]
    #     current_pose_msg.pose.position.z = current_pos[2]

    #     current_pose_msg.pose.orientation.x = current_ori[0]
    #     current_pose_msg.pose.orientation.y = current_ori[1]
    #     current_pose_msg.pose.orientation.z = current_ori[2]
    #     current_pose_msg.pose.orientation.w = current_ori[3]

    #     pub.publish(current_pose_msg)
    #     # rate.sleep()

    #     # show target pose
    #     target_pose_msg = PoseStamped()
    #     target_pose_msg.header.stamp = rospy.Time.now()
    #     target_pose_msg.header.frame_id = "base_footprint"  # Change to your robot's base frame

    #     target_pose_msg.pose.position.x = new_pos[0]
    #     target_pose_msg.pose.position.y = new_pos[1]
    #     target_pose_msg.pose.position.z = new_pos[2]

    #     target_pose_msg.pose.orientation.x = new_ori[0]
    #     target_pose_msg.pose.orientation.y = new_ori[1]
    #     target_pose_msg.pose.orientation.z = new_ori[2]
    #     target_pose_msg.pose.orientation.w = new_ori[3]

    #     pub.publish(target_pose_msg)