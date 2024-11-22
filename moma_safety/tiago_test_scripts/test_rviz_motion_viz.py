import rospy
import numpy as np

from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from moma_safety.tiago.tiago_gym import TiagoGym
from moma_safety.tiago.utils.transformations import add_quats

def publish_target_pose(env, delta_pos, delta_ori):
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

    delete_all_marker = Marker()
    delete_all_marker.action = Marker.DELETEALL
    marker_pub.publish(delete_all_marker)
    rospy.sleep(1)

    # show line between current pose and target pose
    line_marker = Marker()
    line_marker.header.frame_id = "base_footprint"  # Ensure this matches your robot's frame
    line_marker.header.stamp = rospy.Time.now()
    line_marker.ns = "line"
    line_marker.id = 0
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


    while not rospy.is_shutdown():

        # show current pose
        current_pose_msg = PoseStamped()
        current_pose_msg.header.stamp = rospy.Time.now()
        current_pose_msg.header.frame_id = "base_footprint"  # Change to your robot's base frame

        current_pose_msg.pose.position.x = current_pos[0]
        current_pose_msg.pose.position.y = current_pos[1]
        current_pose_msg.pose.position.z = current_pos[2]

        current_pose_msg.pose.orientation.x = current_ori[0]
        current_pose_msg.pose.orientation.y = current_ori[1]
        current_pose_msg.pose.orientation.z = current_ori[2]
        current_pose_msg.pose.orientation.w = current_ori[3]

        pub.publish(current_pose_msg)
        # rate.sleep()

        # show target pose
        target_pose_msg = PoseStamped()
        target_pose_msg.header.stamp = rospy.Time.now()
        target_pose_msg.header.frame_id = "base_footprint"  # Change to your robot's base frame

        target_pose_msg.pose.position.x = new_pos[0]
        target_pose_msg.pose.position.y = new_pos[1]
        target_pose_msg.pose.position.z = new_pos[2]

        target_pose_msg.pose.orientation.x = new_ori[0]
        target_pose_msg.pose.orientation.y = new_ori[1]
        target_pose_msg.pose.orientation.z = new_ori[2]
        target_pose_msg.pose.orientation.w = new_ori[3]

        pub.publish(target_pose_msg)
        # rate.sleep()


if __name__ == '__main__':
    rospy.init_node('target_pose_publisher', anonymous=True)
    env = TiagoGym(
        frequency=10,
        right_arm_enabled=True,
        left_arm_enabled=True,
        right_gripper_type='robotiq2F-140',
        left_gripper_type='robotiq2F-85',
        base_enabled=True,
        torso_enabled=False,
    )
    print("env created")
    try:
        delta_pos = np.array([0.3, 0.0, 0.0])  # Example delta position
        # delta_ori = np.array([0.707, 0.0, 0.707, 0.0])  # Example delta orientation (quaternion)
        delta_ori = np.array([0.0, 0.0, 0.0, 1.0])  # Example delta orientation (quaternion)
        publish_target_pose(env, delta_pos, delta_ori)
    except rospy.ROSInterruptException as e:
        print(e)
        pass
