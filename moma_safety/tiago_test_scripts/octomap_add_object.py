import copy
import sys
import rospy
from geometry_msgs.msg import Pose, Point, Quaternion
import moveit_commander
import numpy as np
import geometry_msgs
from scipy.spatial.transform import Rotation as R

import moma_safety.utils.utils as U
import moma_safety.utils.transform_utils as T # transform_utils
from moma_safety.tiago.tiago_gym import TiagoGym
import moma_safety.utils.vision_utils as VU # vision_utils
from moma_safety.tiago.utils.ros_utils import Publisher, Listener, TFTransformListener

rospy.init_node('move_base_python', anonymous=True)
moveit_commander.roscpp_initialize(sys.argv)

def create_move_group_msg(pose, move_group):
    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.position.x = float(pose[0][0])
    pose_goal.position.y = float(pose[0][1])
    pose_goal.position.z = float(pose[0][2])
    pose_goal.orientation.x = float(pose[1][0])
    pose_goal.orientation.y = float(pose[1][1])
    pose_goal.orientation.z = float(pose[1][2])
    pose_goal.orientation.w = float(pose[1][3])
    return pose_goal

def send_pose_goal(pose_goal, move_group):
    move_group.set_pose_target(pose_goal)
    success = move_group.go(wait=True)
    move_group.stop()
    move_group.clear_pose_targets()
    return success

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
tf_listener = TFTransformListener('/base_footprint')
tf_odom = TFTransformListener('/odom')
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

clicked_points = U.get_user_input(rgb)
# clicked_points = [(384, 192)]
print(clicked_points)
robot = moveit_commander.RobotCommander()

group_names = robot.get_group_names()
print("============ Robot Groups:", robot.get_group_names())

group_name = 'arm_right'
scene = moveit_commander.PlanningSceneInterface()
move_group = moveit_commander.MoveGroupCommander(group_name)

object_pos = pcd[clicked_points[0][1], clicked_points[0][0]] # base_footprint frame

n_object_pos = copy.deepcopy(object_pos)

object_pos[0] -= 0.2
object_pos[1] += 0.03
approach_ori = R.from_rotvec(np.asarray([np.pi/2, 0.0, 0.0])).as_quat()

transform = T.pose2mat(tf_odom.get_transform('/base_footprint'))
object_pose_odom = transform @ T.pose2mat((object_pos, approach_ori))
object_pos_odom = object_pose_odom[:3, 3]
object_ori_odom = R.from_matrix(object_pose_odom[:3, :3]).as_quat()

n_object_pose_odom = transform @ T.pose2mat((n_object_pos, approach_ori))
n_object_pos_odom = n_object_pose_odom[:3, 3]

# move_group_msg = create_move_group_msg((object_pos_odom, object_ori_odom), move_group)
# send_pose_goal(move_group_msg, move_group)

box_pose = geometry_msgs.msg.PoseStamped()
box_pose.header.frame_id = "odom"
box_pose.pose.position.x = n_object_pos_odom[0]
box_pose.pose.position.y = n_object_pos_odom[1]
box_pose.pose.position.z = n_object_pos_odom[2]
box_pose.pose.orientation.x = 0.0
box_pose.pose.orientation.y = 0.0
box_pose.pose.orientation.z = 0.0
box_pose.pose.orientation.w = 1.0
box_name = "box"

def check_box_is_in_scene(
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

scene.add_box(box_name, box_pose, size=(0.1, 0.1, 0.1))
box_in_scene = check_box_is_in_scene(box_name, scene, box_is_known=True, box_is_attached=False, timeout=5)
print(box_in_scene)
print(scene.get_known_object_names())
print(scene.get_attached_objects())
print(scene.get_object_poses([box_name]))
import ipdb; ipdb.set_trace()

grasping_group = 'gripper_right'
touch_links = robot.get_link_names(group=grasping_group)
eef_link = move_group.get_end_effector_link()
scene.attach_box(eef_link, box_name, touch_links=touch_links)
import ipdb; ipdb.set_trace()


move_group_msg = create_move_group_msg((object_pos_odom, object_ori_odom), move_group)
send_pose_goal(move_group_msg, move_group)

scene.remove_attached_object(eef_link, name=box_name)
scene.remove_world_object(box_name)
rospy.sleep(1)
