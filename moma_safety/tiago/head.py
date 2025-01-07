import rospy
import numpy as np

from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R

from moma_safety.tiago.utils.ros_utils import Publisher, create_pose_command, TFTransformListener, Listener
from moma_safety.tiago.utils.camera_utils import Camera
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg  import JointTrajectoryControllerState

from threading import Thread

from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
import actionlib


class TiagoHead:

    def __init__(self, head_policy) -> None:
        self.head_enabled = head_policy is not None
        self.head_policy = head_policy

        self.img_topic = "/xtion/rgb/image_raw"
        self.depth_topic = "/xtion/depth/image_raw"
        self.camera_info_topic = "/xtion/rgb/camera_info"
        self.head_camera = Camera(
            img_topic=self.img_topic,
            depth_topic=self.depth_topic,
            camera_info_topic=self.camera_info_topic,
        )

        self.setup_actors()
        self.setup_listeners()

        self.camera_intrinsics = np.asarray(list(self.head_camera.camera_info.K)).reshape(3,3)
        self.camera_extrinsics = None

    def setup_listeners(self):
        self.camera_reader = TFTransformListener('/base_footprint')
        def process_head(message):
                return message.actual.positions
        self.head_sub = Listener('/head_controller/state', JointTrajectoryControllerState, post_process_func=process_head)

    def setup_actors(self):
        self.head_writer = None
        if self.head_enabled:
            self.head_writer = Publisher('/whole_body_kinematic_controller/gaze_objective_xtion_optical_frame_goal', PoseStamped)
        self.head_pub = Publisher('/head_controller/command', JointTrajectory)

    def write(self, trans, quat):
        if self.head_enabled:
            self.head_writer.write(create_pose_command(trans, quat))

    def get_camera_obs(self):
        return self.head_camera.get_camera_obs()

    def step(self, env_action):
        pos, quat = self.head_policy.get_action(env_action)
        if pos is None:
            return
        self.write(pos, quat)
        return {}

    def reset_step(self, env_action):
        pos, quat = self.head_policy.get_action(env_action, euler=False)
        if pos is None:
            return
        self.write(pos, quat)

    @property
    def camera_extrinsic(self):
        pos, quat = self.camera_reader.get_transform(target_link='/xtion_rgb_optical_frame')
        if pos is None:
            return None
        extr_rotation = R.from_quat(quat).as_matrix()
        R_world_cam = extr_rotation
        T_world_cam = np.array([
            [R_world_cam[0][0], R_world_cam[0][1], R_world_cam[0][2], pos[0]],
            [R_world_cam[1][0], R_world_cam[1][1], R_world_cam[1][2], pos[1]],
            [R_world_cam[2][0], R_world_cam[2][1], R_world_cam[2][2], pos[2]],
            [0, 0, 0, 1]
        ])
        return T_world_cam
    
    def write_head_command(self, head_positions):
        
        # Create the JointTrajectory message
        trajectory_msg = JointTrajectory()
        trajectory_msg.header.seq = 0
        trajectory_msg.header.stamp = rospy.Time(0)
        trajectory_msg.header.frame_id = ''
        trajectory_msg.joint_names = ['head_1_joint', 'head_2_joint']

        # Create a JointTrajectoryPoint
        point = JointTrajectoryPoint()
        point.positions = [head_positions[0], head_positions[1]]
        point.velocities = []  # Optional: empty list
        point.accelerations = []  # Optional: empty list
        point.effort = []  # Optional: empty list
        point.time_from_start = rospy.Duration(1)

        trajectory_msg.points.append(point)

        self.head_pub.write(trajectory_msg)


class TiagoHeadPolicy:

    def get_action(self, env_action, euler=True):
        '''
            if euler is true then env_action[arm] is expected to be a 7 dimensional vector -> pos(3), rot(3), grip(1)
            otherwise, rot(4) is expected as a quat
        '''
        raise NotImplementedError

class FollowHandPolicy(TiagoHeadPolicy):

    def __init__(self, arm='right'):
        super().__init__()
        assert arm in ['right', 'left']

        self.arm = arm

    def get_action(self, env_action, euler=True):
        if env_action[self.arm] is None:
            return None, None

        position = env_action[self.arm][:3]
        return position, [0, 0, 0, 1]


class LookAtFixedPoint(TiagoHeadPolicy):

    def __init__(self, point) -> None:
        super().__init__()

        self.point = point

    def get_action(self, env_action, euler=True):
        position = self.point[:3]

        return position, [0, 0, 0, 1]




