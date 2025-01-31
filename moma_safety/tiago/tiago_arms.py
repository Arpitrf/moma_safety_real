import os
import numpy as np

import rospy
from std_msgs.msg import Header
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import WrenchStamped
from actionlib_msgs.msg import GoalID
from pal_common_msgs.msg import EmptyActionGoal
from controller_manager_msgs.srv import SwitchController

from moma_safety.tiago.utils.ros_utils import Publisher, Listener, TFTransformListener
from moma_safety.tiago.utils.transformations import euler_to_quat, quat_to_euler, add_angles, quat_to_rmat, add_quats
from tracikpy import TracIKSolver
from scipy.spatial.transform import Rotation as R

def joint_process_func(data):
    return np.array(data.actual.positions)

class TiagoArms:

    def __init__(
            self,
            arm_enabled,
            gripper_type,
            side='right',
            torso_enabled=False,
            torso=None,
        ) -> None:
        self.arm_enabled = arm_enabled
        self.side = side
        self.gripper_type = gripper_type

        self.torso = torso
        self.torso_enabled = torso_enabled
        self.ik_base_link = 'base_footprint' if torso_enabled else 'torso_lift_link'

        self.urdf_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'urdf/tiago.urdf')

        self.setup_listeners()
        self.setup_actors()

    def switch_controller(self, start_controllers=None, stop_controllers=None, strictness = 0):
        # Create a proxy to the '/controller_manager/switch_controller' service
        service_proxy = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
        try:
            # Call the service
            response = service_proxy(start_controllers, stop_controllers, strictness)
            rospy.loginfo("Service call succeeded with response: %s", response)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)
        rospy.sleep(1)
        if "impedance" in start_controllers[0]:
            print("Starting impedance controller")
            self.joint_reader = Listener(f'/arm_{self.side}_impedance_controller/state', JointTrajectoryControllerState, post_process_func=joint_process_func)
            self.arm_writer = Publisher(f'/arm_{self.side}_impedance_controller/command', JointTrajectory) 
        else:
            self.joint_reader = Listener(f'/arm_{self.side}_controller/state', JointTrajectoryControllerState, post_process_func=joint_process_func)
            self.arm_writer = Publisher(f'/arm_{self.side}_controller/safe_command', JointTrajectory)

    def setup_listeners(self):

        self.arm_reader = TFTransformListener('/base_footprint')
        self.joint_reader = Listener(f'/arm_{self.side}_controller/state', JointTrajectoryControllerState, post_process_func=joint_process_func)
        def process_force(message):
            return {"force": message.wrench.force, "torque": message.wrench.torque}
        ### this is always for the right arm
        if (self.side == 'right') and (self.gripper_type is not None):
            self.ft_right_sub = Listener(input_topic_name=f'/wrist_right_ft/corrected', input_message_type=WrenchStamped, post_process_func=process_force)
        else:
            self.ft_right_sub = None

    @property
    def arm_pose(self):
        pos, quat = self.arm_reader.get_transform(target_link=f'/arm_{self.side}_tool_link')

        if pos is None:
            return None
        return np.concatenate((pos, quat))

    def setup_actors(self):
        self.arm_writer = None
        if self.arm_enabled:
            self.ik_solver = \
                TracIKSolver(
                    urdf_file=self.urdf_path,
                    base_link=self.ik_base_link,
                    tip_link=f"arm_{self.side}_tool_link",
                    timeout=0.025,
                    epsilon=5e-4,
                    solve_type="Distance"
                )
            self.arm_writer = Publisher(
                f'/arm_{self.side}_controller/safe_command',
                JointTrajectory
            )
            
            self.arm_cancel = Publisher(
                f'/arm_{self.side}_controller/follow_joint_trajectory/cancel',
                GoalID,
            )
            self.gravity_start = rospy.Publisher('/gravity_compensation/goal', EmptyActionGoal, queue_size=1)
            self.gravity_end = rospy.Publisher('/gravity_compensation/cancel', GoalID, queue_size=1)


    def process_action(self, action):
        # convert deltas to absolute positions
        pos_delta, quat_delta = action[:3], action[3:7]

        cur_pos, cur_quat = self.arm_reader.get_transform(
            target_link=f'/arm_{self.side}_tool_link',
            base_link=f'/{self.ik_base_link}'
        )
        # cur_euler = quat_to_euler(cur_quat)
        # target_euler = add_angles(euler_delta, cur_euler)
        # target_quat = euler_to_quat(target_euler)
        target_pos = cur_pos + pos_delta
        target_quat = add_quats(delta=quat_delta, source=cur_quat)
        return target_pos, target_quat
    
    def process_action2(self, action):
        # convert deltas to absolute positions
        pos_delta, euler_delta = action[:3], action[3:6]

        cur_pos, cur_quat = self.arm_reader.get_transform(target_link=f'/arm_{self.side}_tool_link', base_link='/torso_lift_link')
        cur_euler = quat_to_euler(cur_quat)

        target_pos = cur_pos + pos_delta

        target_euler = add_angles(euler_delta, cur_euler)
        target_quat = euler_to_quat(target_euler)
        return target_pos, target_quat

    def create_joint_command(self, joint_goal, duration_scale, teleop=False):
        message = JointTrajectory()
        message.header = Header()

        joint_names = []

        positions = list(self.joint_reader.get_most_recent_msg())
        for i in range(1, 8):
            joint_names.append(f'arm_{self.side}_{i}_joint')
            positions[i-1] = joint_goal[i-1]

        message.joint_names = joint_names

        """ Teleop Changes """
        if teleop:
            duration = duration_scale + 0.7
        else:
            duration = duration_scale

        point = JointTrajectoryPoint(positions=positions, time_from_start=rospy.Duration(duration))
        message.points.append(point)
        return message

    def is_at_joint(self, joint_goal, threshold=5e-3):
        cur_joints = self.joint_reader.get_most_recent_msg()
        return np.linalg.norm(cur_joints - joint_goal) < threshold

    def write(self, joint_goal, duration_scale, threshold=5e-3, delay_scale_factor=1.0, force_z_th=None, teleop=False, timeout=20):
        counter = 0
        if self.arm_writer is not None:

            """ Teleop Changes """
            if teleop:
                # while not self.is_at_joint(joint_goal, threshold):
                pose_command = self.create_joint_command(joint_goal, duration_scale, teleop=teleop)
                self.arm_writer.write(pose_command)
                if self.ft_right_sub is not None:
                    force_vals = self.ft_right_sub.get_most_recent_msg()
                if force_z_th is not None:
                    assert self.side == 'right', "We only have force sensor for right arm."
                    print(f"force value: {force_vals.z} > {force_z_th}")
                    if force_vals.z < force_z_th: # we only use this for elevator.
                        print(f"Force value violated: {force_vals.z} < {force_z_th}")
                        # cancel pose command
                        joint_val = self.joint_reader.get_most_recent_msg()
                        pose_command = self.create_joint_command(joint_val, duration_scale=0.1, teleop=teleop)
                        self.arm_writer.write(pose_command)
                        rospy.sleep(1)
                        # break
                counter += 1
                duration_scale = np.linalg.norm(joint_goal - self.joint_reader.get_most_recent_msg())*delay_scale_factor

            else:
                
                start_time = rospy.get_rostime()
                while not self.is_at_joint(joint_goal, threshold):
                    pose_command = self.create_joint_command(joint_goal, duration_scale, teleop=teleop)
                    self.arm_writer.write(pose_command)
                    if self.ft_right_sub is not None:
                        force_vals = self.ft_right_sub.get_most_recent_msg()
                    if force_z_th is not None:
                        assert self.side == 'right', "We only have force sensor for right arm."
                        print(f"force value: {force_vals.z} > {force_z_th}")
                        if force_vals.z < force_z_th: # we only use this for elevator.
                            print(f"Force value violated: {force_vals.z} < {force_z_th}")
                            # cancel pose command
                            joint_val = self.joint_reader.get_most_recent_msg()
                            pose_command = self.create_joint_command(joint_val, duration_scale=0.1, teleop=teleop)
                            self.arm_writer.write(pose_command)
                            rospy.sleep(1)
                            # break
                    counter += 1
                    rospy.sleep(0.1)
                    duration_scale = np.linalg.norm(joint_goal - self.joint_reader.get_most_recent_msg())*delay_scale_factor
                    if (rospy.get_rostime() - start_time).to_sec() > timeout:
                        break

        return counter

    def find_ik(self, target_pos, target_quat):
        ee_pose = np.eye(4)
        ee_pose[:3, :3] = quat_to_rmat(target_quat)
        ee_pose[:3, 3] = np.array(target_pos)

        joint_init = self.joint_reader.get_most_recent_msg()
        if self.torso_enabled:
            joint_init = np.concatenate((np.asarray([self.torso.get_torso_extension()]), joint_init))
        joint_goal = self.ik_solver.ik(ee_pose, qinit=joint_init)
        # print("ik solved joint positions: ", joint_goal)

        duration_scale = 0
        if joint_goal is not None:
            duration_scale = np.linalg.norm(joint_goal-joint_init)

        return joint_goal, duration_scale

    def step(self, action, delay_scale_factor=1.0, force_z_th=None, teleop=False, timeout=20.0):
        if self.arm_enabled:
            if not teleop:
                target_pos, target_quat = self.process_action(action)
            else:
                target_pos, target_quat = self.process_action2(action)
            joint_goal, duration_scale = self.find_ik(target_pos, target_quat)
            duration_scale *= delay_scale_factor
            # print("found joint_goal", joint_goal, "duraction_scale", duration_scale)

            if joint_goal is not None:
                if self.torso_enabled:
                    self.torso.torso_writer.write(self.torso.create_torso_command(joint_goal[0]))
                    joint_goal = joint_goal[1:]
                self.write(joint_goal, duration_scale, delay_scale_factor=delay_scale_factor, force_z_th=force_z_th, teleop=teleop, timeout=timeout)

            return {
                'joint_goal': joint_goal,
                'duration_scale': duration_scale
            }
        return {}

    def reset(self, action, allowed_delay_scale=4.0, delay_scale_factor=1.5, force_z_th=None, teleop=False):
        if self.arm_enabled:
            assert len(action) == 7

            cur_joints = self.joint_reader.get_most_recent_msg()
            delay_scale = np.linalg.norm(cur_joints - action)
            assert delay_scale < allowed_delay_scale, f"Resetting to a pose that is too far away: {delay_scale:.2f} > {allowed_delay_scale:.2f}"
            # breakpoint()
            self.write(action, delay_scale*delay_scale_factor, delay_scale_factor=delay_scale_factor, force_z_th=force_z_th, teleop=teleop)

    def local_ik_controller(self):
        pass        