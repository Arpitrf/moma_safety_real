import math
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import rospy

from moma_safety.tiago.tiago_gym import TiagoGym
from moma_safety.tiago import RESET_POSES as RP
import moma_safety.utils.transform_utils as T # transform_utils
from moma_safety.tiago.utils.ros_utils import TFTransformListener
from moma_safety.tiago.utils.transformations import quat_diff
from scipy.spatial.transform import Rotation as R


# Example parameters
v_min_x = 0.25      # Minimum velocity in the x direction (m/s)
v_min_y = 0.25      # Minimum velocity in the y direction (m/s)
omega_min = 0.4
v_max_x = 0.5      # Maximum velocity in the x direction (m/s)
v_max_y = 0.5      # Maximum velocity in the y direction (m/s)
omega_max = 0.8  # Maximum angular velocity (rad/s)

# Normalize the yaw difference to be within the range [-pi, pi]
def normalize_angle(angle):
    # Normalize an angle to the range [-pi, pi]
    return (angle + math.pi) % (2 * math.pi) - math.pi

# Function to compute velocities based on the pose error
def compute_velocity_from_error(d_x, d_y, d_yaw, v_max_x, v_max_y, omega_max, v_min_x, v_min_y, omega_min):
    # Normalize d_yaw to be within the range [-pi, pi]
    d_yaw = normalize_angle(d_yaw)
    
    # Compute linear velocity (limit it to max velocities)
    v_x = max(v_min_x, min(v_max_x, abs(d_x)))  # Velocity in the x-direction (based on d_x, but at least v_min_x)
    v_y = max(v_min_y, min(v_max_y, abs(d_y)))  # Velocity in the y-direction (based on d_y, but at least v_min_y)

    # Assign correct signs to the velocities based on the direction of the error
    v_x = v_x if d_x > 0 else -v_x
    v_y = v_y if d_y > 0 else -v_y


    # Compute angular velocity (limit it to max angular velocity)
    omega_z = max(omega_min, min(omega_max, abs(d_yaw)))
    omega_z = omega_z if d_yaw > 0 else -omega_z
    
    return v_x, v_y, omega_z

# Function to update the robot's pose incrementally
def move_to_target(env, current_pose, target_pose, initial_pose_wrt_map, tf_map=None, pos_tolerance=0.01, orn_tolerance=0.02):
    # Calculate the target pose based on the current pose and delta pose
    current_x, current_y, current_yaw = current_pose
    target_x, target_y, target_yaw = target_pose
    
    # # Print the target pose for reference
    # print(f"Target Pose: ({target_x:.2f}, {target_y:.2f}, {target_yaw:.2f})")
    
    # Loop until the robot reaches the target pose (with some tolerance)
    while True:
        # Calculate the error in x, y, and yaw
        d_x = target_x - current_x
        d_y = target_y - current_y
        d_yaw = target_yaw - current_yaw
        # print("Pose error: d_x, d_y, d_yaw: ", d_x, d_y, d_yaw)
        
        # Check if the robot is close enough to the target pose (tolerance)
        if abs(d_x) < pos_tolerance and abs(d_y) < pos_tolerance and abs(d_yaw) < orn_tolerance:
            # print(f"Target reached: ({current_x:.2f}, {current_y:.2f}, {current_yaw:.2f})")
            break
        
        # Compute the velocities based on the current pose error
        v_x, v_y, omega_z = compute_velocity_from_error(d_x, d_y, d_yaw, v_max_x, v_max_y, omega_max, v_min_x, v_min_y, omega_min)
        if abs(d_x) < pos_tolerance: 
            v_x = 0
        if abs(d_y) < pos_tolerance:
            v_y = 0
        if abs(d_yaw) < orn_tolerance:
            omega_z = 0

        # print("Velocities: v_x, v_y, omega_z: ", v_x, v_y, omega_z)
        action = {'right': None, 'left': None, 'base': np.array([v_x, v_y, omega_z])}
        obs, reward, done, info = env.step(action)
        
        # Update the current pose incrementally (this would be a real-time update in actual robot control)
        current_pose_wrt_map = T.pose2mat((tf_map.get_transform(target_link=f'/base_footprint')))
        # current_pos_map = current_pose_wrt_map[:3, 3]
        # current_ori_map = T.mat2quat(current_pose_wrt_map[:3, :3])
        current_pose_wrt_inital_pose = np.linalg.inv(initial_pose_wrt_map) @ current_pose_wrt_map
        current_pos_wrt_initial_pose = current_pose_wrt_inital_pose[:3, 3]
        current_ori_wrt_initial_pose = T.mat2quat(current_pose_wrt_inital_pose[:3, :3])
    
        current_x, current_y = current_pos_wrt_initial_pose[0], current_pos_wrt_initial_pose[1]
        current_yaw = R.from_quat(current_ori_wrt_initial_pose).as_euler('xyz')[2]
                
        # # Print the current pose for debugging purposes
        # print(f"Moving to: ({current_x:.2f}, {current_y:.2f}, {current_yaw:.2f}), Velocities: ({v_x:.2f}, {v_y:.2f}, {omega_z:.2f})")


def move_base_vel(env, action):
    d_x, d_y, d_yaw = action
    delta_pos = [d_x, d_y, 0.0]
    delta_orn = R.from_euler('xyz', [0.0, 0.0, d_yaw]).as_quat()
    delta_pose2d = np.array([d_x, d_y, d_yaw])

    # Obtain target base pose in current base pose frame
    tf_map = TFTransformListener('/map')
    initial_pose_wrt_map = T.pose2mat((tf_map.get_transform(target_link=f'/base_footprint')))
    initial_pos_map = initial_pose_wrt_map[:3, 3]
    initial_ori_map = T.mat2quat(initial_pose_wrt_map[:3, :3])
    initial_yaw = R.from_quat(initial_ori_map).as_euler('xyz')[2]
    # print("initial_yaw: ", initial_yaw)
    initial_pose2d = np.array([initial_pos_map[0], initial_pos_map[1], initial_yaw])
    # print(f"Initial base pos, ori in map: ", initial_pos_map, initial_ori_map)

    delta_pose_wrt_robot = T.pose2mat((delta_pos, delta_orn))
    # Note that since the robot_pose_wrt_map also has translation component, thie output is no lopnger a delta pose but the target pose
    target_pose_wrt_map = initial_pose_wrt_map @ delta_pose_wrt_robot
    target_pos_map = target_pose_wrt_map[:3, 3]
    target_ori_map = T.mat2quat(target_pose_wrt_map[:3, :3])
    target_wrt_map_pose2d = np.array([target_pos_map[0], target_pos_map[1], R.from_quat(target_ori_map).as_euler('xyz')[2]])
    # print(f"Target base pos, ori in map: ", target_pos_map, target_ori_map)

    target_pose_wrt_initial_pose = np.linalg.inv(initial_pose_wrt_map) @ target_pose_wrt_map
    target_pos_wrt_initial_pose  = target_pose_wrt_initial_pose [:3, 3]
    target_ori_wrt_initial_pose  = T.mat2quat(target_pose_wrt_initial_pose [:3, :3])
    target_wrt_initial_pose2d = np.array([target_pos_wrt_initial_pose[0], target_pos_wrt_initial_pose[1], R.from_quat(target_ori_wrt_initial_pose).as_euler('xyz')[2]])
    # print(f"Target base pos, ori in initial frame: ", target_pos_wrt_initial_pose, target_ori_wrt_initial_pose)

    # Call the function to move the robot from the initial pose using the delta pose
    initial_pose2d_wrt_initial_pose = np.array([0.0, 0.0, 0.0])
    move_to_target(env, initial_pose2d_wrt_initial_pose, target_wrt_initial_pose2d, initial_pose_wrt_map=initial_pose_wrt_map, tf_map=tf_map)


    tf_map = TFTransformListener('/map')
    transform = T.pose2mat((tf_map.get_transform(target_link=f'/base_footprint')))
    final_pos_map = transform[:3, 3]
    final_ori_map = T.mat2quat(transform[:3, :3])
    # print(f"Target base pos, ori in map: ", target_pos_map, target_ori_map)
    # print(f"Final base pos, ori in map: ", final_pos_map, final_ori_map)

if __name__ == "__main__":
    rospy.init_node('tiago_test')

    env = TiagoGym(
        frequency=10,
        right_arm_enabled=True,
        left_arm_enabled=False,
        right_gripper_type='robotiq2F-140',
        left_gripper_type='robotiq2F-85',
        base_enabled=True,
        torso_enabled=False,
    )

    # Example delta pose (change in position and yaw)
    d_x = -0.3         # Change in x position (m)
    d_y = 0.0          # Change in y position (m)
    d_yaw_degrees = 0.0
    d_yaw = np.deg2rad(d_yaw_degrees)  # Change in yaw (radians)
    action = [d_x, d_y, d_yaw]
    move_base_vel(env, action)
