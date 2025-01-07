import time
import rospy
import numpy as np
np.set_printoptions(precision=3, suppress=True)


from moma_safety.tiago.tiago_gym import TiagoGym
from moma_safety.tiago import RESET_POSES as RP
import moma_safety.utils.transform_utils as T # transform_utils
from moma_safety.tiago.utils.ros_utils import TFTransformListener
from moma_safety.tiago.utils.transformations import quat_diff
from scipy.spatial.transform import Rotation as R
from controller_manager_msgs.srv import SwitchController
from moma_safety.tiago.utils.transformations import euler_to_quat, quat_to_euler, add_angles, quat_to_rmat, add_quats


def call_switch_controller(start_controllers=None, stop_controllers=None):
    # Create a proxy to the '/controller_manager/switch_controller' service
    service_proxy = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
    
    # Define the controllers to start and stop
    # start_controllers = ['arm_right_impedance_controller', 'arm_left_impedance_controller']
    # stop_controllers = ['arm_right_controller', 'arm_left_controller']
    strictness = 0  # usually 0 for the most flexible switching behavior
    
    try:
        # Call the service
        response = service_proxy(start_controllers, stop_controllers, strictness)
        rospy.loginfo("Service call succeeded with response: %s", response)
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s", e)

from controller_manager_msgs.srv import ListControllers
def list_controllers():
    # Wait for the service to become available
    rospy.wait_for_service('/controller_manager/list_controllers')
    
    try:
        # Create a service proxy
        list_controllers_service = rospy.ServiceProxy('/controller_manager/list_controllers', ListControllers)
        
        # Call the service
        response = list_controllers_service()
        
        # Process and print the response
        for controller in response.controller:
            print(f"Controller Name: {controller.name}")
            print(f"Type: {controller.type}")
            print(f"State: {controller.state}")
            print("-" * 30)
    except rospy.ServiceException as e:
        print(f"Service call failed: {e}")

    breakpoint()


if __name__ == "__main__":
    rospy.init_node('switch_controller_client')

    env = TiagoGym(
        frequency=10,
        right_arm_enabled=True,
        left_arm_enabled=False,
        right_gripper_type='robotiq2F-140',
        left_gripper_type='robotiq2F-85',
        base_enabled=True,
        torso_enabled=False,
    )

    # list_controllers()

    # # reset to a start pose
    # reset_joint_pos = RP.PREGRASP_R_H
    # env.reset(reset_arms=True, reset_pose=reset_joint_pos, allowed_delay_scale=6.0)

    # switch to impedance controller
    start_controllers = ['arm_right_impedance_controller', 'arm_left_impedance_controller']
    stop_controllers = ['arm_right_controller', 'arm_left_controller']
    env.tiago.arms["right"].switch_controller(start_controllers, stop_controllers)

    # target_right_ee_pose = current_right_ee_pose + np.array([0.1, 0, 0, 0, 0, 0, 0])
    start_right_ee_pose = env.tiago.arms["right"].arm_pose
    delta_pos = np.array([-0.2, 0.0, 0.0])
    delta_ori = np.array([0.0, 0.0, 0.0, 1.0])
    gripper_act = np.array([1.0])
    delta_act = np.concatenate(
        (delta_pos, delta_ori, gripper_act)
    )
    print(f'delta_pos: {delta_pos}', f'delta_ori: {delta_ori}')
    action = {'right': None, 'left': None, 'base': None}
    action["right"] = delta_act
    obs, reward, done, info = env.step(action, delay_scale_factor=6.0, timeout=5.0) 
    print("info: ", info["arm_right"]["joint_goal"])

    target_joint_pos = np.array(info["arm_right"]["joint_goal"])
    # while not rospy.is_shutdown():
    rospy.sleep(2)
    current_joint_pos = env.tiago.arms["right"].joint_reader.get_most_recent_msg()
    abs_err = np.abs(target_joint_pos - current_joint_pos)
    print("abs joint error: ", abs_err)
    time.sleep(0.1)

    # time.sleep(2)
    target_right_ee_pos = start_right_ee_pose[:3] + delta_pos
    target_right_ee_orn = R.from_quat(start_right_ee_pose[3:7]) * R.from_quat(delta_ori)
    target_right_ee_orn = target_right_ee_orn.as_quat()
    target_right_ee_pose = np.concatenate((target_right_ee_pos, target_right_ee_orn))
    end_right_ee_pose = env.tiago.arms["right"].arm_pose
    pos_error = np.linalg.norm(end_right_ee_pose[:3] - target_right_ee_pose[:3])
    orn_error = T.get_orientation_diff_in_radian(end_right_ee_pose[3:7], target_right_ee_pose[3:7])
    orn_error = orn_error % (2*np.pi)
    print(f"==== Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees ====")



    # switch back to joint controller
    start_controllers = ['arm_right_controller', 'arm_left_controller']
    stop_controllers = ['arm_right_impedance_controller', 'arm_left_impedance_controller']
    env.tiago.arms["right"].switch_controller(start_controllers, stop_controllers)

    start_right_ee_pose = env.tiago.arms["right"].arm_pose
    delta_pos = np.array([0.2, 0.0, 0.0])
    delta_ori = np.array([0.0, 0.0, 0.0, 1.0])
    gripper_act = np.array([1.0])
    delta_act = np.concatenate(
        (delta_pos, delta_ori, gripper_act)
    )
    print(f'delta_pos: {delta_pos}', f'delta_ori: {delta_ori}')
    action = {'right': None, 'left': None, 'base': None}
    action["right"] = delta_act
    obs, reward, done, info = env.step(action, delay_scale_factor=6.0, timeout=5.0) 
    print("info: ", info["arm_right"]["joint_goal"])

    target_joint_pos = np.array(info["arm_right"]["joint_goal"])
    # while not rospy.is_shutdown():
    rospy.sleep(2)
    current_joint_pos = env.tiago.arms["right"].joint_reader.get_most_recent_msg()
    abs_err = np.abs(target_joint_pos - current_joint_pos)
    print("abs joint error: ", abs_err)
    time.sleep(0.1)

    # time.sleep(2)
    target_right_ee_pos = start_right_ee_pose[:3] + delta_pos
    target_right_ee_orn = R.from_quat(start_right_ee_pose[3:7]) * R.from_quat(delta_ori)
    target_right_ee_orn = target_right_ee_orn.as_quat()
    target_right_ee_pose = np.concatenate((target_right_ee_pos, target_right_ee_orn))
    end_right_ee_pose = env.tiago.arms["right"].arm_pose
    pos_error = np.linalg.norm(end_right_ee_pose[:3] - target_right_ee_pose[:3])
    orn_error = T.get_orientation_diff_in_radian(end_right_ee_pose[3:7], target_right_ee_pose[3:7])
    orn_error = orn_error % (2*np.pi)
    print(f"==== Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees ====")
