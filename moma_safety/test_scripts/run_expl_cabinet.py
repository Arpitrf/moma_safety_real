import os
import yaml
import  pdb
import pickle
import cv2
import imageio
import random

import numpy as np
import torch as th
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T

from scipy.spatial.transform import Rotation as R
from datetime import datetime
from omnigibson.utils.asset_utils import decrypt_file
from omnigibson.utils.ui_utils import KeyboardRobotController, draw_line, clear_debug_drawing
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives

from motion_utils import MotionUtils
# from memory import Memory
from omnigibson.arpit_trial.failure_models import CollisionFailureModel, GraspFailureModel
from omnigibson.arpit_trial.utils.utils import correct_gripper_friction, check_success, set_extrinsic_matrix, hori_concatenate_image

num_tries = 10
num_samples = 15
num_top_samples = 3
epochs = 10
success = False

mu_x = np.zeros(3) + 0.03  # Example: 2-dimensional problem
sigma_x = np.eye(3) * 0.003
mu_y = np.zeros(3) # Example: 2-dimensional problem
sigma_y = np.eye(3) * 0.003
mu_z = np.zeros(3)  # Example: 2-dimensional problem
sigma_z = np.eye(3) * 0.003

temp_prior = th.tensor([
    [0.,    0.,    0., -0.055,  -0.013,  0.,    0.,    0.,    0., -1.0],
    [0.,    0.,    0., -0.074,  -0.037, -0.001, 0.,    0.,    0., -1.0],
    [0.,    0.,    0., -0.072,  -0.021, -0.004, 0.,    0.,    0., -1.0],
    [0.,    0.,    0., -0.07 ,  -0.016, -0.005, 0.,    0.,    0., -1.0],
    [0.,    0.,    0., -0.069,  -0.018, -0.002, 0.,    0.,    0., -1.0],
])

# temp_prior = th.tensor([
#     [ 0.,     0.,     0.,    -0.067, -0.04,   0.,     0.,    -0.,     0.1,    -1.   ],
#     [ 0.,     0.,     0.,    -0.053, -0.058,  0.,     0.,    -0.,     0.1,    -1.   ],
#     [ 0.,     0.,     0.,    -0.033, -0.071,  0.,     0.,    -0.,     0.1,    -1.   ],
#     [ 0.,     0.,     0.,    -0.011, -0.078,  0.,     0.,     0.,     0.1,    -1.   ],
#     [ 0.,     0.,     0.,     0.013, -0.077,  0.,     0.,     0.,     0.1,    -1.   ],
# ])

def teleop(robot, env):
    # Create teleop controller
    action_generator = KeyboardRobotController(robot=robot)
    # Register custom binding to reset the environment
    action_generator.register_custom_keymapping(
        key=lazy.carb.input.KeyboardInput.R,
        description="Reset the robot",
        callback_fn=lambda: env.reset(),
    )
    # Print out relevant keyboard info if using keyboard teleop
    action_generator.print_keyboard_teleop_info()

    max_steps = -1 
    step = 0
    while step != max_steps:
        action, keypress_str = action_generator.get_teleop_action()
        print("action: ", action)
        env.step(action=action)
        if keypress_str == 'TAB':
            return
        step += 1

def visualize_trajectories(actions, start_position, grasp_vector=None, robot=None, env=None, writer=None): 
    # # Hide right gripper so as to visualize the actions
    # robot.links["gripper_right_link"].visible=False
    # robot.links["gripper_right_left_finger_link"].visible=False
    # robot.links["gripper_right_right_finger_link"].visible=False
    
    total_lines = 0
    for i in range(actions.shape[0]):
        trajectory = actions[i, :, 3:6]
        prev_position = start_position

        # for j in range(trajectory.shape[0]):
        for j in range(0, 1):
            direction = trajectory[j]  # Direction vector at this waypoint
            magnitude = np.linalg.norm(direction)  # Magnitude of the direction vector
            direction_normalized = direction / magnitude if magnitude != 0 else direction  # Normalize the direction
            step = magnitude * direction_normalized

            next_position = prev_position + step
            if grasp_vector is not None:
                if grasp_vector[i]:
                    color = (0.0, 1.0, 0.0, 1.0)
                else:
                    color = (1.0, 0.0, 0.0, 1.0)
            # ax.quiver(prev_position[0], prev_position[1], prev_position[2], step[0], step[1], step[2], color=color)
            # convert the positions to world frame
            robot_pos, robot_orn = robot.get_position_orientation()
            robot_to_world = np.eye(4)
            robot_to_world[:3, :3] = R.from_quat(robot_orn).as_matrix()
            robot_to_world[:3, 3] = robot_pos
            prev_position = (robot_to_world @ np.array([prev_position[0], prev_position[1], prev_position[2], 1.0]))[:3]
            next_position = (robot_to_world @ np.array([next_position[0], next_position[1], next_position[2], 1.0]))[:3]
            draw_line(prev_position, next_position, color=color, size=5.0)
            # visualize_marker(start_position=prev_position, end_position=next_position, id=total_lines)
            prev_position = prev_position + step  # Move to the new position
            total_lines += 1
    
    for _ in range(25):
        og.sim.step()
        obs, obs_info = env.get_obs()
        img = obs[f"{env.robots[0].name}"][f"{env.robots[0].name}:eyes:Camera:0"]["rgb"][:, :, :3].numpy() / 255.0
        viewer_img = og.sim.viewer_camera._get_obs()[0]['rgb'][:,:,:3] / 255.0
        concat_img = hori_concatenate_image([viewer_img, img])
        concat_img = concat_img * 255.0
        concat_img = concat_img.astype(np.uint8)
        if writer is not None:
            writer.append_data(concat_img)

def sample_delta_orientation(prior, noise=0.5):
    new_traj = []
    for original_delta_orn in prior:
        original_delta_orn_euler = np.array(R.from_rotvec(original_delta_orn).as_euler("xyz", degrees=False))
        # Decompose the delta orientation into Euler angles
        delta_orn_x, delta_orn_y, delta_orn_z = original_delta_orn_euler  # Assuming (yaw, pitch, roll) order
        
        # Sample noise for each axis
        delta_orn_z = delta_orn_z + np.random.uniform(-noise, noise)
                
        # Return as Euler angles or convert to quaternion/matrix as needed
        new_orientation = R.from_euler('xyz', [delta_orn_x, delta_orn_y, delta_orn_z], degrees=False).as_rotvec()
        new_traj.append(new_orientation)
    return np.array(new_traj)

def sample_from_cone(prior, max_angle=np.pi/18, norm_variance=0.4):
    new_traj = []
    for original_vector in prior:
        original_norm = np.linalg.norm(original_vector)
        original_vector = original_vector / np.linalg.norm(original_vector)  # Normalize input vector

        # Compute the rotation matrix to align [0, 0, 1] with the original vector
        z_axis = np.array([0.0, 0.0, 1.0])
        if np.allclose(original_vector, z_axis):
            rotation_matrix = np.eye(3)  # No rotation needed if already aligned
        else:
            rotation_axis = np.cross(z_axis, original_vector)
            rotation_axis /= np.linalg.norm(rotation_axis)
            angle = np.arccos(np.clip(np.dot(z_axis, original_vector), -1.0, 1.0))
            rotation_matrix = R.from_rotvec(angle * rotation_axis).as_matrix()

        # Sample a random point in the cone aligned with the z-axis
        z = np.cos(max_angle) + (1 - np.cos(max_angle)) * np.random.rand()
        phi = 2 * np.pi * np.random.rand()
        x = np.sqrt(1 - z**2) * np.cos(phi)
        y = np.sqrt(1 - z**2) * np.sin(phi)
        random_point = np.array([x, y, z], dtype=np.float64)

        # Apply the rotation to align the point with the original vector
        noisy_vector = rotation_matrix @ random_point

        # Vary the norm of the noisy vector
        varied_norm = original_norm * (1 + np.random.uniform(-norm_variance, norm_variance))
        noisy_vector *= varied_norm

        # print("np.linalg.norm(noisy_vector): ", np.linalg.norm(noisy_vector))
        new_traj.append(noisy_vector)

    return np.array(new_traj)

def grasp_handle(env, robot, episode_memory=None, action_primitives=None, motion_utils=None, grasp_mode=None):
    grasp_action = -1.0
    # ======================= Move hand to grasp pose ================================    
    # # w.r.t world (side grasp)
    # target_pose_world = th.tensor([
    #     [ 0.93739022,  0.01530303,  0.34794453,  1.26256945], #1.24256945
    #     [-0.34824335,  0.02651403,  0.93702912, -0.15659404],
    #     [ 0.00511398, -0.9995313,   0.03018317,  0.50507379],
    #     [ 0.        ,  0.,          0.,          1.        ],
    # ])
    # # w.r.t world (front grasp)
    # # target_pose_world = th.tensor([
    # #     [ 0.18879972, -0.08792357,  0.97807163,  1.26433671],
    # #     [-0.97119662,  0.1307178,   0.19922347, -0.08407815],
    # #     [-0.14536781, -0.98751319, -0.06071159,  0.50231367],
    # #     [ 0.        ,  0.,          0.,          1.        ],
    # # ])
    # target_pose_world = T.mat2pose(target_pose_world)


    if grasp_mode == "side":
        # w.r.t object (side grasp)
        target_pose_obj = np.array([
            [0.34824231, -0.02651246, -0.93702955, -0.0934068 ],
            [0.9373906,   0.01530303,  0.3479435,  -0.22743054],
            [0.00511456, -0.99953134,  0.03018169, -0.04371384],
            [0.,          0.,          0.,          1.        ],
        ])
    elif grasp_mode == "front":
        # w.r.t object (front grasp)
        target_pose_obj = np.array([
            [ 0.97119581, -0.1307169,  -0.19922798, -0.13592077],
            [ 0.18880397, -0.08792588,  0.9780706,  -0.23566403],
            [-0.14536765, -0.9875131,  -0.06071338, -0.0464736 ],
            [ 0.        ,  0.,          0.,          1.        ],
        ])
    cabinet_pos_world, cabinet_orn_world = env.scene.object_registry("name", "bottom_cabinet").get_position_orientation()
    cabinet_pose_world = np.eye(4)
    cabinet_pose_world[:3, :3] = R.from_quat(cabinet_orn_world).as_matrix()
    cabinet_pose_world[:3, 3] = cabinet_pos_world
    target_pose_world = cabinet_pose_world @ target_pose_obj
    target_pos_world = target_pose_world[:3, 3]
    target_orn_world = R.from_matrix(target_pose_world[:3, :3]).as_quat()
    target_pose_world = (th.tensor(target_pos_world, dtype=th.float32), th.tensor(target_orn_world, dtype=th.float32))

    # pre_target_pose = (target_pose_world[0] + th.tensor([0.0, 0.0, 0.1]), target_pose_world[1]) 
    # execute_controller(action_primitives._move_hand_direct_ik(pre_target_pose, ignore_failure=True, in_world_frame=True), 
    #                    env, 
    #                    robot, 
    #                    grasp_action, 
    #                    episode_memory) 
    
    motion_utils.execute_controller(action_primitives._move_hand_direct_ik(target_pose_world, ignore_failure=True, in_world_frame=True), 
                       grasp_action) 
    for _ in range(40):
        og.sim.step()
    
    # Debugging
    # post_eef_pose = robot.get_relative_eef_pose(arm='right')
    post_eef_pose = robot.eef_links["right"].get_position_orientation()
    pos_error = np.linalg.norm(post_eef_pose[0] - target_pose_world[0])
    orn_error = T.get_orientation_diff_in_radian(post_eef_pose[1], target_pose_world[1])
    print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")
    # =================================================================================

    # ============= Perform grasp ===================
    grasp_action = 1.0
    action = action_primitives._empty_action()
    action[robot.gripper_action_idx["right"]] = grasp_action
    env.step(action)
    for _ in range(40):
        og.sim.step()
    # ==============================================

def custom_reset(env, robot, episode_memory=None): 
    scene_initial_state = env.scene._initial_state
    
    base_yaw = 0
    r_euler = R.from_euler('z', base_yaw, degrees=True) # or -120
    r_quat = R.as_quat(r_euler)
    scene_initial_state['object_registry'][env.robots[0].name]['root_link']['ori'] = r_quat

    # randomizing base pos
    base_pos = np.array([0.67, 0.0, 0.0])
    # base_x_noise = np.random.uniform(-0.15, 0.15)
    # base_y_noise = np.random.uniform(-0.15, 0.15)
    # base_noise = np.array([base_x_noise, base_y_noise, 0.0])
    # base_pos += base_noise 
    scene_initial_state['object_registry'][env.robots[0].name]['root_link']['pos'] = base_pos

    # Reset environment and robot
    env.reset()
    robot.reset()

    # set head joint positions
    head_joints = th.tensor([-0.603, -0.897])
    robot.set_joint_positions(positions=head_joints, indices=robot.camera_control_idx)

    # Step simulator a few times so that the effects of "reset" take place
    for _ in range(10):
        og.sim.step()


def expl(t, actions, motion_utils, robot, env, traj_length, collision_failure_model=None, grasp_mode=None, grasp_failure_model=None):
    print(f"===== waypoint {t} ===== ") 
    if t == traj_length:
        print("Reached end of recursion")
        # inp = input("Press Y if success else N: ")
        # if inp == 'Y':
        #     return False
        # else:
        #     return True
        task_success = check_success(env, robot)
        retval = dict()
        if task_success:
            print("Task succeeded!")
            retval["all_failed"] = False
            retval["stop_expl"] = True
            retval["task_success"] = True
        else:
            retval["all_failed"] = True
            retval["stop_expl"] = False
            retval["task_success"] = False
        return retval
    
    # check all action samples via the model
    safe_list = []
    grasp_failure_model_thresholds = [0.95, 0.95, 0.7, 0.7, 0.7]
    grasp_failure_model_threshold = grasp_failure_model_thresholds[t]
    
    # This is just to visualize the safe/unsafe actions
    if grasp_failure_model is not None:
        for action_sample in actions:
            obs, obs_info = env.get_obs()
            check_grasp = grasp_failure_model.check_grasp(obs, obs_info, action_sample[t], env.robots[0].name, threshold=grasp_failure_model_threshold, waypt=t)
            # print("check_grasp: ", check_grasp)
            if check_grasp == 0.0:
                safe_list.append(False)
            else:
                # print("action: ", action_sample[t][3:6])
                safe_list.append(True)
        print("Number of safe actions: ", sum(safe_list))
        # Visualize the actions
        visualize_trajectories(actions.numpy(), start_position=robot.get_relative_eef_pose(arm='right')[0].numpy(), grasp_vector=safe_list, robot=robot, env=env, writer=motion_utils.writer)
        # uncomment to debug
        # breakpoint()
        clear_debug_drawing()
        # TODO: change this
        # robot.links["gripper_right_link"].visible=True
        # robot.links["gripper_right_left_finger_link"].visible=True
        # robot.links["gripper_right_right_finger_link"].visible=True

    for action_num, action in enumerate(actions):
        if not safe_list[action_num]:
            continue
        # print("===== time step, action ===== ", t, action[t][3:6])
        ee_pose_before = robot.get_relative_eef_pose(arm='right')
        joint_pos_before = robot.get_joint_positions()[robot.arm_control_idx["right"]]
        sim_state_before = og.sim.dump_state()
        
        action_exec = motion_utils.act(action[t], grasp_failure_model=grasp_failure_model, grasp_mode=grasp_mode, grasp_failure_model_threshold=grasp_failure_model_threshold)
        if not action_exec:
            retval = dict()
            retval["all_failed"] = False
            retval["stop_expl"] = True
            retval["task_success"] = False
            return retval
        # In the current implementation I am performing the action (move_primitive) inside the safe action. This will change later.
        retval = expl(t+1, actions, motion_utils, robot, env, traj_length, grasp_failure_model=grasp_failure_model, grasp_mode=grasp_mode)

        if retval["stop_expl"]:
            return retval
        # if not all_failed:
        #     return all_failed
        
        # if task success
        if check_success(env, robot):
            print("Task succeeded!")
            retval["all_failed"] = False
            retval["stop_expl"] = True
            retval["task_success"] = True
            return retval
        
        # undo the last action. For now try making it go back to exact joint positions
        # motion_utils.undo_action(t, action)        
        # ee_pose_after = robot.get_relative_eef_pose(arm='right')
        # pos_error = np.linalg.norm(ee_pose_after[0] - ee_pose_before[0])
        # orn_error = T.get_orientation_diff_in_radian(ee_pose_after[1], ee_pose_before[1])
        # # print(f"Final pos_error and orn error: {pos_error} meters, {np.rad2deg(orn_error)} degrees.")
        # # print("joint_pos_before: ", joint_pos_before)
        # joint_pos_after = robot.get_joint_positions()[robot.arm_control_idx["right"]]
        # # print("joint_pos after rewind: ", joint_pos_after)
        # need_reset = any(abs(joint_pos_before - joint_pos_after) > 0.1)
        # print("need to call sim.load_state?: ", need_reset)
        need_reset = True
        if need_reset:
            og.sim.load_state(sim_state_before)
            for _ in range(10):
                og.sim.step()
            # uncomment to debug
            # breakpoint()
            # inp = input("Press key")
            # if inp == "t":
            #     teleop(robot, env)
            joint_pos_after = robot.get_joint_positions()[robot.arm_control_idx["right"]]
            # print("joint_pos after reset: ", joint_pos_after)
        # input("Undid the action. Press enter to continue")

        # # TODO: Change the sampling logic here
        # if all_failed:
        #     # sample t+1 actions again
        #     actions = sample_actions(t+1, actions, traj_length)

    retval = dict()
    retval["all_failed"] = True
    retval["stop_expl"] = False
    retval["task_success"] = False
    return retval 

def sample_actions(t, actions, traj_length, start_idx):
    if t == traj_length:
        print("No action sampling needed.")
        return actions
    
    x_noise = np.random.multivariate_normal(mu_x, sigma_x, num_samples)
    y_noise = np.random.multivariate_normal(mu_y, sigma_y, num_samples)
    z_noise = np.random.multivariate_normal(mu_z, sigma_z, num_samples)
    episode_pos_noise = np.concatenate((np.expand_dims(x_noise, axis=1), 
                    np.expand_dims(y_noise, axis=1), 
                    np.expand_dims(z_noise, axis=1)), axis=1)
    
    actions_org = temp_prior.clone()
    actions_org = actions_org[None, start_idx:-1]
    actions_org = actions_org.repeat(num_samples, 1, 1)
    actions[:, t:, 3:6] = actions_org[:, t:, 3:6] + episode_pos_noise[:, t:]
    return actions


def get_episode_reward(robot):
    target_pos = [1.1888, -0.1884,  0.8387]
    curr_pos = robot.eef_links["right"].get_position_orientation()[0].numpy()
    dist = np.linalg.norm(target_pos - curr_pos)
    return -dist

def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True


def main():
    set_all_seeds(seed=1)
    log_video = True
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    config["scene"] = dict()
    config["scene"]["type"] = "Scene"

    # robot specific config
    config["robots"][0]["default_arm_pose"] = "horizontal"
    config["robots"][0]["controller_config"]["arm_right"]["name"] = "InverseKinematicsController"
    config["robots"][0]["controller_config"]["arm_right"]["kp"] = 150.0

    # Create and load this object into the simulator
    rot_euler = [0.0, 0.0, -90.0]
    rot_quat = np.array(R.from_euler('XYZ', rot_euler, degrees=True).as_quat())
    # obj_cfg = dict(
    #     type="DatasetObject",
    #     name="fridge",
    #     category="fridge",
    #     model="hivvdf",
    #     position=[1.5, -0.6, 1.0],
    #     # scale=[2.0, 1.0, 1.0],
    #     orientation=rot_quat,
    #     )
    obj_cfg = dict(
        type="DatasetObject",
        name="bottom_cabinet",
        category="bottom_cabinet",
        # visual_only=True,
        model="bycegi",
        position=[1.5, -0.25, 1.0],
        scale=[1.0, 1.0, 1.2],
        orientation=rot_quat,
    )
    config["objects"] = [obj_cfg]

    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    action_primitives = StarterSemanticActionPrimitives(env, enable_head_tracking=False)
    correct_gripper_friction(robot, friction_val=4.0)

    # Set viewer camera
    og.sim.viewer_camera.set_position_orientation(
        th.tensor([0.88,  0.76,  0.98]),
        th.tensor([-0.12,  0.50,  0.83, -0.20]),
    )

    # for saving videos
    if log_video:
        # current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # folder_path = f"outputs/run_{current_time}"
        # os.makedirs(folder_path, exist_ok=True)
        current_date = datetime.now().strftime("%Y-%m-%d")  # Format: YYYY-MM-DD
        current_time = datetime.now().strftime("%H-%M-%S")  # Format: HH-MM-SS
        base_folder = f"{current_date}"
        time_folder = os.path.join(base_folder, current_time)
        folder_path = f"outputs_expl/{time_folder}"
        os.makedirs(folder_path, exist_ok=True)

    writer = None
    if log_video:
        imgio_kargs = {'fps': 10, 'quality': 10, 'macro_block_size': None,  'codec': 'h264',  'ffmpeg_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']}
        output_path = f'{folder_path}/video.mp4'
        writer = imageio.get_writer(output_path, **imgio_kargs)

    motion_utils = MotionUtils(env, robot, action_primitives, writer)

    # collision_failure_model = CollisionFailureModel(robot=robot)
    grasp_failure_model = GraspFailureModel(robot=robot)

    # setting properties of the objects
    bottom_cabinet = env.scene.object_registry("name", "bottom_cabinet")
    bottom_cabinet.root_link.mass = 50.0

    for _ in range(50):
        og.sim.step()

    # Try two modes
    # modes = ["front", "side"]
    modes = ["front"]

    # episode_memory = Memory()
    episode_memory = None
    custom_reset(env, robot, episode_memory)
    state = og.sim.dump_state(serialized=False)

    for mode in modes:
        for attempt in range(num_tries):
            print(f"============= Try {attempt} =============")
            grasp_handle(env=env, robot=robot, episode_memory=episode_memory, action_primitives=action_primitives, motion_utils=motion_utils, grasp_mode=mode)
            grasp_mode = mode
            # breakpoint()
            # add extrinsic matrix to robot state
            set_extrinsic_matrix(robot)
            traj_length = len(temp_prior)

            actions = []
            for _ in range(num_samples):
                sampled_traj_pos = sample_from_cone(temp_prior[:, 3:6].numpy(), max_angle=np.pi/3, norm_variance=0.4)
                sampled_traj = temp_prior.clone()
                sampled_traj_pos = th.from_numpy(sampled_traj_pos)
                sampled_traj[:, 3:6] = sampled_traj_pos

                delta_orn_euler = R.from_rotvec(sampled_traj[0, 6:9]).as_euler("xyz", degrees=True)
                sampled_traj_orn = sample_delta_orientation(temp_prior[:, 6:9], noise=0.2)
                sampled_traj_orn = th.from_numpy(sampled_traj_orn)
                sampled_traj[:, 6:9] = sampled_traj_orn

                actions.append(sampled_traj)

            actions = np.array(actions)
            actions = th.from_numpy(actions)
            print("Start actions shape: ", actions.shape)

            all_failed = expl(t=0, actions=actions, motion_utils=motion_utils, robot=robot, env=env, traj_length=traj_length, grasp_mode=grasp_mode, grasp_failure_model=grasp_failure_model)

            if not all_failed:
                break

            # if all_failed:
            og.sim.load_state(state)

    breakpoint()
    og.shutdown()


if __name__ == "__main__":
    main()