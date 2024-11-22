import os
import argparse
import rospy
import numpy as np
from termcolor import colored


import moma_safety.utils.utils as U
import moma_safety.utils.transform_utils as T # transform_utils
import moma_safety.utils.vision_utils as VU # vision_utils

from moma_safety.tiago.tiago_gym import TiagoGym
from moma_safety.tiago.utils.camera_utils import Camera
from moma_safety.tiago.skills.push_obs_ground import PushObsGrSkill
import moma_safety.tiago.RESET_POSES as RP

from moma_safety.tiago.utils.ros_utils import Publisher, Listener, TFTransformListener
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
rospy.init_node('tiago_test')

def main(args):
    # make sure the head is -0.8
    env = TiagoGym(
        frequency=10,
        right_arm_enabled=True,
        left_arm_enabled=True,
        right_gripper_type='robotiq2F-140',
        left_gripper_type='robotiq2F-85',
        base_enabled=True,
        torso_enabled=True,
    )
    prompt_args = {}
    run_dir='temp'
    os.makedirs(run_dir, exist_ok=True)
    push_obs_gr = PushObsGrSkill(
        oracle_action=args.oracle,
        debug=args.debug,
        run_dir=run_dir,
        prompt_args=prompt_args,
    )
    push_obs_gr.send_head_command(push_obs_gr.default_head_joint_position)
    tf_listener = TFTransformListener('/base_footprint')
    # grasp_h_r = RP.HORIZONTAL_R_H
    # print(colored("MAKE SURE EVERYTHING IS CLEAR.  THIS WILL GO TO RIGHT ARM HORIZONTAL.", "red"))
    # cont = U.confirm_user(True, "Move to reset pose? (y/n)")
    # if not cont:
    #     exit()
    # env.reset(reset_arms=True, reset_pose=grasp_h_r, allowed_delay_scale=6.0, delay_scale_factor=2.0)

    # grasp_h_r = RP.OPEN_DOOR_R
    # grasp_h_r = RP.HOME_L_OPEN_DOOR_R
    grasp_h_r = RP.HOME_L_PUSH_R
    _exec = U.reset_env(env, reset_pose=grasp_h_r, reset_pose_name='HOME_L_PUSH_R_H', delay_scale_factor=1.5)

    obs_pp = VU.get_obs(env, tf_listener)
    rgb, depth, cam_intr, cam_extr, pcd, normals = \
        obs_pp['rgb'], obs_pp['depth'], obs_pp['cam_intr'], obs_pp['cam_extr'], obs_pp['pcd'], obs_pp['normals']

    select_ind = 0
    save_key = f'step_{select_ind:03d}'
    info = {'step_idx': select_ind, 'cam_intr': cam_intr, 'cam_extr': cam_extr, 'save_key': save_key}
    query="Remove obstacle from the ground to clear the path for the robot."
    push_obs_gr.step(
        env=env,
        rgb=rgb,
        depth=depth,
        pcd=pcd,
        normals=normals,
        query=query,
        arm=args.arm,
        execute=args.exec,
        run_vlm=args.run_vlm,
        info=info,
        floor_num=args.floor,
        bld=args.bld,
    )
    rospy.signal_shutdown("Shutdown")
    rospy.spin()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--exec', action='store_true')
    parser.add_argument('--arm', type=str, choices=['right', 'left'])
    parser.add_argument('--run_vlm', action='store_true')
    parser.add_argument('--oracle', action='store_true')
    parser.add_argument('--floor', type=int, default=None)
    parser.add_argument('--bld', type=str, default=None)
    args = parser.parse_args()
    assert args.run_vlm or args.oracle
    main(args)
