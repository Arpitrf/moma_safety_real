import os
import argparse
import rospy
import numpy as np


import moma_safety.utils.utils as U
import moma_safety.utils.transform_utils as T # transform_utils
import moma_safety.utils.vision_utils as VU # vision_utils

from moma_safety.tiago.tiago_gym import TiagoGym
from moma_safety.tiago.utils.camera_utils import Camera
from moma_safety.tiago.skills import TapDoorSkill
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
        right_gripper_type=None, #'robotiq2F-140',
        left_gripper_type='robotiq2F-85',
        base_enabled=True,
        torso_enabled=True,
    )
    prompt_args = {
    }
    run_dir='temp'
    os.makedirs(run_dir, exist_ok=True)
    tap_door_skill = TapDoorSkill(
        oracle_action=args.oracle,
        debug=args.debug,
        run_dir=run_dir,
        prompt_args=prompt_args,
    )
    tf_listener = TFTransformListener('/base_footprint')

    #########
    # first we need to align the robot with the door. This needs to be called in the rw_eval
    # open_door_skill.align_with_door(env, floor_num=args.floor_num, arm=args.arm, execute=args.exec)
    #########
    # check if it is safe to move to the reset pose
    user_input = 'y'#input('Move to reset pose? (y/n): ')
    if user_input != 'y':
        return

    # reset of arm is done inside the run_vlm function
    grasp_h_r = RP.HOME_L_TAP_DOOR_R_INITIAL
    env.reset(reset_arms=True, reset_pose=grasp_h_r, allowed_delay_scale=6.0)
    # grasp_h_r = RP.GRASP_H_R
    # env.reset(reset_arms=True, reset_pose=grasp_h_r, allowed_delay_scale=6.0)

    obs_pp = VU.get_obs(env, tf_listener)
    rgb, depth, cam_intr, cam_extr, pcd, normals = obs_pp['rgb'], obs_pp['depth'], obs_pp['cam_intr'], obs_pp['cam_extr'], obs_pp['pcd'], obs_pp['normals']

    select_ind = 0
    save_key = f'step_{select_ind:03d}'
    info = {'step_idx': select_ind, 'cam_intr': cam_intr, 'cam_extr': cam_extr, 'save_key': save_key}
    query="Tap the door in front of you to open it."
    tap_door_skill.step(
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
    )
    rospy.signal_shutdown("Shutdown")
    rospy.spin()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--exec', action='store_true')
    parser.add_argument('--arm', type=str, default='right', choices=['right', 'left'])
    parser.add_argument('--floor_num', type=int, default=None)
    parser.add_argument('--run_vlm', action='store_true')
    parser.add_argument('--oracle', action='store_true')
    args = parser.parse_args()
    assert args.run_vlm or args.oracle
    assert args.floor_num is not None
    main(args)
