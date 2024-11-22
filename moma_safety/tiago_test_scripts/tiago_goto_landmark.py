import os
import argparse
import rospy
import numpy as np


import moma_safety.utils.utils as U
import moma_safety.utils.transform_utils as T # transform_utils
import moma_safety.utils.vision_utils as VU # vision_utils

from moma_safety.tiago.tiago_gym import TiagoGym
from moma_safety.tiago.utils.camera_utils import Camera
from moma_safety.tiago.skills.goto_landmark import GoToLandmarkSkill
import moma_safety.tiago.RESET_POSES as RP
from moma_safety.tiago.ros_restrict import change_map, set_floor_map

from moma_safety.tiago.utils.ros_utils import Publisher, Listener, TFTransformListener

rospy.init_node('tiago_test')

def main(args):
    # move_base will not work without publishing this empty map.
    floor_num = args.floor
    set_floor_map(floor_num, bld=args.bld)
    pid = change_map(floor_num=floor_num, bld=args.bld, empty=False) # this one will only add he additional map.
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
    print('Environment initialized')
    prompt_args = {
        'raidus_per_pixel': 0.03,
    }
    run_dir='temp'
    os.makedirs(run_dir, exist_ok=True)
    goto_landmark_skill = GoToLandmarkSkill(
        bld=args.bld,
        oracle_action=args.oracle,
        debug=args.debug,
        run_dir=run_dir,
        prompt_args=prompt_args,
    )
    print('Skill initialized')
    tf_listener = TFTransformListener('/base_footprint')
    grasp_h_r = RP.HOME_L_PUSH_R_H
    _exec = U.reset_env(env, reset_pose=grasp_h_r, int_pose=RP.INT_R_H, reset_pose_name='HOME_L_PUSH_R_H', delay_scale_factor=1.5)
    # if not _exec:
    #     return
    print('Reset pose reached')

    # query="I want to color grass in my drawing. Can you get me a marker?"
    # query="I want something fizzy to drink, but I am on diet. Can you pick up the soda?"
    # query="Arrange the chair in my work area. Can you help me with that?"
    query = "navigate to the elevator."
    # query = "navigate to a kitchen or vending machine."
    # query = "navigate to the work area."

    obs_pp = VU.get_obs(env, tf_listener)
    rgb, depth, cam_intr, cam_extr, pcd, normals = obs_pp['rgb'], obs_pp['depth'], obs_pp['cam_intr'], obs_pp['cam_extr'], obs_pp['pcd'], obs_pp['normals']

    select_ind = 0
    save_key = f'step_{select_ind:03d}'
    info = {'step_idx': select_ind, 'cam_intr': cam_intr, 'cam_extr': cam_extr, 'save_key': save_key}
    # query="I need energy to stay up all night. Can you get me something to drink?"
    goto_landmark_skill.step(
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
    # kill all the child processes
    # U.kill_all_child_processes()
    rospy.signal_shutdown("Shutdown")
    rospy.spin()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--exec', action='store_true')
    parser.add_argument('--arm', type=str, default='left', choices=['right', 'left'])
    parser.add_argument('--run_vlm', action='store_true')
    parser.add_argument('--oracle', action='store_true')
    parser.add_argument('--floor', type=int, default=None)
    parser.add_argument('--bld', type=str, default=None)
    args = parser.parse_args()
    assert args.run_vlm or args.oracle
    assert args.floor is not None
    assert args.bld is not None
    main(args)
