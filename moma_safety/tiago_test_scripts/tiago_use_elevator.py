import os
import argparse
import rospy
import numpy as np


import moma_safety.utils.utils as U
import moma_safety.utils.transform_utils as T # transform_utils
import moma_safety.utils.vision_utils as VU # vision_utils

from moma_safety.tiago.tiago_gym import TiagoGym
from moma_safety.tiago.utils.camera_utils import Camera
from moma_safety.tiago.skills.use_elevator import UseElevatorSkill
from moma_safety.tiago.skills.use_elevator import CallElevatorSkill
import moma_safety.tiago.RESET_POSES as RP
from moma_safety.tiago.ros_restrict import change_map, set_floor_map

from moma_safety.tiago.utils.ros_utils import Publisher, Listener, TFTransformListener
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
rospy.init_node('tiago_test')

def main(args):
    env = TiagoGym(
        frequency=10,
        right_arm_enabled=True,
        left_arm_enabled=True,
        right_gripper_type='robotiq2F-140',
        left_gripper_type='robotiq2F-85',
        base_enabled=True,
        torso_enabled=True,
    )
    tf_listener = TFTransformListener('/base_footprint')
    grasp_h_r = RP.PUSH_R_H
    run_dir = 'temp'
    prompt_args = {
        'add_object_boundary': False,
        'add_dist_info': False,
        'add_arrows_for_path': False,
        'radius_per_pixel': 0.03,
    }
    print("init use_eval skill")
    use_elevator = UseElevatorSkill(
        oracle_position=args.oracle,
        debug=args.debug,
        run_dir=run_dir,
        add_histories=True,
        prompt_args=prompt_args,
    )
    print("init call_elev skill")
    call_elevator = CallElevatorSkill(
        oracle_position=args.oracle,
        debug=args.debug,
        run_dir=run_dir,
        add_histories=True,
        prompt_args=prompt_args,
    )
    _exec = U.reset_env(env, reset_pose=grasp_h_r, delay_scale_factor=1.5)
    # if not _exec:
    #     return
    query = None
    if args.floor_num == 1:
        query="Go to the second floor."
    elif args.floor_num == 2:
        query="Go to the first floor."
    else:
        raise NotImplementedError

    if True:
        obs_pp = VU.get_obs(env, tf_listener)
        rgb, depth, cam_intr, cam_extr, pcd, normals = obs_pp['rgb'], obs_pp['depth'], obs_pp['cam_intr'], obs_pp['cam_extr'], obs_pp['pcd'], obs_pp['normals']
        select_ind = 0
        save_key = f'step_{select_ind:03d}'
        info = {'step_idx': select_ind, 'cam_intr': cam_intr, 'cam_extr': cam_extr, 'save_key': save_key, 'floor_num': args.floor_num}
        call_elevator.step(
            env=env,
            rgb=rgb,
            depth=depth,
            pcd=pcd,
            normals=normals,
            query=query,
            arm=args.arm,
            execute=args.exec,
            run_vlm=args.run_vlm,
            bld=args.bld,
            info=info,
        )
    if True:
        obs_pp = VU.get_obs(env, tf_listener)
        rgb, depth, cam_intr, cam_extr, pcd, normals = obs_pp['rgb'], obs_pp['depth'], obs_pp['cam_intr'], obs_pp['cam_extr'], obs_pp['pcd'], obs_pp['normals']
        select_ind = 0
        save_key = f'step_{select_ind:03d}'
        info = {'step_idx': select_ind, 'cam_intr': cam_intr, 'cam_extr': cam_extr, 'save_key': save_key, 'floor_num': args.floor_num}
        is_succes, _, _, return_info = use_elevator.step(
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
            bld=args.bld,
        )

    floor_num = return_info['floor_num'] # change in the floor number
    # floor_num = return_info['floor_num'] # change in the floor number
    # floor_num = 2 # return_info['floor_num'] # change in the floor number
    set_floor_map(floor_num, bld=args.bld)
    pid = change_map(floor_num=floor_num, bld=args.bld, empty=False) # this one will only add he additional map.
    U.reset_env(env, reset_pose=RP.HOME_R, reset_pose_name='HOME_R', delay_scale_factor=1.5)
    # localize the robot
    use_elevator.localize_robot(floor=floor_num, bld=args.bld)
    rospy.signal_shutdown("Shutdown")
    rospy.spin()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--oracle', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--exec', action='store_true')
    parser.add_argument('--run_vlm', action='store_true')
    parser.add_argument('--arm', type=str, default='right', choices=['right', 'left'])
    parser.add_argument('--floor_num', type=int, default=2, choices=[1,2])
    parser.add_argument('--bld', type=str, default=None)
    args = parser.parse_args()
    assert args.bld is not None
    main(args)
