import os
import argparse
import rospy
import numpy as np


import moma_safety.utils.utils as U
import moma_safety.utils.transform_utils as T # transform_utils
import moma_safety.utils.vision_utils as VU # vision_utils

from moma_safety.tiago.tiago_gym import TiagoGym
from moma_safety.tiago.utils.camera_utils import Camera
from moma_safety.tiago.skills.place import PlaceSkill
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
        right_gripper_type=None,#'robotiq2F-140',
        left_gripper_type='robotiq2F-85',
        base_enabled=False,
        torso_enabled=True,
    )
    prompt_args = {
        'add_object_boundary': False,
        'add_dist_info': False,
        'add_arrows_for_path': False,
        'radius_per_pixel': 0.03,
    }
    run_dir='temp'
    place_skill = PlaceSkill(
        oracle_position=args.oracle,
        debug=args.debug,
        run_dir=run_dir,
        z_offset=-0.01,
        prompt_args=prompt_args,
        method=args.method,
    )
    tf_listener = TFTransformListener('/base_footprint')
    grasp_h_r = RP.PICKUP_TABLE_L_HOME_R

    _exec = U.reset_env(env, reset_pose=grasp_h_r, reset_arms=True, delay_scale_factor=1.0)
    # if not _exec:
    #     rospy.spin()
    #     rospy.signal_shutdown("Shutdown")
    #     exit(0)

    obs = env._observation()
    rgb = obs['tiago_head_image'][:, :, ::-1].astype(np.uint8) # BGR -> RGB
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
    select_ind = 0
    save_key = f'step_{select_ind:03d}'
    info = {'step_idx': select_ind, 'cam_intr': cam_intr, 'cam_extr': cam_extr, 'save_key': save_key}
    # query="I need something to drink but I am on diet. Can you pick up something for me?"
    # query="Can you get me something to read? Probably a book from a libray would be nice."
    query="Place the lemon on the plate."
    place_skill.step(
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
    parser.add_argument('--arm', type=str, default='left', choices=['right', 'left'])
    parser.add_argument('--run_vlm', action='store_true')
    parser.add_argument('--oracle', action='store_true')
    parser.add_argument('--method', type=str, default='ours')
    args = parser.parse_args()
    assert args.run_vlm or args.oracle
    main(args)
