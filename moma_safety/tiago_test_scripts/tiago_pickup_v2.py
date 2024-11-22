import os
import argparse
import rospy
import numpy as np


import moma_safety.utils.utils as U
import moma_safety.utils.transform_utils as T # transform_utils
import moma_safety.utils.vision_utils as VU # vision_utils

from moma_safety.tiago.tiago_gym import TiagoGym
from moma_safety.tiago.utils.camera_utils import Camera
from moma_safety.tiago.skills.pickup_v2 import PickupSkill
from moma_safety.tiago.skills.push_button_v2 import PushButtonSkill
import moma_safety.tiago.RESET_POSES as RP

from moma_safety.tiago.utils.ros_utils import Publisher, Listener, TFTransformListener
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
rospy.init_node('tiago_test')

def main(args):
    # make sure the head is -0.8
    env = TiagoGym(
        frequency=10,
        right_arm_enabled=args.arm=='right',
        left_arm_enabled=args.arm=='left',
        right_gripper_type='robotiq2F-140' if args.arm=='right' else None,
        left_gripper_type='robotiq2F-85' if args.arm=='left' else None,
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
    if args.skill == "pickup_v2":
        skill_cls = PickupSkill
    elif args.skill == "push_button_v2":
        skill_cls = PushButtonSkill
    skill = skill_cls(
        oracle_position=args.oracle,
        debug=args.debug,
        run_dir=run_dir,
        prompt_args=prompt_args,
    )
    tf_listener = TFTransformListener('/base_footprint')
    if args.grasp_dir == "top":
        grasp_reset_pose = RP.PICKUP_TABLE_L_H
    elif args.grasp_dir == "front":
        grasp_reset_pose = RP.PICKUP_TABLE_FRONT_L_H

    _exec = U.reset_env(env, reset_pose=grasp_reset_pose, reset_arms=True, delay_scale_factor=1.0)
    if not _exec:
        rospy.spin()
        rospy.signal_shutdown("Shutdown")
        exit(0)

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
    # query="I want to pour some hot water. Can you pick up something by the handle for me?"
    # query = "I want to touch a button. Can you press something for me?"
    query = "Grasp the green marker on the table."
    skill.step(
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
        grasp_dir=args.grasp_dir,
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
    parser.add_argument('--grasp-dir', type=str, default='top', choices=['top', 'front'])
    parser.add_argument('--skill', type=str, default='pickup_v2', choices=['pickup_v2', 'push_button_v2'])
    args = parser.parse_args()
    assert args.run_vlm or args.oracle
    main(args)
