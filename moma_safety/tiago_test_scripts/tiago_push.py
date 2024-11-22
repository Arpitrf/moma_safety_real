import os
import argparse
import rospy
import numpy as np


import moma_safety.utils.utils as U
import moma_safety.utils.transform_utils as T # transform_utils
import moma_safety.utils.vision_utils as VU # vision_utils

from moma_safety.tiago.tiago_gym import TiagoGym
from moma_safety.tiago.utils.camera_utils import Camera
from moma_safety.tiago.skills.push_button import PushButtonSkill
import moma_safety.tiago.RESET_POSES as RP

from moma_safety.tiago.utils.ros_utils import Publisher, Listener, TFTransformListener
rospy.init_node('tiago_test')

def main(args):
    env = TiagoGym(
        frequency=10,
        right_arm_enabled=args.arm=='right',
        left_arm_enabled=args.arm=='left',
        right_gripper_type='robotiq2F-140' if args.arm=='right' else None,
        left_gripper_type='robotiq2F-85' if args.arm=='left' else None,
        base_enabled=False,
        torso_enabled=True,
    )
    tf_listener = TFTransformListener('/base_footprint')
    grasp_h_r = RP.PUSH_R_H if args.arm == 'right' else RP.PUSH_L_H
    # check if it is safe to move to the reset pose
    user_input = input('Move to reset pose? (y/n): ')
    if user_input != 'y':
        return

    push_button = PushButtonSkill(oracle_position=args.oracle, debug=args.debug)
    env.reset(reset_arms=True, reset_pose=grasp_h_r, allowed_delay_scale=6.0)

    obs = env._observation()
    rgb = obs['tiago_head_image']
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
    push_button.step(env, rgb, depth, pcd=pcd, normals=normals, arm=args.arm, execute=args.exec)
    rospy.signal_shutdown("Shutdown")
    rospy.spin()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--oracle', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--exec', action='store_true')

    parser.add_argument('--arm', type=str, default='left', choices=['right', 'left'])
    args = parser.parse_args()
    main(args)
