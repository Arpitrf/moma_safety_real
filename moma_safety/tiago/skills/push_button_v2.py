#!/us/bin/env python3
import sys

import numpy as np

from moma_safety.tiago.skills.pickup_v2 import PickupSkill

import moveit_commander
moveit_commander.roscpp_initialize(sys.argv)

class PushButtonSkill(PickupSkill):
    def __init__(self, *args, **kwargs):
        super().__init__(
            gsam_query="buttons", do_grasp=False, finger_for_pos="opp_from_arm",
            *args, **kwargs)
        self.prompt_args.update(dict(plot_outside_bbox=True))

    def set_offset_maps(self):
        self.dir_to_pos_trans_offset_map = dict(
            top=np.array([0.01, 0.03, .01]),
            # front=np.array([.04, -.02, 0.]),
        )
        self.dir_to_goto_offset_map = dict(
            top=np.array([0.0, 0.0, -.02]),
            # front=np.array([.02, 0., 0.]),
        )

    def get_mean_pos(self, poses, grasp_dir="top"):
        if grasp_dir == "top":
            # highest point (z) is the object
            highest_z_val = np.max(poses[:, 2])
            # x value should be the minimum x value of the object
            x_val = np.mean(poses[:, 0])
            # y value should be the average of min and max y value of the object
            # take ones those y values whose z value is > min_z_val by 1cm
            y_valid = poses[poses[:, 2] > (np.max(poses[:, 2]) - 0.01)]
            y_val = np.mean([np.min(y_valid[:, 1]), np.max(y_valid[:, 1])])
            mean_pos = np.asarray([x_val, y_val, highest_z_val])
        else:
            raise NotImplementedError
        return mean_pos

