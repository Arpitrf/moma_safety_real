import os
import sys
import copy
import numpy as np

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus

from moma_safety.tiago.utils.ros_utils import Publisher, Listener, TFTransformListener
from moma_safety.tiago.skills.base import SkillBase, movebase_code2error
import moma_safety.utils.utils as U
import moma_safety.utils.vision_utils as VU
import moma_safety.utils.transform_utils as T # transform_utils
import moma_safety.pivot.vip_utils as vip_utils
from moma_safety.tiago.prompters.object_bbox import bbox_prompt_img
import moma_safety.tiago.RESET_POSES as RP

from termcolor import colored


class TapDoorSkill(SkillBase):
    def __init__(
            self,
            oracle_action=False,
            debug=False,
            use_vlm=False,
            run_dir=None,
            prompt_args=None,
            skip_ros=False,
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.oracle_action = oracle_action
        self.debug = debug
        if not skip_ros:
            self.setup_listeners()
        self.approach_dist = 0.6
        self.pre_goal_dist = 0.8
        self.goal_dist = 0.7 # to the right or left of the door
        self.vis_dir = os.path.join(run_dir, 'tap_door')
        os.makedirs(self.vis_dir, exist_ok=True)

        arrow_length_per_pixel = prompt_args.get('arrow_length_per_pixel', 0.15)
        radius_per_pixel = prompt_args.get('radius_per_pixel', 0.06)
        self.prompt_args = {
            "add_arrows": prompt_args.get('add_arrows', True),
            "color": (0, 0, 0),
            "mix_alpha": 0.6,
            'thickness': 2,
            'rgb_scale': 255,
            'plot_dist_factor': prompt_args.get('plot_dist_factor', 1.0),
            'rotate_dist': 0.3,
            'radius_per_pixel': radius_per_pixel,
            'arrow_length_per_pixel': arrow_length_per_pixel,
            'add_object_boundary': False,
            'plot_direction': self.method == "ours",
        }
        self.skill_name = "tap_door"
        self.skill_descs = f"""
skill_name: {self.skill_name}
arguments: None
description: Opens the door by tapping the key card access.
""".strip()

    def step(self, env, rgb, depth, pcd, normals, query, arm='right', execute=True, run_vlm=True, info=None, **kwargs):
        '''
            action: Position, Quaternion (xyzw) of the goal
        '''
        if env is not None:
            # get the current position of the robot_base w.r.t. odom
            self.send_head_command(head_positions=[0.0, -0.4])
            obs_pp = VU.get_obs(env, self.tf_base)
            rgb, depth, cam_intr, cam_extr, pcd, normals = obs_pp['rgb'], obs_pp['depth'], obs_pp['cam_intr'], obs_pp['cam_extr'], obs_pp['pcd'], obs_pp['normals']

        text_direction = 'right'
        # because we do not have any parameters at the moment for this skill, we will just use a predefined motion and initialize some variables to avoid errors
        prompt_rgb = rgb.copy()
        response = ''
        return_info = {
            'response': response,
            'model_out': '',
            'error_list': [],
        }

        capture_history = {
            'image': prompt_rgb,
            'query': query,
            'model_response': text_direction,
            'full_response': '',
            'text_direction': text_direction,
            'model_analysis': '',
        }
        door_distance = None
        if env is None:
            return self.on_failure(
                reason_for_failure='Environment is None.', # only for debugging
                reset_required=False,
                capture_history=capture_history,
                return_info=return_info,
            )
        # overrite ori map:  rotate np.asarray([0.0, 0.0, 0.7089131175897102, 0.7052958185819889]) by 180 degrees about z-axis
        # to calculate the overrite ori map - we need to rotate np.asarray([0.0, 0.0, 0.7089131175897102, 0.7052958185819889]) by 180 degrees about z-axis
        # overrite_ori_map = np.asarray([0.0, 0.0, -0.7089131175897102, 0.7052958185819889])
        overrite_ori_map = np.asarray([0.0, 0.0, -0.7089131175897102, 0.7052958185819889])
        # We use move_base to approach the door. Maintain the 30cm distance from the door
        approach_pos_base = np.asarray([0.0, 0.0, 0.0]) # add distance here if needed
        approach_ori_base = np.asarray([0.0, 0.0, 0.0, 1.0])
        transform = T.pose2mat((self.tf_map.get_transform(target_link=f'/base_footprint')))
        approach_pose_map = T.pose2mat((approach_pos_base, approach_ori_base))
        approach_pose_map = transform @ approach_pose_map
        approach_pos_map = approach_pose_map[:3, 3]
        approach_ori_map = T.mat2quat(approach_pose_map[:3, :3])

        if self.debug:
            pcd_wrt_map = np.concatenate((pcd.reshape(-1, 3), np.ones((pcd.reshape(-1,3).shape[0], 1))), axis=1)
            transform = T.pose2mat((self.tf_map.get_transform(target_link=f'/base_footprint')))
            pcd_wrt_map = (transform @ pcd_wrt_map.T).T
            pcd_wrt_map = pcd_wrt_map[:, :3]
            pcd_to_plot = pcd_wrt_map.reshape(-1,3)
            rgb_to_plot = rgb.reshape(-1,3)

            # current pos in grey
            cur_pos_map, _ = self.tf_map.get_transform(target_link='/base_footprint')
            cur_pos_map = np.asarray(cur_pos_map)
            pcd_to_plot = np.concatenate((pcd_to_plot, cur_pos_map.reshape(1,3)), axis=0)
            rgb_to_plot = np.concatenate((rgb_to_plot.reshape(-1,3), np.asarray([[128.0, 128.0, 128.0]])), axis=0) # current pos in grey

            # approach pos in red
            pcd_to_plot = np.concatenate((pcd_to_plot, approach_pos_map.reshape(1,3)), axis=0)
            rgb_to_plot = np.concatenate((rgb_to_plot.reshape(-1,3), np.asarray([[255.0, 0.0, 0.0]])), axis=0) # approach pos in green


            U.plotly_draw_3d_pcd(pcd_to_plot, rgb_to_plot)

        is_success = False
        error = None
        print(colored(f"Door distance: {door_distance}", 'red'))
        print(colored(f"Make sure the door handle is to the {text_direction} of the robot.", 'red'))
        print(colored(f"Make sure there is enough space in the {text_direction} side of the robot so that the arm can move.", "red"))
        if execute:
            user_input = 'y'#input("Press Enter to continue or 'q' to quit: ")
            if user_input == 'q':
                execute = False
        if execute:
            if text_direction == 'left':
                _exec = U.reset_env(env, reset_pose=RP.HOME_R_TAP_DOOR_L, int_pose=RP.INT_L_H, delay_scale_factor=1.5)
            else:
                _exec = U.reset_env(env, reset_pose=RP.HOME_L_TAP_DOOR_R, int_pose=None, delay_scale_factor=1.5)
            print("Moving to the approach position?")
            _continue = 'y'#input("Press Enter to continue or 'q' to quit: ")
            if _continue == 'q':
                return self.on_failure(
                    reason_for_failure="Did not execute the skill.",
                    reset_required=False,
                    capture_history={},
                    return_info={},
                )
            # goal = self.create_move_base_goal((approach_pos_map, overrite_ori_map))
            # state = self.send_move_base_goal(goal)

        if not is_success:
            return self.on_failure(
                reason_for_failure=error,
                reset_required=False,
                capture_history=capture_history,
                return_info=return_info,
            )

        return self.on_success(
            capture_history=capture_history,
            return_info=return_info,
        )
