import os
import copy
import pickle
import argparse
import numpy as np
from easydict import EasyDict
from termcolor import colored

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from control_msgs.msg  import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import moma_safety.utils.utils as U
import moma_safety.utils.transform_utils as T # transform_utils
import moma_safety.utils.vision_utils as VU # vision_utils

from moma_safety.tiago.tiago_gym import TiagoGym
from moma_safety.tiago.utils.camera_utils import Camera
from moma_safety.tiago.skills.selector import SkillSelector
import moma_safety.tiago.RESET_POSES as RP
from moma_safety.tiago.skills import MoveToSkill, PickupSkill, GoToLandmarkSkill, UseElevatorSkill, OpenDoorSkill, PushObsGrSkill, NavigateToPointSkill, CallElevatorSkill
import moma_safety.tiago.prompters.vlms as vlms # GPT4V
from moma_safety.models.wrappers import GroundedSamWrapper
from moma_safety.tiago.ros_restrict import change_map, set_floor_map
from moma_safety.tiago.utils.ros_utils import Publisher, Listener, TFTransformListener

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
rospy.init_node('tiago_test')
def load_skill(skill_id, args, kwargs_to_add):
    if skill_id == 'move':
        prompt_args = {
            'raidus_per_pixel': 0.06,
            'arrow_length_per_pixel': 0.1,
            'plot_dist_factor': 1.0,
            'plot_direction': True,
        }
        skill = MoveToSkill(
            oracle_action=args.oracle,
            debug=False,
            run_dir=args.run_dir,
            prompt_args=prompt_args,
            **kwargs_to_add,
        )
    elif skill_id == 'pick_up_object':
        z_offset = 0.02
        if args.eval_id == 1:
            z_offset = -0.01
        prompt_args = {
            'add_object_boundary': False,
            'add_arrows_for_path': False,
            'radius_per_pixel': 0.04,
        }
        skill = PickupSkill(
            oracle_position=args.oracle,
            debug=args.debug,
            run_dir=args.run_dir,
            prompt_args=prompt_args,
            z_offset=z_offset,
            **kwargs_to_add,
        )
    elif skill_id == 'goto_landmark':
        prompt_args = {
            'raidus_per_pixel': 0.03,
        }
        skill = GoToLandmarkSkill(
            bld=args.bld,
            oracle_action=args.oracle,
            debug=args.debug,
            run_dir=args.run_dir,
            prompt_args=prompt_args,
            **kwargs_to_add,
        )
    elif skill_id == 'open_door':
        prompt_args = {}
        skill = OpenDoorSkill(
            oracle_action=args.oracle,
            debug=args.debug,
            run_dir=args.run_dir,
            prompt_args=prompt_args,
            **kwargs_to_add,
        )
    elif skill_id == 'use_elevator':
        prompt_args = {
            'add_object_boundary': False,
            'add_dist_info': False,
            'add_arrows_for_path': False,
            'radius_per_pixel': 0.03,
        }
        skill = UseElevatorSkill(
            oracle_position=args.oracle,
            debug=args.debug,
            run_dir=args.run_dir,
            prompt_args=prompt_args,
            **kwargs_to_add,
        )
    elif skill_id == 'call_elevator':
        prompt_args = {
            'add_object_boundary': False,
            'add_dist_info': False,
            'add_arrows_for_path': False,
            'radius_per_pixel': 0.03,
        }
        skill = CallElevatorSkill(
            oracle_position=args.oracle,
            debug=args.debug,
            run_dir=args.run_dir,
            prompt_args=prompt_args,
            **kwargs_to_add,
        )
    elif skill_id == 'push_obs_gr':
        prompt_args = {}
        skill = PushObsGrSkill(
            oracle_action=args.oracle,
            debug=args.debug,
            run_dir=args.run_dir,
            prompt_args=prompt_args,
            **kwargs_to_add,
        )
    elif skill_id == 'navigate_to_point_gr':
        prompt_args = {
            'raidus_per_pixel': 0.04,
            'arrow_length_per_pixel': 0.1, # don't need this
            'plot_dist_factor': 1.0, # don't need this
        }
        skill = NavigateToPointSkill(
            oracle_action=args.oracle,
            debug=args.debug,
            run_dir=args.run_dir,
            prompt_args=prompt_args,
            **kwargs_to_add,
        )
    else:
        raise ValueError(f"Unknown skill id: {skill_id}")
    return skill

def main(args):
    run_dir = args.run_dir
    # kwargs_to_add = get_kwargs_to_add()
    kwargs_to_add = {}
    kwargs_to_add['skip_ros'] = True
    run_dir = args.c_dir
    args.run_dir = run_dir

    dataset_name = 'move_forward'
    dataset_dir = os.path.join('../datasets', dataset_name)
    eval_dir = os.path.join(args.run_dir, f'test_reselect')
    os.makedirs(eval_dir, exist_ok=True)

    print("Loading gsam")
    gsam = None
    if args.run_vlm:
        # make sure the head is -0.8
        gsam = GroundedSamWrapper(sam_ckpt_path=os.environ['SAM_CKPT_PATH'])
    print("Gsam loading done")

    skill_id_list = args.skills
    skill_list = []
    for skill_id in skill_id_list:
        skill_list.append(load_skill(skill_id, args, kwargs_to_add=kwargs_to_add))
    skill_name2obj = {}
    skill_descs = []
    for skill in skill_list:
        skill.set_gsam(gsam)
        skill_name2obj[f'{skill.skill_name}'] = skill
        skill_descs.append(skill.skill_descs)

    prompt_args = {
        'n_vlm_evals': 0,
        'add_obj_ind': True,
        'raidus_per_pixel': 0.04,
        'add_dist_info': True,
        'add_object_boundary': False,
    }
    selector_skill = SkillSelector(
        skill_descs=skill_descs,
        skill_names=skill_name2obj.keys(),
        run_dir=eval_dir,
        prompt_args=prompt_args,
        **kwargs_to_add,
    )

    # very random number. change it to a number right after the skill execution fails.
    start_select_ind = 5

    eval_ind = args.rollout_num
    log_dir = os.path.join(args.run_dir, 'logs')
    traj_save_path = f'{log_dir}/eval_{eval_ind:03d}.pkl'
    traj_save = pickle.load(open(traj_save_path, 'rb'))

    # start_select_ind = traj_save['total_skill_exec']
    total_skill_exec = traj_save['total_skill_exec']
    skill_histories = traj_save['skill_histories']
    selection_seq = traj_save['selection_seq']
    skill_histories_orig = copy.deepcopy(skill_histories)
    selection_seq_orig = copy.deepcopy(selection_seq)

    print(f"{len(selection_seq)}/{start_select_ind}")
    print(f"{len(skill_histories['selection'])}/{start_select_ind}")

    task_query = traj_save['task_query']
    prev_exec_failed = False

    dataset = os.listdir(dataset_dir)
    dataset = [d for d in dataset if d.endswith('.pkl')]
    sorted(dataset)

    for select_ind in range(start_select_ind, len(selection_seq)):
        skill_histories['selection'] = skill_histories_orig['selection'][:select_ind]
        selection_seq = selection_seq_orig[:select_ind]
        last_skill_name = selection_seq[-1]

        if skill_histories['selection'][-1]['is_success'] == False:
            if last_skill_name != 'pick_up_object':
                continue
            if not prev_exec_failed:
                prev_exec_failed = True
                continue
            select_ind -= 1
            skill_histories['selection'] = skill_histories_orig['selection'][:select_ind]
            selection_seq = selection_seq_orig[:select_ind]
            last_skill_name = selection_seq[-1]
            print_skill_history = '\n'.join([f"({msg['subtask']}, {msg['skill_name']})" for msg in skill_histories['selection']])
            print(50*"-")
            print(colored(f"Starting from a previous checkpoint", "green"))
            print(colored(f"*** Last Skill Executed: {last_skill_name}\n*** Selected Skill Sequences: {selection_seq}.", "green"))
            print(colored(f"*** Skill Selection History:\n{print_skill_history}", "green"))
            print(50*"-")
            data_ind = 1
            data = dataset[data_ind]
            obs_pp = pickle.load(open(os.path.join(dataset_dir, data), 'rb'))
            rgb, depth, cam_intr, cam_extr, pcd, normals = obs_pp['rgb'], obs_pp['depth'], obs_pp['cam_intr'], obs_pp['cam_extr'], obs_pp['pcd'], obs_pp['normals']
            save_key = f'eval_eval_ind{eval_ind}_select_ind_{select_ind}_data_ind_{data_ind}'
            info = {'step_idx': select_ind, 'cam_intr': cam_intr, 'cam_extr': cam_extr, 'eval_ind': eval_ind, 'save_key': save_key, 'floor_num': args.floor_num}
            is_success, selection_error, selection_history, selection_return_info = \
                    selector_skill.step(
                        env=None,
                        rgb=rgb,
                        depth=depth,
                        pcd=pcd,
                        normals=normals,
                        query=task_query,
                        arm=args.arm,
                        execute=args.exec,
                        run_vlm=args.run_vlm,
                        info=info,
                        history=skill_histories['selection'] if args.add_selection_history else None,
                    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bld', type=str, default=None)
    parser.add_argument('--eval_id', type=int, help="eval_id=1 is the marker task whereas eval_id=2 is the coke task")
    parser.add_argument('--skills', type=str, nargs='+', default=None)
    parser.add_argument('--n_eval', type=int, default=1)
    parser.add_argument('--run_dir', type=str, default=None)
    parser.add_argument('--suffix', type=str, default=None)
    parser.add_argument('--c_dir', type=str, default=None, help="continue from the previous run")
    parser.add_argument('--rollout_num', type=int, default=None, help="resumes from the previous rollout number, wherever it stopped.")
    parser.add_argument('--remove_last', action='store_true', help='starts execution from one skill execution before.')

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--exec', action='store_true')
    parser.add_argument('--arm', type=str, default='left', choices=['right', 'left'])
    parser.add_argument('--run_vlm', action='store_true')
    parser.add_argument('--oracle', action='store_true')
    parser.add_argument('--method', default="ours", choices=["ours", "llm_baseline", "ours_no_markers"])

    parser.add_argument('--n_skill_selection', type=int, default=30, help='Maximum number of skill selection')
    parser.add_argument('--add_selection_history', action='store_true', help='Add history for skill selection')
    parser.add_argument('--add_reasoning', action='store_true', help='Add reasoning to the history')
    parser.add_argument('--floor_num', type=int, help='starting floor number', default=None)

    args = parser.parse_args()
    # default should have: --debug --exec --run_vlm --add_selection_history --floor_num {}
    if args.skills is None:
        args.skills = ['pick_up_object', 'move', 'goto_landmark', 'open_door', 'call_elevator', 'use_elevator', 'push_obs_gr', 'navigate_to_point_gr']
    assert args.run_vlm or args.oracle
    if args.rollout_num is not None:
        assert args.c_dir is not None, "please specify the directory you want to continue the skill execution from."
    assert args.eval_id is not None
    assert args.floor_num is not None, "Please add the correct --floor_num flag."
    assert args.bld is not None, "Please add the correct --bld flag."
    main(args)
