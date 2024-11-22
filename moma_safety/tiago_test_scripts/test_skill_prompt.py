import argparse
import glob
import os
import pickle
from tqdm import tqdm

from moma_safety.tiago.skills import (
    MoveToSkill, PickupSkill, GoToLandmarkSkill, UseElevatorSkill,
    OpenDoorSkill, PushObsGrSkill,
    CallElevatorSkill, NavigateToPointSkill,
)
import moma_safety.utils.utils as U

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# rospy.init_node('moma_safety', anonymous=True)


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
            skip_ros=args.no_robot,
            **kwargs_to_add,
        )
    elif skill_id == 'pick_up_object':
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
            skip_ros=args.no_robot,
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
            skip_ros=args.no_robot,
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
            skip_ros=args.no_robot,
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
            skip_ros=args.no_robot,
            **kwargs_to_add,
        )
    elif skill_id == 'push_obs_gr':
        prompt_args = {}
        skill = PushObsGrSkill(
            oracle_action=args.oracle,
            debug=args.debug,
            run_dir=args.run_dir,
            prompt_args=prompt_args,
            skip_ros=args.no_robot,
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
            skip_ros=args.no_robot,
            **kwargs_to_add,
        )
    else:
        raise ValueError(f"Unknown skill id: {skill_id}")
    return skill


def test_skill(args, task_query, skill=None):
    os.makedirs(args.run_dir, exist_ok=True)
    select_ind = 0
    obs_pp = pickle.load(open(args.prev_traj_pkl, 'rb'))
    rgb, depth, cam_intr, cam_extr, pcd, normals = obs_pp['rgb'], obs_pp['depth'], obs_pp['cam_intr'], obs_pp['cam_extr'], obs_pp['pcd'], obs_pp['normals']
    fname = os.path.splitext(os.path.basename(args.prev_traj_pkl))[0]
    parent_dirname = args.prev_traj_pkl.split("/")[-2]
    save_key = f'eval_baseline_test_{parent_dirname}_{fname}'
    info = {'step_idx': select_ind, 'cam_intr': cam_intr, 'cam_extr': cam_extr, 'eval_ind': 0, 'save_key': save_key}
    common_args = {
        'env': None,
        'rgb': rgb,
        'depth': depth,
        'pcd': pcd,
        'normals': normals,
        'arm': 'left',
        'info': info,
        'history': None,
        'query': task_query, # or task_query
    }
    skill_id2specific_kwargs = dict(
        goto_landmark=dict(
            bld=args.bld,
            floor_num=2,
        ),
        move=dict(),
        pick_up_object=dict(),
        navigate_to_point_gr=dict(),
        open_door=dict(),
        use_elevator=dict(bld=args.bld),
        call_elevator=dict(bld=args.bld),
        push_obs_gr=dict(),
    )
    if args.skill_id in ["call_elevator", "use_elevator"]:
        info['floor_num'] = 2
    kwargs_to_add = {'method': args.method}
    if not skill:
        skill = load_skill(args.skill_id, args, kwargs_to_add)
    step_kwargs = dict(
        **common_args,
        execute=False,
        run_vlm=True,
        debug=args.debug,
        **skill_id2specific_kwargs[args.skill_id])
    skill.step(**step_kwargs)
    return skill


def run_param_predictions(args):
    skill_id2dir = dict(
        open_door="door_direction",
        move="move_*_ablation",
        pick_up_object="move_*_ablation",
    )
    skill_id2query = dict(
        open_door="Open the door in front of you.",
        move="Pick up the diet coke.",
        pick_up_object="I want something fizzy to drink, but I am on diet. Can you pick up the soda?",
    )
    pkls = glob.glob(
        os.path.join(
            args.dataset_dir, skill_id2dir[args.skill_id]
        ) + "/*.pkl", recursive=True)
    skill = None
    for pkl_datapath in tqdm(pkls):
        args.prev_traj_pkl = pkl_datapath
        skill = test_skill(args, skill_id2query[args.skill_id], skill)


if __name__ == "__main__":
    # ROS_HOSTNAME=localhost ROS_MASTER_URI=http://localhost:11311 python tiago_test_scripts/test_skill_prompt.py --run_dir /home/abba/Desktop/rutav/vlm-skill/temp/grounded_param_ablations/llm_baseline/ --dataset-dir /home/abba/Desktop/rutav/vlm-skill/datasets/Datasets/ --skill-id open_door --method=ours_no_markers
    # ROS_HOSTNAME=localhost ROS_MASTER_URI=http://localhost:11311 python tiago_test_scripts/test_skill_prompt.py --run_dir /home/abba/Desktop/rutav/vlm-skill/temp/baseline_debug --prev-traj-pkl /home/abba/Desktop/rutav/vlm-skill/datasets/move_009.pkl --skill-id navigate_to_point_gr --method=llm_baseline
    parser = argparse.ArgumentParser()
    parser.add_argument('--oracle', action='store_true')
    parser.add_argument('--debug', action='store_true')

    parser.add_argument('--run_dir', type=str, default="temp")
    parser.add_argument('--prev-traj-pkl', type=str)
    parser.add_argument('--bld', default='ahg', choices=['ahg', 'mbb'])
    parser.add_argument('--skill-id', required=True)
    parser.add_argument(
        '--method', default="ours",
        choices=["ours", "llm_baseline", "ours_no_markers"])

    # For param predictions
    parser.add_argument('--dataset-dir', type=str, default=None)
    args = parser.parse_args()
    args.no_robot = True

    if args.dataset_dir:
        assert args.dataset_dir
        run_param_predictions(args)
    else:
        assert args.prev_traj_pkl
        task_query = "Go to the kitchen area."
        skill, step_kwargs = test_skill(args, task_query)
