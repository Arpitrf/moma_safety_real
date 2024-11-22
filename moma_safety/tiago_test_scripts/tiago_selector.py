import os
import argparse
import pickle
import rospy
import numpy as np
from termcolor import colored
from easydict import EasyDict

import moma_safety.tiago.prompters.vlms as vlms
import moma_safety.utils.utils as U
import moma_safety.utils.transform_utils as T # transform_utils
import moma_safety.utils.vision_utils as VU # vision_utils

from moma_safety.tiago.tiago_gym import TiagoGym
from moma_safety.tiago.utils.camera_utils import Camera
from moma_safety.tiago.skills.selector import SkillSelector
from moma_safety.tiago.skills import Reasoner
import moma_safety.tiago.RESET_POSES as RP
from moma_safety.tiago.skills import (
    MoveToSkill, PickupSkill, GoToLandmarkSkill, UseElevatorSkill, OpenDoorSkill, PushObsGrSkill,
    CallElevatorSkill, NavigateToPointSkill,
)

from moma_safety.tiago.utils.ros_utils import Publisher, Listener, TFTransformListener
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
rospy.init_node('tiago_test')

reasoner = Reasoner()
def update_history(
        is_success,
        reason_for_failure,
        history_i,
        args,
    ):
    print(colored(f"Success: {is_success}", 'green' if is_success else 'red'))
    history_i['is_success'] = is_success

    # ask user if it is not successful
    success = U.confirm_user(True, 'Is the action successful (y/n)?')
    if success:
        history_i['is_success'] = True
    else:
        history_i['is_success'] = False
        if args.reasoner_type == 'oracle':
            U.clear_input_buffer()
            reason_for_failure = input('Reason for failure: ')
        elif args.reasoner_type == 'model':
            skill_info = history_i['skill_info']
            distance_info = history_i['distance_info']
            reason_for_failure, _ = reasoner.step(
                skill_name='selector',
                history_i=history_i,
                info={
                    'skill_info': skill_info,
                    'distance_info': distance_info,
                },
            )
            if type(reason_for_failure) == list:
                reason_for_failure = reason_for_failure[0]
        else:
            raise NotImplementedError
    history_i['model_analysis'] = reason_for_failure
    history_i['env_reasoning'] = None
    return history_i

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


def get_kwargs_to_add():
    print("Loading VLM")
    vlm = vlms.GPT4V(openai_api_key=os.environ['OPENAI_API_KEY'])
    print("Done.")
    print("Loading transforms")
    if not args.no_robot:
        tf_map = TFTransformListener('/map')
        tf_odom = TFTransformListener('/odom')
        tf_base = TFTransformListener('/base_footprint')
        tf_arm_left = TFTransformListener('/arm_left_tool_link')
        tf_arm_right = TFTransformListener('/arm_right_tool_link')
        print("Done.")
        print("Loading action client for move_base")
        client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        client.wait_for_server()
        print("Done")
        head_pub = Publisher('/head_controller/command', JointTrajectory)
        def process_head(message):
            return message.actual.positions
        head_sub = Listener('/head_controller/state', JointTrajectoryControllerState, post_process_func=process_head)
        kwargs_to_add = {
            'vlm': vlm,
            'tf_map': tf_map,
            'tf_odom': tf_odom,
            'tf_base': tf_base,
            'tf_arm_left': tf_arm_left,
            'tf_arm_right': tf_arm_right,
            'client': client,
            'head_pub': head_pub,
            'head_sub': head_sub,
            'skip_ros': args.no_robot,
        }
    else:
        kwargs_to_add = {
            'vlm': vlm,
            'skip_ros': args.no_robot,
        }
    return kwargs_to_add


def main(args):
    query = None
    if 'push_chair_inside' in args.data_dir:
        # query = "Can you help me arrange the chairs in the reception table?"
        # query = "Could you make the seating in the reception area more orderly?"
        # query = "Make the reception area more welcoming by arranging the chairs"
        # query =  "Can you help me arrange the chairs in the reception area?"
        query = "Arrange the chair to tuck it under the computer table."
    elif 'move_forward_ablation' in args.data_dir:
        query="I need energy to stay up all night. Can you get me something to drink?"
    elif 'diet_soda_variations' in args.data_dir:
        # query_list = ["Could you grab me a drink that's low in calories?", "Do you mind bringing me something diet-friendly to drink?", "Do we have any zero-calorie drinks you could bring me?", "Any chance you could find me a sugar-free soda?", "I want something fizzy to drink, but I am on diet. Can you get me something?"]
        # query = "Could you grab me a drink that's low in calories?"
        # query = "Do you mind bringing me something diet-friendly to drink?"
        # query = "Do we have any zero-calorie drinks you could bring me?"
        query = "Any chance you could find me a sugar-free soda?"
        # query = "I want something fizzy to drink, but I am on diet. Can you get me something?"
    elif 'test_landmark_pred' in args.data_dir:
        # query="Arrange the chair in the work area under the desk. Can you help me with that?"
        # query="I need energy to stay up all night. Can you get me something to drink?"
        query="I want to color grass in my drawing. Can you get me a marker?"
    elif 'diet_soda_variations' in args.data_dir:
        query = "I want something fizzy to drink, but I am on diet. Can you get me something?"
    elif 'hallucination' in args.data_dir:
        # query = "Can you help me arrange the chairs in the reception area?"
        # query = "Could you make the seating chairs in the reception area more orderly?"
        # query = "I want something fizzy to drink, but I am on diet. Can you get me something?"
        # query="I want to color grass in my drawing. Can you get me a marker?"
        # query="I spilled water on the floor. Can you hand me a napkin?"
        # query="I dropped my phone in water. I have dried the phone using a napkin but it is still not working. Can you help me remove all the moisture?"
        query="I dropped my phone in water and dried it off with a napkin, but it's still not working. Do you have any tips for getting rid of all the moisture?"
    else:
        raise NotImplementedError

    args.run_dir = args.run_dir + f'_{args.bld}_floor{args.floor_num}'
    if args.load_hist:
        args.run_dir += '_w_hist'
    args.run_dir += f'_{args.reasoner_type}'

    args.run_dir = os.path.join(args.run_dir, f'{query.replace(" ", "_")}')

    kwargs_to_add = get_kwargs_to_add()
    kwargs_to_add['method'] = args.method

    dataset = [None]
    if args.data_dir:
        dataset = os.listdir(args.data_dir)
        dataset = [d for d in dataset if d.endswith('.pkl')]
        dataset = sorted(dataset)

    # make sure the head is -0.8
    skill_id_list = args.skills
    skill_list = []
    for skill_id in skill_id_list:
        skill_list.append(load_skill(skill_id, args, kwargs_to_add))
    skill_name2obj = {}
    skill_descs = []
    for skill in skill_list:
        skill_name2obj[f'{skill.skill_name}'] = skill
        skill_descs.append(skill.skill_descs)

    if args.no_robot:
        env = None
    else:
        env = TiagoGym(
            frequency=10,
            right_arm_enabled=args.arm=='right',
            left_arm_enabled=args.arm=='left',
            right_gripper_type='robotiq2F-140' if args.arm=='right' else None,
            left_gripper_type='robotiq2F-85' if args.arm=='left' else None,
            base_enabled=False,
            torso_enabled=False,
        )

    prompt_args = {
        'n_vlm_evals': 0,
        'add_obj_ind': True,
        'raidus_per_pixel': 0.04,
        'add_dist_info': True,
        'add_object_boundary': False,
    }
    run_dir = args.run_dir
    os.makedirs(run_dir, exist_ok=True)
    selector_skill = SkillSelector(
        skill_descs=skill_descs,
        skill_names=skill_name2obj.keys(),
        run_dir=run_dir,
        prompt_args=prompt_args,
        add_histories=args.load_hist,
        reasoner_type=args.reasoner_type,
        **kwargs_to_add,
    )
    tf_listener = TFTransformListener('/base_footprint')
    grasp_h_r = RP.PICKUP_TABLE_L

    # check if it is safe to move to the reset pose
    # user_input = input('Move to reset pose? (y/n): ')
    # if user_input != 'y':
    #     return
    # env.reset(reset_arms=True, reset_pose=grasp_h_r, allowed_delay_scale=6.0)

    for i, data in enumerate(dataset):
        if data is not None:
            dataset_path = os.path.join(args.data_dir, data)
            obs_pp = pickle.load(open(dataset_path, 'rb'))
            rgb, depth, cam_intr, cam_extr, pcd, normals = obs_pp['rgb'], obs_pp['depth'], obs_pp['cam_intr'], obs_pp['cam_extr'], obs_pp['pcd'], obs_pp['normals']
        elif args.prev_traj_pkl:
            obs_pp = pickle.load(open(os.path.join(args.prev_traj_pkl), 'rb'))
            rgb, depth, cam_intr, cam_extr, pcd, normals = obs_pp['rgb'], obs_pp['depth'], obs_pp['cam_intr'], obs_pp['cam_extr'], obs_pp['pcd'], obs_pp['normals']
        else:
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
        if data is not None:
            save_key = 'eval_' + data.split('.')[0]
        info = {'step_idx': select_ind, 'cam_intr': cam_intr, 'cam_extr': cam_extr, 'save_key': save_key}
        info['floor_num'] = args.floor_num
        is_success, reason_for_failure, history_i, return_info = selector_skill.step(
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
            history=None,
        )
        skill_name = return_info['skill_name']
        subtask = return_info['subtask']
        common_args = {
            'env': env,
            'rgb': rgb,
            'depth': depth,
            'pcd': pcd,
            'normals': normals,
            'arm': args.arm,
            'info': info,
            'history': None,
            'query': subtask, # or task_query
        }
        if skill_name == 'goto_landmark':
            goto_is_success, goto_reason_for_failure, goto_history, goto_return_info = \
                    skill_name2obj[skill_name].step(
                        **common_args,
                        execute=False,
                        run_vlm=args.run_vlm,
                        debug=args.debug,
                        floor_num=args.floor_num,
                        bld=args.bld,
                    )
        if args.save_hist:
            save_key = info['save_key']
            eval_dir = args.run_dir
            save_dir = eval_dir
            print(save_key) # adjust the save_key before saving the pkl file
            history_path = os.path.join(save_dir, f'history_{save_key}.pkl')
            history_i = update_history(
                is_success,
                reason_for_failure,
                history_i,
                args=args,
            )
            pickle.dump(history_i, open(history_path, 'wb'))

    rospy.signal_shutdown("Shutdown")
    rospy.spin()


if __name__ == '__main__':
    # ROS_HOSTNAME=localhost ROS_MASTER_URI=http://localhost:11311 python tiago_test_scripts/tiago_selector.py --run_vlm --prev-traj-pkl /home/abba/Desktop/rutav/vlm-skill/datasets/move_009.pkl --llm-baseline
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default='../datasets/selector_test')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--exec', action='store_true')
    parser.add_argument('--arm', type=str, default='left', choices=['right', 'left'])
    parser.add_argument('--run_vlm', action='store_true')
    parser.add_argument('--oracle', action='store_true')
    parser.add_argument('--floor_num', type=int, help='starting floor number', default=2)
    parser.add_argument('--load_hist', action='store_true')
    parser.add_argument('--save_hist', action='store_true')
    parser.add_argument('--reasoner_type', type=str, default='oracle', choices=['oracle', 'model'])
    parser.add_argument('--prev-traj-pkl', type=str, default="")  # pkl file to load previous env readings from
    parser.add_argument(
        '--method', default="ours",
        choices=["ours", "llm_baseline", "ours_no_markers"])
    parser.add_argument('--bld', default='ahg', choices=['ahg', 'mbb', 'nhb'])
    args = parser.parse_args()
    args.skills = ['pick_up_object', 'move', 'goto_landmark', 'open_door', 'push_obs_gr']
    assert args.run_vlm or args.oracle
    assert args.run_dir is not None

    if (args.prev_traj_pkl) or (args.data_dir):
        args.no_robot = True
        assert args.exec == False

    main(args)
