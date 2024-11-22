import os
import cv2
import rospy
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from termcolor import colored
from easydict import EasyDict

import moma_safety.utils.utils as U
import moma_safety.tiago.prompters.vlms as vlms # GPT4V
from moma_safety.tiago.skills import Reasoner
from moma_safety.models.wrappers import GroundedSamWrapper
from moma_safety.tiago.skills.push_obs_ground import PushObsGrSkill

# from tiago_test_scripts.test_elevator_segm import create_history_msgs, update_history

rospy.init_node('test_push_obs_gr', anonymous=True)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

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
            reason_for_failure, _ = reasoner.step(
                skill_name='push_object_on_ground',
                history_i=history_i,
                info={},
            )
            if type(reason_for_failure) == list:
                reason_for_failure = reason_for_failure[0]
        else:
            raise NotImplementedError
    history_i['model_analysis'] = reason_for_failure
    history_i['env_reasoning'] = None
    return history_i

# without history
# 8/10, 13/15, 9/10
# with oracle history
# 10/10, 14/15 (new dirs, new scenario)
# with model history
# 10/10, 14/15 (new dirs, new scenario), 8/10 (new task, new scenario)
#
# with oracle history (2 examples)
#
# with model history (2 examples)
# 10/10, 15/15 (new dirs, new scenario), 10/10

prompt_args = {}
save_history_dataset = False
method = 'ours'
load_history_dataset = True
# dataset_name = 'push_obs_ground_3'
# dataset_name = 'push_obs_gr2'

# model_name = 'claude-3-5-sonnet-20240620'
# model_name = 'claude-3-opus-20240229'
# model_name = 'claude-3-sonnet-20240229'
# model_name = 'claude-3-haiku-20240307'
# model_name = 'gpt-4o-2024-05-13'
# model_name = 'gpt-4o-2024-08-06'
model_name = 'gpt-4o-mini-2024-07-18'
vlm = U.get_model(model_name)

dataset_name = 'push_obs_gr'
# dataset_name = 'push_chair_inside'
reasoner_type = 'model'
dataset_dir = os.path.join('../datasets/final_ablations', dataset_name)
# base_dir_name = f'prompt_data_elev_{reasoner_type}'
# base_dir_name = f'{base_dir_name}_w_hist_v2' if load_history_dataset else base_dir_name
base_dir_name = f'{method}_hist{load_history_dataset}_{model_name}'
# eval_dir = dataset_dir.replace(dataset_name, dataset_name + '_eval')
eval_dir = os.path.join(dataset_dir, base_dir_name)
print(colored(f"Eval dir: {eval_dir}", 'green'))
os.makedirs(eval_dir, exist_ok=True)
print("init skill")
skill = PushObsGrSkill(
    method=method,
    oracle_action=False,
    debug=False,
    run_dir=eval_dir,
    prompt_args=prompt_args,
    add_histories=load_history_dataset,
    skip_ros=True,
    vlm=vlm,
)
print("skill initialized")

dataset = os.listdir(dataset_dir)
dataset = [d for d in dataset if d.endswith('.pkl')]
dataset = sorted(dataset)
task_query = "Remove obstacle from the ground to clear the path for the robot."
if dataset_name == 'push_chair_inside':
    task_query = "Arrange the chair in the room neatly."
for i, data in enumerate(dataset):
    obs_pp = pickle.load(open(os.path.join(dataset_dir, data), 'rb'))
    rgb, depth, cam_intr, cam_extr, pcd, normals = obs_pp['rgb'], obs_pp['depth'], obs_pp['cam_intr'], obs_pp['cam_extr'], obs_pp['pcd'], obs_pp['normals']
    data_name = data.split('.')[0]
    select_ind = 0
    save_key = f'eval_{data_name}'
    info = {'step_idx': select_ind, 'cam_intr': cam_intr, 'cam_extr': cam_extr, 'eval_ind': i, 'save_key': save_key}
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
    is_success, reason_for_failure, history_i, return_info = skill.step(
        **common_args,
        execute=False,
        run_vlm=True,
        debug=False,
    )
    args = EasyDict({
        'reasoner_type': reasoner_type,
    })
    if save_history_dataset:
        save_key = info['save_key']
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

