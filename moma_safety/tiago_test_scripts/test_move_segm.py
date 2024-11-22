import os
import cv2
import pickle
import rospy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from termcolor import colored

import moma_safety.utils.utils as U
import moma_safety.tiago.prompters.vlms as vlms # GPT4V
from moma_safety.models.wrappers import GroundedSamWrapper
from moma_safety.tiago.prompters.object_bbox import bbox_prompt_img

from moma_safety.tiago.skills import MoveToSkill

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
        U.clear_input_buffer()
        reason_for_failure = input('Reason for failure: ')
    history_i['model_analysis'] = reason_for_failure
    history_i['env_reasoning'] = None
    return history_i

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# rospy.init_node('moma_safety', anonymous=True)

method = 'ours'
save_history_dataset = False
load_history_dataset = False
samples_per_hist = 2
prompt_args = {
   'raidus_per_pixel': 0.06,
    'arrow_length_per_pixel': 0.1,
    'plot_dist_factor': 1.0,
    'plot_direction': True,
    'add_dist_info': False,
}
dataset_name = 'move_right'
dataset_dir = os.path.join('../datasets/final_ablations/', dataset_name)
eval_dir = os.path.join(dataset_dir, f'{method}_hist{load_history_dataset}_eval')
os.makedirs(eval_dir, exist_ok=True)
skill = MoveToSkill(
    oracle_action=False,
    debug=False,
    run_dir=eval_dir,
    prompt_args=prompt_args,
    add_histories=load_history_dataset,
    skip_ros=True,
    method=method,
)

task_query = None
if dataset_name == 'move_forward':
    task_query = "Move closer to the diet coke can."
dataset = os.listdir(dataset_dir)
dataset = [d for d in dataset if d.endswith('.pkl')]
dataset = sorted(dataset)
history_list = []
history_all_path = os.path.join(eval_dir, 'history_all.pkl')
# print(history_all_path)
# if load_history_dataset:
#     # history_eval_dirs = [os.path.join('../datasets/move_tests2/', 'move_left_diet_coke_eval'), os.path.join('../datasets/move_tests2/', 'move_right_diet_coke_eval')]
#     history_eval_dirs = [os.path.join('../datasets/move_tests2/', 'move_left_diet_coke_eval'), os.path.join('../datasets/move_tests2/', 'move_right_diet_coke_eval'), os.path.join('../datasets/move_tests2/', 'move_forward_diet_coke_eval')]
#     # history_eval_dirs = [os.path.join('../datasets/move_tests/', 'move_test_left_eval'), os.path.join('../datasets/move_tests/', 'move_test_right_eval')]
#     # history_eval_dirs = [os.path.join('../datasets/move_tests/', 'move_test_left_eval'), os.path.join('../datasets/move_tests/', 'move_test_right_eval'), os.path.join('../datasets/move_tests/', 'move_test_forward_eval')]
#     # history_eval_dirs = [os.path.join('../datasets', 'move_right_ablation_eval_test'), os.path.join('../datasets', 'move_left_ablation_eval_test'), os.path.join('../datasets', 'move_forward_ablation_eval_test')]
#     # history_eval_dirs = [os.path.join('../datasets', 'move_forward_ablation_eval_test')]
#     for hist_eval_dir in history_eval_dirs:
#         _history_all_path = os.path.join(hist_eval_dir, 'history_all.pkl')
#         assert os.path.exists(_history_all_path), f"History file not found: {_history_all_path}"

#         _history_list = pickle.load(open(_history_all_path, 'rb'))
#         _success_list = [h for h in _history_list if h['is_success']]
#         _history_list = [h for h in _history_list if not h['is_success']]
#         _history_list = _history_list[:samples_per_hist]
#         _success_list = _success_list[:samples_per_hist]
#         history_list.extend(_history_list)
#         # history_list.extend(_success_list)
#         print(f"Loaded {len(history_list)} failed samples.")

for i, data in enumerate(dataset):
    select_ind = 0
    obs_pp = pickle.load(open(os.path.join(dataset_dir, data), 'rb'))
    rgb, depth, cam_intr, cam_extr, pcd, normals = obs_pp['rgb'], obs_pp['depth'], obs_pp['cam_intr'], obs_pp['cam_extr'], obs_pp['pcd'], obs_pp['normals']
    data_name = data.split('.')[0]
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
        'history': None, #if not load_history_dataset else history_list,
        'query': task_query, # or task_query
    }
    is_success, reason_for_failure, history_i, return_info = skill.step(
        **common_args,
        execute=False,
        run_vlm=True,
        debug=False,
    )
    if save_history_dataset:
        history_i = update_history(
            is_success,
            reason_for_failure,
            history_i,
            args=None,
        )
        history_list.append(history_i)
        pickle.dump(history_list, open(history_all_path, 'wb'))
