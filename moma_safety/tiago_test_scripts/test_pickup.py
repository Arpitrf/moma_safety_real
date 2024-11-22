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

from moma_safety.tiago.skills import PickupSkill


os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# rospy.init_node('moma_safety', anonymous=True)

method = 'ours'
# model_name = 'claude-3-5-sonnet-20240620'
# model_name = 'claude-3-opus-20240229'
# model_name = 'claude-3-sonnet-20240229'
# model_name = 'claude-3-haiku-20240307'
# model_name = 'gpt-4o-2024-08-06'
model_name = 'gpt-4o-mini-2024-07-18'

# pickup_less_distractors:
# ours: less distractors: 9/10 (1 gsam error), 7/10 (2 marker association errors, 1 gsam error)
# ours_no_markers: 10/10, 10/10
# llm: pickup less distractors: 5/10 , 8/10
# ours w/o CoT:
# ours, Claude 3 Haiku: 3/10, 1/10 = 20%
# ours, Claude 3 Sonnet: 2/10, 2/10 = 20%
# ours, Claude 3 Opus: 2/10, 2/10 = 20%
# ours, Claude 3.5 Sonnet: 6/10, 5/10 = 55%
# ours, GPT-4o-mini: 4/10, 4/10 = 40%

# pickup more distractors
# ours: 8/10, 5/10
# ours_no_markers: 10/10, 10/10
# llm: 6/10, 7/10
# ours w/o CoT:
# ours, Claude 3 Haiku: 2/10, 3/10 = 25%
# ours, Claude 3 Sonnet: 1/10, 5/10 = 30%
# ours, Claude 3 Opus: 2/10, 1/10 = 15%
# ours, Claude 3.5 Sonnet: 7/10, 5/10 = 60%
# ours, GPT-4o-mini: 2/10, 6/10 = 40%

# Trends:
# Opus-3 17.5%
# Haiku-3: 22.5%
# Sonnet-3: 25%
# Sonnet-3.5: 57.5%
# Ours: 72.5%
# GPT-4o-mini: 40%

# ours:
save_history_dataset = False
load_history_dataset = False
#samples_per_hist = 2
prompt_args = {
    'add_object_boundary': False,
    'add_arrows_for_path': False,
    'radius_per_pixel': 0.04,
}
# dataset_name = 'pickup1_more_distractors'
dataset_name = 'pickup2_less_dist'
dataset_dir = os.path.join('../datasets/final_ablations/', dataset_name)
eval_dir = os.path.join(dataset_dir, f'{method}_hist{load_history_dataset}_{model_name}_eval')
os.makedirs(eval_dir, exist_ok=True)
vlm = U.get_model(model_name)
skill = PickupSkill(
    # oracle=False,
    debug=False,
    run_dir=eval_dir,
    prompt_args=prompt_args,
    skip_ros=True,
    method=method,
    vlm=vlm,
)

task_query = None
task_query = "Pickup a sugar-free soda can."
if 'pickup2_more_distractors' in dataset_dir:
    task_query = "I want to draw grass in my drawing, can you pick up a marker for me?"
dataset = os.listdir(dataset_dir)
dataset = [d for d in dataset if d.endswith('.png')]
dataset = sorted(dataset)

for i, data in enumerate(dataset):
    select_ind = 0
    # obs_pp = pickle.load(open(os.path.join(dataset_dir, data), 'rb'))
    # rgb, depth, cam_intr, cam_extr, pcd, normals = obs_pp['rgb'], obs_pp['depth'], obs_pp['cam_intr'], obs_pp['cam_extr'], obs_pp['pcd'], obs_pp['normals']

    rgb_path = os.path.join(dataset_dir, data)
    depth, cam_intr, cam_extr, pcd, normals = None, None, None, None, None
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

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
