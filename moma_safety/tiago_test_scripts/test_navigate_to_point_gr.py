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
from moma_safety.tiago.skills.use_elevator import get_button_positions, make_prompt

from moma_safety.tiago.skills import NavigateToPointSkill
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
rospy.init_node('moma_safety', anonymous=True)

prompt_args = {
    'raidus_per_pixel': 0.06,
    'arrow_length_per_pixel': 0.1,
    'plot_dist_factor': 1.0,
    'plot_direction': True,
}
dataset_name = 'test_navigate_to'
dataset_dir = os.path.join('../datasets', dataset_name)
eval_dir = dataset_dir.replace(dataset_name, dataset_name + '_eval')
os.makedirs(eval_dir, exist_ok=True)
print("init skill")
skill = NavigateToPointSkill(
    oracle_action=False,
    debug=False,
    run_dir=eval_dir,
    prompt_args=prompt_args,
    skip_ros=True,
)
print("skill init done")

dataset = os.listdir(dataset_dir)
dataset = [d for d in dataset if d.endswith('.pkl')]
dataset = sorted(dataset)
task_query = "pick up the red cup."
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
    skill.step(
        **common_args,
        execute=False,
        run_vlm=True,
        debug=False,
    )

