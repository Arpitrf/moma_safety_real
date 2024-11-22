import os
import cv2
import rospy
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from termcolor import colored

import moma_safety.utils.utils as U
import moma_safety.tiago.prompters.vlms as vlms # GPT4V
from moma_safety.models.wrappers import GroundedSamWrapper
from moma_safety.tiago.skills.open_door import OpenDoorSkill

rospy.init_node('test_push_obs_gr', anonymous=True)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

prompt_args = {}
method = 'llm_baseline'
dataset_name = 'door_direction'
# dataset_name = 'open_door2'
dataset_dir = os.path.join('../datasets/final_ablations', dataset_name)
# eval_dir = dataset_dir.replace(dataset_name, dataset_name + '_eval')
eval_dir  = os.path.join(dataset_dir, f'{method}')
os.makedirs(eval_dir, exist_ok=True)
print("init skill")
skill = OpenDoorSkill(
    oracle_action=False,
    debug=False,
    run_dir=eval_dir,
    prompt_args=prompt_args,
    skip_ros=True,
    method=method,
)
print("skill initialized")

dataset = os.listdir(dataset_dir)
dataset = [d for d in dataset if d.endswith('.pkl')]
dataset = sorted(dataset)
task_query = "Open the door to clear the path for the robot."
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

