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

from moma_safety.tiago.skills import UseElevatorSkill
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# rospy.init_node('moma_safety', anonymous=True)

prompt_args = {
    'add_object_boundary': False,
    'add_dist_info': False,
    'add_arrows_for_path': False,
    'radius_per_pixel': 0.03,
}
dataset_name = 'test_elevator_mbb_north_out2'
dataset_dir = os.path.join('../datasets', dataset_name)
eval_dir = dataset_dir.replace(dataset_name, dataset_name + '_eval')
os.makedirs(eval_dir, exist_ok=True)
print("init use_eval skill")
skill = UseElevatorSkill(
    oracle_position=False,
    debug=True,
    run_dir=eval_dir,
    skip_ros=True,
    prompt_args=prompt_args,
)

dataset = os.listdir(dataset_dir)
dataset = [d for d in dataset if d.endswith('.pkl')]
task_query = "Go to the second floor."
for i, data in enumerate(dataset):
    print(f"index {i:03d}")
    obs_pp = pickle.load(open(os.path.join(dataset_dir, data), 'rb'))
    rgb, depth, cam_intr, cam_extr, pcd, normals = obs_pp['rgb'], obs_pp['depth'], obs_pp['cam_intr'], obs_pp['cam_extr'], obs_pp['pcd'], obs_pp['normals']
    data_name = data.split('.')[0]
    select_ind = i
    save_key = f'eval_{data_name}'
    info = {'step_idx': select_ind, 'cam_intr': cam_intr, 'cam_extr': cam_extr, 'eval_ind': i, 'save_key': save_key, 'floor_num': 1}
    path = os.path.join(eval_dir, f'depth_{info["save_key"]}.png')
    print(path)
    U.save_image(depth.astype(np.uint8), path)
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
    # skill.press_once(
    #     **common_args,
    #     execute=False,
    #     run_vlm=True,
    #     debug=False,
    #     bld='mbb',
    #     make_prompt_func=make_prompt,
    # )


