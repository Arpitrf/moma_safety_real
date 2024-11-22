import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt

import vlm_skill.utils.utils as U

class JsonDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, img_size=None, bgr=False):
        super(JsonDataset, self).__init__()
        self.dataset_path = dataset_path
        self.dataset = self.load()
        self.img_size = Image.open(self.dataset['data1']['img_path']).size if img_size is None else img_size
        self.bgr = bgr

    def load(self):
        json_file = os.path.join(self.dataset_path, 'dataset.json')
        with open(json_file, 'r') as f:
            dataset = json.load(f)

        for data_key in dataset.keys():
            dataset[data_key]['img_path'] = os.path.join(self.dataset_path, dataset[data_key]['img_path'])
            dataset[data_key]['segm_path'] = U.convert_img_path2segm_path(dataset[data_key]['img_path'], [dataset[data_key]['object_of_interest']])
        return dataset

    def __len__(self):
        return len(self.dataset.keys())

    def __getitem__(self, idx):
        key = f'data{idx+1}'
        data = self.dataset[key]
        # read jpg image in the format (H, W, C) as numpy array
        return_data = {'obs': {}, 'task_prompt': '', 'meta_info': {}}
        return_data['obs']['rgb'] = {}
        return_data['obs']['segm'] = {}
        # use Image.open to read the image and convert it to numpy array
        img = Image.open(data['img_path'])
        # resize the image to the desired size
        img = img.resize(self.img_size)
        if self.bgr:
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        return_data['obs']['rgb']['img'] = np.asarray(img)
        if os.path.exists(data['segm_path']):
            segm = Image.open(data['segm_path'])
            segm = segm.resize(self.img_size)
            return_data['obs']['segm']['img'] = np.asarray(segm)
        else:
            return_data['obs']['segm']['img'] = None
        return_data['task_prompt'] = data['language_task']
        return_data['meta_info'] = {
            'object_of_interest': [data['object_of_interest']],
            'img_path': data['img_path'],
            'labels': None, # We do not have have labels for this dataset
            'index': idx,
        }
        return return_data
