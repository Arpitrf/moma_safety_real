import numpy as np
from collections import defaultdict

class TrajInfo:
    is_success: bool
    reason_for_failure: str
    history_i: dict
    center_coord: list
    opt_time: float
    rgb: np.ndarray
    depth: np.ndarray
    failure_category: dict

    def get_vlm_pred_img(self):
        return self.history_i['img']

    def get_query(self):
        return self.history_i['query']

    def get_env_img(self):
        return self.rgb

    def get_pred_coords(self):
        return self.center_coord

    def get_reason_for_failure(self):
        return self.reason_for_failure

    @property
    def env_feedback(self):
        return  self.get_reason_for_failure()

    @property
    def model_reasoning(self):
        if 'model_analysis' not in self.history_i.keys():
            return ""
        return self.history_i['model_analysis']

    @property
    def query(self):
        return self.get_query()

    @property
    def success(self):
        return self.is_success

    def failure_category2desc(self):
        category = {
            0: "Success Rate (0-1)",
            1: "invalid prediction",
            2: "collision",
            3: "out of FOV",
            4: "planning failure",
            5: "grasping failure"
        }
        return category

    def get_failure_category(self):
        '''
            0: no failure
            1: str: "outside the image" - invalid prediction
            2: str: "Collision" - collision
            3: str: "not in the field of view" - out of FOV
            4: str: "PLANNING_ERROR" - planning failure
            5: str: "POST_CONDITION_ERROR" - grasping failure
        '''
        if self.is_success:
            return 0
        # check using the str if it in the reason_for_failure string
        if "outside the image" in self.reason_for_failure:
            return 1
        if "Collision" in self.reason_for_failure:
            return 2
        if "not in the field of view" in self.reason_for_failure:
            return 3
        if "PLANNING_ERROR" in self.reason_for_failure:
            return 4
        if "POST_CONDITION_ERROR" in self.reason_for_failure:
            return 5
        raise ValueError("Unknown failure category: {}".format(self.reason_for_failure))

def get_traj_info(eval_attempt):
    traj_info = TrajInfo()
    traj_info.is_success = eval_attempt['is_success']
    traj_info.reason_for_failure = eval_attempt['reason_for_failure']
    traj_info.history_i = eval_attempt['history_i']
    traj_info.center_coord = eval_attempt['center_coord']
    traj_info.opt_time = eval_attempt['opt_time']
    traj_info.rgb = eval_attempt['rgb']
    traj_info.depth = eval_attempt['depth']
    return traj_info
