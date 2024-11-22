import os
import sys
import copy
import numpy as np
import pickle

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus

from moma_safety.tiago.utils.ros_utils import Publisher, Listener, TFTransformListener
from moma_safety.tiago.skills.base import SkillBase, movebase_code2error
import moma_safety.utils.utils as U
import moma_safety.utils.transform_utils as T # transform_utils
import moma_safety.tiago.prompters.vip_utils as vip_utils
from moma_safety.tiago.prompters.direction import prompt_move_img, prompt_rotate_img
from moma_safety.tiago.prompters.object_bbox import bbox_prompt_img

from termcolor import colored

def make_prompt(query, info, llm_baseline_info=None, method="ours", histories=None, **kwargs):
    '''
        The below instructions are used to prompt the model for skill parameter prediction and not skill selection.
        query: str is the main (sub)task description
        info: dictionary of information required to prompt the model
    '''
    add_dist_info = False
    bbox_ind2dist = None
    if info['add_dist_info'] == True:
        add_dist_info = True
        obj_bbox_list = info['obj_bbox_list']
        bbox_id2dist = info['bbox_id2dist']
        bbox_ind2dist = [(bbox.obj_id, bbox.dist2robot) for bbox in obj_bbox_list]
    # bbox_id2dir = info['bbox_id2dir']
    if method == "ours":
        visual_instructions = [
            "the image",
            "the direction of the image",
            "The forward direction is moving in the direction of the image (towards the top of the image), and backward is moving in the opposite direction. The left direction is moving to the left of the image, and right is moving to the right of the image. The robot uses its left arm to grab objects and can easily grasp objects on the left side of the image. If the robot moves right, the object will move to the left side of the image. If the robot moves left, the objects will move to the right side of the image. If the robot moves forward, the objects in the front will be closer and move to bottom of the image. If the robot moves backward, the objects in the front will move farther away from the robot, towards the top of the image. Each object is marked with an object id, example, 'B'. Along with it, the image is marked with directions indicating left, forward, and right directions to help you decide the direction.",
            # "The forward direction is moving in the direction of the image (towards the top of the image), and backward is moving in the opposite direction. The left direction is moving to the left of the image, and right is moving to the right of the image. Each object is marked with an object id, example, 'B'. Along with it, the image is marked with directions indicating left, forward, and right directions to help you decide the direction.",
            "describe the scene and each object id. Then,",
            "Make use of the markers (F,L,R) to guide your answer. ",
        ]
    elif method == "llm_baseline":
        visual_instructions = [
            "a description",
            "forward",
            "The forward direction is moving toward the objects on the scene, and backward is moving away from the objects on the scene. The left direction is moving to the left of the scene, and right is moving to the right of the scene. The robot uses its left arm to grab objects and can easily grasp objects on the left side of the scene. If the robot moves right, the object will move to the left side of the scene. If the robot moves left, the objects will move to the right side of the scene. If the robot moves forward, the objects in the front will be closer. If the robot moves backward, the objects in the front of the scene will move farther away from the robot.",
            "",
            "",
        ]
    elif method == "ours_no_markers":
        visual_instructions = [
            "the image",
            "the direction of the image",
            "The forward direction is moving in the direction of the image (towards the top of the image), and backward is moving in the opposite direction. The left direction is moving to the left of the image, and right is moving to the right of the image. The robot uses its left arm to grab objects and can easily grasp objects on the left side of the image. If the robot moves right, the object will move to the left side of the image. If the robot moves left, the objects will move to the right side of the image. If the robot moves forward, the objects in the front will be closer and move to bottom of the image. If the robot moves backward, the objects in the front will move farther away from the robot, towards the top of the image.",
            "describe the scene. Then,",
            "",
        ]
    else:
        raise NotImplementedError


    instructions = f"""
INSTRUCTIONS:
You are tasked to predict the direction in which the robot must move to complete the task. You are provided with {visual_instructions[0]} of the scene, and a description of the task. The robot is currently facing {visual_instructions[1]}. The robot can move in ONLY ONE of the four directions: forward, backward, left, or right by a distance of {info['move_dist']} meters. {visual_instructions[2]}

You are a five-time world champion in this game. Output only one of the directions: forward, backward, left, or right. Do NOT leave it empty. First, summarize all the errors made in previous predictions if provided. Then, {visual_instructions[3]}describe the effect of the robot moving in each direction. {visual_instructions[4]}Then, select the direction that can best help complete the task of reaching near the object of interest. Finally, provide the direction in a valid JSON of this format:
{{"direction_to_move": ""}}
""".strip()
    if add_dist_info:
        instructions += f"""\n
Below is provided the distances to the objects in the scene. Use this information to decide how far the robot is from the desired object."""
        for obj_id, dist in bbox_ind2dist:
            instructions += f"""
- Object id {obj_id} is {dist:.2f} metres from the robot."""
        instructions += f"""\n"""

    if llm_baseline_info:
        instructions += f"""\n
SCENE DESCRIPTION:
{llm_baseline_info['im_scene_desc']}
OBJECT ID DESCRIPTIONS:
{llm_baseline_info['obj_descs']}
"""
    # if bbox_id2dir is not None:
    #     atleast_one_id = False
    #     for obj_id, direction_dict in bbox_id2dir.items():
    #         if direction_dict is not None:
    #             atleast_one_id = True
    #             break
    #     print(f"bbox_id2dir: {bbox_id2dir}")
    #     if atleast_one_id:
    #         instructions += f"""\n
# OBSERVATIONS:"""
    #         for obj_id, direction_dict in bbox_id2dir.items():
    #             if direction_dict is None:
    #                 continue
    #             print(f"Object ID: {obj_id}", direction_dict)
    #             for direction, dist in direction_dict.items():
    #                 instructions += f"""
# - Object {obj_id} is at a distance of {dist:.2f} meters in the {direction} direction."""
    task_prompt = f"""\nTASK DESCRIPTION: {query}"""
    task_prompt += f"""\n
ANSWER: Let's think step by step.""".strip()
    return instructions, task_prompt

def make_history_prompt(history, _type='failure'):
    # assert _type == 'failure', "Only failure history is supported"
    # instructions = f"""
# Below is some of the prediction history that can help you understand the mistakes made in the past. Pay close attention to the mistakes made in the past and try to avoid them in the current prediction. Summarize each of prediction failures in the past. Based on the history, improve your prediction.
# PREDICTION HISTORY:
# """
    instructions = f"""
Below is some of the prediction history that can help you understand the mistakes and successful predictions made in the past. Pay close attention to the mistakes made in the past and try to avoid them in the current prediction. Summarize each of prediction failures in the past. Based on the history, improve your prediction. Note that the object ids are marked differently in each image.
PREDICTION HISTORY:
"""
    history_desc = []
    history_model_analysis = []
    for ind, msg in enumerate(history):
        # assert 'model_analysis' in msg, "Model analysis is required in the history."
        # assert msg['model_analysis'] != '', "Model analysis is required in the history."
        # print(msg['model_analysis'])
        if ('model_analysis' not in msg) or (msg['model_analysis'] == '') or (msg['model_analysis'] is None):
            msg['model_analysis'] = 'The prediction was accurate for successful task completion.'
        example_desc = f"""\n
Example {ind+1}:
- Task Query: {msg['query']}
- Answer: {msg['text_direction']}
""".strip()
        history_desc.append(example_desc)
        history_model_analysis.append(msg['model_analysis'])
    return instructions, history_desc, history_model_analysis

class TurnSkill(SkillBase):
    def __init__(
            self,
            oracle_action=False,
            debug=False,
            run_dir=None,
            prompt_args=None,
            skip_ros=False,
            add_histories=False,
            *args, **kwargs,
        ):
        super().__init__(*args, **kwargs)
        self.move_angle = 45
        self.oracle_action = oracle_action
        self.debug = debug
        self.skip_ros = skip_ros
        self.add_histories = add_histories
        self.history_list = None
        
        self.skill_name = "turn"
        self.skill_descs = f"""
skill_name: turn
arguments: direction
description: Turn the robot base in the specified direction by {self.move_angle} degrees. The direction can be either 'left' or 'right'. The skill should be used to turn the robot for different paths in corridors for navigation.
""".strip()

    def get_param_from_response(self, response, info):
        # TODO
        return None, None

    def step(self, env, rgb, depth, pcd, normals, query, execute=True, run_vlm=True, info=None, history=None, bboxes=None, mask_image=None, **kwargs):
        # TODO
        return None