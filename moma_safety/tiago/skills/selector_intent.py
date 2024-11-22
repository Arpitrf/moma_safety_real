import os
import cv2
import copy
import pickle
import matplotlib.pyplot as plt
import numpy as np

import moma_safety.tiago.prompters.vip_utils as vip_utils
from moma_safety.tiago.skills.base import SkillBase
import moma_safety.utils.utils as U
from moma_safety.tiago.prompters.object_bbox import bbox_prompt_img
import time 
from collections import Counter


def make_prompt_video_and_distance_subtask(skill_descs, distance_prompts, history_prompts=None, info=None, llm_baseline_info=None, method="ours"):
    """
    method arg is not being used for ablation results, just here for consistency
    """
    # Skills descriptions
    skill_desc_str = ""
    for ind, skill_desc in enumerate(skill_descs):
        skill_desc_str += skill_desc + "\n\n"

    # Object descriptions
    add_obj_ind = False
    bbox_ind2dist = None
    if info['add_obj_ind'] == True:
        add_obj_ind = True
        obj_bbox_list = info['obj_bbox_list']
        bbox_id2dist = info['bbox_id2dist']
        bbox_ind2dist = [(bbox.obj_id, bbox.dist2robot) for bbox in obj_bbox_list]

    instructions_object_id = ""
    if add_obj_ind:
        instructions_object_id += f"""
    - The images are marked with object / region id. The objects / region that are marked are the relevant objects to consider for the possible subtask choices, example, 'B'. You should make use of this information when coming up with the next task choices.
    - Avoid using the object id, like 'A', 'C', in the final response. Avoid mentioning the exact distance to the object in the subtask. Describe the object(s) involved in the task using color, nearby objects, or direction like left, right. This is very important.
OBSERVATIONS:"""
        for obj_id, dist in bbox_ind2dist:
            instructions_object_id += f"""
    - Object id {obj_id}: {dist:.2f} meters from the robot."""
    instructions_object_id += f"""\n"""

    # Generate subtask prompt dynamically from `make_tasks_prompt`
    subtask_prompt = f"""First, give a list of possible tasks to perform, using the information of the scene, the relevant objects, and relevant skills. The tasks should be using the skills listed above. 

        Note: 
        - The robot can only pick up and place objects that are within 0.7 meters of the robot. If the distance from the robot to the object is greater than 0.7 meters, then you SHOULD NOT include the pick up and place skill in the task choices!
        - The pick up and place skill should be used on smaller objects, and the navigate skill should be used on furniture, like tables, chairs, etc.
        - Always output navigate_to_point_on_ground skill first, if you see tables! This is very important.

        You should use the robot history: Eliminate the tasks that the robot has already performed. If the robot has picked up an object, it will not perform the task again!

        {history_prompts}

        Formulate your results in the format of multiple-choice questions. 

        Example 1: Given that I am farther away and the robot is moving, the possible subtasks to perform are:
            A) Navigate to the desk with pens on top of it.
            B) Navigate to the brown colored door.

        Example 2: Given that I am near the table, the possible subtasks to perform are:
            A) Place the apple in the pink bowl.
            B) Pick up the screwdriver with blue handle.
            
        Example 3: Given that I am near the table, the possible subtasks to perform are:
            A) Pick up the blue bowl with pink stripes.
            B) Pick up the apple.
            D) Pick up the purple bowl.

        Example 4: Given that I am in the corridor, the possible subtasks to perform are:
            A) Go to the kitchen.
            B) Go to the classroom.

        """.strip()

    distance_prompt, distance_prompt_1_object_direction, distance_prompt_2_robot_movement, distance_prompt_3_change_object_dist = distance_prompts

    prompt_tuning = {
    
    "nothing": "3",

    "distance_object": f"""
    3. You are given a history of the robot's movement. This contains the distance of the robot end effector position to each of the objects:
    {distance_prompt}
    Answer: where is the robot moving towards? Which object in the task choice option is the robot moving towards?
    4""".strip(),

    "robot_only": f"""
    You are given a history of the robot's movement. This contains the direction and orientation change of the robot's movement:
    {distance_prompt_2_robot_movement}
    """.strip()
    }

    examples = {
        "nothing": "",
        "few prompts": """
        Example reasoning w/ history 1: The robot is holding a book in its hand. Therefore it will not be picking up the book. It observes a shelf; it is likely that it will NAVIGATE to the shelf.

        Example reasoning w/ history 2: The robot is holding a food in its hand. Therefore it will not be picking up the food. It observes a bowl on a table; it is likely that it will NAVIGATE to the table.

        Example reasoning w/ history 3: The robot is holding a trash bag in its hand. it is likely that it will GO TO a trash can.

        Example reasoning w/ history 4: Given the robot's previous action of picking up the lemon, it will likely navigate to the table to place the lemon."""
    }

    # Final combined instruction set
    instructions = f"""
        INSTRUCTIONS:

        You are given a sequence of images of the scene. The images are taken from the camera on a mobile robot that is moving its base. Your goal is to determine the robot's intent based on this sequence of robot observations. You want to first come up with a list of potential task choices, then make use of the list of skills, the history of the robot's movement, and the list of task choices to determine the human's goal.

        The list of skills that the robot has are below. The tasks are using the skills listed here.
        {skill_desc_str}

        MARKERS ON THE IMAGE:
        {instructions_object_id}

        HISTORY OF PAST EXECUTIONS: You should make use of this information for decision making.

        {history_prompts}

        Think step by step, keep in mind the following points:

        1. {subtask_prompt}
        
        2. Focus on the images, and see if there is a change in robot's point of view; see how it is moving and changing its position, or if the gripper is getting closer to one of the objects, or turning towards one of the landmarks. {prompt_tuning['robot_only']}.Then, given the images and the robot's movement, summarize the previous the robot's movement.

        3. Then, summarize the previous executions made by the robot and feedback received from the human or environment. 
        
        Finally, answer: What is the robot trying to do? Choose from the list of possible task choices.
        
        Example reasoning 1: The robot is moving towards the left, where there is a table with a bowl on it. Since it has already picked up an object, it most likely wants to place the object on the bowl. However, the distance to the table is farther for the robot (> 0.7 meters) to place the object. We should first navigate to the table with a bowl on it.

        Example reasoning 2: The robot arm is moving closer towards the apple. The apple is already within the reach of the robot, that is, less than 0.7 meters. Therefore, it is likely that the robot will pick up the apple.

        Example reasoning 3: The robot is moving towards the bookshelf with a book in its hand. It is most likely trying to place the book on the bookshelf. However, the robot is far away from the bookshelf (> 0.7 meters). We should first navigate to the bookshelf.

        Example reasoning 4: The robot is moving towards the table which has a book on it. The robot tried to pick up the book before, but it failed due to IK solver issues. Since the robot is far away from the table, we should first navigate to the table with a book on it using the navigate skill.

        Example reasoning 5: The robot is near the book shelf which has one thriller and one comedy book. The robot tried to pick up the comedy book but the human stopped it. It is likely that the robot will try to pick up the thriller book.

        Example reasoning 6: The robot is moving towards the bookshelf with a book in its hand. The robot tried to place the book on the book holder, but it failed due to IK solver issues. Since the robot is far away from the book holder, we should first navigate to the bookshelf using the navigate skill.

        {examples['nothing']}


        Provide the skill name in a valid JSON format. Your answer at the end in a valid JSON of this format: {{"subtask": "", "skill_name": ""}}
        - Avoid using the object id in the final JSON response. Describe the object(s) involved in the sub-task instead of using the object id in the JSON response. This is very important.

        ANSWER: Let's think step by step.\n""".strip()

    return instructions, ''



def make_prompt_video_and_distance(skill_descs, distance_prompt, subtask_prompt, info=None, llm_baseline_info=None, method="ours"):
    """
    method arg is not being used for ablation results, just here for consistency
    """
    # Skills descriptions
    skill_desc_str = ""
    for ind, skill_desc in enumerate(skill_descs):
        skill_desc_str += skill_desc + "\n\n"

    # Object descriptions
    add_obj_ind = False
    bbox_ind2dist = None
    if info['add_obj_ind'] == True:
        add_obj_ind = True
        obj_bbox_list = info['obj_bbox_list']
        bbox_id2dist = info['bbox_id2dist']
        bbox_ind2dist = [(bbox.obj_id, bbox.dist2robot) for bbox in obj_bbox_list]

    instructions_object_id = ""
    if add_obj_ind:
        instructions_object_id += f"""
    - If there are signs for the landmarks in the building (e.g. Kitchen, Restoom, Lab, Elevator), you should use them to help you with the task choices. You should read the name of the landmark and their directions. 
    - If you see corridors or hallways, you should always output the GoToLandmark skill.
    - The images are marked with object / region id. The objects / region that are marked are the relevant objects to consider for the possible subtask choices. You should make use of this information when coming up with the next task choices.
    - Avoid using the object id in the final response. Describe the object(s) involved in the task instead of using the object id in the response. This is very important.
OBSERVATIONS:"""
        for obj_id, dist in bbox_ind2dist:
            instructions_object_id += f"""
    - Object id {obj_id}: {dist:.2f} meters from the robot."""
    instructions_object_id += f"""\n"""


    instructions = f"""
        INSTRUCTIONS:

        You are given a sequence of images of the scene. The images are taken from the camera on a mobile robot that is moving its base. Your goal is to determine the robot's intent based on this sequence of robot observations. You want to make use of the list of skills, the history of the robot's movement, and the list of task choices to determine the human's goal.

        Possible task choices:

        {subtask_prompt}

        The list of skills that the robot has are below. The tasks are using the skills listed here.
        {skill_desc_str}

        Think step by step, keep in mind the following points:

        1. Focus on the robot's changes in its point of view. See how it is moving and changing its position, and how it is getting closer to one of the objects. 
        
        Answer: where is the robot moving towards?

        2. You are given a history of the robot's movement. This contains the distance of the robot end effector position to each of the objects:

        {distance_prompt}

        Answer: where is the robot moving towards? Which object in the task choice option is the robot moving towards?

        3. Finally, with the two answers combined, answer: What is the robot trying to do? Choose from the list of possible task choices.

        Provide the skill name in a valid JSON format. Your answer at the end in a valid JSON of this format: {{"subtask": "", "skill_name": ""}}
        -Avoid using the object id in the final JSON response. Describe the object(s) involved in the sub-task instead of using the object id in the JSON response. This is very important.

        ANSWER: """.strip()

    return instructions, ''


def make_prompt_video(skill_descs, distance_prompt, subtask_prompt, info=None, llm_baseline_info=None, method="ours"):
    """
    method arg is not being used for ablation results, just here for consistency
    """
    # Skills descriptions
    skill_desc_str = ""
    for ind, skill_desc in enumerate(skill_descs):
        skill_desc_str += skill_desc + "\n\n"

    instructions = f"""
        INSTRUCTIONS:

        You are given a sequence of images of the scene. The images are taken from the camera on a mobile robot that is moving its base. Your goal is to determine the robot's intent based on this sequence of robot observations. You want to make use of the list of skills, the history of the robot's movement, and the list of task choices to determine the human's goal.

        Focus on the robot's changes in its point of view. See how it is moving and changing its position, and how it is getting closer to one of the objects. First answer: what is the robot moving towards?

        The list of skills that the robot has are below. The tasks are using the skills listed here.
        {skill_desc_str}

        What is the robot trying to do? Choose from the following options.

        Options:
        {subtask_prompt}

        Provide the skill name in a valid JSON format. Your answer at the end in a valid JSON of this format: {{"subtask": "", "skill_name": ""}}
        -Avoid using the object id in the final JSON response. Describe the object(s) involved in the sub-task instead of using the object id in the JSON response. This is very important.

        ANSWER: """.strip()

    return instructions, ''



def make_prompt(skill_descs, distance_prompt, subtask_prompt, info=None, llm_baseline_info=None, method="ours"):
    """
    method arg is not being used for ablation results, just here for consistency
    """
    # Skills descriptions
    skill_desc_str = ""
    for ind, skill_desc in enumerate(skill_descs):
        skill_desc_str += skill_desc + "\n\n"

    # Object descriptions
    add_obj_ind = False
    bbox_ind2dist = None
    if info['add_obj_ind'] == True:
        add_obj_ind = True
        obj_bbox_list = info['obj_bbox_list']
        bbox_id2dist = info['bbox_id2dist']
        bbox_ind2dist = [(bbox.obj_id, bbox.dist2robot) for bbox in obj_bbox_list]

    instructions_object_id = ""
    if add_obj_ind:
        instructions_object_id += f"""
    - The images are marked with object / region id. The objects / region that are marked are the relevant objects to consider for the possible subtask choices. You should make use of this information when coming up with the next task choices.
    - Avoid using the object id in the final response. Describe the object(s) involved in the task instead of using the object id in the response. This is very important.
OBSERVATIONS:"""
        for obj_id, dist in bbox_ind2dist:
            instructions_object_id += f"""
- Object id {obj_id}: {dist:.2f} meters from the robot."""
    instructions_object_id += f"""\n"""

    instructions = f"""
        INSTRUCTIONS:

        You are given a picture of the scene. Your goal is to determine the human's intent based on the history of the robot's movement. You want to make use of the list of skills, the history of the robot's movement, the changes of distance to each of the objects, and the list of task choices to determine the human's goal.

        IMPORTANT NOTE: 
        - The most important information you should use is the history of the robot's movement, and the changes of distance to each of the objects. Change in distance is more important than the absolute distance itself. Distance greater than 0.1 meters is considered important. 
        - If the robot is moving closer to multiple objects, then the object that is closer to the robot is more important.


        Then, using this sub-task, identitfy the skill that the robot must execute to complete the sub-task. You do NOT have to predict the arguments of the skill. 
        Select ONLY from the list of skills that the robot has and is feasible. The sub-task and skill_name are two different things. Sub-task is a high-level description of the task that the robot must complete. skill_name is the name of the skill that the robot must execute to complete the sub-task.

        The list of skills that the robot has are below. The tasks are using the skills listed here.
        {skill_desc_str}

        You are given a history of the robot's movement controlled by a human. This contains the distance of the robot end effector position to each of the objects, indicating where the human want to go:

        {distance_prompt}
        
        What is the human trying to do? Choose from the following options.

        Options:
        {subtask_prompt}

        Provide the skill name in a valid JSON format. Your answer at the end in a valid JSON of this format: {{"subtask": "", "skill_name": ""}}
        -Avoid using the object id in the final JSON response. Describe the object(s) involved in the sub-task instead of using the object id in the JSON response. This is very important.

        ANSWER: """.strip()

    return instructions, ''

def _determine_object_direction(robot_pos, object_pos, object_name):
    # Extract x and y positions for both robot and object
    robot_x, robot_y = robot_pos[:2]
    object_x, object_y = object_pos[:2]
    
    # Check if object position is NaN
    if np.isnan(object_x) or np.isnan(object_y):
        return f"Object id {object_name} position cannot be determined."
    
    # Calculate relative position between the object and the robot
    relative_x = object_x - robot_x
    relative_y = object_y - robot_y
    
    # Determine horizontal direction (left or right) based on the relative y position
    if relative_y < -0.05:
        horizontal_direction = "right"
    elif relative_y > 0.05:
        horizontal_direction = "left"
    else:
        horizontal_direction = "straight ahead"
    
    # Determine vertical direction (forward or backward) based on the relative x position
    if relative_x > 0.05:
        vertical_direction = "in front"
    elif relative_x < -0.05:
        vertical_direction = "behind"
    else:
        vertical_direction = "at the same x level"
    
    # print(f"object_name: {object_name}")
    # print(f"relative_x: {relative_x}, relative_y: {relative_y}, horizontal_direction: {horizontal_direction}, vertical_direction: {vertical_direction}")
    
    # Output the combined direction
    if horizontal_direction == "straight ahead":
        return f"Object id {object_name} is located {vertical_direction}."
    else:
        return f"Object id {object_name} is to the {horizontal_direction} and {vertical_direction} of the robot."


def _determine_robot_movement_direction(robot_base_pos_begin, robot_base_pos_end):

    # Extract x and y positions for the beginning and end
    robot_x_begin, robot_y_begin = robot_base_pos_begin[:2]
    robot_x_end, robot_y_end = robot_base_pos_end[:2]

    robot_base_ori_begin, robot_base_ori_end = robot_base_pos_begin[2], robot_base_pos_end[2]
    
    # Calculate the relative movement (delta x and delta y)
    relative_x = robot_x_end - robot_x_begin
    relative_y = robot_y_end - robot_y_begin
    
    # Determine horizontal movement direction (left or right)
    if relative_y < -0.08:
        horizontal_direction = "right"
    elif relative_y > 0.08:
        horizontal_direction = "left"
    else:
        horizontal_direction = "stationary"

    # Determine vertical movement direction (forward or backward)
    if relative_x > 0.08:
        vertical_direction = "forward"
    elif relative_x < -0.08:
        vertical_direction = "backward"
    else:
        vertical_direction = "stationary"
    
    # import pdb; pdb.set_trace()

    # calculate orientation change
    relative_ori = robot_base_ori_end - robot_base_ori_begin
    if relative_ori > 5:  # assume degrees
        orientation_direction = "left"
    elif relative_ori < -5:
        orientation_direction = "right" # TODO: double direction  
    else:
        orientation_direction = "stationary"

    if orientation_direction == "left":
        return "The robot is turning left."
    elif orientation_direction == "right":
        return "The robot is turning right." 

    # Construct the movement direction output
    if vertical_direction == "stationary" and horizontal_direction == "stationary":
        return "The robot is stationary."
    elif vertical_direction == "stationary":
        return f"The robot is moving {horizontal_direction}."
    elif horizontal_direction == "stationary":
        return f"The robot is moving {vertical_direction}."
    else:
        return f"The robot is moving {vertical_direction} and {horizontal_direction}."



def make_history_trace_prompt(robot_history, info=None, method="ours"):
    """
    robot_history: history of robot base position and left arm position.
    info: contains object locations relative to the base.
    """
    
    if info['add_obj_ind'] == True:
        bbox_id2pos = info['bbox_id2pos']

        robot_history_base = [np.array(entry['base']) for entry in robot_history]
        relative_left_to_base = [np.array(entry['left']) for entry in robot_history]

        # calculate the change in robot left gripper position: robot_history_base + relative_left_to_base
        robot_history_left = [robot_history_base[i] + relative_left_to_base[i] for i in range(len(robot_history_base))]

        print(robot_history_left)

        # calculate relative distance between left arm and the base

        selected_history = robot_history_left
        # change ori from radian to degree
        selected_history = [np.array([pos[0], pos[1], pos[2]*180/np.pi]) for pos in selected_history]

        distances_over_time = []
        for robot_base_pos in selected_history:
            distances_at_timestep = {}
            for bbox_id, object_pos in bbox_id2pos.items():
                dist = np.linalg.norm(robot_base_pos - object_pos)  # Calculate Euclidean distance
                distances_at_timestep[bbox_id] = dist
            distances_over_time.append(distances_at_timestep)

        # Access distances at the first and last timesteps
        first_timestep_distances = distances_over_time[0]
        last_timestep_distances = distances_over_time[-1]
        
        distance_prompt_1_object_direction = """
        Summary of object locations relative to the robot base:
        """
        for bbox_id, object_pos in bbox_id2pos.items():
            direction = _determine_object_direction(selected_history[0], object_pos, bbox_id)
            distance_prompt_1_object_direction += f"{direction}\n\n"

        distance_prompt_2_robot_movement = """
        Summary of robot movement over time:
        """

        robot_direction = _determine_robot_movement_direction(selected_history[0], selected_history[-1])
        distance_prompt_2_robot_movement += f"{robot_direction}\n\n"

        # Format the change in distance prompt
        distance_prompt_3_change_object_dist = "Initial and final distances between the robot and the objects, and absolute change from first to last timestep:\n"
        
        for obj_id, first_dist in first_timestep_distances.items():
            # Get the corresponding distance at the last timestep, if the object exists
            if obj_id in last_timestep_distances:
                last_dist = last_timestep_distances[obj_id]
                # Calculate the absolute change in distance
                change_in_dist = abs(last_dist - first_dist)
                
                # Determine if the object is further or nearer
                if last_dist > first_dist:
                    distance_change_str = f"further by {change_in_dist:.2f} meters"
                else:
                    distance_change_str = f"nearer by {change_in_dist:.2f} meters"
                    
                distance_prompt_3_change_object_dist += (
                    f"Object id {obj_id}: Initial distance = {first_dist:.2f} meters, "
                    f"Final distance = {last_dist:.1f} meters, "
                    f"{distance_change_str}.\n"
                )
            else:
                # Handle case where object is not present in the last timestep
                distance_prompt_3_change_object_dist += f"Object id {obj_id}: Not present at the last timestep.\n"

        # Combine all distance prompts
        distance_prompt = f"""
        {distance_prompt_1_object_direction}
        {distance_prompt_2_robot_movement}
        {distance_prompt_3_change_object_dist}
        """.strip()

        return distance_prompt, distance_prompt_1_object_direction, distance_prompt_2_robot_movement, distance_prompt_3_change_object_dist

    return ""

def make_tasks_prompt(skill_descs, info=None, method="ours"):

    # Skills descriptions
    skill_desc_str = ""
    for ind, skill_desc in enumerate(skill_descs):
        skill_desc_str += skill_desc + "\n\n"

    # Object descriptions
    add_obj_ind = False
    bbox_ind2dist = None
    if info['add_obj_ind'] == True:
        add_obj_ind = True
        obj_bbox_list = info['obj_bbox_list']
        bbox_id2dist = info['bbox_id2dist']
        bbox_ind2dist = [(bbox.obj_id, bbox.dist2robot) for bbox in obj_bbox_list]

    instructions_object_id = ""
    if add_obj_ind:
        instructions_object_id += f"""
    - The images are marked with object / region id. The objects / region that are marked are the relevant objects to consider for the possible subtask choices. You should make use of this information when coming up with the next task choices.
    - Avoid using the object id in the final response. Describe the object(s) involved in the task instead of using the object id in the response. This is very important.
    - Some skills like Turn does not require an object.
OBSERVATIONS:"""
        for obj_id, dist in bbox_ind2dist:
            instructions_object_id += f"""
- Object id {obj_id}: {dist:.2f} meters from the robot."""
    instructions_object_id += f"""\n"""

    # Full instruction
    instructions = f"""
        INSTRUCTIONS:
        You are given an image of the scene, along with a description of the scene and visible objects. Your goal is to give a list of possible tasks to perform, using the information of relevant objects, and relevant skills.

        {instructions_object_id}

        The list of skills that the robot has are:
        {skill_desc_str}
        The tasks should be using the skills listed above.

        Note: 
        - The robot can only pick up objects that are within 0.7 meters of the robot. If the distance from the robot to the object is greater than 0.7 meters, then you SHOULD USE THE NAVIGATE SKILL!
        - The pick up skill should be used on objects, and the navigate skill should be used on furnitures, like tables, chairs, etc.
        - When you see corridors, you should always output the Turn option.

        Finally, formulate your results in the format of multiple-choice questions. 

        For example:

        The possible subtasks to perform are:
            A) Nagivate to the desk.
            B) Navigate to the door.

        The possible subtasks to perform are:
            A) Pick up the blue bowl
            B) Pick up the apple
            C) Pick up the can
            D) Pick up the purple bowl

        The possible subtasks to perform are:
            A) Turn left.
            B) Turn right.

        The possible subtasks to perform are:
            A) Open the door.
            B) Navigate to the box.

        Answer: Think step by step.""".strip()

    return instructions, ''


def make_history_prompt(history):
    instructions = f"""
Below is the execution history from previous time-steps of the same episode. Pay close attention to your previous execution and success/failure feedback from the environment. Give a summary of what you have done, and what your current state is. Based on the history, you can improve your predictions.

PREVIOUS TIME-STEP HISTORY:
""".strip()
    history_desc = []
    history_model_analysis = []
    for ind, msg in enumerate(history):
        example_desc = f"""\n
    TIME-STEP: {ind+1}
    SUBTASK: {{"subtask": "{msg['subtask']}"}}
    SKILL NAME: {{"skill_name": "{msg['skill_name']}"}}
    SKILL SUCCESS: {msg['is_success']}
    """.strip()
        if not msg['is_success']:
            example_desc += f"""\n
    FEEDBACK: {msg['env_reasoning']}
    """.strip()

        history_model_analysis.append(msg['model_analysis'])
        history_desc.append(example_desc)

    return instructions + "\n\n" + "\n\n".join(history_desc) #, history_model_analysis
    # return instructions + "\n\n" + history_desc #, history_model_analysis


class SkillIntentSelector(SkillBase):
    def __init__(
        self,
        skill_descs: list[str],
        skill_names: list[str],
        run_dir: str,
        prompt_args: dict,
        add_histories: bool = False,
        reasoner_type: str = 'model',
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.skill_descs = skill_descs
        print(self.skill_descs)
        self.skill_names = skill_names
        self.n_vlm_evals = prompt_args.pop('n_vlm_evals', 0)
        self.add_obj_ind = prompt_args.pop('add_obj_ind', True)
        radius_per_pixel = prompt_args.pop('radius_per_pixel', 0.03)
        self.skill_name = 'selector_intent'
        self.prompt_args = {
            "color": (0, 0, 0),
            "mix_alpha": 0.6,
            'thickness': 2,
            'rgb_scale': 255,
            'add_dist_info': prompt_args.get('add_dist_info', True),
            'add_object_boundary': prompt_args.get('add_object_boundary', False),
            'radius_per_pixel': radius_per_pixel,
        }

        self.vis_dir = os.path.join(run_dir, 'selector')
        os.makedirs(self.vis_dir, exist_ok=True)
        self.add_histories = add_histories
        self.reasoner_type = reasoner_type
        if self.add_histories:
            history_eval_dirs = self.get_history_dirs()
            history_list = []
            for hist_eval_dir in history_eval_dirs:
                samples_per_hist = 1
                _history_all_path = os.path.join(hist_eval_dir, 'history_all.pkl')
                if hist_eval_dir.endswith('.pkl'):
                    _history_all_path = hist_eval_dir
                assert os.path.exists(_history_all_path), f"History file not found: {_history_all_path}"
                _history_list = pickle.load(open(_history_all_path, 'rb'))
                if not isinstance(_history_list, list):
                    _history_list = [_history_list]
                # _success_list = [h for h in _history_list if h['is_success']]
                _history_list = [h for h in _history_list if not h['is_success']]
                _history_list = _history_list[:samples_per_hist]
                # _success_list = _success_list[:samples_per_hist]
                history_list.extend(_history_list)
            self.history_list = history_list
            print(f"Loaded {len(history_list)} failed samples.")

    def get_param_from_response(self, response, query, info):
        error_list = []
        return_info = {}
        return_info['response'] = response
        subtask = ''
        try:
            subtask = vip_utils.extract_json(response, 'subtask')
        except Exception as e:
            print(f"Error: {e}")
            subtask = query
            error = 'Missing subtask information in the JSON response.'
            error_list.append(error)

        skill_name = ''
        try:
            skill_name = vip_utils.extract_json(response, 'skill_name')
        except Exception as e:
            print(f"Error: {e}")
            skill_name = None
            error = 'Missing skill name in the JSON response.'
            error_list.append(error)

        if (skill_name is not None) and (skill_name not in self.skill_names):
            error = f"Skill name {skill_name} is not in the list of skills."
            error_list.append(error)

        return_info['error_list'] = error_list
        return_info['subtask'] = subtask
        return_info['skill_name'] = skill_name
        return subtask, skill_name, return_info

    def step_video_subtask(
        self,
        env,
        encoded_image_lst,
        rgb_lst,
        depth,
        pcd,
        normals,
        robot_history,
        query,
        run_vlm=True,
        info=None,
        history=None,
        n_retries=1, # we query the model multiple times to avoid errors
        **kwargs,
    ):

        time_start = time.time()

        info = copy.deepcopy(info)
        step_idx = info['step_idx']
        e_value = 'incorrect'

        time_copy = time.time()

        time_gsam = time.time()

        # history_msgs = None # this is for episodic history
        # cross_history_msgs = None
        # if self.add_histories:
        #     cross_history_msgs = self.create_history_msgs(
        #         self.history_list,
        #         func=make_cross_history_prompt,
        #         func_kwargs={},
        #     )
        #     history_msgs = cross_history_msgs
        # if (history is not None) and (len(history)>0):
        #     ep_history_msgs = None
        #     if self.method == 'llm_baseline':
        #         ep_history_msgs = self.create_language_history_msgs(
        #             history,
        #             func=make_history_prompt,
        #             func_kwargs={},
        #         )
        #     else:
        #         ep_history_msgs = self.create_history_msgs(
        #             history,
        #             func=make_history_prompt,
        #             func_kwargs={},
        #         )
        #     if history_msgs is None:
        #         history_msgs = ep_history_msgs
        #     else:
        #         history_msgs.extend(ep_history_msgs)

        history_prompts = None
        if (history is not None) and (len(history)>0):
            history_prompts = make_history_prompt(history)
        

        distance_prompts = make_history_trace_prompt(robot_history, info)

        time_prompts = time.time()
        
        time_subtask = time.time()

        time_log = []
        results_log = []

        time_start_vlm = time.time()

        response = self.vlm_runner_video(
            encoded_image_lst=encoded_image_lst,
            history_msgs=None,
            make_prompt_func=make_prompt_video_and_distance_subtask,
            make_prompt_func_kwargs={
                'skill_descs': self.skill_descs,
                'distance_prompts': distance_prompts,
                'history_prompts': history_prompts,
                'info': info,
            }
        )

        time_end_vlm = time.time()
        time_log.append(time_end_vlm - time_start_vlm)

        if type(response) == str:
            subtask, skill_name, return_info = self.get_param_from_response(response, query=query, info=info)
        
            capture_history = {
                'image': encoded_image_lst[-1],
                'query': query,
                'model_response': [subtask, skill_name],
                'full_response': response,
                'subtask': subtask,
                'skill_name': skill_name,
                'skill_info': self.skill_descs,
                # 'distance_info': distance_str,
                'model_analysis': '', # this will be added by an external evaluator
            }

        else:

            subtasks_info = [self.get_param_from_response(r, query=query, info=info) for r in response]
            subtasks = [subtask for subtask, skill_name, return_info in subtasks_info]
            # convert everything in lower case
            subtasks = [subtask.lower() for subtask in subtasks]

            subtask_counter = Counter(subtasks)
            most_common_subtask, count = subtask_counter.most_common(1)[0]

            for subtask_tuple in subtasks_info:
                if subtask_tuple[0].lower() == most_common_subtask:
                    result_tuple = subtask_tuple
                    break
            
            subtask, skill_name, return_info = result_tuple

            capture_history = {
                'image': encoded_image_lst[-1],
                'query': query,
                'model_response': [subtask, skill_name],
                'full_response': response,
                'subtask': subtask,
                'skill_name': skill_name,
                'skill_info': self.skill_descs,
                'model_analysis': ''
            }

            if count < 4:
                error = f"There is insufficient information to determine the subtask. I should wait for more information."
                return self.on_failure(
                    reason_for_failure=error,
                    reset_required=False,
                    capture_history=capture_history,
                    return_info=return_info,
                )

            # print most common subtask and count
            print("\n\nMost common subtask: ", most_common_subtask, " with count: ", count, "\n\n")


        results_log.append({"subtask": subtask, "skill_name": skill_name})

        time_end_all = time.time()

        # print times with colors
        time_copy_log = time_copy - time_start
        time_gsam_log = time_gsam - time_copy
        time_prompts_log = time_prompts - time_gsam
        time_subtask_log = time_subtask - time_prompts
        time_vlm_log = time_end_vlm - time_subtask
        time_end_log = time_end_all - time_end_vlm
        time_overall_log = time_end_all - time_start
        
        # print time with colors
        print(f"\033[1;32;40mTime for copying: {time_copy_log:.2f} seconds\033[0m")
        print(f"\033[1;32;40mTime for GSAM: {time_gsam_log:.2f} seconds\033[0m")
        print(f"\033[1;32;40mTime for prompts: {time_prompts_log:.2f} seconds\033[0m")
        print(f"\033[1;32;40mTime for subtask: {time_subtask_log:.2f} seconds\033[0m")
        print(f"\033[1;32;40mTime for VLM: {time_vlm_log:.2f} seconds\033[0m")
        print(f"\033[1;32;40mTime for end: {time_end_log:.2f} seconds\033[0m")
        print(f"\033[1;32;40mTime for overall: {time_overall_log:.2f} seconds\033[0m")

        print("\n\nTime log: ", time_log, "\n\n")
        print("\n\nResults log: \n", results_log, "\n\n")

        return_info.update({ # this will be reused in the pickup skill to avoid gsam queries
            # 'bboxes': bboxes,
            # 'mask_image': mask_image,
            'time': [
                time_copy_log,
                time_gsam_log,
                time_prompts_log,
                time_subtask_log,
                time_vlm_log,
                time_end_log,
                time_overall_log,
            ]
        })

        error = None
        if len(return_info['error_list']) > 0:
            error = "Following errors have been produced: "
            for e in return_info['error_list']:
                error += f"{e}, "
            error = error[:-2]
            return self.on_failure(
                reason_for_failure=error,
                reset_required=False,
                capture_history=capture_history,
                return_info=return_info,
            )

        return self.on_success(
            capture_history=capture_history,
            return_info=return_info,
        )


    def step_video(
        self,
        env,
        encoded_image_lst,
        rgb_lst,
        depth,
        pcd,
        normals,
        robot_history,
        query,
        run_vlm=True,
        info=None,
        history=None,
        n_retries=1, # we query the model multiple times to avoid errors
        **kwargs,
    ):
        
        # keep time for individual functions 

        time_start = time.time()

        info = copy.deepcopy(info)
        step_idx = info['step_idx']
        e_value = 'incorrect'

        time_copy = time.time()

        # encoded_image_lst = []
        # for rgb_id in range(len(rgb_lst)):

        #     rgb = rgb_lst[rgb_id]

        #     im_copy_start = time.time()
        #     im = rgb.copy()
        #     im_copy_end = time.time()

        #     img_size = min(im.shape[0], im.shape[1])

        #     self.prompt_args.update({
        #         'radius': int(img_size * self.prompt_args['radius_per_pixel']),
        #         'fontsize': int(img_size * 30 * self.prompt_args['radius_per_pixel']),
        #     })
        #     info.update({'add_obj_ind': self.add_obj_ind})

        #     # Start the overall timing
        #     start_total = time.time()

        #     # only do this for the last image
        #     if self.add_obj_ind and rgb_id == len(rgb_lst) - 1:
        #         gsam_query = ['objects.']
                
        #         # Time the object bounding box retrieval
        #         start_bbox_retrieval = time.time()
                
        #         for _ in range(2):
        #             bboxes, mask_image = self.get_object_bboxes(rgb, query=gsam_query)
        #             if len(bboxes) > 0:
        #                 break
        #             else:
        #                 gsam_query = ['all objects and floor']
                
        #         # End time for bbox retrieval
        #         end_bbox_retrieval = time.time()
        #         print(f"Bounding box retrieval took: {end_bbox_retrieval - start_bbox_retrieval:.4f} seconds")

        #         if len(bboxes) == 0:
        #             # this should not happen
        #             ipdb.set_trace()
        #             error = "No objects found in the scene."
        #             self.on_failure(
        #                 reason_for_failure=error,
        #                 reset_required=False,
        #                 capture_history={},
        #                 return_info={},
        #             )

        #         # Time the overlay image creation
        #         start_overlay_creation = time.time()

        #         # used mainly for debugging
        #         overlay_image = U.overlay_xmem_mask_on_image(
        #             rgb.copy(),
        #             np.array(mask_image),
        #             use_white_bg=False,
        #             rgb_alpha=0.3
        #         )
                
        #         # End time for overlay creation
        #         end_overlay_creation = time.time()
        #         print(f"Overlay creation took: {end_overlay_creation - start_overlay_creation:.4f} seconds")

        #         # save the overlay image for debugging
        #         U.save_image(overlay_image, os.path.join(self.vis_dir, f'overlay_image_{info["save_key"]}_{rgb_id}.png'))

        #         # Time for bbox processing
        #         start_bbox_processing = time.time()

        #         bbox_id2dist = {}
        #         bbox_id2pos = {}
        #         for bbox in bboxes:
        #             center = (bbox[1] + bbox[3]) // 2, (bbox[2] + bbox[4]) // 2
        #             pos_wrt_base = pcd[center[1], center[0]]
        #             dist = np.linalg.norm(pos_wrt_base[:2])
        #             bbox_id2dist[bbox[0]] = dist
        #             bbox_id2pos[bbox[0]] = pos_wrt_base
        #             print(bbox[0], bbox_id2pos[bbox[0]])

        #         # End time for bbox processing
        #         end_bbox_processing = time.time()
        #         print(f"Bounding box processing took: {end_bbox_processing - start_bbox_processing:.4f} seconds")

        #         info.update({
        #             'bbox_ignore_ids': [0],
        #             'bbox_id2dist': bbox_id2dist,
        #             'bbox_id2pos': bbox_id2pos,
        #         })

        #         # Time for prompt image creation
        #         start_prompt_img = time.time()

        #         prompt_rgb, obj_bbox_list = bbox_prompt_img(
        #             im=rgb.copy(),
        #             info=info,
        #             bboxes=bboxes,
        #             prompt_args=self.prompt_args,
        #         )
                
        #         # End time for prompt image creation
        #         end_prompt_img = time.time()
        #         print(f"Prompt image creation took: {end_prompt_img - start_prompt_img:.4f} seconds")

        #         info['obj_bbox_list'] = obj_bbox_list
        #         U.save_image(prompt_rgb, os.path.join(self.vis_dir, f'prompt_img_{info["save_key"]}_{rgb_id}.png'))

        #     else:
        #         im_copy_start = time.time()
                
        #         prompt_rgb = rgb.copy()

        #         im_copy_end = time.time()

        #     start_encoding = time.time()

        #     encoded_image = U.encode_image(prompt_rgb)
        #     encoded_image_lst.append(encoded_image)

        #     end_encoding = time.time()

        #     print(f"Encoding took: {end_encoding - start_encoding:.4f} seconds")

        #     # End the overall timing
        #     end_total = time.time()
        #     print(f"Total execution time: {end_total - start_total:.4f} seconds")

        time_gsam = time.time()


        history_msgs = None # this is for episodic history
        cross_history_msgs = None
        if self.add_histories:
            cross_history_msgs = self.create_history_msgs(
                self.history_list,
                func=make_cross_history_prompt,
                func_kwargs={},
            )
            history_msgs = cross_history_msgs
        if (history is not None) and (len(history)>0):
            ep_history_msgs = None
            if self.method == 'llm_baseline':
                ep_history_msgs = self.create_language_history_msgs(
                    history,
                    func=make_history_prompt,
                    func_kwargs={},
                )
            else:
                ep_history_msgs = self.create_history_msgs(
                    history,
                    func=make_history_prompt,
                    func_kwargs={},
                )
            if history_msgs is None:
                history_msgs = ep_history_msgs
            else:
                history_msgs.extend(ep_history_msgs)

        distance_prompt = make_history_trace_prompt(robot_history, info)

        time_prompts = time.time()

        subtask_prompt = self.obtain_task_choices(
                            env=env,
                            encoded_image_lst=encoded_image_lst,
                            rgb=encoded_image_lst[-1], 
                            depth=depth,
                            pcd=pcd,
                            normals=normals,
                            query=None,
                            info=info,
                            history=history_msgs,
                            **kwargs,
                        )
        
        time_subtask = time.time()

        time_log = []
        results_log = []

        # print("\n\nQuerying the model, Trial ", _, "\n\n")  

        time_start_vlm = time.time()

        response = self.vlm_runner_video(
            encoded_image_lst=encoded_image_lst,
            history_msgs=history_msgs,
            make_prompt_func=make_prompt_video_and_distance,
            make_prompt_func_kwargs={
                'skill_descs': self.skill_descs,
                'distance_prompt': distance_prompt,
                'subtask_prompt': subtask_prompt,
                'info': info,
            }
        )

        time_end_vlm = time.time()
        time_log.append(time_end_vlm - time_start_vlm)

#         #### creating the distance information string for capturing history
#         bbox_ind2dist = [(bbox.obj_id, bbox.dist2robot) for bbox in obj_bbox_list]
#         distance_str = ""
#         for obj_i
# d, dist in bbox_ind2dist:
#             distance_str += f"""
# - Object id {obj_id} is {dist:.2f} metres from the robot."""
#         ####

        subtask, skill_name, return_info = self.get_param_from_response(response, query=query, info=info)

        results_log.append({"subtask": subtask, "skill_name": skill_name})

        capture_history = {
            'image': encoded_image_lst[-1],
            'query': query,
            'model_response': [subtask, skill_name],
            'full_response': response,
            'subtask': subtask,
            'skill_name': skill_name,
            'skill_info': self.skill_descs,
            # 'distance_info': distance_str,
            'model_analysis': '', # this will be added by an external evaluator
        }
        # self.save_model_output(
        #     rgb=prompt_rgb,
        #     response=response,
        #     subtitles=[f'Task Query: {query}', f'Subtask: {subtask}\nSkill: {skill_name}'],
        #     img_file=os.path.join(self.vis_dir, f'output_{info["save_key"]}.png'),
        # )

        time_end_all = time.time()

        # print times with colors
        time_copy_log = time_copy - time_start
        time_gsam_log = time_gsam - time_copy
        time_prompts_log = time_prompts - time_gsam
        time_subtask_log = time_subtask - time_prompts
        time_vlm_log = time_end_vlm - time_subtask
        time_end_log = time_end_all - time_end_vlm
        time_overall_log = time_end_all - time_start
        
        # print time with colors
        print(f"\033[1;32;40mTime for copying: {time_copy_log:.2f} seconds\033[0m")
        print(f"\033[1;32;40mTime for GSAM: {time_gsam_log:.2f} seconds\033[0m")
        print(f"\033[1;32;40mTime for prompts: {time_prompts_log:.2f} seconds\033[0m")
        print(f"\033[1;32;40mTime for subtask: {time_subtask_log:.2f} seconds\033[0m")
        print(f"\033[1;32;40mTime for VLM: {time_vlm_log:.2f} seconds\033[0m")
        print(f"\033[1;32;40mTime for end: {time_end_log:.2f} seconds\033[0m")
        print(f"\033[1;32;40mTime for overall: {time_overall_log:.2f} seconds\033[0m")

        print("\n\nTime log: ", time_log, "\n\n")
        print("\n\nResults log: \n", results_log, "\n\n")
        # count different options times
        count_dict = {}
        for res in results_log:
            if res['subtask'] not in count_dict:
                count_dict[res['subtask']] = 0
            count_dict[res['subtask']] += 1
        print("\n\nCount dict: ", count_dict, "\n\n")

        return_info.update({ # this will be reused in the pickup skill to avoid gsam queries
            # 'bboxes': bboxes,
            # 'mask_image': mask_image,
            'time': [
                time_copy_log,
                time_gsam_log,
                time_prompts_log,
                time_subtask_log,
                time_vlm_log,
                time_end_log,
                time_overall_log,
            ]
        })

        error = None
        if len(return_info['error_list']) > 0:
            error = "Following errors have been produced: "
            for e in return_info['error_list']:
                error += f"{e}, "
            error = error[:-2]
            return self.on_failure(
                reason_for_failure=error,
                reset_required=False,
                capture_history=capture_history,
                return_info=return_info,
            )

        return self.on_success(
            capture_history=capture_history,
            return_info=return_info,
        )

    def step(
        self,
        env,
        rgb,
        depth,
        pcd,
        normals,
        robot_history,
        query,
        run_vlm=True,
        info=None,
        history=None,
        n_retries=5, # we query the model multiple times to avoid errors
        **kwargs,
    ):
        
        time_start = time.time()

        info = copy.deepcopy(info)
        step_idx = info['step_idx']
        e_value = 'incorrect'
        im = rgb.copy()
        img_size = min(im.shape[0], im.shape[1])

        self.prompt_args.update({
            'radius': int(img_size * self.prompt_args['radius_per_pixel']),
            'fontsize': int(img_size * 30 * self.prompt_args['radius_per_pixel']),
        })
        info.update({'add_obj_ind': self.add_obj_ind})

        time_copy = time.time()

        if self.add_obj_ind:
            gsam_query = ['objects.']
            for _ in range(2):
                bboxes, mask_image = self.get_object_bboxes(rgb, query=gsam_query)
                if len(bboxes) > 0:
                    break
                else:
                    gsam_query = ['all objects and floor']
            if len(bboxes) == 0:
                # this should not happen
                import ipdb; ipdb.set_trace()
                error = "No objects found in the scene."
                self.on_failure(
                    reason_for_failure=error,
                    reset_required=False,
                    capture_history={},
                    return_info={},
                )

            # used mainly for debugging
            overlay_image = U.overlay_xmem_mask_on_image(
                rgb.copy(),
                np.array(mask_image),
                use_white_bg=False,
                rgb_alpha=0.3
            )
            # save the overlay image for debugging
            U.save_image(overlay_image, os.path.join(self.vis_dir, f'overlay_image_{info["save_key"]}.png'))
            bbox_id2dist = {}
            bbox_id2pos = {}
            for bbox in bboxes:
                center = (bbox[1] + bbox[3]) // 2, (bbox[2] + bbox[4]) // 2
                pos_wrt_base = pcd[center[1], center[0]]
                dist = np.linalg.norm(pos_wrt_base[:2])
                bbox_id2dist[bbox[0]] = dist
                bbox_id2pos[bbox[0]] = pos_wrt_base
                print(bbox[0], bbox_id2pos[bbox[0]])

            info.update({
                'bbox_ignore_ids': [0],
                'bbox_id2dist': bbox_id2dist,
                'bbox_id2pos': bbox_id2pos,
            })
            prompt_rgb, obj_bbox_list = bbox_prompt_img(
                im=rgb.copy(),
                info=info,
                bboxes=bboxes,
                prompt_args=self.prompt_args,
            )
            info['obj_bbox_list'] = obj_bbox_list
            U.save_image(prompt_rgb, os.path.join(self.vis_dir, f'prompt_img_{info["save_key"]}.png'))
        else:
            prompt_rgb = rgb.copy()

        encoded_image = U.encode_image(prompt_rgb)

        time_gsam = time.time()

        history_msgs = None # this is for episodic history
        cross_history_msgs = None
        if self.add_histories:
            cross_history_msgs = self.create_history_msgs(
                self.history_list,
                func=make_cross_history_prompt,
                func_kwargs={},
            )
            history_msgs = cross_history_msgs
        if (history is not None) and (len(history)>0):
            ep_history_msgs = None
            if self.method == 'llm_baseline':
                ep_history_msgs = self.create_language_history_msgs(
                    history,
                    func=make_history_prompt,
                    func_kwargs={},
                )
            else:
                ep_history_msgs = self.create_history_msgs(
                    history,
                    func=make_history_prompt,
                    func_kwargs={},
                )
            if history_msgs is None:
                history_msgs = ep_history_msgs
            else:
                history_msgs.extend(ep_history_msgs)

        distance_prompt = make_history_trace_prompt(robot_history, info)

        time_prompts = time.time()

        subtask_prompt = self.obtain_task_choices(
                            env=env,
                            rgb=rgb,
                            depth=depth,
                            pcd=pcd,
                            normals=normals,
                            query=None,
                            info=info,
                            history=history_msgs,
                            **kwargs,
                        )
        
        time_subtask = time.time()

        time_log = []
        results_log = []

        for _ in range(n_retries):

            print("\n\nQuerying the model, Trial ", _, "\n\n")  

            time_start_vlm = time.time()

            response = self.vlm_runner(
                encoded_image=encoded_image,
                history_msgs=history_msgs,
                make_prompt_func=make_prompt,
                make_prompt_func_kwargs={
                    'skill_descs': self.skill_descs,
                    'distance_prompt': distance_prompt,
                    'subtask_prompt': subtask_prompt,
                    'info': info,
                }
            )

            time_end_vlm = time.time()
            time_log.append(time_end_vlm - time_start_vlm)

            #### creating the distance information string for capturing history
            bbox_ind2dist = [(bbox.obj_id, bbox.dist2robot) for bbox in obj_bbox_list]
            distance_str = ""
            for obj_id, dist in bbox_ind2dist:
                distance_str += f"""
- Object id {obj_id} is {dist:.2f} metres from the robot."""
            ####

            subtask, skill_name, return_info = self.get_param_from_response(response, query=query, info=info)

            results_log.append({"subtask": subtask, "skill_name": skill_name})

            capture_history = {
                'image': prompt_rgb,
                'query': query,
                'model_response': [subtask, skill_name],
                'full_response': response,
                'subtask': subtask,
                'skill_name': skill_name,
                'skill_info': self.skill_descs,
                'distance_info': distance_str,
                'model_analysis': '', # this will be added by an external evaluator
            }
            self.save_model_output(
                rgb=prompt_rgb,
                response=response,
                subtitles=[f'Task Query: {query}', f'Subtask: {subtask}\nSkill: {skill_name}'],
                img_file=os.path.join(self.vis_dir, f'output_{info["save_key"]}.png'),
            )
            # if len(return_info['error_list']) == 0:
            #     break
        
        time_end_all = time.time()

        # print times with colors
        time_copy_log = time_copy - time_start
        time_gsam_log = time_gsam - time_copy
        time_prompts_log = time_prompts - time_gsam
        time_subtask_log = time_subtask - time_prompts
        time_vlm_log = time_end_vlm - time_subtask
        time_end_log = time_end_all - time_end_vlm
        time_overall_log = time_end_all - time_start
        
        # print time with colors
        print(f"\033[1;32;40mTime for copying: {time_copy_log:.2f} seconds\033[0m")
        print(f"\033[1;32;40mTime for GSAM: {time_gsam_log:.2f} seconds\033[0m")
        print(f"\033[1;32;40mTime for prompts: {time_prompts_log:.2f} seconds\033[0m")
        print(f"\033[1;32;40mTime for subtask: {time_subtask_log:.2f} seconds\033[0m")
        print(f"\033[1;32;40mTime for VLM: {time_vlm_log:.2f} seconds\033[0m")
        print(f"\033[1;32;40mTime for end: {time_end_log:.2f} seconds\033[0m")
        print(f"\033[1;32;40mTime for overall: {time_overall_log:.2f} seconds\033[0m")


        print("\n\nTime log: ", time_log, "\n\n")

        print("\n\nResults log: \n", results_log, "\n\n")
        # count different options times
        count_dict = {}
        for res in results_log:
            if res['subtask'] not in count_dict:
                count_dict[res['subtask']] = 0
            count_dict[res['subtask']] += 1
        print("\n\nCount dict: ", count_dict, "\n\n")
        
        return_info.update({ # this will be reused in the pickup skill to avoid gsam queries
            'bboxes': bboxes,
            'mask_image': mask_image,

            'time': [
                time_copy_log,
                time_gsam_log,
                time_prompts_log,
                time_subtask_log,
                time_vlm_log,
                time_end_log,
                time_overall_log,
            ]

        })

        error = None
        if len(return_info['error_list']) > 0:
            error = "Following errors have been produced: "
            for e in return_info['error_list']:
                error += f"{e}, "
            error = error[:-2]
            return self.on_failure(
                reason_for_failure=error,
                reset_required=False,
                capture_history=capture_history,
                return_info=return_info,
            )

        return self.on_success(
            capture_history=capture_history,
            return_info=return_info,
        )

    def obtain_task_choices(
        self,
        env,
        encoded_image_lst,
        rgb,
        depth,
        pcd,
        normals,
        query,
        run_vlm=True,
        info=None,
        history=None,
        n_retries=3, # we query the model multiple times to avoid errors
        **kwargs,
    ):
        info = copy.deepcopy(info)
        step_idx = info['step_idx']
        e_value = 'incorrect'
        # # im = rgb.copy()
        # # img_size = min(im.shape[0], im.shape[1])
        # img_size = 480

        # self.prompt_args.update({
        #     'radius': int(img_size * self.prompt_args['radius_per_pixel']),
        #     'fontsize': int(img_size * 30 * self.prompt_args['radius_per_pixel']),
        # })
        # info.update({'add_obj_ind': self.add_obj_ind})

        encoded_image = encoded_image_lst[-1] # TODO: need change

        response = self.vlm_runner(
            encoded_image=encoded_image,
            history_msgs=history,
            make_prompt_func=make_tasks_prompt,
            make_prompt_func_kwargs={
                'skill_descs': self.skill_descs,
                'info': info,
            }
        )

        return response
