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

def make_prompt(skill_descs, task_desc, info=None, llm_baseline_info=None, method="ours"):
    """
    method arg is not being used for ablation results, just here for consistency
    """
    floor_num = info['floor_num']
    add_obj_ind = False
    bbox_ind2dist = None
    if info['add_obj_ind'] == True:
        add_obj_ind = True
        obj_bbox_list = info['obj_bbox_list']
        bbox_id2dist = info['bbox_id2dist']
        bbox_ind2dist = [(bbox.obj_id, bbox.dist2robot) for bbox in obj_bbox_list]
    skill_desc_str = ""
    for ind, skill_desc in enumerate(skill_descs):
        skill_desc_str += skill_desc + "\n\n"

    if llm_baseline_info:
        visual_instructions = [
            "Describe the distance to each of the objects.",
            "We describe the objects on the scene by their object id.",
            "Scene",
            "along with a description of the scene and visible objects",
            "analyzing the object descriptions",
        ]
    else:
        visual_instructions = [
            "First, describe the scene in the image. Describe each marked object briefly along with the distance to each of the object.",
            "The images are marked with object id.",
            "Scene in the image",
            "along with image of the scene",
            "looking at the objects in the image",
        ]

    task_goal_instruction = [

    """First, given the general instruction, infer the two things:
    1. What are the possible overall goal that the human want to do? For example, given the general instruction is to eat, and the scene has different food items, the human is likely to want to eat one of them.
    2. What ares the possible next subtasks towards this overall goal that the human wants to achieve, given the task progress that you already achieved? For example, the human might want to pick up food item 1, or pick up food item 2. Use your commonsense of everyday tasks to determine the task sequence. For example, you usually first pick up a food item, then place it onto some container to give to the human; You should not give the human an empty plate, etc.
    """,

    """
    First, given the overall goal and the list of subtasks you have completed, infer what are the possible next subgoals that the human is trying to complete.
    Some examples:
    - Overall goal: Help human to be less thristy. Subgoals completed: Pick up something to drink. Next possible subgoals: Place the leverage on a plate.
    """

    ]

    instructions = f"""
INSTRUCTIONS:
You are given a description of the task that the robot must execute along with a description of the scene and visible objects. Given the current scene, break down the task into multiple sub-tasks to successfully complete the main task. Describe the distance to each of the objects. Describe where the robot is present in the building. 

{task_goal_instruction[1]}

Next, look at the trajectory of human joystick input that you are also given. The joystick input is a 2D arrow pointing to a certain direction. You should infer what the human wants you to do, given this joystick input. You should use the joystick information, as well as the image of the scene to understand the direction of the robot in relationship to the objects in the scene. You should answer the following questions one by one: 
1. Describe where the joystick is pointing to on the 2D space. It can be a composition of left, right, up and down. It can also be a curvature, indicating a certain motion.
2. Given this direction, infer what is the task, among the previous subgoals, is more likely to be the one human want to complete?

Some Examples:
- If the human is moving joystick forward, and there is an object in the forward view: Most likely to grasp the object
- If the robot is already holding some object, and the direction where human joystick moves to has some container: Most likely to put objects in container

3. What should be the skill the robot should perform to complete the task?
4. Finally, output one subtask that is most likely that the human wants to perform.

IMPORTANT GUIDELINES TO NOTE:
    - You should especially pay attention to where the arrow of the joystick is pointing to. Make sure the direction is correct!
    - Pay attention to the progress you already made, and given the overall goal and the subtask you have achieved, infer the next action based on human joystick action. For example, if you are already successfully holding a food item, you should go on to find a container, rather than picking up another food. 
    - Whenever you are able to perform the pick_up_object, or place_object action, you should always do so. Do not move, unless it is necessary!
    - You should see the distance of a certain item is to you. If you are close enough to an object, do not move closer to the object. Just perform the pick up action. You can pick up objects that are approximately 0.7 meters. If an object is closer to this distance, just pick it up, do not move!

Provide your answer at the end in a valid JSON of this format: {{"subtask": "", "skill_name": ""}}

The list of skills that the robot has are:
{skill_desc_str}
""".strip()
    instructions += f"""\n
OTHER GENERAL GUIDELINES:
    - If the task involves in a particular area, always check if the robot is in the correct area. If not, navigate to the correct area.
    - If you do NOT see objects relevant to the task, describe potential locations where it can be found. Go to a location where the object can be found.
    - Avoid assuming objects if not present in the scene.
    - Avoid picking up objects that are too far away from the robot. Robot can only pick up objects that are approximately 0.7 meters. Always specify the distance of the object if selecting pick_up_object skill in the summary.
    - Avoid pushing objects that are too far away from the robot. Robot can only push objects that are withing 3.0 meters. Always specify the distance of the object if selecting push_object_on_ground skill in the summary. Note that the approximate distance to  push an object is within 3.0 meters but the distance to pick up an object is 0.7 meters.
    - Use navigate_to_point_on_ground skill to move to a point located on the ground near the object of interest to move inside the room, especially when the object is far away from the robot and ground is visible.
    - If it is already close to the object, use move skill instead of navigate_to_point_on_ground skill. Use move skill to adjust the robot's base position by few centimeters to reach the object.
    - Always pay attention to the information provided due to failure of skill execution.
    - If there is a failure due to collision, there may be an obstacle in front of the robot that may or may not be visible. Use push_object_on_ground skill to clear the path.
    - The push_object_on_ground will select the object and direction to push. All the objects may not be visible to you. Hence, providing names of the objects in the subtask can lead to selecting incorrect object by the push_object_on_ground skill.
    - If the collision is due to a door that you see, open the door using open_door skill before navigating further through the door.
    - Make sure to specify the destination floor in the subtask for call_elevator and use_elevator skills."""
    if add_obj_ind:
        instructions += f"""
    - {visual_instructions[1]} Below is the distance of each object with the robot. Use this information to decide the feasibility of manipulating the object.
    - Avoid using the object id in the final JSON response. Describe the object(s) involved in the sub-task instead of using the object id in the JSON response. This is very important.
OBSERVATIONS:"""
        for obj_id, dist in bbox_ind2dist:
            instructions += f"""
- Object id {obj_id} is {dist:.2f} meters from the robot."""
    instructions += f"""\n"""

    # Add scene description and obj descriptions if language only.
    if llm_baseline_info:
        instructions += f"""
- {visual_instructions[2]}: {llm_baseline_info['im_scene_desc']}
        """
        instructions += f"""
- Object ID descriptions: {llm_baseline_info['obj_descs']}"""

    task_prompt = f"""\nTASK DESCRIPTION: {task_desc}"""
    task_prompt += f"""\n

TIME-STEP: {info['step_idx']+1}
ANSWER: Let's think step by step.""".strip()
    return instructions, task_prompt

def make_history_prompt(history):
    instructions = f"""
Below is the execution history from previous time-steps of the same episode. Pay close attention to your previous predictions, success/failure feedback from the environment. Provide a summary of each of the errors in previous time-steps in your response. Avoid repeating the same errors. Based on the history, you can improve your predictions.
PREVIOUS TIME-STEP HISTORY:
""".strip()
    history_desc = []
    history_model_analysis = []
    for ind, msg in enumerate(history):
        example_desc = f"""\n
    TIME-STEP: {ind+1}
    DESCRIPTION: {msg['query']}
    ANSWER: {{"subtask": "{msg['model_response'][0]}", "skill_name": "{msg['model_response'][1]}"}}
    SKILL SUCCESS: {msg['is_success']}
    """.strip()
        if not msg['is_success']:
            example_desc += f"""\n
    FEEDBACK: {msg['env_reasoning']}
    """.strip()
        # if msg['skill_name'] == "goto_landmark":
        #     example_desc += f"""\n
    # SKILL RESPONSE: We have naviageted to a place that looks like {msg['skill_response'][-1]}.
    # """.strip()
        if ('model_analysis' not in msg) or (msg['model_analysis'] == ""):
            if not msg['is_success']:
                msg_to_add = "I made an error in my reasoning."
            else:
                msg_to_add = "The prediction is appropriate to complete the skill."
            msg['model_analysis'] = msg_to_add
        history_model_analysis.append(msg['model_analysis'])
        history_desc.append(example_desc)

    return instructions, history_desc, history_model_analysis

def make_cross_history_prompt(history):
    instructions = f"""
Below is the execution history from previous trials and not the current trial. The task may or may not be different from the current task. Pay close attention to the summary of the feedback. Avoid repeating the same errors. Based on the history, you can improve your predictions.
SUMMARY OF PREVIOUS TRIALS:
""".strip()
    history_desc = []
    history_model_analysis = []
    for ind, msg in enumerate(history):
        example_desc = f""
        # example_desc = f"""\n
    # DESCRIPTION: {msg['query']}
    # ANSWER: {{"subtask": "{msg['model_response'][0]}", "skill_name": "{msg['model_response'][1]}"}}
    # SKILL SUCCESS: {msg['is_success']}
    # """.strip()
        if not msg['is_success']:
            example_desc += f"""\n
    SUMMARY {ind+1}: {msg['env_reasoning']}
    """.strip()

        if ('model_analysis' not in msg) or (msg['model_analysis'] == ""):
            if not msg['is_success']:
                msg_to_add = "I made an error in my reasoning."
            else:
                msg_to_add = "The prediction is appropriate to complete the skill."
            msg['model_analysis'] = msg_to_add
        history_model_analysis.append(msg['model_analysis'])
        history_desc.append(example_desc)

    return instructions, history_desc, history_model_analysis

class SkillSelector(SkillBase):
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
        self.skill_name = 'selector'
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

    def get_history_dirs(self):
        history_eval_dirs = None
        if self.reasoner_type == 'oracle':
            # ##  For the oracle evaluation
            # The robot must the chair outside the desk inside it. The subtask should not include object id like Chair B, instead, it should simply include the chair in front of the desk. Finally, the chair in front of the desk is approximate 1 meter away from the robot. This implies that the robot can directly push the chair without further adjusting the base. The robot should have pushed the chair.
            # base_dir = "/home/pal/Desktop/rutav/datasets/push_chair_inside/selector_eval_oracle/"
            # history_eval_dirs = [os.path.join(base_dir, 'eval_id000.pkl')]

            # base_dir = "/home/pal/Desktop/rutav/datasets/push_chair_inside2/selector_eval_oraclev2/"
            # history_eval_dirs = [os.path.join(base_dir, 'eval_id001.pkl')]

            base_dir = "/home/pal/Desktop/rutav/datasets/push_chair_inside2/selector_eval_oraclev3/"
            history_eval_dirs = [os.path.join(base_dir, 'eval_id000.pkl')]
        elif self.reasoner_type == 'model':
            ##  For the model evaluation
            # base_dir = "/home/pal/Desktop/rutav/datasets/push_chair_inside2/selector_eval_model/"
            # base_dir = "/home/pal/Desktop/rutav/datasets/push_chair_inside2/selector_eval_modelv2/"
            # history_eval_dirs = [os.path.join(base_dir, 'eval_id000.pkl')]

            # base_dir = "/home/pal/Desktop/rutav/datasets/hallucination/eval_ahg_floor1_w_hist_model/"
            # history_eval_dirs.extend([os.path.join(base_dir, 'eval_id000.pkl')])

            base_dir = "/home/pal/Desktop/rutav/datasets/push_chair_inside2/selector_eval_modelv3/"
            history_eval_dirs = [os.path.join(base_dir, 'eval_id000.pkl')]
        else:
            raise NotImplementedError(f"Reasoner type {reasoner_type} not implemented for skill {self.skill_name}.")

        return history_eval_dirs

    def create_language_history_msgs(
            self,
            history,
            func, # function to create the prompt
            func_kwargs, # kwargs for the function
            image_key=None,
        ):
        history_msgs = []
        history_inst, history_desc, history_model_analysis = func(history, **func_kwargs)
        history_imgs = [None] * len(history_desc)

        history_msgs = self.vlm.create_msg_history(
            history_instruction=history_inst,
            history_desc=history_desc,
            history_model_analysis=history_model_analysis,
            history_imgs=history_imgs,
        )
        return history_msgs

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

    def step(
        self,
        env,
        rgb,
        image_joystick,
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
        im = rgb.copy()
        img_size = min(im.shape[0], im.shape[1])

        self.prompt_args.update({
            'radius': int(img_size * self.prompt_args['radius_per_pixel']),
            'fontsize': int(img_size * 30 * self.prompt_args['radius_per_pixel']),
        })
        info.update({'add_obj_ind': self.add_obj_ind})

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
            for bbox in bboxes:
                center = (bbox[1] + bbox[3]) // 2, (bbox[2] + bbox[4]) // 2
                pos_wrt_base = pcd[center[1], center[0]]
                dist = np.linalg.norm(pos_wrt_base[:2])
                bbox_id2dist[bbox[0]] = dist

            info.update({
                'bbox_ignore_ids': [0],
                'bbox_id2dist': bbox_id2dist,
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

        encoded_image_joystick = U.encode_image(image_joystick.copy())

        for _ in range(n_retries):
            response = self.vlm_runner(
                encoded_image=encoded_image,
                encoded_image_joystick=encoded_image_joystick,
                history_msgs=history_msgs,
                make_prompt_func=make_prompt,
                make_prompt_func_kwargs={
                    'task_desc': query,
                    'skill_descs': self.skill_descs,
                    'info': info,
                }
            )
            #### creating the distance information string for capturing history
            bbox_ind2dist = [(bbox.obj_id, bbox.dist2robot) for bbox in obj_bbox_list]
            distance_str = ""
            for obj_id, dist in bbox_ind2dist:
                distance_str += f"""
- Object id {obj_id} is {dist:.2f} metres from the robot."""
            ####

            subtask, skill_name, return_info = self.get_param_from_response(response, query=query, info=info)
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
            if len(return_info['error_list']) == 0:
                break
        return_info.update({ # this will be reused in the pickup skill to avoid gsam queries
            'bboxes': bboxes,
            'mask_image': mask_image,
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