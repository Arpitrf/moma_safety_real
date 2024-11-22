import os
import cv2

from moma_safety.tiago.prompters.vip_utils import extract_json

# this function is used only to generate reasoning prompt for the sub-task completion.
def make_prompt_for_reasoning(query, info):
    instructions = f"""
INSTRUCTIONS:
You are given a description of the task that the robot must perform. The robot has executed an action in the environment to achieve the task, the action may or may not have been executed successfully. Along with the task description and robot action, you are provided with an image of the scene before and after the intended action has been taken. If the action taken by robot failed to execute, you are provided with the feedback received from the environment. It is possible that even though the action is executed successfully, the task is not finished yet. Your goal is to evaluate whether the task has been completed successfully using this information. First, describe the elements in the image that are relevant to the task and must be considered to evaluate the action. Then, describe the difference between the images before and after the action. Finally, provide your evaluation of the task completion. This evaluation will be used to make better action prediction in the future.

You are a five-times world champion in this game. You have a keen eye for detail and can spot the smallest of differences in the images.
Provide your answer at the end in a valid JSON of this format: {{"evaluation": ""}}.
""".strip()

    prompt=f"""
TASK DESCRIPTION: {query}
ROBOT ACTION: {info['robot_action']}
ACTION SUCCESS: {info['is_success']}"""
    if info['is_success'] == False:
        prompt += f"""
FEEDBACK: {info['env_reasoning']}"""
    prompt += f"""
ANSWER: Let's think step by step."""
    return instructions, prompt

def run_reasoner(
        vlm,
        history_i,
    ):
    # history_i must have before and after images, query,
    """Perform one reasoning pass given samples."""
    img = history_i['b_rgb']
    img = history_i['a_rgb']
    query = history_i['query']
    # feedback = history_i['env_reasoning']
    # is_success = history_i['is_success']

    instructions, prompt = make_prompt_for_reasoning(query, feedback)

    # convert to bgr and encode image
    encoded_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, encoded_image = cv2.imencode('.png', encoded_image)

    prompt_seq = [prompt, encoded_image]
    response = vlm.query(instructions, prompt_seq)

    try:
        short_analysis = extract_json(response, 'evaluation')
    except Exception as e:
        short_analysis = ""

    return short_analysis
