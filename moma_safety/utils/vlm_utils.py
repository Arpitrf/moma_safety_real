import io
import os
import cv2
import rospy
import base64
import numpy as np

from PIL import Image
from openai import OpenAI
from moma_safety.tiago.tiago_gym import TiagoGym

client = OpenAI(api_key=os.environ['ARPIT_OPENAI_KEY'])

def get_prompt_nav_success(object_name):
    prompt = \
    f"""
    A robot has been asked to navigate to [object]. Here the [object] can be a complete object (like cup or apple) or it can be a specific part of an object (like white handle). We provide an image of robot's egocentric view (taken from robot's head camera) after the navigation. 
    Your job is to answer whether the robot has successfully navigated to the [object]. For a successful navigation to [object] the [object] should be clearly visible in the image.

    Please answer in the following format only:

    [Yes/No].[Reason in one sentence].

    Now, given that the robot is asked to navigate to {object_name}, please answer whether or not the robot successfully navigated to the object in the aforementioned format.
    """
    return prompt

def get_content_for_example(task_description):
    content = []
    prompt = \
    f"""
    Here is an example:
    """
    content.append(prompt)
    
    img = encode_image("resources/manip_segments_example_success/open fridge.png")
    content.append({"image": img, "resize": 512})
    
    prompt = \
    f"""
    Yes. The fridge door appears open.
    """
    content.append(prompt)

    return content

def get_prompt_manip_success(object_name, task_description):
    prompt = \
    f"""
    A robot has been asked perform the following task: {task_description}. We provide an image of robot's egocentric view (taken from robot's head camera) after the robot completes the manipulation. 
    Your job is to answer whether the robot has successfully completed the task.

    Please answer in the following format only:

    [Yes/No].[Reason in one sentence].
    """
    return prompt

def get_content(current_img, object_name, task_description):
    content = []
    
    prompt = get_prompt_manip_success(object_name, task_description)
    content.append(prompt)
    
    # if object_name == "fridge handle":
    #     content_example = get_content_for_example(task_description)
    #     content += content_example
    
    prompt = \
    f"""
    Now, given that the robot is asked to {task_description}, please answer whether or not the robot successfully completed the task in the aforementioned format.
    """
    content.append(prompt)
    
    content.append({"image": current_img, "resize": 512})
    
    return content

def request_api(content):
    # prompt_file = f"prompts/nav_success_detection.txt"
    # with open(prompt_file, 'r') as file:
    #     file_content = file.read()

    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": content,
            # "content": [
            #     prompt,
            #     {"image": img, "resize": 512}, #768
            # ],
        },
    ]
    params = {
        "model": "gpt-4o",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 200,
    }
    result = client.chat.completions.create(**params)
    return result.choices[0].message.content

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    

def check_manip_success_using_vlm(rgb, object_name, task_description):
    # process the rgb image
    if rgb.dtype != np.uint8:
        rgb = cv2.convertScaleAbs(rgb)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    img_PIL = Image.fromarray(rgb)
    # convert the image to bytes and then to base64
    buffer = io.BytesIO()
    img_PIL.save(buffer, format="PNG")  # You can specify the format: PNG, JPEG, etc.
    img_bytes = buffer.getvalue()
    base64_image = base64.b64encode(img_bytes).decode("utf-8")

    # prompt = get_prompt_manip_success(object_name, task_description)
    content = get_content(base64_image, object_name, task_description)
    breakpoint()
    retval = request_api(content)
    print("retval: ", retval)
    breakpoint()
    return retval


if __name__ == "__main__":
    rospy.init_node('tiago_test')

    env = TiagoGym(
        frequency=10,
        right_arm_enabled=True,
        left_arm_enabled=False,
        right_gripper_type='robotiq2F-140',
        left_gripper_type=None,
        base_enabled=True,
        torso_enabled=False,
    )
    
    obs = env._observation()
    img = obs['tiago_head_image']
    object_name = "fridge handle"
    task_description = "open fridge"
    check_manip_success_using_vlm(img, object_name=object_name, task_description=task_description)
