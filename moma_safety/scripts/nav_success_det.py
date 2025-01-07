import io
import os
import cv2
import time
import rospy
import torch
import base64
import datetime

import open3d as o3d
import numpy as np
import supervision as sv
import groundingdino.datasets.transforms as T

from PIL import Image
from pathlib import Path
import matplotlib
matplotlib.use("agg", force=True)
from matplotlib import pyplot as plt

from torchvision.ops import box_convert
from moma_safety.utils.env_variables import *
from moma_safety.tiago import RESET_POSES as RP
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, load_image, predict
from moma_safety.tiago.tiago_gym import TiagoGym
from moma_safety.grasp import obtain_mask, get_pcd, grasp, check_grasp_reachability, obtain_grasp_modes
from moma_safety.utils.object_config import object_config as OC 

"""
Hyper parameters
"""
# change later
# BOX_THRESHOLD = 0.75 # 0.35 originally
# TEXT_THRESHOLD = 0.75 # 0.25 originally
BOX_THRESHOLD = 0.45 # 0.35 originally
TEXT_THRESHOLD = 0.35 # 0.25 originally
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("grounded_sam_outputs/grounded_sam2_local_demo")
DUMP_JSON_RESULTS = True


class NavSuccessDetection:
    def __init__(self, env, obj_name=None, exec_on_robot=False):
        self.env = env
        self.obj_name = obj_name
        self.exec_on_robot = exec_on_robot

        # build grounding dino model
        self.grounding_model = load_model(
            model_config_path=GROUNDING_DINO_CONFIG, 
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
            device=DEVICE
        )

    def set_object_name(self, obj_name):
        self.obj_name = obj_name

    def check_grasp_modes_reachability(self, mask, pcd):
        grasp_modes = obtain_grasp_modes(pcd, self.env, obj_name=self.obj_name, select_mode=False)
        # grasp_modes = grasp(self.env, obj_name=self.obj_name, exec_on_robot=self.exec_on_robot, mask=mask, pcd=pcd, select_mode=False)
        grasp_reachable = []
        obj_graspable = False
        for grasp_mode in grasp_modes:
            retval = check_grasp_reachability(self.env, grasp_mode, self.obj_name)
            grasp_reachable.append(retval)
        print("sum(grasp_reachable): ", sum(grasp_reachable))
        if sum(grasp_reachable) > 1:
            obj_graspable = True
        
        return obj_graspable

    def check_object_visibility_dino2(self):
        obj_visible_gdino = False
        text = OC[self.obj_name]["text_description"] # VERY important: text queries need to be lowercased + end with a dot
        print("text: ", text)

        obj_visible_gdino_list = []
        for i in range(5):
            obs = self.env._observation()
            depth = obs['tiago_head_depth']
            rgb = obs['tiago_head_image']
            if rgb.dtype != np.uint8:
                rgb = cv2.convertScaleAbs(rgb)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            img_source = rgb.copy()
            img_PIL = Image.fromarray(img_source)

            transform = T.Compose(
                    [
                        T.RandomResize([800], max_size=1333),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ]
                )
            img_transformed, _ = transform(img_PIL, None)

            torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()
            boxes, confidences, labels = predict(
                model=self.grounding_model,
                image=img_transformed,
                caption=text,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
            )
            # process the box prompt for SAM 2
            h, w, _ = img_source.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            print("boxes: ", input_boxes.shape)
            # to detect if object is found or not
            if input_boxes.shape[0] > 0:
                obj_visible_gdino_list.append(True)

            # Visualize
            class_names = labels
            class_ids = np.array(list(range(len(class_names))))
            detections = sv.Detections(
                xyxy=input_boxes,  # (n, 4)
                mask=None,  # (n, h, w)
                class_id=class_ids
            )
            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(scene=img_source.copy(), detections=detections)
            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            plt.imsave(f"resources/tmp_outputs/dino_output_{i}.jpg", annotated_frame)


        if sum(obj_visible_gdino_list) > 2:
            obj_visible_gdino = True
            print("object found")

        return obj_visible_gdino

    def check_object_reachability(self, select_object=False):
        obj_reachable = True
        mask = obtain_mask(self.env, select_object=select_object)
        try:
            pcd = get_pcd(self.env, mask)
        except:
            return False, mask, None 
        # 1. Test distances first
        points = np.array(pcd.points)
        distances = np.linalg.norm(points[:, :2], axis=1)
        print("min, max: ", min(distances), max(distances))

        # Sort points by distance
        distances_sorted = np.sort(distances)[:20] # 20 is a hyperparameter
        near_point_distance = np.median(distances_sorted, axis=0)
        min_th = OC[self.obj_name]["min_th"]
        max_th = OC[self.obj_name]["max_th"]
        if near_point_distance < min_th:
            print("Base pose is too close to object: ", near_point_distance)
            obj_reachable = False
        elif near_point_distance > max_th:
            print("Base pose is too far from object: ", near_point_distance)  
            obj_reachable = False

        return obj_reachable, mask, pcd

    def check_nav_success(self, check_grasp_reachability_flag=False, select_object_for_mask=False):
        # 1. Check Dino2 can detect the object or not
        visibility_dino_test = self.check_object_visibility_dino2()
        # change later
        inp = input("press Y if visibile else N")
        if inp == "Y":
            visibility_dino_test = True
        else:
            visibility_dino_test = False
        if not visibility_dino_test: return False

        # # 2. GPT4o
        # # image_path = "/home/pal/arpit/Grounded-SAM-2/notebooks/images/IMG_3144.jpg"
        # # object_name = "white handle"
        # # base64_image = encode_image(image_path)
        # # convert the image to bytes and then to base64
        # buffer = io.BytesIO()
        # img_PIL.save(buffer, format="PNG")  # You can specify the format: PNG, JPEG, etc.
        # img_bytes = buffer.getvalue()
        # base64_image = base64.b64encode(img_bytes).decode("utf-8")
        # response = request_api(base64_image, object_name=text)
        # print("response: ", response)
        # TODO: process the response to get the answer and the reason

        # 3. Distances of the object
        object_reachability_test, mask, pcd = self.check_object_reachability(select_object=select_object_for_mask)
        if not object_reachability_test: return False

        # 4. Check grasp reachability
        grasp_modes_reachable_test = True
        if check_grasp_reachability_flag:
            grasp_modes_reachable_test = self.check_grasp_modes_reachability(mask, pcd)
            if not grasp_modes_reachable_test: return False

        print("visibility_dino_test, object_reachability_test, grasp_modes_reachable_test: ", visibility_dino_test, object_reachability_test, grasp_modes_reachable_test)
        breakpoint()

        if visibility_dino_test and object_reachability_test and grasp_modes_reachable_test:
            return True
        return False
            
if __name__ == "__main__":
    rospy.init_node('tiago_test')

    obj_name = "fridge"
    env = TiagoGym(
        frequency=10,
        right_arm_enabled=True,
        left_arm_enabled=False,
        right_gripper_type='robotiq2F-140',
        left_gripper_type='robotiq2F-85',
        base_enabled=True,
        torso_enabled=False,
    )
    exec_on_robot = False
    nav_success_det = NavSuccessDetection(env, obj_name, exec_on_robot)
    env.tiago.head.write_head_command(OC[obj_name]["head_joint_pos"])

    # Move Tiago to reset pose
    if exec_on_robot:
        # open gripper. 1 is open and 0 is close
        env.tiago.gripper['right'].step(OC[obj_name]["gripper_open_pos"])
        time.sleep(2)
        reset_joint_pos = RP.PREGRASP_R_H
        reset_joint_pos["right"][-1] = OC[obj_name]["gripper_open_pos"]
        env.reset(reset_arms=True, reset_pose=reset_joint_pos, allowed_delay_scale=6.0)

    # In a while loop so that I can change base poses and check the results
    while True:
        nav_success = nav_success_det.check_nav_success(select_object_for_mask=True, 
                                                        check_grasp_reachability_flag=OC[obj_name]["check_grasp_reachability_flag"])
        breakpoint()
