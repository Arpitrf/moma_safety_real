import cv2
import rospy
import torch
import open3d as o3d
import numpy as np
import groundingdino.datasets.transforms as T
import supervision as sv
from PIL import Image
from torchvision.ops import box_convert
import matplotlib
matplotlib.use("agg", force=True)
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from pointnet2.models import action_lidar
from moma_safety.utils.env_variables import *
from moma_safety.utils.object_config import object_config as OC
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, load_image, predict
from moma_safety.safety_models.utils import *
from moma_safety.tiago.tiago_gym import TiagoGym
# from pointnet2.data_utils.utils import generate_point_cloud_from_depth


class BaseCollisionModel():
    def __init__(self, env):  
        '''MODEL LOADING'''
        num_class = 1
        experiment_dir = "/home/pal/arpit/Pointnet_Pointnet2_pytorch/pointnet2/log/classification/run_base_collision"

        self.classifier = action_lidar.get_model()
        self.classifier = self.classifier.cuda()

        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/model_epoch_240.pth')
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.eval()

        self.env = env

    def get_scan(self):
        obs = self.env._observation()
        scan = obs["scan"]
        print("obs_scan: ", scan.shape)
        scan[scan > 1.5] = 10.0
        scan[scan != 10.0] /= 2.0
        # breakpoint()
        return scan
    
    def get_set_scan(self):
        self.scan = torch.from_numpy(self.get_scan()).unsqueeze(0)
    
    def predict(self, action, threshold=0.2):
        scans = self.scan
        actions = action[None, ...]
        actions = torch.from_numpy(np.array(actions)).to(dtype=torch.float64)

        scans, actions = scans.type(torch.FloatTensor).cuda(), actions.type(torch.FloatTensor).cuda()

        if torch.isnan(scans).any():
            print("NaN values detected in points")
            breakpoint()
        if torch.isnan(actions).any():
            print("NaN values detected in actions")
            breakpoint()
        
        # ---------------------------
        vote_num = 1    
        for _ in range(vote_num):
            pred = self.classifier(scans, actions)
            # vote_pool += pred
        # pred = vote_pool / vote_num
        # pred_choice = pred.data.max(1)[1]

        probabilities = torch.sigmoid(pred)
        # Convert probabilities to binary predictions (0 or 1)
        pred_choice = (probabilities >= threshold).float()

        print(f"pred_choice, pred_prob: ", pred_choice.item(), probabilities.item(), action)
        # -----------------------------

        return pred_choice.item(), probabilities.item()


if __name__ == "__main__":
    rospy.init_node('tiago_test')
    
    env = TiagoGym(
        frequency=10,
        right_arm_enabled=True,
        left_arm_enabled=False,
        right_gripper_type='robotiq2F-140',
        left_gripper_type='robotiq2F-85',
        base_enabled=True,
        torso_enabled=False,
    )

    obj_name = "shelf" 
    collision_model = ArmCollisionModel(env)
    collision_model.get_set_points(obj_name)
    # env =