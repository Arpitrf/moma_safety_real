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
from pointnet2.models import action_pointnet2_cls_ssg
from moma_safety.utils.env_variables import *
from moma_safety.utils.object_config import object_config as OC
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, load_image, predict
from moma_safety.safety_models.utils import *
from moma_safety.tiago.tiago_gym import TiagoGym
# from pointnet2.data_utils.utils import generate_point_cloud_from_depth

BOX_THRESHOLD = 0.25 # 0.35 originally
TEXT_THRESHOLD = 0.15 # 0.25 originally
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def mouseclick_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global clicked_points
        clicked_points.append([x, y])

class ObjDroppingDetector():
    def __init__(self, env):  
        '''MODEL LOADING'''
        num_class = 1
        # experiment_dir = "/home/pal/arpit/Pointnet_Pointnet2_pytorch/pointnet2/log/classification/run_place_in_shelf_obj_dropping"
        experiment_dir = "/home/pal/arpit/Pointnet_Pointnet2_pytorch/pointnet2/log/classification/run_obj_dropping"

        self.classifier = action_pointnet2_cls_ssg.get_model(num_class, normal_channel=False)
        self.classifier = self.classifier.cuda()

        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/model_epoch_180.pth')
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.eval()

        self.env = env

    def get_set_points(self, object_name):
        self.points = torch.from_numpy(self.get_pcd(object_name)).unsqueeze(0)
        print("points shape: ", self.points.shape)  
    
    def predict(self, threshold=0.5, object_name=None):
        points = self.points
        action = np.concatenate((np.zeros(9), np.array([1.0])))
        actions = action[None, ...]
        actions = torch.from_numpy(np.array(actions)).to(dtype=torch.float64)
            
        # # remove later
        # votes = 10
        # actions_tile = actions.repeat(votes, 1)
        # pos_noise = torch.empty(votes, 3).uniform_(-0.005, 0.005)
        # actions_tile[:, 3:6] = actions_tile[:, 3:6] + pos_noise

        points, actions = points.type(torch.FloatTensor).cuda(), actions.type(torch.FloatTensor).cuda()
        # actions_tile = actions_tile.type(torch.FloatTensor).cuda()
        points = points.transpose(2, 1)
        # print("points shape: ", points.shape)
        # print("actions shape: ", actions, actions.shape)
        
        # ---------------------------
        vote_num = 1    
        for _ in range(vote_num):
            pred, _ = self.classifier(points, actions)
            # vote_pool += pred
        # pred = vote_pool / vote_num
        # pred_choice = pred.data.max(1)[1]

        probabilities = torch.sigmoid(pred)
        # Convert probabilities to binary predictions (0 or 1)
        pred_choice = (probabilities >= threshold).float()

        print(f"pred_choice, pred_prob: ", pred_choice.item(), probabilities.item(), action[3:6])
        # -----------------------------

        return pred_choice.item(), probabilities.item()

    def generate_point_cloud_from_depth(self, depth_image, intrinsic_matrix, mask, extrinsic_matrix):
        """
        Generate a point cloud from a depth image and intrinsic matrix.
        
        Parameters:
        - depth_image: np.array, HxW depth image (in meters).
        - intrinsic_matrix: np.array, 3x3 intrinsic matrix of the camera.
        
        Returns:
        - point_cloud: Open3D point cloud.
        """
        
        # Get image dimensions
        height, width = depth_image.shape

        # Create a meshgrid of pixel coordinates
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # Flatten the pixel coordinates and depth values
        u_flat = u.flatten()
        v_flat = v.flatten()
        depth_flat = depth_image.flatten()
        mask_flat = mask.flatten()

        # # Filter points where the mask is 1
        # valid_indices = np.where(mask_flat == 1)
        
        # Filter points where the mask is 1 AND depth is valid (not inf and not 0)
        valid_indices = np.where(
            (mask_flat == 1) & 
            (np.isfinite(depth_flat)) &  # Remove inf values
            (depth_flat > 0)             # Remove 0 or negative values
        )[0]

        # Apply the mask to the pixel coordinates and depth
        u_valid = u_flat[valid_indices]
        v_valid = v_flat[valid_indices]
        depth_valid = depth_flat[valid_indices]

        # Generate normalized pixel coordinates in homogeneous form
        pixel_coords = np.vstack((u_valid, v_valid, np.ones_like(u_valid)))

        # Compute inverse intrinsic matrix
        intrinsic_inv = np.linalg.inv(intrinsic_matrix)

        # Apply the inverse intrinsic matrix to get normalized camera coordinates
        cam_coords = intrinsic_inv @ pixel_coords

        # Multiply by depth to get 3D points in camera space
        cam_coords *= depth_valid
        # breakpoint()

        # # Reshape the 3D coordinates
        # x = cam_coords[0].reshape(height, width)
        # y = cam_coords[1].reshape(height, width)
        # z = depth_image

        # # Stack the coordinates into a single 3D point array
        # points = np.dstack((x, y, z)).reshape(-1, 3)

        # breakpoint()
        points = np.vstack((cam_coords[0], cam_coords[1], depth_valid)).T
        # points = np.vstack((cam_coords[0], -depth_valid, -cam_coords[1])).T
        # points = np.vstack((-cam_coords[1], -depth_valid, cam_coords[0])).T

        # pad points so that the total number of points are 128*128
        target_size=(height*width, 3)
        N_i = points.shape[0]
        pad_rows = target_size[0] - N_i
        padding = ((0, pad_rows), (0, 0))
        points = np.pad(points, padding, mode='edge')

        # print("points shape: ", points.shape)

        # remove later
        # points = points[points[:, 2] > 0.5]
        # print("points: ", points[:, 2])


        # transform points to world frame
        # make points homogeneous
        points = points / 1000.0    
        points = np.hstack((points, np.ones((points.shape[0], 1))))
        points = extrinsic_matrix @ points.T
        points = points.T
        # remove homogeneous coordinate
        points = points[:, :3]

        keep_ratio = 0.7  # Keep % of points randomly
        num_points = len(points)
        mask = np.random.choice([True, False], size=num_points, p=[keep_ratio, 1-keep_ratio])
        points = points[mask]

        # # Create an Open3D point cloud object
        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(points)

        return points

            
    def get_pcd(self, object_name):
        obs = self.env._observation()
        depth = obs['tiago_head_depth'].squeeze()
        rgb = obs['tiago_head_image']
        intr = np.asarray(list(self.env.cameras['tiago_head'].camera_info.K)).reshape(3,3)
        extrinsic_matrix = self.env.tiago.head.camera_extrinsic
        # extrinsic_matrix = np.eye(4)
        
        # mask not working that well as of now.
        # mask = obtain_mask(self.env, object_name)
        mask = np.ones_like(depth)

        points = self.generate_point_cloud_from_depth(depth, intr, mask, extrinsic_matrix)        
        
        # # remove points that are too close to the floor
        # points = points[points[:, 2] > 0.15]
        # points = points[points[:, 1] > -0.6]

        # remove later. for drawer task
        points = points[points[:, 2] > 0.15]
        if object_name in ["drawer"]:
            points = points[points[:, 1] > -0.7]
            points = points[points[:, 1] < -0.2]
        if object_name not in ["ledge"]:
            points = points[points[:, 0] < 1.2]

        # # show pcd in open3d
        # o3d_pcd = o3d.geometry.PointCloud()
        # o3d_pcd.points = o3d.utility.Vector3dVector(points)
        # o3d.visualization.draw_geometries([o3d_pcd])

        return points
    

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