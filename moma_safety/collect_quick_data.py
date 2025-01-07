import cv2
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import rospy
import pickle
import matplotlib.pyplot as plt

from tiago.tiago_gym import TiagoGym
from moma_safety.tiago import RESET_POSES as RP
import moma_safety.utils.transform_utils as T # transform_utils
from moma_safety.tiago.utils.ros_utils import TFTransformListener
from moma_safety.tiago.utils.transformations import quat_diff
from scipy.spatial.transform import Rotation as R


rospy.init_node('tiago_test')

env = TiagoGym(
    frequency=10,
    right_arm_enabled=True,
    left_arm_enabled=True,
    right_gripper_type='robotiq2F-140',
    left_gripper_type='robotiq2F-140',
    base_enabled=True,
    torso_enabled=False,
)

current_right_arm_joint_angles = env.tiago.arms["right"].joint_reader.get_most_recent_msg()
current_right_ee_pose = env.tiago.arms["right"].arm_pose
obs = env._observation()
print(obs.keys())
depth = obs['tiago_head_depth']
rgb = obs['tiago_head_image']
print("rgb: ", rgb.shape)
print("depth: ", depth.shape)

cam_intr = np.asarray(list(env.cameras['tiago_head'].camera_info.K)).reshape(3,3)
# extr_position, extr_quat = env.tiago.head.get_camera_extrinsic
# extr_rotation = R.from_quat(extr_quat).as_matrix()
# R_world_cam = extr_rotation
# T_world_cam = np.array([
#     [R_world_cam[0][0], R_world_cam[0][1], R_world_cam[0][2], extr_position[0]],
#     [R_world_cam[1][0], R_world_cam[1][1], R_world_cam[1][2], extr_position[1]],
#     [R_world_cam[2][0], R_world_cam[2][1], R_world_cam[2][2], extr_position[2]],
#     [0, 0, 0, 1]
# ])
T_world_cam = env.tiago.head.camera_extrinsic

save_dict = dict()
save_dict = {
    "current_right_arm_joint_angles": current_right_arm_joint_angles,
    "current_right_ee_pose": current_right_ee_pose,
    "cam_intr": cam_intr,
    "cam_extr": T_world_cam,
    "rgb": rgb,
    "depth": depth,
}

rgb = cv2.convertScaleAbs(rgb)
rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
plt.imsave('data/rgb.jpg', rgb)

save_path = "data"
counter = 1
with open(f'{save_path}/{counter:04d}.pickle', 'wb') as handle:
    pickle.dump(save_dict, handle)