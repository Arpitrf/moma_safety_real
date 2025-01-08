import numpy as np
np.set_printoptions(precision=3, suppress=True)
import rospy
import pickle
import os
import cv2
import imageio

import matplotlib.pyplot as plt
from datetime import datetime
from scipy.spatial.transform import Rotation as R

from tiago.tiago_gym import TiagoGym
from moma_safety.tiago import RESET_POSES as RP
import moma_safety.utils.transform_utils as T # transform_utils
from moma_safety.tiago.utils.ros_utils import TFTransformListener
from moma_safety.tiago.utils.transformations import quat_diff


rospy.init_node('tiago_test')

env = TiagoGym(
    frequency=10,
    right_arm_enabled=True,
    left_arm_enabled=True,
    right_gripper_type='robotiq2F-140',
    left_gripper_type=None,
    base_enabled=True,
    torso_enabled=False,
)



current_time = datetime.now()
date_folder = current_time.strftime("%Y-%m-%d")
time_folder = current_time.strftime("%H-%M-%S")
save_path = os.path.join("data/test_data", date_folder, time_folder)
# save_path = os.path.join("data/test_data", current_time.strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(save_path, exist_ok=True)

rospy.sleep(2)

cam_intr = np.asarray(list(env.cameras['tiago_head'].camera_info.K)).reshape(3,3)
T_world_cam = env.tiago.head.camera_extrinsic

# fps=360
imgio_kargs = {'fps': 30, 'quality': 10, 'macro_block_size': None,  'codec': 'h264',  'ffmpeg_params': ['-vf', 'crop=trunc(iw/2)*2:trunc(ih/2)*2']}
output_path = f'{save_path}/video.mp4'
writer = imageio.get_writer(output_path, **imgio_kargs)


# time.sleep(10)
rgb_list = []
depth_list = []
current_right_arm_joint_angles_list = []
current_right_ee_pose_list = []
counter = 0
print("================STARTING CAPTURE===================")
# rum this loop for 10 seconds
start_time = rospy.get_time()
counter = 0
# while rospy.get_time() - start_time < 10:
while rospy.get_time() - start_time < 25:
    obs = env._observation()
    depth = obs['tiago_head_depth']
    rgb = obs['tiago_head_image']
    # print("counter: ", counter)
    # counter += 1
    # plt.imshow(rgb)
    # plt.show()
    # rgb_list.append(rgb)
    # depth_list.append(depth)

    if rgb.dtype != np.uint8:
        rgb = cv2.convertScaleAbs(rgb)
    rgb_corrected = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    writer.append_data(rgb_corrected)
    rospy.sleep(0.05)

# save_dict = dict()
# save_dict = {
#     "cam_intr": cam_intr,
#     "cam_extr": T_world_cam,
#     "rgb": rgb_list,
#     "depth": depth_list,
# }
# with open(f'{save_path}/dict.pickle', 'wb') as handle:
#     pickle.dump(save_dict, handle)