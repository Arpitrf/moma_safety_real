import os
import numpy as np
import rospy
from PIL import Image
import matplotlib.pyplot as plt

import moma_safety.utils.utils as U
from moma_safety.models.wrappers import GroundedSamWrapper
from moma_safety.tiago.tiago_gym import TiagoGym

rospy.init_node("gsam")
gsam = GroundedSamWrapper(sam_ckpt_path=os.environ["SAM_CKPT_PATH"])
env = TiagoGym(
    frequency=10,
    right_arm_enabled=False,
    left_arm_enabled=False,
    right_gripper_type=None,
    left_gripper_type=None,
    base_enabled=False,
    torso_enabled=False,
)
while True:
    obs = env._observation()
    rgb = obs['tiago_head_image'][:, :, ::-1].astype(np.uint8) # BGR -> RGB
    gsam_output = gsam.segment(rgb, ['food items', 'snacks', 'drinks'])
    overlay_image = U.overlay_xmem_mask_on_image(
        rgb.copy(),
        np.array(gsam_output),
        use_white_bg=False,
        rgb_alpha=0.3
    )
    plt.imshow(overlay_image)
    plt.show()
