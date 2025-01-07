import numpy as np
np.set_printoptions(precision=3, suppress=True)
import rospy

from moma_safety.tiago.tiago_gym import TiagoGym
from moma_safety.tiago import RESET_POSES as RP
import moma_safety.utils.transform_utils as T # transform_utils
from moma_safety.tiago.utils.ros_utils import TFTransformListener
from moma_safety.tiago.utils.transformations import quat_diff
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

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
print("obs_scan: ", obs["scan"][545:])
plt.plot(obs["scan"][545:])
plt.ylim(0, 1.5)
plt.show()


# # Generate the angles for the 545 readings (from 0 to 180 degrees, step size 0.33)
# angles = np.linspace(0, 180, 545)

# # Example sensor readings (replace this with your actual data)
# # Assuming the readings are stored in an array 'sensor_readings' of length 545
# sensor_readings = obs["scan"][:545]  # Replace this with actual data

# # Convert angles from degrees to radians
# angles_rad = np.deg2rad(angles)

# # Create the polar plot
# plt.figure(figsize=(8, 8))
# ax = plt.subplot(111, projection='polar')

# # Plot the data on the polar axis
# ax.plot(angles_rad, sensor_readings, marker='o', linestyle='-', color='b')

# # Add labels and title
# ax.set_title("Sensor Readings Polar Plot", va='bottom')
# ax.set_xlabel("Angle (degrees)")
# ax.set_ylabel("Sensor Readings")

# # Show the plot
# plt.show()

breakpoint()
