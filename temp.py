import numpy as np

# B = np.array([
#     [0, 0, -1], 
#     [1, 0, 0], 
#     [0, -1, 0]
# ])
# A = np.array([
#     [1, 0, 0], 
#     [0, 0, 1], 
#     [0, -1, 0]
# ])

# # print(np.linalg.inv(B))
# X = np.linalg.inv(B) @ A
# print(X)

# ============== second day ==============

# B = np.array([
#     [0, 0, -1], 
#     [1, 0, 0], 
#     [0, -1, 0]
# ])
# A = np.array([
#     [1, 0, 0], 
#     [0, 0, -1], 
#     [0, 1, 0]
# ])

# # print(np.linalg.inv(B))
# X = np.linalg.inv(B) @ A
# print(X)

from scipy.spatial.transform import Rotation as R
R_z_45 = np.array([
    [0.70710678, -0.70710678, 0],
    [0.70710678, 0.70710678, 0],
    [0, 0, 1]
])
angle = R.from_matrix(R_z_45[:3, :3]).as_euler("xyz")[2]
print(angle)
breakpoint()