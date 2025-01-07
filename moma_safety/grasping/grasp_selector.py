import copy
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

import moma_safety.grasping.grasp_pose_generator as gpg
from moma_safety.grasping.grasp_classifier import GraspClassifier

class GraspSelector(object):
    def __init__(self, object_frame, point_cloud_with_normals):
        super(GraspSelector, self).__init__()
        
        # Initializations
        self.object_frame_in_world_frame = object_frame
        self.point_cloud_with_normals = point_cloud_with_normals
        # Distance from sampled point in cloud to designated ee frame
        #self.dist_from_point_to_ee_link = -0.02
        self.dist_from_point_to_ee_link = -0.01

        # Gaussian Process classifier wrapper
        self.clf = GraspClassifier()


        self.generateGraspPosesWorldFrame(point_cloud_with_normals)

    def visualizeGraspPoses(self, grasp_poses):
        # Given a 4x4 transformation matrix, create coordinate frame mesh at the pose
        #     and scale down.
        def o3dTFAtPose(pose, scale_down=10):
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
            scaling_maxtrix = np.ones((4,4))
            scaling_maxtrix[:3, :3] = scaling_maxtrix[:3, :3]/scale_down
            scaled_pose = pose*scaling_maxtrix
            axes.transform(scaled_pose)
            return axes
        world_frame_axes = o3dTFAtPose(np.eye(4))
        models = [world_frame_axes, self.point_cloud_with_normals]
        for i, grasp_pose in enumerate(grasp_poses):
            grasp_axes = o3dTFAtPose(grasp_pose, scale_down=100)
            models.append(grasp_axes)
            # Create a sphere
            grasp_pos = np.array(grasp_pose[:3, 3])
            for j in range(i+1):
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
                sphere_pos = grasp_pos + np.array([0.0, 0.0, 0.01*j])
                sphere.translate(sphere_pos)
                models.append(sphere)
            
            # o3d.visualization.draw_geometries([self.point_cloud_with_normals, grasp_axes])
        try: 
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            for model in models:
                vis.add_geometry(model)
            vis.run()
            # o3d.visualization.draw_geometries(models)
        finally:
            vis.destroy_window()
    """
    Initialize the grasp pose generator and generate a grasp pose at each point
        in the provided cloud in the world frame.

    @param point_cloud_with_normals: o3d point cloud with estimated normals

    """
    def generateGraspPosesWorldFrame(self, point_cloud_with_normals):
        self.grasp_generator = gpg.GraspPoseGenerator(point_cloud_with_normals, rotation_values_about_approach=[0])
        self.grasp_poses = []
        for i in range(np.asarray(point_cloud_with_normals.points).shape[0]):
            self.grasp_poses += self.grasp_generator.proposeGraspPosesAtCloudIndex(i)

    # Transform a grasp pose in the world frame to the object frame (used in training.)
    def graspPoseWorldFrameToObjFrame(self, grasp_pose):
        return np.matmul(np.linalg.inv(self.object_frame_in_world_frame), grasp_pose)

    # Transform a grasp pose in the world frame to a list of len(7) representing
    #     the pose in the object frame (position and quaternion)
    def graspPoseWorldFrameToClassifierInput(self, grasp_pose):
        # World frame to object frame
        grasp_pose_in_obj_frame = self.graspPoseWorldFrameToObjFrame(grasp_pose)
        # 4x4 matrix to position, quaternion
        grasp_pose_obj_frame_pos, grasp_pose_obj_frame_quat = mat2PosQuat(grasp_pose_in_obj_frame)
        # [3], [4] to [7]
        example_vector = list(grasp_pose_obj_frame_pos) + list(grasp_pose_obj_frame_quat)
        return example_vector

    """
    Add a new example to the classifier's training set, retrain the classifier

    @param grasp_pose_mat: (4x4) numpy array representing grasp pose, with z-axis
        being the approach direction.
    """
    def updateClassifier(self, grasp_pose_mat, label):
        # grasp pose to list of len(7) representing pose in the object frame.
        example_vector = self.graspPoseWorldFrameToClassifierInput(grasp_pose_mat)
        print("Re-training classifier with additional example {} and label {}.".format(example_vector, label))
        self.clf.addBinaryLabeledExample(example_vector, label)
        self.clf.trainClassifier()

    def getRankedGraspPoses(self):
        # If the classifier has yet to be trained, return all poses in default order
        if self.clf.clf is None:
            return copy.deepcopy(self.grasp_poses)
        else:
            grasp_poses_classifier_input = [self.graspPoseWorldFrameToClassifierInput(pose) for pose in self.grasp_poses]
            scores = self.clf.predictSuccessProbabilities(grasp_poses_classifier_input)
            # Indices corresponding to scores sorted from largest to smallest
            sorted_grasp_indices = scores.argsort()[::-1]
            sorted_grasp_poses = [self.grasp_poses[i] for i in sorted_grasp_indices]
            print("Top grasp has a success probability of", scores[sorted_grasp_indices[0]])
            return sorted_grasp_poses