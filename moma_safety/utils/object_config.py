import numpy as np

object_config = {
    "pringles": {
        "min_th": 0.5,
        "max_th": 0.7,
        "gripper_open_pos": 0.9,
        "gripper_closed_pos": 0.0,
        "right_tooltip_ee_offset": np.array([-0.24, 0, 0]),
        "right_tooltip_ee_offset_pregrasp": np.array([-0.30, 0, 0]),
        "add_grasp_modes": False,
        "head_joint_pos": np.array([-0.5, -0.7]),
        "manip_head_joint_pos": np.array([-0.5, -0.7]),
        "text_description": "pringles.",
        "check_grasp_reachability_flag": True,
        "post_grasp_pregrasp_pose_flag": True,
        "use_impedance_controller": False,
        "use_safe_list": False,
        "check_arm_collision": True,
        "check_obj_dropping": True,
        "check_grasp_loss": False,
        "check_ft": False,
        "manip_num_samples": 20,
        "zero_out_z": False,
        
    },
    "can": {
        "min_th": 0.5,
        "max_th": 0.8,
        "gripper_open_pos": 0.9,
        "gripper_closed_pos": 0.0,
        "right_tooltip_ee_offset": np.array([-0.24, 0, 0]),
        "right_tooltip_ee_offset_pregrasp": np.array([-0.30, 0, 0]),
        "add_grasp_modes": False,
        "head_joint_pos": np.array([-0.5, -0.7]),
        "manip_head_joint_pos": np.array([-0.5, -0.7]),
        "text_description": "soup can.",
        "check_grasp_reachability_flag": True,
        "post_grasp_pregrasp_pose_flag": True,
        "use_impedance_controller": False,
        "use_safe_list": False,
        "check_arm_collision": True,
        "check_obj_dropping": True,
        "check_grasp_loss": False,
        "check_ft": False,
        "manip_num_samples": 20,
        "zero_out_z": False,
    },
    "ranch": {
        "min_th": 0.5,
        "max_th": 0.8,
        "gripper_open_pos": 0.9,
        "gripper_closed_pos": 0.0,
        "right_tooltip_ee_offset": np.array([-0.24, 0, 0]),
        "right_tooltip_ee_offset_pregrasp": np.array([-0.30, 0, 0]),
        "add_grasp_modes": False,
        "head_joint_pos": np.array([-0.5, -0.7]),
        "manip_head_joint_pos": np.array([-0.5, -0.7]),
        "text_description": "ranch bottle on table.",
        "check_grasp_reachability_flag": True,
        "post_grasp_pregrasp_pose_flag": True,
        "use_impedance_controller": False,
        "use_safe_list": False,
        "check_arm_collision": True,
        "check_obj_dropping": True,
        "check_grasp_loss": False,
        "check_ft": False,
        "manip_num_samples": 20,
        "zero_out_z": False,
    },
    "cup": {
        "min_th": 0.5,
        "max_th": 0.8,
        "gripper_open_pos": 0.9,
        "gripper_closed_pos": 0.0,
        "right_tooltip_ee_offset": np.array([-0.22, 0, 0]),
        "right_tooltip_ee_offset_pregrasp": np.array([-0.30, 0, 0]),
        "add_grasp_modes": False,
        "head_joint_pos": np.array([-0.5, -0.7]),
        "manip_head_joint_pos": np.array([-0.5, -0.7]),
        "text_description": "blue cup.",
        "check_grasp_reachability_flag": True,
        "post_grasp_pregrasp_pose_flag": True,
        "use_impedance_controller": False,
        "use_safe_list": False,
        "check_arm_collision": True,
        "check_obj_dropping": True,
        "check_grasp_loss": False,
        "check_ft": False,
        "manip_num_samples": 20,
        "zero_out_z": False,
    },
    "jello": {
        "min_th": 0.5,
        "max_th": 0.8,
        "gripper_open_pos": 0.9,
        "gripper_closed_pos": 0.0,
        "right_tooltip_ee_offset": np.array([-0.20, 0, 0]),
        "right_tooltip_ee_offset_pregrasp": np.array([-0.30, 0, 0]),
        "add_grasp_modes": False,
        "head_joint_pos": np.array([-0.5, -0.8]),
        "manip_head_joint_pos": np.array([-0.6, -0.8]),
        "text_description": "jello.",
        "check_grasp_reachability_flag": True,
        "post_grasp_pregrasp_pose_flag": True,
        "use_impedance_controller": False,
        "use_safe_list": False,
        "check_arm_collision": True,
        "check_obj_dropping": True,
        "check_grasp_loss": False,
        "check_ft": False,
        "manip_num_samples": 15,
        "zero_out_z": False,
    },
    "pot": {
        "min_th": 0.5,
        "max_th": 0.8,
        "gripper_open_pos": 0.7,
        "gripper_closed_pos": 0.0,
        "right_tooltip_ee_offset": np.array([-0.26, 0, 0]),
        "right_tooltip_ee_offset_pregrasp": np.array([-0.30, 0, 0]),
        "add_grasp_modes": True,
        "head_joint_pos": np.array([-0.9, -0.7]),
        "manip_head_joint_pos": np.array([-0.9, -0.7]),
        "text_description": "pot with white handle.",
        "check_grasp_reachability_flag": True,
        "post_grasp_pregrasp_pose_flag": True,
        "use_impedance_controller": False,
        "use_safe_list": False,
        "check_arm_collision": True,
        "check_obj_dropping": True,
        "check_grasp_loss": False,
        "check_ft": False,
        "manip_num_samples": 20,
        "zero_out_z": False,
    },
    # -0.35 AND -0.28 FOR CABINETS AND DRAWERS
    # TODO: correct this
    "drawer": {
        "min_th": 0.5,
        "max_th": 0.9,
        "gripper_open_pos": 0.5,
        "gripper_closed_pos": 0.0,
        "right_tooltip_ee_offset": np.array([-0.27, 0, 0]),
        "right_tooltip_ee_offset_pregrasp": np.array([-0.35, 0, 0]),
        "add_grasp_modes": True,
        "head_joint_pos": np.array([-0.5, -0.7]),
        "manip_head_joint_pos": np.array([-0.6, -0.7]),
        "text_description": "white handle.",
        "check_grasp_reachability_flag": True,
        "post_grasp_pregrasp_pose_flag": False,
        "use_impedance_controller": False,
        "use_safe_list": False,
        "check_arm_collision": False,
        "check_obj_dropping": False,
        "check_grasp_loss": True,
        "check_ft": True,
        "manip_num_samples": 15,
        "zero_out_z": True,
    },
    "shelf": {
        "min_th": 0.4,
        "max_th": 0.9,
        "gripper_open_pos": 0.9,
        "gripper_closed_pos": 0.0,
        "right_tooltip_ee_offset": np.array([-0.28, 0, 0]),
        "right_tooltip_ee_offset_pregrasp": np.array([-0.38, 0, 0]),
        "add_grasp_modes": False,
        "head_joint_pos": np.array([-0.5, -0.7]),
        "manip_head_joint_pos": np.array([-0.9, -0.6]),
        "text_description": "shelves.",
        "safety_model_text_description": "brown shelves. black robot part. white robot part. bottle.",
        "check_grasp_reachability_flag": False,
        "post_grasp_pregrasp_pose_flag": False,
        "use_impedance_controller": False,
        "use_safe_list": False,
        "check_arm_collision": True,
        "check_obj_dropping": True,
        "check_grasp_loss": False,
        "check_ft": False,
        "manip_num_samples": 20,
        "zero_out_z": False,
    },
    "fridge handle": {
        "min_th": 0.7,
        "max_th": 1.0,
        "gripper_open_pos": 0.7,
        "gripper_closed_pos": 0.0,
        "right_tooltip_ee_offset": np.array([-0.25, 0, 0]),
        "right_tooltip_ee_offset_pregrasp": np.array([-0.35, 0, 0]),
        "add_grasp_modes": True,
        "head_joint_pos": np.array([-0.4, -0.6]),
        "manip_head_joint_pos": np.array([-0.5, -0.6]),
        "text_description": "red colored handle.",
        "check_grasp_reachability_flag": True,
        "post_grasp_pregrasp_pose_flag": False,
        "use_impedance_controller": True,
        "use_safe_list": False,
        "check_arm_collision": False,
        "check_obj_dropping": False,
        "check_grasp_loss": True,
        "check_ft": True,
        "manip_num_samples": 15,
        "zero_out_z": True,
    },
    "ledge": {
        "min_th": 0.4,
        "max_th": 1.0,
        "gripper_open_pos": 0.9,
        "gripper_closed_pos": 0.0,
        "right_tooltip_ee_offset": np.array([-0.28, 0, 0]),
        "right_tooltip_ee_offset_pregrasp": np.array([-0.38, 0, 0]),
        "add_grasp_modes": False,
        "head_joint_pos": np.array([-0.8, -0.2]),
        "manip_head_joint_pos": np.array([-1.1, -0.2]),
        "text_description": "wall-mounted shelf.",
        "safety_model_text_description": "wall-mounted shelf.",
        "check_grasp_reachability_flag": False,
        "post_grasp_pregrasp_pose_flag": False,
        "use_impedance_controller": False,
        "use_safe_list": False,
        "check_arm_collision": True,
        "check_obj_dropping": True,
        "check_grasp_loss": False,
        "check_ft": False,
        "manip_num_samples": 20,
        "zero_out_z": False,
    },
    "faucet handle": {
        "min_th": 0.5,
        "max_th": 0.9,
        "gripper_open_pos": 0.5,
        "gripper_closed_pos": 0.0,
        "right_tooltip_ee_offset": np.array([-0.27, 0, 0]),
        "right_tooltip_ee_offset_pregrasp": np.array([-0.35, 0, 0]),
        "add_grasp_modes": True,
        "head_joint_pos": np.array([-0.5, -0.7]),
        "manip_head_joint_pos": np.array([-0.6, -0.7]),
        "text_description": "blue handle.",
        "check_grasp_reachability_flag": True,
        "post_grasp_pregrasp_pose_flag": False,
        "use_impedance_controller": True,
        "use_safe_list": False,
        "check_arm_collision": False,
        "check_obj_dropping": False,
        "check_grasp_loss": True,
        "check_ft": True,
        "manip_num_samples": 15,
        "zero_out_z": True,
    },
    "fridge": {
        "min_th": 0.2,
        "max_th": 0.9,
        "gripper_open_pos": 0.7,
        "gripper_closed_pos": 0.0,
        "right_tooltip_ee_offset": np.array([-0.28, 0, 0]),
        "right_tooltip_ee_offset_pregrasp": np.array([-0.38, 0, 0]),
        "add_grasp_modes": False,
        "head_joint_pos": np.array([-0.5, -0.7]),
        "manip_head_joint_pos": np.array([-0.9, -0.6]),
        "text_description": "white colored fridge.",
        "safety_model_text_description": "white colored fridge.",
        "check_grasp_reachability_flag": False,
        "post_grasp_pregrasp_pose_flag": False,
        "use_impedance_controller": False,
        "use_safe_list": False,
        "check_arm_collision": True,
        "check_obj_dropping": True,
        "check_grasp_loss": False,
        "check_ft": False,
        "manip_num_samples": 20,
        "zero_out_z": False,
    },
}


