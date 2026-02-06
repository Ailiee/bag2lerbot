#!/usr/bin/env python3
"""
Custom State-Action Mapping Example for LeRobot Converter

This file demonstrates how to create custom state/action mappings for
different robot configurations when converting to LeRobot format.
"""

import logging
import numpy as np
from typing import Dict, List
from dataclasses import dataclass, field
from typing import Callable, Optional

# Names must match the concatenation order in combine_ur_dual_arm_state/action
STATE_NAMES = [
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_joint_6",
    "left_gripper",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_joint_6",
    "right_gripper",
    "left_eef_x",
    "left_eef_y",
    "left_eef_z",
    "left_eef_rx",
    "left_eef_ry",
    "left_eef_rz",
    "right_eef_x",
    "right_eef_y",
    "right_eef_z",
    "right_eef_rx",
    "right_eef_ry",
    "right_eef_rz",
]

ACTION_NAMES = STATE_NAMES.copy()

# Expected dimensions for this UR dual-arm setup
STATE_DIM = len(STATE_NAMES)
ACTION_DIM = len(ACTION_NAMES)



@dataclass
class StateActionMapping:
    """Define how to map HDF5 data to LeRobot state and action tensors."""
    
    # State components to combine
    state_components: List[str] = field(default_factory=list)
    
    # Action components to combine  
    action_components: List[str] = field(default_factory=list)
    
    # Custom combine functions
    state_combine_fn: Optional[Callable] = None
    action_combine_fn: Optional[Callable] = None
    
    # Normalization parameters
    normalize: bool = True
    state_stats: Optional[Dict[str, Dict[str, float]]] = None
    action_stats: Optional[Dict[str, Dict[str, float]]] = None


def combine_ur_dual_arm_state(components: Dict[str, np.ndarray]) -> np.ndarray:
    """Custom state combination for UR dual-arm robot."""
    state_parts = []
    missing = []
    
    # Left arm joints (6 DOF for UR5e)
    if "left_ur5e/joint_positions" in components:
        left_joints = components["left_ur5e/joint_positions"]
        state_parts.append(left_joints[:6])

    # Left gripper position
    if "left_ur5e/gripper_position" in components:
        left_gripper = components["left_ur5e/gripper_position"]
        state_parts.append(left_gripper[:1])

    
    # Right arm joints (6 DOF for UR5e)
    if "right_ur5e/joint_positions" in components:
        right_joints = components["right_ur5e/joint_positions"]
        state_parts.append(right_joints[:6])
    
    # Right gripper position
    if "right_ur5e/gripper_position" in components:
        right_gripper = components["right_ur5e/gripper_position"]
        state_parts.append(right_gripper[:1])
    
    # Left end-effector position (6D: x, y, z, rx, ry, rz)
    if "left_ur5e/eef_pose" in components:
        left_ee_pos = components["left_ur5e/eef_pose"]
        state_parts.append(left_ee_pos[:6])

    # Right end-effector position (6D: x, y, z, rx, ry, rz)
    if "right_ur5e/eef_pose" in components:
        right_ee_pos = components["right_ur5e/eef_pose"]
        state_parts.append(right_ee_pos[:6])

    # Concatenate all parts
    # Total: 6 (left joints) + 1 (left gripper) + 6 (right joints) + 1 (right gripper) + 6 (left eef) + 6 (right eef) = 26 dimensions
    return np.concatenate(state_parts, axis=-1).astype(np.float32)

def combine_ur_dual_arm_action(components: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Custom action combination for UR dual-arm robot.
    
    This function defines the specific order and structure of the action vector
    for your UR dual-arm setup.
    
    Note: Action is typically the next state (state_{t+1}), so we use the same structure as state.
    For learning from demonstrations, action is usually the next observed state.
    
    Args:
        components: Dictionary mapping HDF5 paths to numpy arrays
        
    Returns:
        Combined action vector with consistent ordering
    """
    action_parts = []
    missing = []
    
    # Left arm joint commands (6 DOF)
    if "left_ur5e/joint_positions" in components:
        left_joints = components["left_ur5e/joint_positions"]
        action_parts.append(left_joints[:6])

    # Left gripper command
    if "left_ur5e/gripper_position" in components:
        left_gripper = components["left_ur5e/gripper_position"]
        action_parts.append(left_gripper[:1])
    
    # Right arm joint commands (6 DOF)
    if "right_ur5e/joint_positions" in components:
        right_joints = components["right_ur5e/joint_positions"]
        action_parts.append(right_joints[:6])
    
    # Right gripper command
    if "right_ur5e/gripper_position" in components:
        right_gripper = components["right_ur5e/gripper_position"]
        action_parts.append(right_gripper[:1])
        
    # Left end-effector position (6D: x, y, z, rx, ry, rz)
    if "left_ur5e/eef_pose" in components:
        left_ee_pos = components["left_ur5e/eef_pose"]
        action_parts.append(left_ee_pos[:6])
    
    # Right end-effector position (6D: x, y, z, rx, ry, rz)
    if "right_ur5e/eef_pose" in components:
        right_ee_pos = components["right_ur5e/eef_pose"]
        action_parts.append(right_ee_pos[:6])

    return np.concatenate(action_parts, axis=-1).astype(np.float32)

def get_state_action_mapping() -> StateActionMapping:
    """
    Main function called by the converter to get custom mapping.
    
    Modify this function to return your specific robot's mapping.
    
    Returns:
        StateActionMapping configuration for your robot
    """
    
    state_components = [
        "left_ur5e/joint_positions",      # 6D: joints
        "left_ur5e/gripper_position",     # 1D: gripper
        "right_ur5e/joint_positions",     # 6D: joints
        "right_ur5e/gripper_position",    # 1D: gripper
        "left_ur5e/eef_pose",             # 6D: x, y, z, rx, ry, rz
        "right_ur5e/eef_pose",            # 6D: x, y, z, rx, ry, rz
    ]
    
    action_components = [
        "left_ur5e/joint_positions",      # 6D: joints
        "left_ur5e/gripper_position",     # 1D: gripper
        "right_ur5e/joint_positions",     # 6D: joints
        "right_ur5e/gripper_position",    # 1D: gripper
        "left_ur5e/eef_pose",             # 6D: x, y, z, rx, ry, rz
        "right_ur5e/eef_pose",            # 6D: x, y, z, rx, ry, rz
    ]
    
    # Optional: Define normalization statistics
    # These would typically be computed from your training data
    state_stats = {
        "mean": np.zeros(STATE_DIM),
        "std": np.ones(STATE_DIM),
        "min": np.full(STATE_DIM, -np.inf),
        "max": np.full(STATE_DIM, np.inf)
    }
    
    action_stats = {
        "mean": np.zeros(ACTION_DIM),
        "std": np.ones(ACTION_DIM),
        "min": np.full(ACTION_DIM, -np.inf),
        "max": np.full(ACTION_DIM, np.inf)
    }
    
    return StateActionMapping(
        state_components=state_components,
        action_components=action_components,
        state_combine_fn=combine_ur_dual_arm_state,
        action_combine_fn=combine_ur_dual_arm_action,
        normalize=True,
        state_stats=state_stats,
        action_stats=action_stats
    )


# def get_key_mappings() -> Dict[str, str]:
#     """No key remapping needed for UR when using processor-native keys."""
#     return {}

# def get_single_arm_mapping() -> StateActionMapping:
#     """Mapping for single-arm robot configuration."""
    
#     def combine_single_arm_state(components: Dict[str, np.ndarray]) -> np.ndarray:
#         state_parts = []
        
#         # Arm joints
#         if "arm_states/joint_positions" in components:
#             joints = components["arm_states/joint_positions"]
#             state_parts.append(joints)
        
#         # Gripper
#         if "gripper_state/position" in components:
#             gripper = components["gripper_state/position"]
#             state_parts.append(np.atleast_1d(gripper))
        
#         # End-effector
#         if "arm_states/end_effector_pose" in components:
#             ee_pose = components["arm_states/end_effector_pose"]
#             state_parts.append(ee_pose)
        
#         return np.concatenate(state_parts, axis=-1).astype(np.float32)
    
#     def combine_single_arm_action(components: Dict[str, np.ndarray]) -> np.ndarray:
#         action_parts = []
        
#         # Joint commands
#         if "arm_states/joint_commands" in components:
#             commands = components["arm_states/joint_commands"]
#             action_parts.append(commands)
        
#         # Gripper command
#         if "gripper_state/command" in components:
#             gripper_cmd = components["gripper_state/command"]
#             action_parts.append(np.atleast_1d(gripper_cmd))
        
#         return np.concatenate(action_parts, axis=-1).astype(np.float32)
    
#     return StateActionMapping(
#         state_components=[
#             "arm_states/joint_positions",
#             "gripper_state/position",
#             "arm_states/end_effector_pose"
#         ],
#         action_components=[
#             "arm_states/joint_commands",
#             "gripper_state/command"
#         ],
#         state_combine_fn=combine_single_arm_state,
#         action_combine_fn=combine_single_arm_action
#     )

# def get_mobile_manipulator_mapping() -> StateActionMapping:
#     """Mapping for mobile manipulator (base + arm)."""
    
#     def combine_mobile_state(components: Dict[str, np.ndarray]) -> np.ndarray:
#         state_parts = []
        
#         # Base pose (x, y, theta)
#         if "base/pose" in components:
#             base_pose = components["base/pose"]
#             state_parts.append(base_pose)
        
#         # Base velocity
#         if "base/velocity" in components:
#             base_vel = components["base/velocity"]
#             state_parts.append(base_vel)
        
#         # Arm joints
#         if "arm/joint_positions" in components:
#             joints = components["arm/joint_positions"]
#             state_parts.append(joints)
        
#         # Gripper
#         if "gripper/position" in components:
#             gripper = components["gripper/position"]
#             state_parts.append(np.atleast_1d(gripper))
        
#         return np.concatenate(state_parts, axis=-1).astype(np.float32)
    
#     def combine_mobile_action(components: Dict[str, np.ndarray]) -> np.ndarray:
#         action_parts = []
        
#         # Base velocity commands
#         if "base/velocity_command" in components:
#             base_cmd = components["base/velocity_command"]
#             action_parts.append(base_cmd)
        
#         # Arm commands
#         if "arm/joint_commands" in components:
#             arm_cmd = components["arm/joint_commands"]
#             action_parts.append(arm_cmd)
        
#         # Gripper command
#         if "gripper/command" in components:
#             gripper_cmd = components["gripper/command"]
#             action_parts.append(np.atleast_1d(gripper_cmd))
        
#         return np.concatenate(action_parts, axis=-1).astype(np.float32)
    
#     return StateActionMapping(
#         state_components=[
#             "base/pose",
#             "base/velocity",
#             "arm/joint_positions",
#             "gripper/position"
#         ],
#         action_components=[
#             "base/velocity_command",
#             "arm/joint_commands",
#             "gripper/command"
#         ],
#         state_combine_fn=combine_mobile_state,
#         action_combine_fn=combine_mobile_action
#     )
