#!/usr/bin/env python3
"""
Custom State-Action Mapping Example for LeRobot Converter

This file demonstrates how to create custom state/action mappings for
different robot configurations when converting to LeRobot format.
"""

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
    "left_joint_7",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_joint_6",
    "right_joint_7",
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
    "waist_x",
    "waist_y",
    "waist_z",
    "neck_pitch",
    "neck_yaw",
    "gripper_left",
    "gripper_right"
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


def combine_state(components: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Custom state combination for UR dual-arm robot.
    
    This function defines the specific order and structure of the state vector
    for your UR dual-arm setup.
    
    Args:
        components: Dictionary mapping component paths to numpy arrays
        
    Returns:
        Combined state vector with consistent ordering
    """
    state_parts = []
    
    # Left arm joints (6 DOF for UR5e)
    joints = None
    eef = None
    
    if "driver/q_pos" in components:
        joints = components["driver/q_pos"]
    # Right arm joints (6 DOF for UR5e)
    if "end/eef" in components:
        eef = components["end/eef"]
        
    if joints is not None:
        state_parts.append(joints[:14])  # Ensure 14 joints
        
    if eef is not None:
        state_parts.append(eef[:12])  # Ensure 12 joints
        
    if joints is not None and eef is None:
        # Fallback if only joints available? Or raise error?
        pass

    if joints is not None:
        state_parts.append(joints[14:])
        
    if not state_parts:
        raise ValueError("No state components found (missing driver/q_pos and end/eef)")
        
    # Concatenate all parts
    # Total: 7 + 7 + 6 + 6 + 3 + 2 + 2 = 33 dimensions
    return np.concatenate(state_parts, axis=-1).astype(np.float32)

def combine_action(components: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Custom action combination for UR dual-arm robot.
    
    This function defines the specific order and structure of the action vector
    for your UR dual-arm setup.
    
    Args:
        components: Dictionary mapping component paths to numpy arrays
        
    Returns:
        Combined action vector with consistent ordering
    """
    action_parts = []
    
    # Left arm joint commands (6 DOF)
    joints = None
    eef = None
    
    if "driver/q_pos" in components:
        joints = components["driver/q_pos"]

    # Right arm joint commands (6 DOF)
    if "end/eef" in components:
        eef = components["end/eef"]
    
    if joints is not None:
        action_parts.append(joints[:14])  # Ensure 14 joints
        
    if eef is not None:
        action_parts.append(eef[:12])  # Ensure 12 joints
        
    if joints is not None:
        action_parts.append(joints[14:])
        
    if not action_parts:
        raise ValueError("No action components found")
      
    # Concatenate all parts
    # Total: 7 + 7 + 6 + 6 + 3 + 2 + 2 = 33 dimensions
    return np.concatenate(action_parts, axis=-1).astype(np.float32)

def get_state_action_mapping() -> StateActionMapping:
    """
    Main function called by the converter to get custom mapping.
    
    Modify this function to return your specific robot's mapping.
    
    Returns:
        StateActionMapping configuration for your robot
    """
    
    # Define which HDF5 paths contain state data
    state_components = [
        # Joint states
        "driver/q_pos",
        "end/eef",          
    ]
    
    # Define which HDF5 paths contain action data
    action_components = [
        # Joint commands
        "driver/q_pos",
        "end/eef",
    ]
    
    # Optional: Define normalization statistics
    # These would typically be computed from your training data
    state_stats = {
        "mean": np.zeros(STATE_DIM),  # 26-dimensional state
        "std": np.ones(STATE_DIM),
        "min": np.full(STATE_DIM, -np.inf),
        "max": np.full(STATE_DIM, np.inf)
    }
    
    action_stats = {
        "mean": np.zeros(ACTION_DIM),  # 26-dimensional action
        "std": np.ones(ACTION_DIM),
        "min": np.full(ACTION_DIM, -np.inf),
        "max": np.full(ACTION_DIM, np.inf)
    }
    
    return StateActionMapping(
        state_components=state_components,
        action_components=action_components,
        state_combine_fn=combine_state,
        action_combine_fn=combine_action,
        normalize=True,
        state_stats=state_stats,
        action_stats=action_stats
    )
