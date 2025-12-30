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
    if "driver/q_pos" in components:
        joints = components["driver/q_pos"]
    # Right arm joints (6 DOF for UR5e)
    if "end/eef" in components:
        eef = components["end/eef"]
        
    state_parts.append(joints[:14])  # Ensure 14 joints
    state_parts.append(eef[:12])  # Ensure 12 joints
    state_parts.append(joints[14:])
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
    if "driver/q_pos" in components:
        joints = components["driver/q_pos"]

    # Right arm joint commands (6 DOF)
    if "end/eef" in components:
        eef = components["end/eef"]
        
    action_parts.append(joints[:14])  # Ensure 14 joints
    action_parts.append(eef[:12])  # Ensure 12 joints
    action_parts.append(joints[14:])
      
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
        "mean": np.zeros(33),  # 26-dimensional state
        "std": np.ones(33),
        "min": np.full(33, -np.inf),
        "max": np.full(33, np.inf)
    }
    
    action_stats = {
        "mean": np.zeros(33),  # 26-dimensional action
        "std": np.ones(33),
        "min": np.full(33, -np.inf),
        "max": np.full(33, np.inf)
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
