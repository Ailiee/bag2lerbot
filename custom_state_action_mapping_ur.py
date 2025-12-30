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


def combine_ur_dual_arm_state(components: Dict[str, np.ndarray]) -> np.ndarray:
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
    if "left_ur5e/joint_positions" in components:
        left_joints = components["left_ur5e/joint_positions"]
        if left_joints.ndim == 0:
            left_joints = np.atleast_1d(left_joints)
        state_parts.append(left_joints[:6])  # Ensure 6 joints
    
    # Right arm joints (6 DOF for UR5e)
    if "right_ur5e/joint_positions" in components:
        right_joints = components["right_ur5e/joint_positions"]
        if right_joints.ndim == 0:
            right_joints = np.atleast_1d(right_joints)
        state_parts.append(right_joints[:6])  # Ensure 6 joints
    
    # Left end-effector position (6D)
    if "left_ur5e/end_eff" in components:
        left_ee_pos = components["left_ur5e/end_eff"]
        state_parts.append(left_ee_pos[:6])
    
    # Right end-effector position (6D)
    if "right_ur5e/end_eff" in components:
        right_ee_pos = components["right_ur5e/end_eff"]
        state_parts.append(right_ee_pos[:6])
    
    # Left gripper position
    if "left_gripper/gripper_position" in components:
        left_gripper = components["left_gripper/gripper_position"]
        state_parts.append(np.atleast_1d(left_gripper))

    # Right gripper position
    if "right_gripper/gripper_position" in components:
        right_gripper = components["right_gripper/gripper_position"]
        state_parts.append(np.atleast_1d(right_gripper))

    # Concatenate all parts
    # Total: 6 + 6 + 6 + 6 + 1 + 1= 26 dimensions
    return np.concatenate(state_parts, axis=-1).astype(np.float32)

def combine_ur_dual_arm_action(components: Dict[str, np.ndarray]) -> np.ndarray:
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
    if "left_ur5e/joint_positions" in components:
        left_commands = components["left_ur5e/joint_positions"]
        if left_commands.ndim == 0:
            left_commands = np.atleast_1d(left_commands)
        action_parts.append(left_commands[:6])
    
    # Right arm joint commands (6 DOF)
    if "right_ur5e/joint_positions" in components:
        right_commands = components["right_ur5e/joint_positions"]
        if right_commands.ndim == 0:
            right_commands = np.atleast_1d(right_commands)
        action_parts.append(right_commands[:6])
        
    # Left end-effector position (6D)
    if "left_ur5e/end_eff" in components:
        left_ee_pos = components["left_ur5e/end_eff"]
        action_parts.append(left_ee_pos[:6])
    
    # Right end-effector position (6D)
    if "right_ur5e/end_eff" in components:
        right_ee_pos = components["right_ur5e/end_eff"]
        action_parts.append(right_ee_pos[:6])
    
    # Left gripper command
    if "left_gripper/gripper_position" in components:
        left_gripper_cmd = components["left_gripper/gripper_position"]
        action_parts.append(np.atleast_1d(left_gripper_cmd))
    
    # Right gripper command
    if "right_gripper/gripper_position" in components:
        right_gripper_cmd = components["right_gripper/gripper_position"]
        action_parts.append(np.atleast_1d(right_gripper_cmd))
    
    # Concatenate all parts
    # Total: 6 + 6 + 6 + 6 + 1 + 1 = 14 dimensions
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
        "left_ur5e/joint_positions",
        "right_ur5e/joint_positions",          
        # End-effector poses
        "left_ur5e/end_eff",
        "right_ur5e/end_eff",
        # Gripper states
        "left_gripper/gripper_position",
        "right_gripper/gripper_position",
        # Optional: Joint velocities if needed
        # "left_ur5e/joint_velocities",
        # "right_ur5e/joint_velocities",
    ]
    
    # Define which HDF5 paths contain action data
    action_components = [
        # Joint commands
        "left_ur5e/joint_positions",
        "right_ur5e/joint_positions",
        # End-effector poses
        "left_ur5e/end_eff",
        "right_ur5e/end_eff",        
        # Gripper commands
        "left_gripper/gripper_position",
        "right_gripper/gripper_position",
    ]
    
    # Optional: Define normalization statistics
    # These would typically be computed from your training data
    state_stats = {
        "mean": np.zeros(26),  # 26-dimensional state
        "std": np.ones(26),
        "min": np.full(26, -np.inf),
        "max": np.full(26, np.inf)
    }
    
    action_stats = {
        "mean": np.zeros(26),  # 26-dimensional action
        "std": np.ones(26),
        "min": np.full(26, -np.inf),
        "max": np.full(26, np.inf)
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

def get_single_arm_mapping() -> StateActionMapping:
    """Mapping for single-arm robot configuration."""
    
    def combine_single_arm_state(components: Dict[str, np.ndarray]) -> np.ndarray:
        state_parts = []
        
        # Arm joints
        if "arm_states/joint_positions" in components:
            joints = components["arm_states/joint_positions"]
            state_parts.append(joints)
        
        # Gripper
        if "gripper_state/position" in components:
            gripper = components["gripper_state/position"]
            state_parts.append(np.atleast_1d(gripper))
        
        # End-effector
        if "arm_states/end_effector_pose" in components:
            ee_pose = components["arm_states/end_effector_pose"]
            state_parts.append(ee_pose)
        
        return np.concatenate(state_parts, axis=-1).astype(np.float32)
    
    def combine_single_arm_action(components: Dict[str, np.ndarray]) -> np.ndarray:
        action_parts = []
        
        # Joint commands
        if "arm_states/joint_commands" in components:
            commands = components["arm_states/joint_commands"]
            action_parts.append(commands)
        
        # Gripper command
        if "gripper_state/command" in components:
            gripper_cmd = components["gripper_state/command"]
            action_parts.append(np.atleast_1d(gripper_cmd))
        
        return np.concatenate(action_parts, axis=-1).astype(np.float32)
    
    return StateActionMapping(
        state_components=[
            "arm_states/joint_positions",
            "gripper_state/position",
            "arm_states/end_effector_pose"
        ],
        action_components=[
            "arm_states/joint_commands",
            "gripper_state/command"
        ],
        state_combine_fn=combine_single_arm_state,
        action_combine_fn=combine_single_arm_action
    )

def get_mobile_manipulator_mapping() -> StateActionMapping:
    """Mapping for mobile manipulator (base + arm)."""
    
    def combine_mobile_state(components: Dict[str, np.ndarray]) -> np.ndarray:
        state_parts = []
        
        # Base pose (x, y, theta)
        if "base/pose" in components:
            base_pose = components["base/pose"]
            state_parts.append(base_pose)
        
        # Base velocity
        if "base/velocity" in components:
            base_vel = components["base/velocity"]
            state_parts.append(base_vel)
        
        # Arm joints
        if "arm/joint_positions" in components:
            joints = components["arm/joint_positions"]
            state_parts.append(joints)
        
        # Gripper
        if "gripper/position" in components:
            gripper = components["gripper/position"]
            state_parts.append(np.atleast_1d(gripper))
        
        return np.concatenate(state_parts, axis=-1).astype(np.float32)
    
    def combine_mobile_action(components: Dict[str, np.ndarray]) -> np.ndarray:
        action_parts = []
        
        # Base velocity commands
        if "base/velocity_command" in components:
            base_cmd = components["base/velocity_command"]
            action_parts.append(base_cmd)
        
        # Arm commands
        if "arm/joint_commands" in components:
            arm_cmd = components["arm/joint_commands"]
            action_parts.append(arm_cmd)
        
        # Gripper command
        if "gripper/command" in components:
            gripper_cmd = components["gripper/command"]
            action_parts.append(np.atleast_1d(gripper_cmd))
        
        return np.concatenate(action_parts, axis=-1).astype(np.float32)
    
    return StateActionMapping(
        state_components=[
            "base/pose",
            "base/velocity",
            "arm/joint_positions",
            "gripper/position"
        ],
        action_components=[
            "base/velocity_command",
            "arm/joint_commands",
            "gripper/command"
        ],
        state_combine_fn=combine_mobile_state,
        action_combine_fn=combine_mobile_action
    )

if __name__ == "__main__":
    """Test the mapping functions."""
    
    # Test dual-arm mapping
    mapping = get_state_action_mapping()
    print("Dual-arm UR Robot Mapping:")
    print(f"  State components: {len(mapping.state_components)}")
    print(f"  Action components: {len(mapping.action_components)}")
    
    # Test with dummy data
    dummy_state_components = {
        "left_ur5e/joint_positions": np.random.randn(6),
        "right_ur5e/joint_positions": np.random.randn(6),
        "left_gripper/gripper_position": np.array([100]),
        "right_gripper/gripper_position": np.array([100]),
        "left_ur5e/end_eff": np.random.randn(6),
        "right_ur5e/end_eff": np.random.randn(6),
    }
    
    combined_state = mapping.state_combine_fn(dummy_state_components)
    print(f"  Combined state shape: {combined_state.shape}")
    print(f"  State dimensions: {combined_state.shape[0]}")
    
    dummy_action_components = {
        "left_ur5e/joint_positions": np.random.randn(6),
        "right_ur5e/joint_positions": np.random.randn(6),
        "left_gripper/gripper_position": np.array([100]),
        "right_gripper/gripper_position": np.array([100]),
        "left_ur5e/end_eff": np.random.randn(6),
        "right_ur5e/end_eff": np.random.randn(6),
    }
    
    combined_action = mapping.action_combine_fn(dummy_action_components)
    print(f"  Combined action shape: {combined_action.shape}")
    print(f"  Action dimensions: {combined_action.shape[0]}")
    
    print("\nMapping test completed successfully!")