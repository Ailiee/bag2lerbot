#!/usr/bin/env python3
"""
Custom State-Action Mapping for Franka Dual-Arm Robot

This file defines how to map HDF5 data to LeRobot state and action tensors
for Franka dual-arm robot configuration.
"""

import numpy as np
from typing import Dict, List
from dataclasses import dataclass, field
from typing import Callable, Optional
import h5py

# Names must match the concatenation order in combine_ur_dual_arm_state/action
STATE_NAMES = [
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_joint_6",
    "left_joint_7",
    "left_gripper",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_joint_6",
    "right_joint_7",
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


def pose_to_6d(pose: np.ndarray) -> np.ndarray:
    """
    Convert pose (position + quaternion) to 6D representation.

    Args:
        pose: Array of shape (7,) containing [x, y, z, qx, qy, qz, qw]

    Returns:
        Array of shape (6,) containing [x, y, z, roll, pitch, yaw]
    """
    if pose.ndim == 0:
        pose = np.atleast_1d(pose)

    if len(pose) >= 7:
        # Extract position (x, y, z)
        position = pose[:3]

        # Extract quaternion (qx, qy, qz, qw)
        qx, qy, qz, qw = pose[3:7]

        # Convert quaternion to Euler angles (roll, pitch, yaw)
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.concatenate([position, [roll, pitch, yaw]]).astype(np.float32)
    elif len(pose) >= 6:
        # Already in 6D format
        return pose[:6].astype(np.float32)
    else:
        # Fallback: pad with zeros
        result = np.zeros(6, dtype=np.float32)
        result[:len(pose)] = pose[:len(pose)]
        return result


def combine_franka_dual_arm_state(components: Dict[str, np.ndarray]) -> np.ndarray:
    """ Custom state combination for Franka dual-arm robot."""
    
    def _get_component(keys: List[str]) -> Optional[np.ndarray]:
        for key in keys:
            if key in components:
                return components[key]
        return None

    state_parts = []

    left_joints = _get_component([
        "left_franka_joint_states/left_franka/joint_positions",
        "left_franka_joint_states",
        "left_franka/joint_positions",
        "left/franka/joint_positions",
    ])
    if left_joints is None:
        left_joints = np.zeros(7, dtype=np.float32)
    else:
        left_joints = np.atleast_1d(left_joints)[:7]
    state_parts.append(left_joints)

    left_gripper = _get_component([
        "left_franka_gripper/left_franka/gripper_position",
        "left_franka_gripper",
        "left_franka/gripper_position",
        "left/franka/gripper_position",
    ])
    if left_gripper is None:
        left_gripper = np.zeros(1, dtype=np.float32)
    else:
        left_gripper = np.atleast_1d(left_gripper)[:1]
    state_parts.append(left_gripper)

    right_joints = _get_component([
        "right_franka_joint_states/right_franka/joint_positions",
        "right_franka_joint_states",
        "right_franka/joint_positions",
        "right/franka/joint_positions",
    ])
    if right_joints is None:
        right_joints = np.zeros(7, dtype=np.float32)
    else:
        right_joints = np.atleast_1d(right_joints)[:7]
    state_parts.append(right_joints)

    right_gripper = _get_component([
        "right_franka_gripper/right_franka/gripper_position",
        "right_franka_gripper",
        "right_franka/gripper_position",
        "right/franka/gripper_position",
    ])
    if right_gripper is None:
        right_gripper = np.zeros(1, dtype=np.float32)
    else:
        right_gripper = np.atleast_1d(right_gripper)[:1]
    state_parts.append(right_gripper)

    left_ee_pose = _get_component([
        "left_franka_eef/left_franka/eef_pose",
        "left_franka_eef",
        "left_franka/eef_pose",
        "left/franka/eef_pose",
    ])
    if left_ee_pose is None:
        left_ee_pose = np.zeros(6, dtype=np.float32)
    else:
        left_ee_pose = pose_to_6d(left_ee_pose)
    state_parts.append(left_ee_pose)

    right_ee_pose = _get_component([
        "right_franka_eef/right_franka/eef_pose",
        "right_franka_eef",
        "right_franka/eef_pose",
        "right/franka/eef_pose",
    ])
    if right_ee_pose is None:
        right_ee_pose = np.zeros(6, dtype=np.float32)
    else:
        right_ee_pose = pose_to_6d(right_ee_pose)
    state_parts.append(right_ee_pose)

    return np.concatenate(state_parts, axis=-1).astype(np.float32)


def combine_franka_dual_arm_action(
    components: Dict[str, np.ndarray],
    next_state_components: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """
    Custom action combination for Franka dual-arm robot.
    
    This function defines the specific order and structure of the action vector
    for Franka dual-arm setup.
    
    Action structure:
    - Left arm joint commands (7 DOF)
    - Right arm joint commands (7 DOF)
    - Left end-effector pose commands (6D: x, y, z, roll, pitch, yaw)
    - Right end-effector pose commands (6D: x, y, z, roll, pitch, yaw)
    - Left gripper command (1D)
    - Right gripper command (1D)
    
    Total: 7 + 7 + 6 + 6 + 1 + 1 = 28 dimensions
    
    Args:
        components: Dictionary mapping component paths to numpy arrays
        
    Returns:
        Combined action vector with consistent ordering
    """
    def _get_component(src: Dict[str, np.ndarray], keys: List[str]) -> Optional[np.ndarray]:
        for key in keys:
            if key in src:
                return src[key]
        return None

    action_parts = []

    left_commands = _get_component(components, [
        "left_franka_joint_states/left_franka/joint_positions",
        "left_franka_joint_states",
        "left_franka/joint_positions",
        "left/franka/joint_positions",
    ])
    if left_commands is None:
        left_commands = np.zeros(7, dtype=np.float32)
    else:
        left_commands = np.atleast_1d(left_commands)[:7]
    action_parts.append(left_commands)

    left_gripper_cmd = _get_component(components, [
        "left_franka_gripper/left_franka/gripper_position",
        "left_franka_gripper",
        "left_franka/gripper_position",
        "left/franka/gripper_position",
    ])
    if left_gripper_cmd is None:
        left_gripper_cmd = np.zeros(1, dtype=np.float32)
    else:
        left_gripper_cmd = np.atleast_1d(left_gripper_cmd)[:1]
    action_parts.append(left_gripper_cmd)

    right_commands = _get_component(components, [
        "right_franka_joint_states/right_franka/joint_positions",
        "right_franka_joint_states",
        "right_franka/joint_positions",
        "right/franka/joint_positions",
    ])
    if right_commands is None:
        right_commands = np.zeros(7, dtype=np.float32)
    else:
        right_commands = np.atleast_1d(right_commands)[:7]
    action_parts.append(right_commands)

    right_gripper_cmd = _get_component(components, [
        "right_franka_gripper/right_franka/gripper_position",
        "right_franka_gripper",
        "right_franka/gripper_position",
        "right/franka/gripper_position",
    ])
    if right_gripper_cmd is None:
        right_gripper_cmd = np.zeros(1, dtype=np.float32)
    else:
        right_gripper_cmd = np.atleast_1d(right_gripper_cmd)[:1]
    action_parts.append(right_gripper_cmd)


    eef_source = next_state_components if next_state_components is not None else components
    left_ee_pose = _get_component(eef_source, [
        "left_franka_eef/left_franka/eef_pose",
        "left_franka_eef",
        "left_franka/eef_pose",
        "left/franka/eef_pose",
    ])
    if left_ee_pose is None:
        left_ee_pose = np.zeros(6, dtype=np.float32)
    else:
        left_ee_pose = pose_to_6d(left_ee_pose)
    action_parts.append(left_ee_pose)

    right_ee_pose = _get_component(eef_source, [
        "right_franka_eef/right_franka/eef_pose",
        "right_franka_eef",
        "right_franka/eef_pose",
        "right/franka/eef_pose",
    ])
    if right_ee_pose is None:
        right_ee_pose = np.zeros(6, dtype=np.float32)
    else:
        right_ee_pose = pose_to_6d(right_ee_pose)
    action_parts.append(right_ee_pose)



    return np.concatenate(action_parts, axis=-1).astype(np.float32)




def get_state_action_mapping() -> StateActionMapping:
    """
    Main function called by the converter to get custom mapping.
    
    Modify this function to return your specific robot's mapping.
    
    Returns:
        StateActionMapping configuration for Franka dual-arm robot
    """
    
    # Define which HDF5 paths contain state data
    # HDF5 结构是嵌套的：state/left_franka_joint_states/left_franka/joint_positions
    # 但 extract_components 不支持嵌套路径，所以我们需要使用实际的 dataset 路径
    # 根据 ros2_to_lerobot_converter.py，实际的 dataset 路径是：left_franka_joint_states/left_franka/joint_positions
    state_components = [
        # 嵌套路径格式（group/dataset）- 这是实际的数据路径
        "left_franka_joint_states/left_franka/joint_positions",
        "left_franka_gripper/left_franka/gripper_position",
        "right_franka_joint_states/right_franka/joint_positions",
        "right_franka_gripper/right_franka/gripper_position",
        "left_franka_eef/left_franka/eef_pose",
        "right_franka_eef/right_franka/eef_pose",
    ]
    
    # Define which HDF5 paths contain action data
    action_components = [
        # 嵌套路径格式（group/dataset）- 这是实际的数据路径
        "left_franka_joint_states/left_franka/joint_positions",
        "left_franka_gripper/left_franka/gripper_position",
        "right_franka_joint_states/right_franka/joint_positions",
        "right_franka_gripper/right_franka/gripper_position",
        "left_franka_eef/left_franka/eef_pose",
        "right_franka_eef/right_franka/eef_pose",
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
        state_combine_fn=combine_franka_dual_arm_state,
        action_combine_fn=combine_franka_dual_arm_action,
        normalize=True,
        state_stats=state_stats,
        action_stats=action_stats
    )


