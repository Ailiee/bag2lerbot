#!/usr/bin/env python3
"""
Custom Message Processors Template for ROS2 to LeRobot Converter

This template provides examples of how to implement custom message processors
for robot-specific ROS message types. Users should modify this file according
to their specific robot's message definitions.

Author: Your Name
Date: 2024
"""

from typing import Any, Dict, List, Tuple
import numpy as np

# Import the base MessageProcessor class
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ros2_to_lerobot_converter import MessageProcessor

class UrStatesProcessor(MessageProcessor):
    """
    Processor for device_interfaces/msg/UrStates messages.
    
    This processor extracts end-effector pose and joint states.
    """
    
    def process(self, msg: Any, timestamp: int) -> Dict[str, Any]:
        """
        Process a UrStates message.
        
        Args:
            msg: device_interfaces/msg/UrStates message
            timestamp: Message timestamp in nanoseconds
            
        Returns:
            Dictionary containing processed robot state
        """
        data = {
            'timestamp': timestamp,
            'state': {},
            'action': {}
        }
        
        # Extract end-effector pose (position and orientation)
        if hasattr(msg, 'eef_pos') and hasattr(msg.eef_pos, 'positions'):
            eef_positions = msg.eef_pos.positions
            if len(eef_positions) == 6:
                data['state']['end_eff'] = np.array(eef_positions[:6], dtype=np.float32)

        # Extract joint angles
        if hasattr(msg, 'joints') and hasattr(msg.joints, 'angles'):
            data['state']['joint_positions'] = np.array(msg.joints.angles, dtype=np.float32)
            
        return data

    def get_state_action_mapping(self) -> Tuple[List[str], List[str]]:
        """
        Return the mapping of data fields to state and action.
        """
        state_fields = [
            'end_eff',
            'joint_position'
        ]
        action_fields = []
        return state_fields, action_fields


class GripperStateProcessor(MessageProcessor):
    """
    Processor for dh_gripper_driver/msg/GripperState messages.
    
    This processor extracts gripper state and action commands.
    """
    
    def process(self, msg: Any, timestamp: int) -> Dict[str, Any]:
        """
        Process a GripperState message.
        
        Args:
            msg: dh_gripper_driver/msg/GripperState message
            timestamp: Message timestamp in nanoseconds
            
        Returns:
            Dictionary containing processed gripper data
        """
        data = {
            'timestamp': timestamp,
            'state': {},
            'action': {}
        }
        
        # State information
        if hasattr(msg, 'position'):
            data['state']['gripper_position'] = float(msg.position)

        # Action information
        if hasattr(msg, 'target_position'):
            data['action']['gripper_position'] = float(msg.target_position)

        return data

    def get_state_action_mapping(self) -> Tuple[List[str], List[str]]:
        """
        Return the mapping of data fields to state and action.
        """
        state_fields = [
            'gripper_position',
        ]
        action_fields = [
            'gripper_position',
        ]
        return state_fields, action_fields


def get_message_processors() -> Dict[str, MessageProcessor]:
    """
    Factory function to create and return message processors.
    
    This function is called by the converter to get custom processors.
    Modify this function to return processors for your robot's message types.
    
    Returns:
        Dictionary mapping message type names to processor instances
    """
    processors = {   
        # Custom processors for UR and Gripper
        'UrStates': UrStatesProcessor(),
        'GripperState': GripperStateProcessor(),
    }
    
    return processors


def transform_coordinate_frame(
    position: np.ndarray,
    from_frame: str = 'ros',
    to_frame: str = 'lerobot'
) -> np.ndarray:
    """
    Transform position between coordinate frames.
    
    Args:
        position: 3D position vector
        from_frame: Source coordinate frame
        to_frame: Target coordinate frame
        
    Returns:
        Transformed position vector
    """
    if from_frame == to_frame:
        return position
    
    # Example transformation (modify based on your needs)
    if from_frame == 'ros' and to_frame == 'lerobot':
        # ROS typically uses: x-forward, y-left, z-up
        # LeRobot might use different convention
        # This is just an example - adjust based on actual conventions
        x, y, z = position
        return np.array([x, -y, z], dtype=np.float32)
    
    return position


def normalize_joint_positions(
    positions: np.ndarray,
    joint_limits: Dict[str, Tuple[float, float]]
) -> np.ndarray:
    """
    Normalize joint positions to [-1, 1] range based on joint limits.
    
    Args:
        positions: Array of joint positions
        joint_limits: Dictionary of joint_name -> (min, max) limits
        
    Returns:
        Normalized joint positions
    """
    normalized = np.zeros_like(positions)
    
    for i, (joint_name, (min_val, max_val)) in enumerate(joint_limits.items()):
        if i < len(positions):
            # Normalize to [-1, 1]
            normalized[i] = 2 * (positions[i] - min_val) / (max_val - min_val) - 1
            normalized[i] = np.clip(normalized[i], -1, 1)
    
    return normalized

