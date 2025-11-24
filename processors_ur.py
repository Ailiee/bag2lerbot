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

