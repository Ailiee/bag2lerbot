#!/usr/bin/env python3
"""
Custom Message Processors with Integrated Configuration

Configuration is now embedded in the processor file, eliminating the need for separate YAML files.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import logging
from scipy.spatial.transform import Rotation as R

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ros2_to_lerobot_converter import (
    MessageProcessor, 
    ConverterConfig,
    TopicConfig,
    CameraConfig,
    RobotStateConfig
)


class ConfigProvider(MessageProcessor):
    """
    Provides converter configuration - replaces config.yaml
    """
    
    def __init__(self):
        """Initialize with your robot configuration."""
        self._config = self._create_config()
    
    def _create_config(self) -> ConverterConfig:
        """Create converter configuration."""
        config = ConverterConfig()
        
        # Camera configuration
        config.cameras = [
            CameraConfig(
                camera_id="camera_h",
                topics=[
                    TopicConfig(
                        name="/head/color/image_raw/compressed",
                        type="sensor_msgs/msg/CompressedImage",
                        frequency=30.0,
                        compressed=True,
                        modality="rgb"
                    )
                ]
            ),
            CameraConfig(
                camera_id="camera_l",
                topics=[
                    TopicConfig(
                        name="/left/color/image_raw/compressed",
                        type="sensor_msgs/msg/CompressedImage",
                        frequency=30.0,
                        compressed=True,
                        modality="rgb"
                    )
                ]
            ),
            CameraConfig(
                camera_id="camera_r",
                topics=[
                    TopicConfig(
                        name="/right/color/image_raw/compressed",
                        type="sensor_msgs/msg/CompressedImage",
                        frequency=30.0,
                        compressed=True,
                        modality="rgb"
                    )
                ]
            )
        ]
        
        # Robot state configuration
        # Note: actual_joints contains 7 values (6 joints + 1 gripper)
        # actual_tcp_pose contains end-effector pose (6D: x, y, z, rx, ry, rz)
        config.robot_state = RobotStateConfig(
            topics=[
                TopicConfig(
                    name="/left/ur5e/actual_joints",
                    type="sensor_msgs/msg/JointState",
                    frequency=500.0
                ),
                TopicConfig(
                    name="/right/ur5e/actual_joints",
                    type="sensor_msgs/msg/JointState",
                    frequency=500.0
                ),
                TopicConfig(
                    name="/left/ur5e/actual_tcp_pose",
                    type="geometry_msgs/msg/PoseStamped",
                    frequency=500.0
                ),
                TopicConfig(
                    name="/right/ur5e/actual_tcp_pose",
                    type="geometry_msgs/msg/PoseStamped",
                    frequency=500.0
                ),
            ]
        )
        
        # Synchronization settings
        config.sync_tolerance_ms = 5000.0
        config.sync_reference = None  # Auto-select
        
        # Output settings
        config.chunk_size = 1000
        config.compression = 'gzip'
        config.compression_opts = 4
        
        return config
    
    def get_converter_config(self) -> ConverterConfig:
        """Return the converter configuration."""
        return self._config
    
    def register_custom_types(self, reader: Any, typestore: Any) -> None:
        """
        Register custom ROS2 message types.
        
        For standard ROS2 messages (sensor_msgs/msg/JointState, geometry_msgs/msg/PoseStamped),
        this is usually not needed as they are already registered in the typestore.
        """
        # Standard ROS2 messages should already be registered
        # If you have custom messages, register them here
        from pathlib import Path
        from rosbags.typesys import get_types_from_msg
        
        # Only register custom message types if needed
        msg_files = {
            # Add custom message types here if you have them
        }
        
        add_types = {}
        for msg_name, msg_path in msg_files.items():
            msg_file = Path(msg_path)
            if msg_file.exists():
                try:
                    msg_text = msg_file.read_text()
                    add_types.update(get_types_from_msg(msg_text, name=msg_name))
                except Exception:
                    pass
        
        # Register custom types if any
        if add_types:
            try:
                typestore.register(add_types)
            except Exception:
                pass
    
    def process(self, msg: Any, timestamp: int) -> Dict[str, Any]:
        """Not used - this is a config provider only."""
        return {}
    
    def get_state_action_mapping(self) -> Tuple[List[str], List[str]]:
        """Not used - this is a config provider only."""
        return [], []


class URJointStateProcessor(MessageProcessor):
    """Processor for sensor_msgs/msg/JointState messages from UR5e robot."""
    
    def __init__(self, arm_side: str = None):
        """
        Initialize processor for a specific arm side.
        
        Args:
            arm_side: 'left' or 'right', or None for auto-detection
        """
        self.arm_side = arm_side
    
    def _detect_arm_side(self, msg: Any = None, topic: str = None) -> str:
        """Detect arm side from topic name or message content (strict match)."""
        if self.arm_side:
            return self.arm_side

        # Prefer topic when available
        if topic:
            topic_l = topic.lower()
            if "/left/" in topic_l or "left" in topic_l:
                return "left"
            if "/right/" in topic_l or "right" in topic_l:
                return "right"

        # Helper: token-based match to avoid false positives
        def _tokens(s: str) -> set:
            import re
            return set([t for t in re.split(r"[^a-z0-9]+", s.lower()) if t])

        if msg is not None:
            # JointState name list
            if hasattr(msg, "name") and msg.name:
                name_tokens = set()
                for n in msg.name:
                    name_tokens |= _tokens(str(n))
                if "left" in name_tokens:
                    return "left"
                if "right" in name_tokens:
                    return "right"
            # Header frame_id
            if hasattr(msg, "header") and hasattr(msg.header, "frame_id"):
                frame_tokens = _tokens(str(msg.header.frame_id))
                if "left" in frame_tokens:
                    return "left"
                if "right" in frame_tokens:
                    return "right"

        return "left"  # Default to left
    
    def process(self, msg: Any, timestamp: int, topic: str = None) -> Dict[str, Any]:
        """Process a JointState message."""
        arm_side = self._detect_arm_side(msg=msg, topic=topic)
        
        data = {
            'timestamp': timestamp,
            'state': {},
            'action': {}
        }
        
        # Extract timestamp from message header if available
        if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
            stamp = msg.header.stamp
            if hasattr(stamp, 'sec') and hasattr(stamp, 'nanosec'):
                data['timestamp'] = stamp.sec * 1_000_000_000 + stamp.nanosec
        
        # Extract joint positions (contains joints + gripper)
        # UR5e actual_joints contains 7 values: 6 joints + 1 gripper
        if hasattr(msg, 'position') and msg.position is not None and len(msg.position) > 0:
            try:
                positions = np.array(msg.position, dtype=np.float32)
                
                if len(positions) >= 14:
                    # First 6 are joints
                    state_joint_positions = positions[:6]
                    # 7th is gripper
                    state_gripper_position = positions[6:7]

                    action_joint_positions = positions[7:13]
                    action_gripper_position = positions[13:14]
                    
                    data['state'][f'{arm_side}_ur5e/joint_positions'] = state_joint_positions
                    data['state'][f'{arm_side}_ur5e/gripper_position'] = state_gripper_position
                    data['action'][f'{arm_side}_ur5e/joint_positions'] = action_joint_positions
                    data['action'][f'{arm_side}_ur5e/gripper_position'] = action_gripper_position

            except Exception as e:
                # If extraction fails, log the error but continue
                # This prevents empty data points from being added to streams
                logging.warning(f"Failed to extract joint positions from {topic}: {e}")
                pass
        
        # Only return data if we have valid state/action data
        # This prevents empty data points from being added to streams
        if not data['state'] and not data['action']:
            return None
        
        return data

    def get_state_action_mapping(self) -> Tuple[List[str], List[str]]:
        """Return the mapping of data fields to state and action."""
        # Return generic fields, will be resolved based on arm_side during processing
        state_fields = [
            'left_ur5e/joint_positions', 'left_ur5e/gripper_position',
            'right_ur5e/joint_positions', 'right_ur5e/gripper_position'
        ]
        action_fields = [
            'left_ur5e/joint_positions', 'left_ur5e/gripper_position',
            'right_ur5e/joint_positions', 'right_ur5e/gripper_position'
        ]
        return state_fields, action_fields


class UREEFPoseProcessor(MessageProcessor):
    """Processor for geometry_msgs/msg/PoseStamped messages from UR5e end-effector.
    
    Extracts end-effector pose (x, y, z, rx, ry, rz) from quaternion.
    """
    
    def __init__(self, arm_side: str = None):
        """
        Initialize processor for a specific arm side.
        
        Args:
            arm_side: 'left' or 'right', or None for auto-detection
        """
        self.arm_side = arm_side
    
    def _detect_arm_side(self, msg: Any = None, topic: str = None) -> str:
        """Detect arm side from topic name or message content (strict match)."""
        if self.arm_side:
            return self.arm_side

        if topic:
            topic_l = topic.lower()
            if "/left/" in topic_l or "left" in topic_l:
                return "left"
            if "/right/" in topic_l or "right" in topic_l:
                return "right"

        def _tokens(s: str) -> set:
            import re
            return set([t for t in re.split(r"[^a-z0-9]+", s.lower()) if t])

        if msg is not None and hasattr(msg, "header") and hasattr(msg.header, "frame_id"):
            frame_tokens = _tokens(str(msg.header.frame_id))
            if "left" in frame_tokens:
                return "left"
            if "right" in frame_tokens:
                return "right"

        return "left"  # Default to left
    
    def process(self, msg: Any, timestamp: int, topic: str = None) -> Dict[str, Any]:
        """Process a PoseStamped message."""
        arm_side = self._detect_arm_side(msg=msg, topic=topic)
        
        data = {
            'timestamp': timestamp,
            'state': {},
            'action': {}
        }
        
        # Extract timestamp from message header if available
        if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
            stamp = msg.header.stamp
            if hasattr(stamp, 'sec') and hasattr(stamp, 'nanosec'):
                data['timestamp'] = stamp.sec * 1_000_000_000 + stamp.nanosec
        
        # Extract position (x, y, z)
        try:
            if hasattr(msg, 'pose') and hasattr(msg.pose, 'position'):
                pos = msg.pose.position
                position = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
                
                # Extract orientation and convert quaternion to rotation vector
                if hasattr(msg.pose, 'orientation'):
                    orient = msg.pose.orientation
                    # Convert quaternion (qx, qy, qz, qw) to rotation vector (rx, ry, rz)
                    rot_vec = self._quaternion_to_euler(orient.x, orient.y, orient.z, orient.w)
                    
                    # Combine position and orientation (6D: x, y, z, rx, ry, rz)
                    eef_pose = np.concatenate([position, rot_vec], dtype=np.float32)
                    data['state'][f'{arm_side}_ur5e/eef_pose'] = eef_pose
        except Exception as e:
            # If extraction fails, return empty data (will be filtered out)
            pass
        
        # Only return data if we have valid state/action data
        # This prevents empty data points from being added to streams
        if not data['state'] and not data['action']:
            return None
        
        return data
    
    def _quaternion_to_euler(self, qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        """Convert quaternion to rotation vector (rx, ry, rz)."""
        quat = np.array([qx, qy, qz, qw], dtype=np.float64)
        r = R.from_quat(quat)       # 四元数 -> 旋转
        rot_vec = r.as_rotvec()     # 旋转 -> 旋转向量 [rx, ry, rz]
        return rot_vec.astype(np.float32)

    def get_state_action_mapping(self) -> Tuple[List[str], List[str]]:
        """Return the mapping of data fields to state and action."""
        state_fields = ['left_ur5e/eef_pose', 'right_ur5e/eef_pose']
        action_fields = ['left_ur5e/eef_pose', 'right_ur5e/eef_pose']
        return state_fields, action_fields


def get_message_processors() -> Dict[str, MessageProcessor]:
    """
    Factory function to create and return message processors.
    
    Returns:
        Dictionary mapping message type names or topic patterns to processor instances
    """
    processors = {
        # Config provider (REQUIRED - provides configuration)
        'ConfigProvider': ConfigProvider(),
        
        '/left/ur5e/actual_joints': URJointStateProcessor('left'),
        '/right/ur5e/actual_joints': URJointStateProcessor('right'),

        '/left/ur5e/actual_tcp_pose': UREEFPoseProcessor('left'),
        '/right/ur5e/actual_tcp_pose': UREEFPoseProcessor('right'),

        'JointState': URJointStateProcessor(),  # Will auto-detect from topic
        'PoseStamped': UREEFPoseProcessor(),    # Will auto-detect from topic
    }
    
    return processors

