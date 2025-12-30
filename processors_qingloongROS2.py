#!/usr/bin/env python3
"""
Custom Message Processors with Integrated Configuration

Configuration is now embedded in the processor file, eliminating the need for separate YAML files.
"""

from typing import Any, Dict, List, Tuple
import numpy as np

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
                        type="sensor_msgs/CompressedImage",
                        frequency=60.0,
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
                        type="sensor_msgs/CompressedImage",
                        frequency=60.0,
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
                        type="sensor_msgs/CompressedImage",
                        frequency=60.0,
                        compressed=True,
                        modality="rgb"
                    )
                ]
            )
        ]
        
        # Robot state configuration
        config.robot_state = RobotStateConfig(
            topics=[
                TopicConfig(
                    name="/driver_pvt",
                    type="driver_pvt/msg/DriverPVT",
                    frequency=200.0
                ),
                TopicConfig(
                    name="/end_pos",
                    type="end_pos/msg/EndPos",
                    frequency=200.0
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
        
        Method 1: Load from .msg files (if you have source)
        Method 2: Load from bag (fallback)
        """
        from pathlib import Path
        from rosbags.typesys import get_types_from_msg
        
        # Method 1: Load from .msg files (优先)
        msg_files = {
            'driver_pvt/msg/Joint': '/workspace/openloong2lerobot/qinglong_msg/Joint.msg',
            'driver_pvt/msg/Limb': '/workspace/openloong2lerobot/qinglong_msg/Limb.msg',
            'driver_pvt/msg/DriverPVT': '/workspace/openloong2lerobot/qinglong_msg/DriverPVT.msg',
            'end_pos/msg/EndPos': '/workspace/openloong2lerobot/qinglong_msg/EndPos.msg',
            # 添加你的其他消息文件
            # 'driver_pvt/msg/Limb': 'src/driver_pvt/msg/Limb.msg',
            # 'driver_pvt/msg/Joint': 'src/driver_pvt/msg/Joint.msg',
        }
        
        add_types = {}
        for msg_name, msg_path in msg_files.items():
            msg_file = Path(msg_path)
            if msg_file.exists():
                try:
                    msg_text = msg_file.read_text()
                    add_types.update(get_types_from_msg(msg_text, name=msg_name))
                except Exception as e:
                    pass  # 如果文件不存在，使用方法2
        
        # 注册从文件加载的类型
        if add_types:
            typestore.register(add_types)
            # print(f"✓ Registered: {list(add_types.keys())}")

        # Method 2: Load from bag (fallback - 如果没有.msg文件)
        connections = reader.connections
        connections_list = connections.values() if isinstance(connections, dict) else connections
        
        for connection in connections_list:
            try:
                typestore.register(connection.msgdef)
            except Exception:
                pass  # 已经注册过
    
    def process(self, msg: Any, timestamp: int) -> Dict[str, Any]:
        """Not used - this is a config provider only."""
        return {}
    
    def get_state_action_mapping(self) -> Tuple[List[str], List[str]]:
        """Not used - this is a config provider only."""
        return [], []


class qingloongStatesProcessor(MessageProcessor):
    """Processor for device_interfaces/msg/UrStates messages."""
    
    def process(self, msg: Any, timestamp: int) -> Dict[str, Any]:
        """Process a UrStates message."""
        data = {
            'timestamp': timestamp,
            'state': {},
            'action': {}
        }
        # Extract joint pose
        q_pos = []
        q_pos_exp = []
        if hasattr(msg, 'limbs'):
            limb_list = msg.limbs
            for limb_idx in range(2,8):
                limb = limb_list[limb_idx]
                for joint in limb.joints:
                    q_pos.append(joint.q)
                    q_pos_exp.append(joint.q_exp)
            # 修正
            if q_pos[-1] > 90.0:
                q_pos[-1] = 0.0
            if q_pos[-2] > 90.0:
                q_pos[-2] = 0.0
            data['state']['q_pos'] = np.array(q_pos, dtype=np.float32)
            data['action']['q_pos'] = np.array(q_pos_exp, dtype=np.float32)
            
        return data

    def get_state_action_mapping(self) -> Tuple[List[str], List[str]]:
        """Return the mapping of data fields to state and action."""
        state_fields = ['q_pos']
        action_fields = ['q_pos']
        return state_fields, action_fields


class qingloongEEFProcessor(MessageProcessor):
    """Processor for dh_gripper_driver/msg/GripperState messages."""
    
    def process(self, msg: Any, timestamp: int) -> Dict[str, Any]:
        """Process a GripperState message."""
        data = {
            'timestamp': timestamp,
            'state': {},
            'action': {}
        }
        
        # State information
        if hasattr(msg, 'ee_pose_l') and hasattr(msg, 'ee_pose_r'):
            data['state']['eef'] = np.concatenate(
                (
                    np.asarray(msg.ee_pose_l[:6], dtype=np.float32),
                    np.asarray(msg.ee_pose_r[:6], dtype=np.float32),
                ),
                axis=0,
            )

        # Action information
        if hasattr(msg, 'ee_pose_l_exp') and hasattr(msg, 'ee_pose_r_exp'):
            data['action']['eef'] = np.concatenate(
                (
                    np.asarray(msg.ee_pose_l_exp[:6], dtype=np.float32),
                    np.asarray(msg.ee_pose_r_exp[:6], dtype=np.float32),
                ),
                axis=0,
                )

        return data

    def get_state_action_mapping(self) -> Tuple[List[str], List[str]]:
        """Return the mapping of data fields to state and action."""
        state_fields = ['eef']
        action_fields = ['eef']
        return state_fields, action_fields


def get_message_processors() -> Dict[str, MessageProcessor]:
    """
    Factory function to create and return message processors.
    
    Returns:
        Dictionary mapping message type names to processor instances
    """
    processors = {
        # Config provider (REQUIRED - provides configuration)
        'ConfigProvider': ConfigProvider(),
        
        # Custom processors for UR and Gripper
        'DriverPVT': qingloongStatesProcessor(),
        'EndPos': qingloongEEFProcessor(),
    }
    
    return processors

