#!/usr/bin/env python3
"""
Custom Message Processors for Franka Dual-Arm Robot

Configuration is embedded in the processor file, eliminating the need for separate YAML files.
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


def _infer_topic_from_stack(max_depth: int = 6) -> str | None:
    """Best-effort inference of ROS topic from call stack frames."""
    try:
        frame = sys._getframe(1)
    except Exception:
        return None

    depth = 0
    while frame is not None and depth < max_depth:
        if "conn" in frame.f_locals:
            conn = frame.f_locals.get("conn")
            topic = getattr(conn, "topic", None)
            if isinstance(topic, str) and topic:
                return topic
        if "topic" in frame.f_locals:
            topic = frame.f_locals.get("topic")
            if isinstance(topic, str) and topic:
                return topic
        frame = frame.f_back
        depth += 1

    return None


class ConfigProvider(MessageProcessor):
    """
    Provides converter configuration - replaces config.yaml
    """
    
    def __init__(self):
        """Initialize with Franka dual-arm robot configuration."""
        self._config = self._create_config()
    
    def _create_config(self) -> ConverterConfig:
        """Create converter configuration for Franka dual-arm robot."""
        config = ConverterConfig()
        
        # Camera configuration
        config.cameras = [
            CameraConfig(
                camera_id="camera_head",
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
                camera_id="camera_left",
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
                camera_id="camera_right",
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
        config.robot_state = RobotStateConfig(
            topics=[
                TopicConfig(
                    name="/left/franka/joint_states_desired",
                    type="sensor_msgs/msg/JointState",
                    frequency=100.0
                ),
                TopicConfig(
                    name="/right/franka/joint_states_desired",
                    type="sensor_msgs/msg/JointState",
                    frequency=100.0
                ),
                TopicConfig(
                    name="/left/franka/eef_pose",
                    type="geometry_msgs/msg/PoseStamped",
                    frequency=100.0
                ),
                TopicConfig(
                    name="/right/franka/eef_pose",
                    type="geometry_msgs/msg/PoseStamped",
                    frequency=100.0
                ),
                TopicConfig(
                    name="/left/franka/gripper_state",
                    type="sensor_msgs/msg/JointState",
                    frequency=100.0
                ),
                TopicConfig(
                    name="/right/franka/gripper_state",
                    type="sensor_msgs/msg/JointState",
                    frequency=100.0
                )
            ]
        )
        
        # Synchronization settings
        config.sync_tolerance_ms = 50.0
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
        
        For standard ROS2 messages, this is usually not needed as they are
        already registered in the typestore.
        """
        # Standard ROS2 messages should already be registered
        # If you have custom messages, register them here
        pass
    
    def process(self, msg: Any, timestamp: int) -> Dict[str, Any]:
        """Not used - this is a config provider only."""
        return {}
    
    def get_state_action_mapping(self) -> Tuple[List[str], List[str]]:
        """Not used - this is a config provider only."""
        return [], []


class FrankaJointStateProcessor(MessageProcessor):
    """Processor for sensor_msgs/msg/JointState messages from Franka robot."""
    
    def __init__(self, arm_side: str = None):
        """
        Initialize processor for a specific arm side.
        
        Args:
            arm_side: 'left' or 'right', or None for auto-detection
        """
        self.arm_side = arm_side
        # Franka robot has 7 joints (excluding gripper)
        # Common joint name patterns for Franka
        self.joint_name_patterns = [
            'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
            'panda_joint5', 'panda_joint6', 'panda_joint7',
            # Alternative naming
            'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7'
        ]
        # Explicit action joint patterns (e.g., fr3_exp_joint1..7)
        self.action_joint_name_patterns = [
            'fr3_exp_joint1', 'fr3_exp_joint2', 'fr3_exp_joint3', 'fr3_exp_joint4',
            'fr3_exp_joint5', 'fr3_exp_joint6', 'fr3_exp_joint7'
        ]
        # Explicit state joint patterns (e.g., fr3_joint1..7)
        self.state_joint_name_patterns = [
            'fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4',
            'fr3_joint5', 'fr3_joint6', 'fr3_joint7'
        ]
    
    def _detect_arm_side(self, msg: Any = None, topic: str = None) -> str:
        """Detect arm side from topic name or message content (UR-style strict match)."""
        if self.arm_side:
            return self.arm_side

        if not topic:
            topic = _infer_topic_from_stack()

        if topic:
            topic_l = topic.lower()
            if "/left/" in topic_l or "left" in topic_l:
                return "left"
            if "/right/" in topic_l or "right" in topic_l:
                return "right"

        def _tokens(s: str) -> set:
            import re
            return set([t for t in re.split(r"[^a-z0-9]+", s.lower()) if t])

        if msg is not None:
            if hasattr(msg, "name") and msg.name:
                name_tokens = set()
                for n in msg.name:
                    name_tokens |= _tokens(str(n))
                if "left" in name_tokens:
                    return "left"
                if "right" in name_tokens:
                    return "right"
            if hasattr(msg, "header") and hasattr(msg.header, "frame_id"):
                frame_tokens = _tokens(str(msg.header.frame_id))
                if "left" in frame_tokens:
                    return "left"
                if "right" in frame_tokens:
                    return "right"

        return "left"
    
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
        
        # Extract joint positions
        if hasattr(msg, 'position') and msg.position is not None and len(msg.position) > 0:
            if hasattr(msg, 'name'):
                if isinstance(msg.name, (list, tuple)):
                    joint_names_list = list(msg.name)
                elif msg.name is None:
                    joint_names_list = []
                else:
                    joint_names_list = [msg.name]
            else:
                joint_names_list = []

            positions = list(msg.position)

            # Handle gripper joint names if present in JointState
            gripper_indices = [
                i for i, n in enumerate(joint_names_list)
                if "gripper" in str(n).lower() and i < len(positions)
            ]
            if gripper_indices:
                state_idx = gripper_indices[0]
                data['state'][f'{arm_side}_franka/gripper_position'] = np.array(
                    [float(positions[state_idx])], dtype=np.float32
                )
                if len(gripper_indices) > 1:
                    action_idx = gripper_indices[1]
                    data['action'][f'{arm_side}_franka/gripper_position'] = np.array(
                        [float(positions[action_idx])], dtype=np.float32
                    )

            state_positions: List[float] = []
            action_positions: List[float] = []

            # 1) Try to extract by explicit name patterns (fr3_joint* / fr3_exp_joint*)
            found_state = {}
            found_action = {}
            for i, joint_name in enumerate(joint_names_list):
                if i >= len(positions):
                    break
                joint_name_lower = str(joint_name).lower()
                for pattern_idx, pattern in enumerate(self.state_joint_name_patterns):
                    if pattern in joint_name_lower:
                        if pattern_idx not in found_state:
                            found_state[pattern_idx] = float(positions[i])
                        break
                for pattern_idx, pattern in enumerate(self.action_joint_name_patterns):
                    if pattern in joint_name_lower:
                        if pattern_idx not in found_action:
                            found_action[pattern_idx] = float(positions[i])
                        break

            if len(found_state) > 0:
                state_positions = [found_state.get(i, 0.0) for i in range(7)]
            if len(found_action) > 0:
                action_positions = [found_action.get(i, 0.0) for i in range(7)]

            # 2) Fallback: use generic patterns for state (panda_joint*, joint*)
            if len(state_positions) == 0:
                found_joints = {}
                for i, joint_name in enumerate(joint_names_list):
                    if i >= len(positions):
                        break
                    for pattern_idx, pattern in enumerate(self.joint_name_patterns[:7]):
                        if pattern in str(joint_name).lower():
                            if pattern_idx not in found_joints:
                                found_joints[pattern_idx] = float(positions[i])
                                break
                if len(found_joints) > 0:
                    state_positions = [found_joints.get(i, 0.0) for i in range(7)]

            # 3) Fallback: if positions length >= 14, split first 7 / last 7
            if len(state_positions) == 0 and len(positions) >= 7:
                state_positions = [float(positions[i]) for i in range(7)]
            if len(action_positions) == 0:
                if len(positions) >= 14:
                    action_positions = [float(positions[i]) for i in range(7, 14)]
                elif len(positions) >= 7:
                    # If no explicit action available, mirror state for backward compatibility
                    action_positions = [float(positions[i]) for i in range(7)]

            # 4) Pad if needed
            if len(state_positions) > 0:
                state_padded = np.zeros(7, dtype=np.float32)
                state_padded[:len(state_positions)] = state_positions
                data['state'][f'{arm_side}_franka/joint_positions'] = state_padded

            if len(action_positions) > 0:
                action_padded = np.zeros(7, dtype=np.float32)
                action_padded[:len(action_positions)] = action_positions
                data['action'][f'{arm_side}_franka/joint_positions'] = action_padded
        
        return data
    
    def get_state_action_mapping(self) -> Tuple[List[str], List[str]]:
        """Return the mapping of data fields to state and action."""
        # Return generic fields, will be resolved based on arm_side during processing
        state_fields = ['left_franka/joint_positions', 'right_franka/joint_positions']
        action_fields = ['left_franka/joint_positions', 'right_franka/joint_positions']
        return state_fields, action_fields


class FrankaGripperStateProcessor(MessageProcessor):
    """Processor for sensor_msgs/msg/JointState messages from Franka gripper."""
    
    def __init__(self, arm_side: str = None):
        """
        Initialize processor for a specific arm side.
        
        Args:
            arm_side: 'left' or 'right', or None for auto-detection
        """
        self.arm_side = arm_side
    
    def _detect_arm_side(self, msg: Any = None, topic: str = None) -> str:
        """Detect arm side from topic name or message content (UR-style strict match)."""
        if self.arm_side:
            return self.arm_side

        if not topic:
            topic = _infer_topic_from_stack()

        if topic:
            topic_l = topic.lower()
            if "/left/" in topic_l or "left" in topic_l:
                return "left"
            if "/right/" in topic_l or "right" in topic_l:
                return "right"

        def _tokens(s: str) -> set:
            import re
            return set([t for t in re.split(r"[^a-z0-9]+", s.lower()) if t])

        if msg is not None:
            if hasattr(msg, "name") and msg.name:
                name_tokens = set()
                for n in msg.name:
                    name_tokens |= _tokens(str(n))
                if "left" in name_tokens:
                    return "left"
                if "right" in name_tokens:
                    return "right"
            if hasattr(msg, "header") and hasattr(msg.header, "frame_id"):
                frame_tokens = _tokens(str(msg.header.frame_id))
                if "left" in frame_tokens:
                    return "left"
                if "right" in frame_tokens:
                    return "right"

        return "left"
    
    def process(self, msg: Any, timestamp: int, topic: str = None) -> Dict[str, Any]:
        """Process a JointState message from gripper."""
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
        
        # Extract gripper position
        # state: position[0], action: position[1] (if available)
        if hasattr(msg, 'position') and msg.position is not None and len(msg.position) > 0:
            gripper_state = float(msg.position[0])
            data['state'][f'{arm_side}_franka/gripper_position'] = np.array(
                [gripper_state], dtype=np.float32
            )
            if len(msg.position) > 1:
                gripper_action = float(msg.position[1])
                data['action'][f'{arm_side}_franka/gripper_position'] = np.array(
                    [gripper_action], dtype=np.float32
                )
        
        return data
    
    def get_state_action_mapping(self) -> Tuple[List[str], List[str]]:
        """Return the mapping of data fields to state and action."""
        state_fields = ['left_franka/gripper_position', 'right_franka/gripper_position']
        action_fields = ['left_franka/gripper_position', 'right_franka/gripper_position']
        return state_fields, action_fields


class FrankaEEFPoseProcessor(MessageProcessor):
    """Processor for geometry_msgs/msg/PoseStamped messages from Franka end-effector."""
    
    def __init__(self, arm_side: str = None):
        """
        Initialize processor for a specific arm side.
        
        Args:
            arm_side: 'left' or 'right', or None for auto-detection
        """
        self.arm_side = arm_side
    
    def _detect_arm_side(self, msg: Any = None, topic: str = None) -> str:
        """Detect arm side from topic name or message content (UR-style strict match)."""
        if self.arm_side:
            return self.arm_side

        if not topic:
            topic = _infer_topic_from_stack()

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

        return "left"
    
    def _quaternion_to_euler(self, qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        """
        Convert quaternion to Euler angles (roll, pitch, yaw) - ZYX order.
        
        Returns:
            numpy array of shape (3,) containing [roll, pitch, yaw]
        """
        import math

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw], dtype=np.float32)
    
    def pose_to_array(self, pose: Any) -> np.ndarray:
        """
        Convert Pose message to 6D numpy array [x, y, z, roll, pitch, yaw].
        
        Args:
            pose: geometry_msgs/msg/Pose object
            
        Returns:
            numpy array of shape (6,)
        """
        if hasattr(pose, 'position') and hasattr(pose, 'orientation'):
            pos = pose.position
            ori = pose.orientation
            position = np.array(
                [float(pos.x), float(pos.y), float(pos.z)],
                dtype=np.float32,
            )
            euler = self._quaternion_to_euler(
                float(ori.x), float(ori.y), float(ori.z), float(ori.w)
            )
            return np.concatenate([position, euler], dtype=np.float32)
        return np.zeros(6, dtype=np.float32)
    
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
        
        # Extract pose
        if hasattr(msg, 'pose'):
            pose_array = self.pose_to_array(msg.pose)
            data['state'][f'{arm_side}_franka/eef_pose'] = pose_array
        
        return data
    
    def get_state_action_mapping(self) -> Tuple[List[str], List[str]]:
        """Return the mapping of data fields to state and action."""
        state_fields = ['left_franka/eef_pose', 'right_franka/eef_pose']
        action_fields = ['left_franka/eef_pose', 'right_franka/eef_pose']
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
        
        # Topic-specific processors (will be matched by topic name in converter)
        '/left/franka/joint_states_desired': FrankaJointStateProcessor('left'),
        '/right/franka/joint_states_desired': FrankaJointStateProcessor('right'),
        '/left/franka/eef_pose': FrankaEEFPoseProcessor('left'),
        '/right/franka/eef_pose': FrankaEEFPoseProcessor('right'),
        '/left/franka/gripper_state': FrankaGripperStateProcessor('left'),
        '/right/franka/gripper_state': FrankaGripperStateProcessor('right'),
        
        # Generic processors (fallback, matched by message type)
        'JointState': FrankaJointStateProcessor(),  # Will auto-detect from topic
        'PoseStamped': FrankaEEFPoseProcessor(),    # Will auto-detect from topic
    }
    
    return processors

