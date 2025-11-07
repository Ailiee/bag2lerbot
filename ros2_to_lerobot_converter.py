#!/usr/bin/env python3
"""
Unified ROS2 to LeRobot 3.0 Dataset Converter Framework

This framework provides a flexible and extensible solution for converting ROS2 bag data
to LeRobot 3.0 dataset format with timestamp synchronization.

Author: LeRobot Conversion Framework
Date: 2024
"""

from __future__ import annotations

import argparse
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union

from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import h5py
import numpy as np
import rosbag2_py
import yaml
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage, Image
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize CV bridge for ROS image conversion
cv_bridge = CvBridge()


@dataclass
class TopicConfig:
    """Configuration for a single ROS topic."""
    name: str
    type: str
    frequency: float
    compressed: bool = False
    modality: Optional[str] = None  # 'rgb', 'depth', etc. for images
    
    def __post_init__(self):
        """Validate topic configuration."""
        if not self.name:
            raise ValueError("Topic name cannot be empty")
        if self.frequency <= 0:
            raise ValueError(f"Frequency must be positive, got {self.frequency}")


@dataclass
class CameraConfig:
    """Configuration for a camera sensor."""
    camera_id: str
    topics: List[TopicConfig]
    
    def __post_init__(self):
        """Validate camera configuration."""
        if not self.camera_id:
            raise ValueError("Camera ID cannot be empty")
        if not self.topics:
            raise ValueError(f"Camera {self.camera_id} must have at least one topic")


@dataclass
class RobotStateConfig:
    """Configuration for robot state topics."""
    topics: List[TopicConfig]
    state_fields: List[str] = field(default_factory=list)
    action_fields: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate robot state configuration."""
        if not self.topics:
            logger.warning("No robot state topics configured")


class ConverterConfig:
    """Main converter configuration loaded from YAML."""
    
    def __init__(self, config_path: Path):
        """Load and parse configuration from YAML file."""
        self.config_path = config_path
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Parse camera configurations
        self.cameras: List[CameraConfig] = []
        if 'cameras' in config_data:
            for cam_data in config_data['cameras']:
                topics = [
                    TopicConfig(
                        name=topic['name'],
                        type=topic['type'],
                        frequency=topic['frequency'],
                        compressed=topic.get('compressed', False),
                        modality=topic.get('modality', 'rgb')
                    )
                    for topic in cam_data['topics']
                ]
                self.cameras.append(
                    CameraConfig(
                        camera_id=cam_data['camera_id'],
                        topics=topics
                    )
                )
        
        # Parse robot state configuration
        robot_data = config_data.get('robot_state', {})
        state_topics = [
            TopicConfig(
                name=topic['name'],
                type=topic['type'],
                frequency=topic['frequency']
            )
            for topic in robot_data.get('topics', [])
        ]
        
        self.robot_state = RobotStateConfig(
            topics=state_topics,
            state_fields=robot_data.get('state_fields', []),
            action_fields=robot_data.get('action_fields', [])
        )
        
        # Synchronization settings
        sync_config = config_data.get('synchronization', {})
        self.sync_tolerance_ms = sync_config.get('tolerance_ms', 50)
        self.sync_reference = sync_config.get('reference_topic', None)
        
        # Output settings
        output_config = config_data.get('output', {})
        self.chunk_size = output_config.get('chunk_size', 1000)
        self.compression = output_config.get('compression', 'gzip')
        self.compression_opts = output_config.get('compression_opts', 4)
        
        logger.info(f"Loaded configuration from {config_path}")
        logger.info(f"  - {len(self.cameras)} camera(s) configured")
        logger.info(f"  - {len(self.robot_state.topics)} robot state topic(s) configured")
        logger.info(f"  - Sync tolerance: {self.sync_tolerance_ms}ms")


class MessageProcessor(ABC):
    """Abstract base class for processing specific ROS message types."""
    
    @abstractmethod
    def process(self, msg: Any, timestamp: int) -> Dict[str, Any]:
        """Process a ROS message and return extracted data.
        
        Args:
            msg: ROS message object
            timestamp: Message timestamp in nanoseconds
            
        Returns:
            Dictionary containing processed data
        """
        pass
    
    @abstractmethod
    def get_state_action_mapping(self) -> Tuple[List[str], List[str]]:
        """Return lists of field names that map to state and action.
        
        Returns:
            Tuple of (state_fields, action_fields)
        """
        pass


class ImageProcessor:
    """Process image messages from ROS topics."""
    
    def __init__(self, output_dir: Path, dataset_name: str):
        """Initialize image processor.
        
        Args:
            output_dir: Base directory to save processed images
            dataset_name: Name of the dataset (for organizing output)
        """
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_counters: Dict[str, int] = {}  # Track frame index for each camera stream
        
    def process_compressed_image(
        self, 
        msg: CompressedImage, 
        timestamp: int,
        camera_id: str,
        modality: str
    ) -> Dict[str, Any]:
        """Process a compressed image message.
        
        Args:
            msg: CompressedImage message
            timestamp: Message timestamp in nanoseconds
            camera_id: Camera identifier
            modality: Image modality (rgb, depth, etc.)
            
        Returns:
            Dictionary with image metadata
        """
        # Decode compressed image
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.warning(f"Failed to decode compressed image from {camera_id}/{modality}")
            return {}
        
        # Convert BGR to RGB if needed
        if modality == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Save image
        image_path = self._save_image(img, timestamp, camera_id, modality)
        
        return {
            'timestamp': timestamp,
            'path': str(image_path),
            'shape': img.shape,
            'dtype': str(img.dtype)
        }
    
    def process_raw_image(
        self,
        msg: Image,
        timestamp: int,
        camera_id: str,
        modality: str
    ) -> Dict[str, Any]:
        """Process a raw image message.
        
        Args:
            msg: Image message
            timestamp: Message timestamp in nanoseconds
            camera_id: Camera identifier
            modality: Image modality (rgb, depth, etc.)
            
        Returns:
            Dictionary with image metadata
        """
        # Convert ROS image to numpy array
        try:
            img = cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            logger.warning(f"Failed to convert image from {camera_id}/{modality}: {e}")
            return {}
        
        # Save image
        image_path = self._save_image(img, timestamp, camera_id, modality)
        
        return {
            'timestamp': timestamp,
            'path': str(image_path),
            'shape': img.shape,
            'dtype': str(img.dtype)
        }
    
    def _save_image(
        self,
        img: np.ndarray,
        timestamp: int,
        camera_id: str,
        modality: str
    ) -> Path:
        """Save image to disk directly in final directory structure.
        
        Args:
            img: Image array
            timestamp: Timestamp in nanoseconds
            camera_id: Camera identifier
            modality: Image modality
            
        Returns:
            Path to saved image (relative to output_dir)
        """
        # Create directory structure - directly in final location
        camera_dir = self.output_dir / self.dataset_name / 'images' / camera_id / modality
        camera_dir.mkdir(parents=True, exist_ok=True)
        
        # Use frame counter to generate sequential filenames
        stream_key = f"{camera_id}/{modality}"
        if stream_key not in self.frame_counters:
            self.frame_counters[stream_key] = 0
        
        frame_idx = self.frame_counters[stream_key]
        filename = f"frame_{frame_idx:06d}.png"
        image_path = camera_dir / filename
        
        # Save image
        cv2.imwrite(str(image_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        self.frame_counters[stream_key] += 1
        
        return image_path.relative_to(self.output_dir)


class TimestampSynchronizer:
    """Synchronize data streams based on timestamps."""
    
    def __init__(self, tolerance_ms: float = 50.0):
        """Initialize synchronizer.
        
        Args:
            tolerance_ms: Maximum time difference in milliseconds for synchronization
        """
        self.tolerance_ns = int(tolerance_ms * 1_000_000)
        
    def find_nearest_index(
        self,
        target_timestamp: int,
        timestamps: np.ndarray
    ) -> Optional[int]:
        """Find index of nearest timestamp within tolerance.
        
        Args:
            target_timestamp: Target timestamp in nanoseconds
            timestamps: Array of timestamps to search
            
        Returns:
            Index of nearest timestamp, or None if none within tolerance
        """
        if len(timestamps) == 0:
            return None
        
        # Binary search for insertion point
        idx = np.searchsorted(timestamps, target_timestamp)
        
        candidates = []
        if idx > 0:
            candidates.append((idx - 1, abs(timestamps[idx - 1] - target_timestamp)))
        if idx < len(timestamps):
            candidates.append((idx, abs(timestamps[idx] - target_timestamp)))
        
        if not candidates:
            return None
        
        # Find closest within tolerance
        best_idx, best_diff = min(candidates, key=lambda x: x[1])
        
        if best_diff <= self.tolerance_ns:
            return best_idx
        
        return None
    
    def synchronize_streams(
        self,
        reference_timestamps: np.ndarray,
        target_streams: Dict[str, np.ndarray]
    ) -> Dict[str, List[Optional[int]]]:
        """Synchronize multiple streams to a reference stream.
        
        Args:
            reference_timestamps: Reference timestamps
            target_streams: Dictionary of stream_name -> timestamps
            
        Returns:
            Dictionary of stream_name -> list of synchronized indices
        """
        sync_indices = {}
        
        for stream_name, stream_timestamps in target_streams.items():
            indices = []
            
            for ref_ts in tqdm(
                reference_timestamps,
                desc=f"Synchronizing {stream_name}",
                leave=False
            ):
                idx = self.find_nearest_index(ref_ts, stream_timestamps)
                indices.append(idx)
            
            sync_indices[stream_name] = indices
            
            # Log synchronization statistics
            valid_count = sum(1 for idx in indices if idx is not None)
            logger.info(
                f"  {stream_name}: {valid_count}/{len(indices)} "
                f"({100*valid_count/len(indices):.1f}%) frames synchronized"
            )
        
        return sync_indices


class DataExtractor:
    """Extract data from ROS bag files."""
    
    def __init__(self, bag_path: Path, config: ConverterConfig, output_dir: Path, dataset_name: str):
        """Initialize data extractor.
        
        Args:
            bag_path: Path to ROS bag file/directory
            config: Converter configuration
            output_dir: Output directory for extracted data
            dataset_name: Name of the dataset
        """
        self.bag_path = bag_path
        self.config = config
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize processors
        self.image_processor = ImageProcessor(output_dir, dataset_name)
        self.message_processors: Dict[str, MessageProcessor] = {}
        
        # Data storage
        self.data_streams: Dict[str, List[Dict]] = {}
        self.timestamps: Dict[str, List[int]] = {}
    
    def extract(self) -> Tuple[Dict[str, List[Dict]], Dict[str, List[int]]]:
        """Extract all data from the bag file.
        
        Returns:
            Tuple of (data_streams, timestamps)
        """
        self._cleanup_sqlite_journals()

        # Create reader
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(
            uri=str(self.bag_path),
            storage_id='sqlite3'
        )
        if hasattr(storage_options, 'read_only'):
            storage_options.read_only = True
        converter_options = rosbag2_py.ConverterOptions('', '')
        reader.open(storage_options, converter_options)
        
        # Get topic metadata
        topic_metadata = reader.get_metadata().topics_with_message_count
        topic_type_map = {tm.topic_metadata.name: tm.topic_metadata.type for tm in topic_metadata}
        
        # Process messages
        total_messages = sum(tm.message_count for tm in topic_metadata)
        
        with tqdm(total=total_messages, desc="Extracting data") as pbar:
            while reader.has_next():
                topic, data, timestamp = reader.read_next()
                
                # Skip topics not in configuration
                if not self._is_configured_topic(topic):
                    pbar.update(1)
                    continue
                
                # Get message type and deserialize
                msg_type = self._get_message_type(topic_type_map[topic])
                msg = deserialize_message(data, msg_type)
                
                # Process message based on type
                processed_data = self._process_message(msg, topic, timestamp)
                
                if processed_data:
                    # Store data
                    stream_name = self._get_stream_name(topic)
                    if stream_name not in self.data_streams:
                        self.data_streams[stream_name] = []
                        self.timestamps[stream_name] = []
                    
                    self.data_streams[stream_name].append(processed_data)
                    self.timestamps[stream_name].append(timestamp)
                
                pbar.update(1)
        
        # Convert timestamps to numpy arrays
        for stream_name in self.timestamps:
            self.timestamps[stream_name] = np.array(self.timestamps[stream_name])
        
        logger.info(f"Extracted {len(self.data_streams)} data streams")
        
        return self.data_streams, self.timestamps

    def _cleanup_sqlite_journals(self):
        """Remove leftover SQLite journal/wal files that force write access."""
        journal_patterns = ["*.db3-journal", "*.db3-wal", "*.sqlite3-journal", "*.sqlite3-wal"]
        removed_files: List[Path] = []

        def try_remove(path: Path):
            if not path.exists() or not path.is_file():
                return
            try:
                path.unlink()
                removed_files.append(path)
            except OSError as exc:
                logger.warning(
                    "Unable to remove SQLite journal file %s: %s",
                    path,
                    exc
                )

        if self.bag_path.is_dir():
            for pattern in journal_patterns:
                for journal_file in self.bag_path.rglob(pattern):
                    try_remove(journal_file)
        else:
            # Direct .db3 file path
            base = self.bag_path
            if base.suffix == ".db3":
                for suffix in ["-journal", "-wal"]:
                    try_remove(base.with_suffix(base.suffix + suffix))

        if removed_files:
            logger.info(
                "Removed stale SQLite sidecar files: %s",
                ", ".join(str(p.name) for p in removed_files)
            )
    
    def _is_configured_topic(self, topic: str) -> bool:
        """Check if topic is in configuration."""
        # Check camera topics
        for camera in self.config.cameras:
            for topic_config in camera.topics:
                if topic_config.name == topic:
                    return True
        
        # Check robot state topics
        for topic_config in self.config.robot_state.topics:
            if topic_config.name == topic:
                return True
        
        return False
    
    def _get_stream_name(self, topic: str) -> str:
        """Get stream name for a topic."""
        # Check camera topics
        for camera in self.config.cameras:
            for topic_config in camera.topics:
                if topic_config.name == topic:
                    return f"camera/{camera.camera_id}/{topic_config.modality}"
        
        # Check robot state topics
        for topic_config in self.config.robot_state.topics:
            if topic_config.name == topic:
                # Use full topic path (remove leading slash and replace / with _)
                safe_topic_name = topic.lstrip('/').replace('/', '_')
                return f"robot/{safe_topic_name}"
        
        return topic
    
    def _get_message_type(self, type_string: str):
        """Get ROS message class from type string."""
        # Import message type dynamically
        parts = type_string.split('/')
        module_name = f"{parts[0]}.msg"
        class_name = parts[-1]
        
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)
    
    def _process_message(
        self,
        msg: Any,
        topic: str,
        timestamp: int
    ) -> Optional[Dict[str, Any]]:
        """Process a ROS message based on its type."""
        # Check if it's an image message
        if isinstance(msg, CompressedImage):
            camera_id, modality = self._get_camera_info(topic)
            if camera_id:
                return self.image_processor.process_compressed_image(
                    msg, timestamp, camera_id, modality
                )
        elif isinstance(msg, Image):
            camera_id, modality = self._get_camera_info(topic)
            if camera_id:
                return self.image_processor.process_raw_image(
                    msg, timestamp, camera_id, modality
                )
        else:
            # Use custom message processor if registered
            processor_key = type(msg).__name__
            if processor_key in self.message_processors:
                return self.message_processors[processor_key].process(msg, timestamp)
            else:
                logger.warning(f"No processor for message type {processor_key}")
        
        return None
    
    def _get_camera_info(self, topic: str) -> Tuple[Optional[str], Optional[str]]:
        """Get camera ID and modality for a topic."""
        for camera in self.config.cameras:
            for topic_config in camera.topics:
                if topic_config.name == topic:
                    return camera.camera_id, topic_config.modality
        return None, None
    
    def register_message_processor(
        self,
        message_type: str,
        processor: MessageProcessor
    ):
        """Register a custom message processor.
        
        Args:
            message_type: ROS message type name
            processor: Message processor instance
        """
        self.message_processors[message_type] = processor
        logger.info(f"Registered processor for {message_type}")


class LeRobotDatasetWriter:
    """Write synchronized data to LeRobot 3.0 format."""
    
    def __init__(
        self,
        output_dir: Path,
        dataset_name: str,
        config: ConverterConfig
    ):
        """Initialize dataset writer.
        
        Args:
            output_dir: Output directory
            dataset_name: Name of the dataset
            config: Converter configuration
        """
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.config = config
        
    def write(
        self,
        synchronized_data: Dict[str, List[Dict]],
        synchronized_indices: Dict[str, List[Optional[int]]],
        original_timestamps: Dict[str, np.ndarray],
        message_processors: Optional[Dict[str, MessageProcessor]] = None
    ):
        """Write synchronized data to LeRobot format.
        
        Args:
            synchronized_data: Synchronized data streams
            synchronized_indices: Synchronization indices
            original_timestamps: Original timestamps for each stream
            message_processors: Optional message processors for state/action mapping
        """
        dataset_dir = self.output_dir / self.dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Write HDF5 file with robot state/action data
        self._write_hdf5(
            dataset_dir,
            synchronized_data,
            synchronized_indices,
            original_timestamps,
            message_processors
        )
        
        # Write metadata
        self._write_metadata(dataset_dir, synchronized_data)
        
        # Images are already in the correct location, no need to copy
        logger.info("Images already saved in final location, skipping copy step")
        
        logger.info(f"Dataset written to {dataset_dir}")
    
    def _write_hdf5(
        self,
        dataset_dir: Path,
        synchronized_data: Dict[str, List[Dict]],
        synchronized_indices: Dict[str, List[Optional[int]]],
        original_timestamps: Dict[str, np.ndarray],
        message_processors: Optional[Dict[str, MessageProcessor]] = None
    ):
        """Write HDF5 file with state and action data."""
        h5_path = dataset_dir / 'data.h5'
        
        with h5py.File(h5_path, 'w') as h5f:
            # Write metadata
            h5f.attrs['dataset_name'] = self.dataset_name
            h5f.attrs['version'] = 'v3.0'
            
            # Process robot state/action data
            state_group = h5f.create_group('state')
            action_group = h5f.create_group('action')
            
            # Process each robot stream
            for stream_name, indices in synchronized_indices.items():
                if not stream_name.startswith('robot/'):
                    continue
                
                # Get synchronized data for this stream
                stream_data_list = synchronized_data.get(stream_name, [])
                if not stream_data_list:
                    continue
                    
                # Get valid indices
                valid_indices = [i for i in indices if i is not None and i < len(stream_data_list)]
                if not valid_indices:
                    continue
                
                # Collect synchronized data
                synced_data = [stream_data_list[i] for i in valid_indices]
                
                # Extract state and action data from the synchronized data
                # The data structure comes from MessageProcessor.process()
                state_data = {}
                action_data = {}
                
                for data_point in synced_data:
                    if 'state' in data_point:
                        for key, value in data_point['state'].items():
                            if key not in state_data:
                                state_data[key] = []
                            state_data[key].append(value)
                    
                    if 'action' in data_point:
                        for key, value in data_point['action'].items():
                            if key not in action_data:
                                action_data[key] = []
                            action_data[key].append(value)
                
                # Align state and action data based on user request
                for key, state_values in state_data.items():
                    if key not in action_data and state_values:
                        # If a state member exists and corresponding action member doesn't,
                        # create action from state by shifting.
                        # action_t = state_{t+1}
                        action_values = list(state_values)
                        action_values.pop(0)  # Remove first element
                        action_values.append(action_values[-1])  # Duplicate last element
                        action_data[key] = action_values
                        logger.info(f"Generated action '{key}' from state data.")

                
                # Extract the robot stream identifier
                robot_stream_id = stream_name[6:]  # Remove 'robot/' prefix
                
                # Create subgroups based on stream structure
                # Example: "left_ur5e_states" -> "left_ur5e" subgroup
                parts = robot_stream_id.split('_')
                if len(parts) >= 2:
                    # Create hierarchical structure for clarity
                    # e.g., "left_ur5e_states" -> state/left_ur5e/
                    subgroup_name = '_'.join(parts[:-1])
                else:
                    subgroup_name = robot_stream_id
                
                # Write state data
                if state_data:
                    if subgroup_name not in state_group:
                        state_subgroup = state_group.create_group(subgroup_name)
                    else:
                        state_subgroup = state_group[subgroup_name]
                    
                    for field_name, field_data in state_data.items():
                        # Convert to numpy array
                        field_array = np.array(field_data)
                        
                        # Write dataset
                        state_subgroup.create_dataset(
                            field_name,
                            data=field_array,
                            compression=self.config.compression,
                            compression_opts=self.config.compression_opts
                        )
                        logger.debug(f"  Wrote state/{subgroup_name}/{field_name}: shape={field_array.shape}")
                    
                    # Write timestamps for this state group
                    timestamps = original_timestamps[stream_name][valid_indices]
                    state_subgroup.create_dataset(
                        'timestamp',
                        data=timestamps,
                        compression=self.config.compression,
                        compression_opts=self.config.compression_opts
                    )
                
                # Write action data
                if action_data:
                    if subgroup_name not in action_group:
                        action_subgroup = action_group.create_group(subgroup_name)
                    else:
                        action_subgroup = action_group[subgroup_name]
                    
                    for field_name, field_data in action_data.items():
                        # Convert to numpy array
                        field_array = np.array(field_data)
                        
                        # Write dataset
                        action_subgroup.create_dataset(
                            field_name,
                            data=field_array,
                            compression=self.config.compression,
                            compression_opts=self.config.compression_opts
                        )
                        logger.debug(f"  Wrote action/{subgroup_name}/{field_name}: shape={field_array.shape}")
                    
                    # Write timestamps for this action group
                    timestamps = original_timestamps[stream_name][valid_indices]
                    action_subgroup.create_dataset(
                        'timestamp',
                        data=timestamps,
                        compression=self.config.compression,
                        compression_opts=self.config.compression_opts
                    )
            
            # Write camera timestamps
            camera_group = h5f.create_group('camera')
            for stream_name, indices in synchronized_indices.items():
                if not stream_name.startswith('camera/'):
                    continue
                
                # Parse stream name
                parts = stream_name.split('/')
                camera_id = parts[1]
                modality = parts[2]
                
                # Get synchronized timestamps
                valid_indices = [i for i in indices if i is not None]
                if not valid_indices:
                    continue
                
                timestamps = original_timestamps[stream_name][valid_indices]
                
                # Create camera subgroup if needed
                if camera_id not in camera_group:
                    cam_group = camera_group.create_group(camera_id)
                else:
                    cam_group = camera_group[camera_id]
                
                # Write timestamps
                cam_group.create_dataset(
                    f'{modality}_timestamp',
                    data=timestamps,
                    compression=self.config.compression,
                    compression_opts=self.config.compression_opts
                )
        
        logger.info(f"HDF5 data written to {h5_path}")
    
    def _write_metadata(self, dataset_dir: Path, synchronized_data: Dict[str, List[Dict]]):
        """Write dataset metadata."""
        metadata_path = dataset_dir / 'metadata.yaml'
        
        metadata = {
            'dataset_name': self.dataset_name,
            'version': 'v3.0',
            'streams': list(synchronized_data.keys()),
            'num_frames': len(next(iter(synchronized_data.values()))),
            'configuration': {
                'sync_tolerance_ms': self.config.sync_tolerance_ms,
                'compression': self.config.compression
            }
        }
        
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        
        logger.info(f"Metadata written to {metadata_path}")


class ROS2ToLeRobotConverter:
    """Main converter class orchestrating the conversion process."""
    
    def __init__(
        self,
        config_path: Path,
        output_base_dir: Path
    ):
        """Initialize converter.
        
        Args:
            config_path: Path to YAML configuration file
            output_base_dir: Base output directory for converted datasets
        """
        self.config = ConverterConfig(config_path)
        self.output_base_dir = Path(output_base_dir)
        
        # Create base output directory
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
    
    def convert_single(
        self,
        bag_path: Path,
        dataset_name: str,
        message_processors: Optional[Dict[str, MessageProcessor]] = None
    ) -> Path:
        """Convert a single ROS bag to synchronized dataset.
        
        Args:
            bag_path: Path to ROS bag file/directory
            dataset_name: Name for this specific dataset
            message_processors: Optional custom message processors
            
        Returns:
            Path to the output dataset directory
        """
        output_dir = self.output_base_dir / dataset_name
        
        logger.info(f"Starting conversion of {bag_path} -> {dataset_name}")
        
        # Step 1: Extract data from bag
        logger.info("Step 1: Extracting data from ROS bag...")
        extractor = DataExtractor(bag_path, self.config, self.output_base_dir, dataset_name)
        
        # Register custom message processors if provided
        if message_processors:
            for msg_type, processor in message_processors.items():
                extractor.register_message_processor(msg_type, processor)
        
        data_streams, timestamps = extractor.extract()
        
        # Step 2: Synchronize streams
        logger.info("Step 2: Synchronizing data streams...")
        synchronizer = TimestampSynchronizer(self.config.sync_tolerance_ms)
        
        # Determine reference stream
        if self.config.sync_reference:
            reference_stream = self.config.sync_reference
        else:
            # Use stream with highest frequency as reference
            reference_stream = min(
                timestamps.keys(),
                key=lambda k: len(timestamps[k])
            )
        
        logger.info(f"Using {reference_stream} as reference stream")
        
        reference_timestamps = timestamps[reference_stream]
        other_streams = {k: v for k, v in timestamps.items() if k != reference_stream}
        
        sync_indices = synchronizer.synchronize_streams(
            reference_timestamps,
            other_streams
        )
        
        # Add reference stream to sync indices (all indices valid)
        sync_indices[reference_stream] = list(range(len(reference_timestamps)))
        
        # Step 3: Write synchronized dataset
        logger.info("Step 3: Writing synchronized dataset...")
        writer = LeRobotDatasetWriter(self.output_base_dir, dataset_name, self.config)
        writer.write(data_streams, sync_indices, timestamps, message_processors)
        
        logger.info(f"Conversion complete for {dataset_name}!")
        
        # Print summary statistics
        self._print_summary(data_streams, sync_indices, dataset_name)
        
        return output_dir
    
    def convert_batch(
        self,
        bag_paths: List[Path],
        dataset_names: Optional[List[str]] = None,
        message_processors: Optional[Dict[str, MessageProcessor]] = None
    ) -> List[Path]:
        """Convert multiple ROS bags to synchronized datasets.
        
        Args:
            bag_paths: List of paths to ROS bag files/directories
            dataset_names: Optional list of names for datasets (auto-generated if None)
            message_processors: Optional custom message processors
            
        Returns:
            List of paths to output dataset directories
        """
        if dataset_names and len(dataset_names) != len(bag_paths):
            raise ValueError(
                f"Number of dataset names ({len(dataset_names)}) must match "
                f"number of bag paths ({len(bag_paths)})"
            )
        
        # Auto-generate names if not provided
        if not dataset_names:
            dataset_names = [f"episode_{i:04d}" for i in range(len(bag_paths))]
        
        output_dirs = []
        
        logger.info(f"Starting batch conversion of {len(bag_paths)} ROS bags")

        max_workers = min(len(bag_paths), max(1, os.cpu_count()//2 or 1))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_info = {
                executor.submit(
                    self.convert_single,
                    bag_path,
                    dataset_name,
                    message_processors
                ): (bag_path, dataset_name)
                for bag_path, dataset_name in zip(bag_paths, dataset_names)
            }

            with tqdm(total=len(future_to_info), desc="Converting bags") as progress:
                for future in as_completed(future_to_info):
                    bag_path, dataset_name = future_to_info[future]
                    try:
                        output_dir = future.result()
                        output_dirs.append(output_dir)
                    except Exception as e:
                        logger.error(f"Failed to convert {bag_path}: {e}")
                    finally:
                        progress.update(1)
        
        logger.info(f"Batch conversion complete: {len(output_dirs)}/{len(bag_paths)} successful")
        
        return output_dirs
    
    def convert_directory(
        self,
        bags_directory: Path,
        pattern: str = "*.db3",
        message_processors: Optional[Dict[str, MessageProcessor]] = None
    ) -> List[Path]:
        """Convert all ROS bags in a directory.
        
        Args:
            bags_directory: Directory containing ROS bag files
            pattern: Glob pattern for finding bag files
            message_processors: Optional custom message processors
            
        Returns:
            List of paths to output dataset directories
        """
        bags_directory = Path(bags_directory)
        
        bag_paths: List[Path] = []
        dataset_names: List[str] = []

        episode_dirs = sorted(p for p in bags_directory.iterdir() if p.is_dir())

        for episode_dir in episode_dirs:
            raw_data_dir = episode_dir / "data" / "record" / "raw_data"
            bag_path: Optional[Path] = None

            if raw_data_dir.is_dir():
                metadata_path = raw_data_dir / "metadata.yaml"
                if metadata_path.exists():
                    bag_path = raw_data_dir
                else:
                    nested_bag_dirs = sorted(
                        child for child in raw_data_dir.iterdir()
                        if child.is_dir() and (child / "metadata.yaml").exists()
                    )
                    if nested_bag_dirs:
                        bag_path = nested_bag_dirs[0]
                    else:
                        db3_files = sorted(raw_data_dir.glob(pattern))
                        if db3_files:
                            bag_path = raw_data_dir

            if bag_path is not None:
                bag_paths.append(bag_path)
                clean_name = episode_dir.name.replace(' ', '_').replace('-', '_').replace('.', '_')
                dataset_names.append(clean_name)
            else:
                logger.warning(
                    f"No ROS bags found under {raw_data_dir} for episode {episode_dir.name}"
                )

        if not bag_paths:
            logger.warning(f"No ROS bags found in {bags_directory}")
            return []

        logger.info(f"Found {len(bag_paths)} ROS bags in {bags_directory}")

        return self.convert_batch(bag_paths, dataset_names, message_processors)
    
    def _print_summary(
        self,
        data_streams: Dict[str, List[Dict]],
        sync_indices: Dict[str, List[Optional[int]]],
        dataset_name: str
    ):
        """Print conversion summary statistics."""
        print("\n" + "="*60)
        print(f"CONVERSION SUMMARY: {dataset_name}")
        print("="*60)
        
        print(f"\nOutput directory: {self.output_base_dir / dataset_name}")
        
        print("\nData Streams:")
        for stream_name in sorted(data_streams.keys()):
            original_count = len(data_streams[stream_name])
            if stream_name in sync_indices:
                valid_count = sum(1 for idx in sync_indices[stream_name] if idx is not None)
                sync_rate = 100 * valid_count / len(sync_indices[stream_name])
                print(f"  {stream_name:30s}: {original_count:6d} frames, "
                      f"{sync_rate:5.1f}% synchronized")
            else:
                print(f"  {stream_name:30s}: {original_count:6d} frames (reference)")
        
        print("\n" + "="*60)


def main():
    """Main entry point for the converter."""
    parser = argparse.ArgumentParser(
        description="Convert ROS2 bag data to synchronized datasets"
    )
    
    # Mode selection
    subparsers = parser.add_subparsers(dest='mode', help='Conversion mode')
    
    # Single bag mode
    single_parser = subparsers.add_parser('single', help='Convert a single ROS bag')
    single_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    single_parser.add_argument(
        "--bag",
        type=str,
        required=True,
        help="Path to ROS2 bag file or directory"
    )
    single_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for converted dataset"
    )
    single_parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Name for the output dataset"
    )
    single_parser.add_argument(
        "--custom-processor",
        type=str,
        help="Path to Python file with custom message processors"
    )
    
    # Batch mode
    batch_parser = subparsers.add_parser('batch', help='Convert multiple ROS bags')
    batch_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    batch_parser.add_argument(
        "--bags-dir",
        type=str,
        required=True,
        help="Directory containing ROS2 bag files"
    )
    batch_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Base output directory for converted datasets"
    )
    batch_parser.add_argument(
        "--pattern",
        type=str,
        default="*.db3",
        help="Pattern for finding bag files (default: *.db3)"
    )
    batch_parser.add_argument(
        "--custom-processor",
        type=str,
        help="Path to Python file with custom message processors"
    )
    
    # Directory mode (process all bags in subdirectories)
    dir_parser = subparsers.add_parser('directory', help='Convert all bags in subdirectories')
    dir_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    dir_parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Base directory containing subdirectories with ROS bags"
    )
    dir_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Base output directory for converted datasets"
    )
    dir_parser.add_argument(
        "--custom-processor",
        type=str,
        help="Path to Python file with custom message processors"
    )
    
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        return
    
    # Load custom processors if provided
    message_processors = None
    if args.custom_processor:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "custom_processors",
            args.custom_processor
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get processors from module
        if hasattr(module, 'get_message_processors'):
            message_processors = module.get_message_processors()
    
    # Execute based on mode
    if args.mode == 'single':
        # Single bag conversion
        converter = ROS2ToLeRobotConverter(
            config_path=Path(args.config),
            output_base_dir=Path(args.output_dir)
        )
        
        converter.convert_single(
            bag_path=Path(args.bag),
            dataset_name=args.dataset_name,
            message_processors=message_processors
        )
        
    elif args.mode == 'batch':
        # Batch conversion
        converter = ROS2ToLeRobotConverter(
            config_path=Path(args.config),
            output_base_dir=Path(args.output_dir)
        )
        
        converter.convert_directory(
            bags_directory=Path(args.bags_dir),
            pattern=args.pattern,
            message_processors=message_processors
        )
        
    elif args.mode == 'directory':
        # Directory structure conversion
        converter = ROS2ToLeRobotConverter(
            config_path=Path(args.config),
            output_base_dir=Path(args.output_dir)
        )
        
        base_dir = Path(args.base_dir)
        
        # Find all subdirectories with bag data
        bag_dirs = []
        for subdir in base_dir.iterdir():
            if subdir.is_dir():
                # Check if it contains ROS bag data
                record_dir = subdir / 'data' / 'record'
                if record_dir.exists():
                    # Find the actual bag directory
                    for item in record_dir.iterdir():
                        if item.is_dir() and (item / 'metadata.yaml').exists():
                            bag_dirs.append(item)
                            break
                elif (subdir / 'metadata.yaml').exists():
                    # Direct bag directory
                    bag_dirs.append(subdir)
        
        if bag_dirs:
            logger.info(f"Found {len(bag_dirs)} bag directories to process")
            
            # Generate dataset names from directory names
            dataset_names = []
            for bag_dir in bag_dirs:
                # Use parent directories to create meaningful names
                parts = []
                current = bag_dir
                for _ in range(3):  # Get up to 3 levels of directory names
                    if current.parent != base_dir and current != base_dir:
                        parts.append(current.name)
                        current = current.parent
                    else:
                        break
                
                # Create name from parts
                if parts:
                    name = '_'.join(reversed(parts))
                else:
                    name = bag_dir.name
                
                # Clean the name
                name = name.replace(' ', '_').replace('-', '_').replace('.', '_')
                dataset_names.append(name)
            
            converter.convert_batch(
                bag_paths=bag_dirs,
                dataset_names=dataset_names,
                message_processors=message_processors
            )
        else:
            logger.warning(f"No ROS bag directories found in {base_dir}")


if __name__ == "__main__":
    main()