#!/usr/bin/env python3
"""
ROS2 to LeRobot 3.0 Converter - Minimal Worker Optimization

Changes:
1. Add --workers parameter to control parallel workers
2. Optimize default worker count from CPU_COUNT/2 to 16
"""

from __future__ import annotations
import argparse
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import h5py
import numpy as np
import yaml
from rosbags.rosbag2 import Reader as RosbagReader
from rosbags.serde import deserialize_cdr
from rosbags.typesys import get_typestore, Stores
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TopicConfig:
    name: str
    type: str
    frequency: float
    compressed: bool = False
    modality: Optional[str] = None


@dataclass
class CameraConfig:
    camera_id: str
    topics: List[TopicConfig]


@dataclass
class RobotStateConfig:
    topics: List[TopicConfig]


class ConverterConfig:
    def __init__(self):
        self.cameras: List[CameraConfig] = []
        self.robot_state = RobotStateConfig(topics=[])
        self.sync_tolerance_ms: float = 5000.0
        self.sync_reference: Optional[str] = None
        self.chunk_size: int = 1000
        self.compression: str = 'gzip'
        self.compression_opts: int = 4


class MessageProcessor(ABC):
    @abstractmethod
    def process(self, msg: Any, timestamp: int) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_state_action_mapping(self) -> Tuple[List[str], List[str]]:
        pass
    
    def get_converter_config(self) -> ConverterConfig:
        raise NotImplementedError("No processor provides converter configuration")
    
    def register_custom_types(self, reader: Any, typestore: Any) -> None:
        """
        Optional: Register custom message types from bag.
        Override this in robot-specific processors.
        
        Args:
            reader: RosbagReader instance
            typestore: Typestore to register types into
        """
        pass
    
    def validate_image_topics(self, data_streams: Dict[str, List[Dict]], 
                            config: ConverterConfig) -> List[str]:
        errors = []
        for camera in config.cameras:
            for topic_config in camera.topics:
                stream_name = f"camera/{camera.camera_id}/{topic_config.modality}"
                if stream_name not in data_streams or len(data_streams[stream_name]) == 0:
                    errors.append(f"Camera topic '{topic_config.name}' has no messages")
        return errors


class ImageProcessor:
    def __init__(self, output_dir: Path, dataset_name: str):
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_counters: Dict[str, int] = {}
        
    def _extract_header_timestamp(self, msg: Any) -> Optional[int]:
        """Extract timestamp from message header if available."""
        if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
            stamp = msg.header.stamp
            return stamp.sec * 10**9 + stamp.nanosec
        return None
        
    def process_compressed_image(self, msg: Any, timestamp: int, camera_id: str, modality: str) -> Dict[str, Any]:
        actual_timestamp = self._extract_header_timestamp(msg) or timestamp
        
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return {}
        if modality == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_path = self._save_image(img, actual_timestamp, camera_id, modality)
        return {'timestamp': actual_timestamp, 'path': str(image_path), 'shape': img.shape, 'dtype': str(img.dtype)}
    
    def process_raw_image(self, msg: Any, timestamp: int, camera_id: str, modality: str) -> Dict[str, Any]:
        try:
            height, width = msg.height, msg.width
            data = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding == 'rgb8':
                img = data.reshape((height, width, 3))
            elif msg.encoding == 'bgr8':
                img = data.reshape((height, width, 3))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                return {}
        except Exception:
            return {}
        image_path = self._save_image(img, timestamp, camera_id, modality)
        return {'timestamp': timestamp, 'path': str(image_path), 'shape': img.shape, 'dtype': str(img.dtype)}
    
    def _save_image(self, img: np.ndarray, timestamp: int, camera_id: str, modality: str) -> Path:
        camera_dir = self.output_dir / self.dataset_name / 'images' / camera_id / modality
        camera_dir.mkdir(parents=True, exist_ok=True)
        stream_key = f"{camera_id}/{modality}"
        if stream_key not in self.frame_counters:
            self.frame_counters[stream_key] = 0
        frame_idx = self.frame_counters[stream_key]
        filename = f"frame_{frame_idx:06d}.png"
        image_path = camera_dir / filename
        cv2.imwrite(str(image_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.frame_counters[stream_key] += 1
        return image_path.relative_to(self.output_dir)


class TimestampSynchronizer:
    def __init__(self, tolerance_ms: float = 50.0):
        self.tolerance_ns = int(tolerance_ms * 1_000_000)
        
    def find_nearest_index(self, target_timestamp: int, timestamps: np.ndarray) -> Optional[int]:
        if len(timestamps) == 0:
            return None
        idx = np.searchsorted(timestamps, target_timestamp)
        candidates = []
        if idx > 0:
            candidates.append((idx - 1, abs(timestamps[idx - 1] - target_timestamp)))
        if idx < len(timestamps):
            candidates.append((idx, abs(timestamps[idx] - target_timestamp)))
        if not candidates:
            return None
        best_idx, best_diff = min(candidates, key=lambda x: x[1])
        return best_idx if best_diff <= self.tolerance_ns else None
    
    def synchronize_streams(self, reference_timestamps: np.ndarray, 
                          target_streams: Dict[str, np.ndarray]) -> Dict[str, List[Optional[int]]]:
        sync_indices = {}
        for stream_name, stream_timestamps in target_streams.items():
            indices = [self.find_nearest_index(ref_ts, stream_timestamps) for ref_ts in reference_timestamps]
            sync_indices[stream_name] = indices
        return sync_indices


class DataExtractor:
    def __init__(self, bag_path: Path, config: ConverterConfig, output_dir: Path, dataset_name: str):
        self.bag_path = bag_path
        self.config = config
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_processor = ImageProcessor(output_dir, dataset_name)
        self.message_processors: Dict[str, MessageProcessor] = {}
        self.data_streams: Dict[str, List[Dict]] = {}
        self.timestamps: Dict[str, List[int]] = {}
    
    def extract(self) -> Tuple[Dict[str, List[Dict]], Dict[str, List[int]]]:
        # Create typestore
        typestore = get_typestore(Stores.ROS2_HUMBLE)
        
        with RosbagReader(self.bag_path) as reader:
            # Let processors register custom types if needed
            for processor in self.message_processors.values():
                if hasattr(processor, 'register_custom_types'):
                    processor.register_custom_types(reader, typestore)
            
            connections = reader.connections
            connections_list = connections.values() if isinstance(connections, dict) else connections
            total_messages = sum(conn.msgcount for conn in connections_list 
                               if self._is_configured_topic(conn.topic))
            
            # Process messages without progress bar
            for connection, timestamp, rawdata in reader.messages():
                if not self._is_configured_topic(connection.topic):
                    continue
                
                try:
                    msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                except Exception:
                    continue
                
                processed_data = self._process_message(msg, connection.topic, timestamp)
                if processed_data:
                    stream_name = self._get_stream_name(connection.topic)
                    if stream_name not in self.data_streams:
                        self.data_streams[stream_name] = []
                        self.timestamps[stream_name] = []
                    self.data_streams[stream_name].append(processed_data)
                    
                    # 修改开始：优先使用 processed_data 中的 timestamp (例如来自消息 Header)
                    # 如果没有,则回退使用 bag 录制时间戳
                    actual_timestamp = processed_data.get('timestamp', timestamp)
                    self.timestamps[stream_name].append(actual_timestamp)
                    # 修改结束
        
        for stream_name in self.timestamps:
            self.timestamps[stream_name] = np.array(self.timestamps[stream_name])
        
        return self.data_streams, self.timestamps
    
    def _is_configured_topic(self, topic: str) -> bool:
        for camera in self.config.cameras:
            for topic_config in camera.topics:
                if topic_config.name == topic:
                    return True
        for topic_config in self.config.robot_state.topics:
            if topic_config.name == topic:
                return True
        return False
    
    def _get_stream_name(self, topic: str) -> str:
        for camera in self.config.cameras:
            for topic_config in camera.topics:
                if topic_config.name == topic:
                    return f"camera/{camera.camera_id}/{topic_config.modality}"
        for topic_config in self.config.robot_state.topics:
            if topic_config.name == topic:
                return f"robot/{topic.lstrip('/').replace('/', '_')}"
        return topic
    
    def _process_message(self, msg: Any, topic: str, timestamp: int) -> Optional[Dict[str, Any]]:
        # Check if this topic is a camera topic first
        camera_id, modality = self._get_camera_info(topic)
        if camera_id:
            # Determine if compressed or raw by checking message attributes
            if hasattr(msg, 'format'):  # CompressedImage has 'format' field
                return self.image_processor.process_compressed_image(msg, timestamp, camera_id, modality)
            elif hasattr(msg, 'encoding'):  # Raw Image has 'encoding' field
                return self.image_processor.process_raw_image(msg, timestamp, camera_id, modality)
        
        # Try custom message processors
        msg_type_name = type(msg).__name__
        processor = None
        candidate_names = [msg_type_name]
        if '__' in msg_type_name:
            candidate_names.append(msg_type_name.split('__')[-1])
        for name in candidate_names:
            if name in self.message_processors:
                processor = self.message_processors[name]
                break
        if processor:
            return processor.process(msg, timestamp)
        
        return None
    
    def _get_camera_info(self, topic: str) -> Tuple[Optional[str], Optional[str]]:
        for camera in self.config.cameras:
            for topic_config in camera.topics:
                if topic_config.name == topic:
                    return camera.camera_id, topic_config.modality
        return None, None
    
    def register_message_processor(self, message_type: str, processor: MessageProcessor):
        self.message_processors[message_type] = processor


class LeRobotDatasetWriter:
    def __init__(self, output_dir: Path, dataset_name: str, config: ConverterConfig):
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.config = config
        
    def write(self, synchronized_data: Dict[str, List[Dict]], synchronized_indices: Dict[str, List[Optional[int]]],
              original_timestamps: Dict[str, np.ndarray], message_processors: Optional[Dict[str, MessageProcessor]] = None):
        dataset_dir = self.output_dir / self.dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        self._write_hdf5(dataset_dir, synchronized_data, synchronized_indices, original_timestamps, message_processors)
        self._write_metadata(dataset_dir, synchronized_data)
    
    def _write_hdf5(self, dataset_dir: Path, synchronized_data: Dict[str, List[Dict]],
                   synchronized_indices: Dict[str, List[Optional[int]]], original_timestamps: Dict[str, np.ndarray],
                   message_processors: Optional[Dict[str, MessageProcessor]] = None):
        h5_path = dataset_dir / 'data.h5'
        with h5py.File(h5_path, 'w') as h5f:
            h5f.attrs['dataset_name'] = self.dataset_name
            h5f.attrs['version'] = 'v3.0'
            state_group = h5f.create_group('state')
            action_group = h5f.create_group('action')
            robot_buffers: Dict[str, Dict[str, Dict[str, List[np.ndarray]]]] = {}
            robot_timestamps: Dict[str, np.ndarray] = {}

            for stream_name, indices in synchronized_indices.items():
                if not stream_name.startswith('robot/'):
                    continue
                stream_data_list = synchronized_data.get(stream_name, [])
                if not stream_data_list:
                    continue
                valid_indices = [i for i in indices if i is not None and i < len(stream_data_list)]
                if not valid_indices:
                    continue
                
                synced_data = [stream_data_list[i] for i in valid_indices]
                state_data, action_data = {}, {}
                
                for data_point in synced_data:
                    if 'state' in data_point:
                        for key, value in data_point['state'].items():
                            state_data.setdefault(key, []).append(value)
                    if 'action' in data_point:
                        for key, value in data_point['action'].items():
                            action_data.setdefault(key, []).append(value)
                
                for key, state_values in state_data.items():
                    if key not in action_data and state_values:
                        action_values = list(state_values)
                        action_values.pop(0)
                        action_values.append(action_values[-1])
                        action_data[key] = action_values
                
                robot_stream_id = stream_name[6:]
                parts = robot_stream_id.split('_')
                subgroup_name = '_'.join(parts[:-1]) if len(parts) >= 2 else robot_stream_id
                subgroup_buffers = robot_buffers.setdefault(subgroup_name, {'state': {}, 'action': {}})

                for field_name, field_values in state_data.items():
                    subgroup_buffers['state'].setdefault(field_name, []).extend(field_values)
                for field_name, field_values in action_data.items():
                    subgroup_buffers['action'].setdefault(field_name, []).extend(field_values)

                timestamps = original_timestamps[stream_name][valid_indices]
                if subgroup_name not in robot_timestamps or len(timestamps) > len(robot_timestamps[subgroup_name]):
                    robot_timestamps[subgroup_name] = timestamps

            for subgroup_name, buffers in robot_buffers.items():
                state_values = buffers['state']
                action_values = buffers['action']

                if state_values:
                    state_subgroup = state_group.create_group(subgroup_name) if subgroup_name not in state_group else state_group[subgroup_name]
                    for field_name, field_list in state_values.items():
                        if not field_list:
                            continue
                        stacked = [np.asarray(v) for v in field_list]
                        field_array = np.stack(stacked) if stacked and stacked[0].ndim > 0 else np.asarray(stacked)
                        state_subgroup.create_dataset(
                            field_name,
                            data=field_array,
                            compression=self.config.compression,
                            compression_opts=self.config.compression_opts,
                        )
                    if subgroup_name in robot_timestamps:
                        state_subgroup.create_dataset(
                            'timestamp',
                            data=robot_timestamps[subgroup_name],
                            compression=self.config.compression,
                            compression_opts=self.config.compression_opts,
                        )

                if action_values:
                    action_subgroup = action_group.create_group(subgroup_name) if subgroup_name not in action_group else action_group[subgroup_name]
                    for field_name, field_list in action_values.items():
                        if not field_list:
                            continue
                        stacked = [np.asarray(v) for v in field_list]
                        field_array = np.stack(stacked) if stacked and stacked[0].ndim > 0 else np.asarray(stacked)
                        action_subgroup.create_dataset(
                            field_name,
                            data=field_array,
                            compression=self.config.compression,
                            compression_opts=self.config.compression_opts,
                        )
                    if subgroup_name in robot_timestamps:
                        action_subgroup.create_dataset(
                            'timestamp',
                            data=robot_timestamps[subgroup_name],
                            compression=self.config.compression,
                            compression_opts=self.config.compression_opts,
                        )
            
            camera_group = h5f.create_group('camera')
            for stream_name, indices in synchronized_indices.items():
                if not stream_name.startswith('camera/'):
                    continue
                parts = stream_name.split('/')
                camera_id, modality = parts[1], parts[2]
                valid_indices = [i for i in indices if i is not None]
                if not valid_indices:
                    continue
                timestamps = original_timestamps[stream_name][valid_indices]
                cam_group = camera_group.create_group(camera_id) if camera_id not in camera_group else camera_group[camera_id]
                cam_group.create_dataset(f'{modality}_timestamp', data=timestamps,
                                       compression=self.config.compression, compression_opts=self.config.compression_opts)
    
    def _write_metadata(self, dataset_dir: Path, synchronized_data: Dict[str, List[Dict]]):
        metadata = {
            'dataset_name': self.dataset_name,
            'version': 'v3.0',
            'streams': list(synchronized_data.keys()),
            'num_frames': len(next(iter(synchronized_data.values()))),
            'configuration': {'sync_tolerance_ms': self.config.sync_tolerance_ms, 'compression': self.config.compression}
        }
        with open(dataset_dir / 'metadata.yaml', 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)


class ROS2ToLeRobotConverter:
    def __init__(self, output_base_dir: Path, message_processors: Dict[str, MessageProcessor]):
        config = None
        for processor in message_processors.values():
            try:
                config = processor.get_converter_config()
                break
            except NotImplementedError:
                continue
        if config is None:
            raise ValueError("No processor provides converter configuration")
        self.config = config
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.message_processors = message_processors
    
    def convert_single(self, bag_path: Path, dataset_name: str) -> Path:
        output_dir = self.output_base_dir / dataset_name
        
        extractor = DataExtractor(bag_path, self.config, self.output_base_dir, dataset_name)
        for msg_type, processor in self.message_processors.items():
            extractor.register_message_processor(msg_type, processor)
        
        data_streams, timestamps = extractor.extract()
        
        # Validate image topics
        for processor in self.message_processors.values():
            errors = processor.validate_image_topics(data_streams, self.config)
            if errors:
                if output_dir.exists():
                    shutil.rmtree(output_dir)
                raise ValueError(f"Missing image topics")
        
        synchronizer = TimestampSynchronizer(self.config.sync_tolerance_ms)
        reference_stream = self.config.sync_reference or min(timestamps.keys(), key=lambda k: len(timestamps[k]))
        
        reference_timestamps = timestamps[reference_stream]
        other_streams = {k: v for k, v in timestamps.items() if k != reference_stream}
        sync_indices = synchronizer.synchronize_streams(reference_timestamps, other_streams)
        sync_indices[reference_stream] = list(range(len(reference_timestamps)))
        
        writer = LeRobotDatasetWriter(self.output_base_dir, dataset_name, self.config)
        writer.write(data_streams, sync_indices, timestamps, self.message_processors)
        
        # Calculate statistics
        num_cameras = len([k for k in data_streams.keys() if k.startswith('camera/')])
        num_frames = len(reference_timestamps)
        
        # Print success message
        tqdm.write(f"  ✓ {dataset_name}: {num_cameras} cameras, {num_frames} frames")
        
        return output_dir
    
    def convert_batch(self, bag_paths: List[Path], dataset_names: Optional[List[str]] = None, 
                     max_workers: int = 16) -> List[Path]:
        """
        Convert multiple bags in parallel.
        
        Args:
            bag_paths: List of bag paths to convert
            dataset_names: Optional list of dataset names
            max_workers: Number of parallel workers (default: 16)
                        - Recommended: 8-16 for HDD, 16-24 for SSD
        """
        if dataset_names and len(dataset_names) != len(bag_paths):
            raise ValueError("Mismatch between bag paths and dataset names")
        dataset_names = dataset_names or [f"episode_{i:04d}" for i in range(len(bag_paths))]
        
        output_dirs = []
        failed_bags = []
        # Cap workers by number of bags
        max_workers = min(len(bag_paths), max_workers)
        
        print(f"\n{'='*80}")
        print(f"Starting conversion: {len(bag_paths)} bags | {max_workers} parallel workers")
        print(f"{'='*80}\n")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_info = {executor.submit(self.convert_single, bag_path, dataset_name): (bag_path, dataset_name)
                            for bag_path, dataset_name in zip(bag_paths, dataset_names)}
            
            with tqdm(total=len(future_to_info), desc="Progress", 
                     ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as progress:
                for future in as_completed(future_to_info):
                    bag_path, dataset_name = future_to_info[future]
                    try:
                        output_dirs.append(future.result())
                    except Exception as e:
                        failed_bags.append((dataset_name, bag_path, str(e)))
                    finally:
                        progress.update(1)
        
        # Summary
        print(f"\n{'='*80}")
        print(f"✓ Conversion complete: {len(output_dirs)}/{len(bag_paths)} successful")
        
        if failed_bags:
            print(f"\n✗ {len(failed_bags)} failed:")
            for dataset_name, bag_path, error in failed_bags:
                print(f"  • {dataset_name}")
                print(f"    → {bag_path}")
                error_short = error.split('\n')[0][:60]
                print(f"    → {error_short}")
        
        print(f"{'='*80}\n")
        
        return output_dirs
    
    def convert_directory(self, bags_directory: Path, pattern: str = "*.mcap", 
                         max_workers: int = 16) -> List[Path]:
        """
        Convert all bags in a directory.
        
        Args:
            bags_directory: Directory containing bag folders
            pattern: Bag file pattern (default: "*.mcap")
            max_workers: Number of parallel workers (default: 16)
        """
        bags_directory = Path(bags_directory)
        bag_paths, dataset_names = [], []
        
        for episode_dir in sorted(p for p in bags_directory.iterdir() if p.is_dir()):
            raw_data_dir = episode_dir / "record" / "raw_data"
            bag_path = None
            
            if raw_data_dir.is_dir():
                if (raw_data_dir / "metadata.yaml").exists():
                    bag_path = raw_data_dir
                else:
                    nested = sorted(c for c in raw_data_dir.iterdir() if c.is_dir() and (c / "metadata.yaml").exists())
                    if nested:
                        bag_path = nested[0]
                    elif list(raw_data_dir.glob(pattern)):
                        bag_path = raw_data_dir
            
            if bag_path:
                bag_paths.append(bag_path)
                dataset_names.append(episode_dir.name.replace(' ', '_').replace('-', '_').replace('.', '_'))
        
        if not bag_paths:
            logger.warning(f"No ROS bags found in {bags_directory}")
            return []
        
        logger.info(f"Found {len(bag_paths)} ROS bags")
        return self.convert_batch(bag_paths, dataset_names, max_workers=max_workers)


def main():
    parser = argparse.ArgumentParser(description="ROS2 to LeRobot Converter")
    subparsers = parser.add_subparsers(dest='mode')
    
    single_parser = subparsers.add_parser('single')
    single_parser.add_argument("--bag", required=True)
    single_parser.add_argument("--output-dir", required=True)
    single_parser.add_argument("--dataset-name", required=True)
    single_parser.add_argument("--custom-processor", required=True)
    
    batch_parser = subparsers.add_parser('batch')
    batch_parser.add_argument("--bags-dir", required=True)
    batch_parser.add_argument("--output-dir", required=True)
    batch_parser.add_argument("--custom-processor", required=True)
    batch_parser.add_argument("--workers", type=int, default=16, 
                            help="Number of parallel workers (default: 16, recommended: 8-24 depending on disk speed)")
    
    args = parser.parse_args()
    if not args.mode:
        parser.print_help()
        return
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("custom_processors", args.custom_processor)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    message_processors = module.get_message_processors()
    
    if args.mode == 'single':
        converter = ROS2ToLeRobotConverter(Path(args.output_dir), message_processors)
        converter.convert_single(Path(args.bag), args.dataset_name)
    elif args.mode == 'batch':
        converter = ROS2ToLeRobotConverter(Path(args.output_dir), message_processors)
        converter.convert_directory(Path(args.bags_dir), max_workers=args.workers)

if __name__ == "__main__":
    main()