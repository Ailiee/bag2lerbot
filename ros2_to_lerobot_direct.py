#!/usr/bin/env python3
"""
Optimized ROS2 to LeRobot Converter (Single-Pass + Separate Mode + Merge)

Key Optimizations over the original:
1. Single-pass bag reading (vs. two passes for timestamp scan + data collection)
2. Immediate video encoding after collecting frames (releases memory faster)  
3. Uses PyAV for video encoding (avoids subprocess overhead)
4. Proper numpy array copying to avoid memory reference issues
5. Compatible with LeRobot _save_episode_video(temp_path=...) API

Usage:
    python ros2_to_lerobot_optimized.py \
        --bags-dir /path/to/bags \
        --output-dir /path/to/output \
        --repo-id your/repo_id \
        --robot-type your_robot \
        --custom-processor /path/to/processor.py \
        --mapping-file /path/to/mapping.py \
        --fps 30 --workers 4
"""

import argparse
import importlib.util
import logging
import os
import shutil
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# ROS2 bag reading
try:
    from rosbags.rosbag2 import Reader as RosbagReader
    from rosbags.typesys import get_typestore, Stores
except ImportError:
    print("Error: 'rosbags' package not found. Install with: pip install rosbags")
    sys.exit(1)

# Video encoding - prefer PyAV for efficiency
try:
    import av
    USE_PYAV = True
except ImportError:
    USE_PYAV = False
    import subprocess
    print("Warning: PyAV not found, falling back to FFmpeg subprocess (slower)")

# LeRobot imports
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.compute_stats import compute_episode_stats
except ImportError:
    print("Error: 'lerobot' package not found. Please install it first.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Utility Classes
# =============================================================================

@dataclass
class StateActionMapping:
    """Define how to map data to LeRobot state and action tensors."""
    state_components: List[str] = field(default_factory=list)
    action_components: List[str] = field(default_factory=list)
    state_combine_fn: Optional[Any] = None
    action_combine_fn: Optional[Any] = None
    normalize: bool = True


class TimestampSynchronizer:
    """
    Synchronizes multiple streams using Nearest Neighbor to the Reference stream.
    """
    def synchronize_streams(self, ref_timestamps: np.ndarray, 
                           other_streams: Dict[str, np.ndarray]) -> Dict[str, List[int]]:
        """
        Match every timestamp in ref_timestamps with the NEAREST timestamp in each other stream.
        """
        sync_indices = {}
        
        for topic, stream_ts in other_streams.items():
            if len(stream_ts) == 0:
                sync_indices[topic] = [None] * len(ref_timestamps)
                continue
            
            # Efficient Nearest Neighbor using searchsorted
            idx_right = np.searchsorted(stream_ts, ref_timestamps, side='right')
            idx_left = idx_right - 1
            
            idx_right = np.clip(idx_right, 0, len(stream_ts) - 1)
            idx_left = np.clip(idx_left, 0, len(stream_ts) - 1)
            
            diff_left = np.abs(stream_ts[idx_left] - ref_timestamps)
            diff_right = np.abs(stream_ts[idx_right] - ref_timestamps)
            
            nearest_indices = np.where(diff_left <= diff_right, idx_left, idx_right)
            sync_indices[topic] = nearest_indices.tolist()
            
        return sync_indices


# =============================================================================
# Image / Video Utilities
# =============================================================================

def get_message_timestamp(msg, default_ts: int) -> int:
    """Extract timestamp from message header (in nanoseconds)."""
    if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
        return msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
    if hasattr(msg, 'timestamp'):
        ts = msg.timestamp
        if hasattr(ts, 'sec') and hasattr(ts, 'nanosec'):
            return ts.sec * 10**9 + ts.nanosec
    return default_ts


def decode_compressed_rgb(image_bytes: bytes) -> np.ndarray:
    """Decode compressed image bytes into RGB numpy array."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to decode compressed image")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def encode_video_pyav(
    images: List[np.ndarray],
    video_path: Path,
    fps: int,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    g: int = 2,
    crf: int = 30,
    preset: int = 12,
) -> None:
    """Encode video using PyAV (fast, no subprocess)."""
    if len(images) == 0:
        raise ValueError("No images provided for video encoding")
    
    height, width = images[0].shape[:2]
    
    # Handle codec/pix_fmt compatibility
    if vcodec in ("libsvtav1", "hevc") and pix_fmt == "yuv444p":
        pix_fmt = "yuv420p"
    
    video_options = {"g": str(g), "crf": str(crf)}
    if vcodec == "libsvtav1":
        video_options["preset"] = str(preset)
    
    video_path.parent.mkdir(parents=True, exist_ok=True)
    
    with av.open(str(video_path), "w") as output:
        stream = output.add_stream(vcodec, fps, options=video_options)
        stream.pix_fmt = pix_fmt
        stream.width = width
        stream.height = height
        
        for img_array in images:
            if img_array.shape[2] == 4:  # RGBA -> RGB
                img_array = img_array[:, :, :3]
            frame = av.VideoFrame.from_ndarray(img_array, format='rgb24')
            for packet in stream.encode(frame):
                output.mux(packet)
        
        # Flush encoder
        for packet in stream.encode():
            output.mux(packet)


def encode_video_ffmpeg(
    images: List[np.ndarray],
    video_path: Path,
    fps: int,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",
    gop: int = 2,
    crf: int = 30,
) -> None:
    """Encode video using FFmpeg subprocess (fallback)."""
    if len(images) == 0:
        raise ValueError("No images provided")
    
    height, width = images[0].shape[:2]
    video_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-c:v", vcodec,
        "-pix_fmt", pix_fmt,
        "-g", str(gop),
        "-crf", str(crf),
    ]
    
    if vcodec == "libsvtav1":
        cmd.extend(["-preset", "8"])
    else:
        cmd.extend(["-preset", "fast"])
    
    cmd.append(str(video_path))
    
    process = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, bufsize=10**8
    )
    
    try:
        for img in images:
            if img.shape[2] == 4:
                img = img[:, :, :3]
            process.stdin.write(img.astype(np.uint8).tobytes())
        process.stdin.close()
    except Exception as e:
        process.kill()
        raise RuntimeError(f"FFmpeg encoding failed: {e}")
    
    process.wait()
    if process.returncode != 0:
        stderr = process.stderr.read().decode() if process.stderr else ""
        raise RuntimeError(f"FFmpeg exited with code {process.returncode}: {stderr}")


def encode_video(images: List[np.ndarray], video_path: Path, fps: int, 
                 vcodec: str = "libsvtav1", crf: int = 30) -> None:
    """Encode video using best available method."""
    if USE_PYAV:
        encode_video_pyav(images, video_path, fps, vcodec=vcodec, crf=crf)
    else:
        encode_video_ffmpeg(images, video_path, fps, vcodec=vcodec, crf=crf)


def save_video_to_dataset(
    dataset: LeRobotDataset,
    video_key: str,
    episode_index: int,
    temp_video_path: Path,
) -> dict:
    """
    Save a pre-encoded video to the LeRobot dataset.
    
    Completely manual implementation that bypasses _save_episode_video
    to avoid any internal file lookups.
    """
    if not temp_video_path.exists():
        raise FileNotFoundError(f"Source video file not found: {temp_video_path}")
    
    # Determine chunk structure
    chunks_size = getattr(dataset.meta, 'chunks_size', 1000)
    chunk_index = episode_index // chunks_size
    file_index = episode_index % chunks_size
    
    # Create target directory structure
    target_dir = dataset.root / "videos" / video_key / f"chunk-{chunk_index:03d}"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Target video file path
    target_video_path = target_dir / f"file-{file_index:03d}.mp4"
    
    # Copy the video file
    logger.info(f"  Copying: {temp_video_path} -> {target_video_path}")
    shutil.copy2(temp_video_path, target_video_path)
    
    # Verify the copy
    if not target_video_path.exists():
        raise RuntimeError(f"Failed to copy video: target does not exist: {target_video_path}")
    
    source_size = temp_video_path.stat().st_size
    target_size = target_video_path.stat().st_size
    if source_size != target_size:
        raise RuntimeError(f"Video copy size mismatch: source={source_size}, target={target_size}")
    
    logger.info(f"  Verified: {target_video_path} ({target_size} bytes)")
    
    # Return empty metadata - the video path is implicit from the directory structure
    return {}


# =============================================================================
# Module Loading
# =============================================================================

def load_module_from_path(name: str, path: str):
    """Dynamically load a Python module from file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =============================================================================
# Core Conversion Logic - SINGLE PASS
# =============================================================================

def convert_bag_single_pass(
    bag_path: Path,
    output_dir: Path,
    dataset_name: str,
    processor_path: str,
    mapping_path: str,
    repo_id: str,
    robot_type: str,
    task_desc: str,
    fps: int,
    vcodec: str,
    crf: int,
) -> dict:
    """
    Convert a SINGLE bag into a SINGLE episode dataset using SINGLE-PASS reading.
    
    Key optimization: Read bag only once, collect timestamps AND data simultaneously.
    """
    result = {
        'bag': str(bag_path),
        'success': False,
        'frames': 0,
        'error': None,
        'dataset_path': None
    }
    
    try:
        # 1. Load configuration
        typestore = get_typestore(Stores.ROS2_HUMBLE)
        proc_mod = load_module_from_path("custom_processor", processor_path)
        message_processors = proc_mod.get_message_processors()
        config = message_processors['ConfigProvider'].get_converter_config()
        map_mod = load_module_from_path("custom_mapping", mapping_path)
        mapping = map_mod.get_state_action_mapping()
        
        # 2. Build topic configuration
        target_topics = set()
        topic_to_config = {}
        camera_topics = {}  # topic -> camera_id
        
        for t in config.robot_state.topics:
            target_topics.add(t.name)
            topic_to_config[t.name] = t
            
        for cam in config.cameras:
            for t in cam.topics:
                target_topics.add(t.name)
                topic_to_config[t.name] = t
                camera_topics[t.name] = cam.camera_id
        
        # =====================================================================
        # SINGLE PASS: Collect timestamps AND raw data simultaneously
        # =====================================================================
        topic_data = {t: [] for t in target_topics}  # (timestamp, data_dict)
        image_shape_cache = {}
        
        with RosbagReader(bag_path) as reader:
            # Register custom types
            for processor in message_processors.values():
                if hasattr(processor, 'register_custom_types'):
                    processor.register_custom_types(reader, typestore)
            
            connections = [c for c in reader.connections if c.topic in target_topics]
            
            for conn, ts, rawdata in reader.messages(connections=connections):
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                real_ts = get_message_timestamp(msg, ts)
                
                # Process message based on type
                if conn.topic in camera_topics:
                    cam_id = camera_topics[conn.topic]
                    t_cfg = topic_to_config[conn.topic]
                    
                    if "CompressedImage" in t_cfg.type:
                        # Store compressed bytes (decode later during video encoding)
                        img_bytes = bytes(msg.data)
                        if cam_id not in image_shape_cache:
                            sample = decode_compressed_rgb(img_bytes)
                            image_shape_cache[cam_id] = sample.shape
                        topic_data[conn.topic].append((real_ts, {
                            'type': 'image',
                            'camera_id': cam_id,
                            'storage': 'compressed',
                            'data': img_bytes
                        }))
                    elif hasattr(msg, 'encoding'):
                        # Raw image - MUST COPY to avoid memory overwrite
                        if msg.encoding in ('rgb8', 'bgr8'):
                            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                                msg.height, msg.width, 3
                            ).copy()  # CRITICAL: .copy() to avoid reference issues
                            if msg.encoding == 'bgr8':
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            image_shape_cache.setdefault(cam_id, img.shape)
                            topic_data[conn.topic].append((real_ts, {
                                'type': 'image',
                                'camera_id': cam_id,
                                'storage': 'array',
                                'data': img
                            }))
                else:
                    # Robot state/action data
                    proc = (message_processors.get(conn.msgtype.split('/')[-1]) or 
                            message_processors.get(conn.msgtype))
                    if proc:
                        payload = proc.process(msg, ts)
                        topic_data[conn.topic].append((real_ts, {
                            'type': 'state_action',
                            'payload': payload
                        }))
        
        # 3. Sort by timestamp and extract timestamps array
        topic_timestamps = {}
        topic_sorted_data = {}
        for topic, data_list in topic_data.items():
            if data_list:
                data_list.sort(key=lambda x: x[0])
                topic_timestamps[topic] = np.array([d[0] for d in data_list])
                topic_sorted_data[topic] = [d[1] for d in data_list]
            else:
                topic_timestamps[topic] = np.array([])
                topic_sorted_data[topic] = []
        
        # Clear original data to free memory
        del topic_data
        
        # 4. Synchronize streams
        if hasattr(config, 'sync_reference') and config.sync_reference:
            ref_topic = config.sync_reference
        else:
            candidate = {t: len(ts) for t, ts in topic_timestamps.items() if len(ts) > 0}
            if not candidate:
                result['error'] = "No data found in any monitored topics"
                return result
            ref_topic = min(candidate, key=candidate.get)
        
        ref_ts = topic_timestamps[ref_topic]
        other_streams = {k: v for k, v in topic_timestamps.items() if k != ref_topic}
        
        synchronizer = TimestampSynchronizer()
        sync_indices = synchronizer.synchronize_streams(ref_ts, other_streams)
        sync_indices[ref_topic] = list(range(len(ref_ts)))
        
        num_frames = len(ref_ts)
        if num_frames == 0:
            result['error'] = "No frames found after synchronization"
            return result
        
        # 5. Assemble synchronized frames
        frame_buffer = [{'state': {}, 'action': {}, 'images': {}} for _ in range(num_frames)]
        
        for topic, indices in sync_indices.items():
            sorted_data = topic_sorted_data[topic]
            for frame_idx, data_idx in enumerate(indices):
                if data_idx is None or data_idx >= len(sorted_data):
                    continue
                
                data = sorted_data[data_idx]
                if data['type'] == 'image':
                    frame_buffer[frame_idx]['images'][data['camera_id']] = {
                        'storage': data['storage'],
                        'data': data['data']
                    }
                elif data['type'] == 'state_action':
                    payload = data['payload']
                    if 'state' in payload:
                        frame_buffer[frame_idx]['state'].update(payload['state'])
                    if 'action' in payload:
                        frame_buffer[frame_idx]['action'].update(payload['action'])
        
        # Clear intermediate data
        del topic_sorted_data, topic_timestamps

        # 5.5 Backfill action with next-state when action key is missing
        state_data = {}
        action_data = {}
        for frame in frame_buffer:
            for key, value in frame['state'].items():
                state_data.setdefault(key, []).append(value)
            for key, value in frame['action'].items():
                action_data.setdefault(key, []).append(value)

        for key, state_values in state_data.items():
            if key not in action_data and state_values:
                action_values = list(state_values)
                if len(action_values) > 1:
                    action_values.pop(0)
                    action_values.append(action_values[-1])
                action_data[key] = action_values

        if action_data:
            for i, frame in enumerate(frame_buffer):
                for key, values in action_data.items():
                    if key in frame['action']:
                        continue
                    if i < len(values):
                        frame['action'][key] = values[i]
        
        # 6. Extract images for video encoding ONLY (not for add_frame)
        camera_images = {cam.camera_id: [] for cam in config.cameras}
        
        for frame in frame_buffer:
            for cam in config.cameras:
                cid = cam.camera_id
                if cid in frame['images']:
                    entry = frame['images'][cid]
                    if entry['storage'] == 'compressed':
                        img = decode_compressed_rgb(entry['data'])
                    else:
                        img = entry['data']
                    camera_images[cid].append(img)
            # Clear image data from frame to save memory (we don't need it for add_frame anymore)
            frame['images'] = {}
        
        # Encode videos to temp files - EACH VIDEO IN SEPARATE DIRECTORY
        # This is critical because _save_episode_video may delete the temp directory
        temp_base_dir = Path(tempfile.mkdtemp())
        video_paths = {}
        
        for cid, images in camera_images.items():
            if images:
                # Create separate temp directory for each video (same as reference code)
                temp_video_dir = Path(tempfile.mkdtemp(dir=temp_base_dir))
                video_path = temp_video_dir / f"{cid}.mp4"
                encode_video(images, video_path, fps, vcodec=vcodec, crf=crf)
                video_paths[cid] = video_path
                logger.debug(f"Encoded {len(images)} frames for camera {cid} -> {video_path}")
        
        # Release image memory IMMEDIATELY after encoding
        del camera_images
        
        # 7. Define features from first valid frame
        s0_dict = frame_buffer[0]['state']
        a0_dict = frame_buffer[0]['action']
        
        # Apply aliasing if needed
        for alias_from, alias_to in [('q_pos', 'driver/q_pos'), ('eef', 'end/eef')]:
            if alias_from in s0_dict:
                s0_dict[alias_to] = s0_dict[alias_from]
            if alias_from in a0_dict:
                a0_dict[alias_to] = a0_dict[alias_from]
        
        s0 = np.asarray(mapping.state_combine_fn(s0_dict), dtype=np.float32)
        a0 = np.asarray(mapping.action_combine_fn(a0_dict), dtype=np.float32)
        
        features = {
            "observation.state": {"dtype": "float32", "shape": s0.shape, "names": None},
            "action": {"dtype": "float32", "shape": a0.shape, "names": None},
        }
        
        for cam in config.cameras:
            cid = cam.camera_id
            if cid in image_shape_cache:
                h, w, c = image_shape_cache[cid]
                features[f"observation.images.{cid}"] = {
                    "dtype": "video",
                    "shape": (h, w, c),
                    "names": ["height", "width", "channels"]
                }
        
        # 8. Create dataset
        # IMPORTANT: Set image_writer_threads=0 to SKIP writing temporary images
        # This saves huge I/O overhead since we already have pre-encoded videos
        episode_dir = output_dir / dataset_name
        if episode_dir.exists():
            shutil.rmtree(episode_dir)
        
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=episode_dir,
            robot_type=robot_type,
            fps=fps,
            features=features,
            use_videos=True,
            image_writer_threads=0,  # CRITICAL: Skip temporary image writing!
        )
        
        # 9. Add frames with PLACEHOLDER images (real data already in pre-encoded videos)
        # This is much faster since no disk I/O for images
        logger.info(f"Adding {len(frame_buffer)} frames (using placeholder images)...")
        for i, frame_data in enumerate(frame_buffer):
            s_dict = frame_data['state']
            a_dict = frame_data['action']
            
            # Apply aliasing
            for alias_from, alias_to in [('q_pos', 'driver/q_pos'), ('eef', 'end/eef')]:
                if alias_from in s_dict:
                    s_dict[alias_to] = s_dict[alias_from]
                if alias_from in a_dict:
                    a_dict[alias_to] = a_dict[alias_from]
            
            s = np.asarray(mapping.state_combine_fn(s_dict), dtype=np.float32)
            a = np.asarray(mapping.action_combine_fn(a_dict), dtype=np.float32)
            
            # Create frame dict with PLACEHOLDER images (to pass validation)
            frame_dict = {
                "observation.state": s,
                "action": a,
                "task": task_desc or "Manipulation Task",
            }
            
            # Add placeholder images - just for schema validation, not written to disk
            for cam in config.cameras:
                cid = cam.camera_id
                feature_key = f"observation.images.{cid}"
                if feature_key in features:
                    h, w, c = features[feature_key]["shape"]
                    frame_dict[feature_key] = np.zeros((h, w, c), dtype=np.uint8)
            
            dataset.add_frame(frame_dict)
            
            # Release memory
            frame_buffer[i] = None
        
        logger.info(f"Frames added successfully")
        
        # 10. Save episode with pre-encoded videos
        dataset._wait_image_writer()
        
        episode_buffer = dataset.episode_buffer
        episode_length = episode_buffer.pop("size")
        tasks_list = episode_buffer.pop("task")
        episode_tasks = list(set(tasks_list))
        episode_index = 0
        
        episode_buffer["index"] = np.arange(0, episode_length)
        episode_buffer["episode_index"] = np.zeros((episode_length,), dtype=np.int32)
        
        dataset.meta.save_episode_tasks(episode_tasks)
        episode_buffer["task_index"] = np.array([
            dataset.meta.get_task_index(t) for t in tasks_list
        ])
        
        # Stack non-video features for saving
        for key, ft in dataset.features.items():
            if key in ["index", "episode_index", "task_index"]:
                continue
            if ft["dtype"] in ["image", "video"]:
                continue
            if key in episode_buffer:
                episode_buffer[key] = np.stack(episode_buffer[key])
        
        # Compute stats - only for non-video features (same as reference code)
        logger.info(f"Computing episode stats...")
        non_video_features = {
            k: v for k, v in dataset.features.items() 
            if v["dtype"] not in ["image", "video"]
        }
        non_video_buffer = {
            k: v for k, v in episode_buffer.items()
            if k not in dataset.meta.video_keys
        }
        ep_stats = compute_episode_stats(non_video_buffer, non_video_features)
        logger.info(f"Episode stats computed successfully")
        
        # No need to delete temporary images - we didn't write any (image_writer_threads=0)
        
        # Save pre-encoded videos using _save_episode_video (same as reference code)
        logger.info(f"Saving {len(video_paths)} pre-encoded videos...")
        episode_metadata = {}
        
        for cid, temp_video_path in video_paths.items():
            video_key = f"observation.images.{cid}"
            logger.info(f"  Saving video: {video_key}")
            
            if not temp_video_path.exists():
                raise FileNotFoundError(f"Source video not found: {temp_video_path}")
            
            # Use _save_episode_video with temp_path parameter (same as reference code)
            video_metadata = dataset._save_episode_video(
                video_key=video_key,
                episode_index=episode_index,
                temp_path=temp_video_path,
            )
            episode_metadata.update(video_metadata)
        logger.info(f"Videos saved successfully")
        
        # Remove video features from episode_buffer before saving data
        for video_key in list(episode_buffer.keys()):
            if video_key in dataset.meta.video_keys:
                del episode_buffer[video_key]
        
        # Save episode data to parquet files
        logger.info(f"Saving episode data...")
        ep_data_metadata = dataset._save_episode_data(episode_buffer)
        episode_metadata.update(ep_data_metadata)
        logger.info(f"Episode data saved")
        
        # Save episode metadata
        logger.info(f"Saving episode metadata...")
        dataset.meta.save_episode(
            episode_index, episode_length, episode_tasks, ep_stats, episode_metadata
        )
        logger.info(f"Episode metadata saved")
        
        # Update video info
        logger.info(f"Updating video info...")
        for video_key in dataset.meta.video_keys:
            dataset.meta.update_video_info(video_key)
        logger.info(f"Video info updated")
        
        logger.info(f"Clearing episode buffer and finalizing...")
        dataset.clear_episode_buffer(delete_images=False)
        dataset.finalize()
        logger.info(f"Finalized successfully")
        
        # Cleanup temp base directory
        shutil.rmtree(temp_base_dir, ignore_errors=True)
        
        result['success'] = True
        result['frames'] = num_frames
        result['dataset_path'] = str(episode_dir)
        
    except Exception as e:
        result['error'] = f"{e}\n{traceback.format_exc()}"
        logger.error(f"Failed processing {bag_path}: {e}")
    
    return result


def convert_bag_worker(args: tuple) -> dict:
    """Wrapper for ProcessPoolExecutor."""
    return convert_bag_single_pass(*args)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Optimized ROS2 to LeRobot Converter (Single-Pass)"
    )
    
    # Input/Output
    parser.add_argument("--bags-dir", type=Path, help="Directory containing ROS bags")
    parser.add_argument("--bag", type=Path, help="Single bag file/directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    
    # Config files
    parser.add_argument("--custom-processor", required=True, help="Processor module path")
    parser.add_argument("--mapping-file", required=True, help="Mapping module path")
    
    # Dataset metadata
    parser.add_argument("--repo-id", required=True, help="HuggingFace repo ID")
    parser.add_argument("--robot-type", required=True, help="Robot type name")
    parser.add_argument("--task-description", default="Manipulation Task", help="Task description")
    parser.add_argument("--fps", type=int, default=30, help="Dataset FPS")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    
    # Video encoding
    parser.add_argument("--vcodec", default="libsvtav1", help="Video codec")
    parser.add_argument("--crf", type=int, default=30, help="Video quality (CRF)")
    
    args = parser.parse_args()
    
    # Collect bags
    bags = []
    if args.bag:
        bags.append(Path(args.bag))
    elif args.bags_dir:
        bd = Path(args.bags_dir)
        metas = sorted(bd.rglob("metadata.yaml"))
        if metas:
            bags = [p.parent for p in metas]
        else:
            bags = sorted(bd.rglob("*.mcap")) + sorted(bd.rglob("*.db3"))
    
    if not bags:
        print("No bags found!")
        sys.exit(1)
    
    print(f"Found {len(bags)} bags. Using {args.workers} workers.")
    print(f"PyAV available: {USE_PYAV}")
    
    # Setup directories
    output_root = Path(args.output_dir)
    # Keep temp episodes outside final output dir to avoid pre-creating it
    separate_dir = output_root.parent / f"{output_root.name}_separate_episodes"
    separate_dir.mkdir(parents=True, exist_ok=True)
    
    # Build tasks
    tasks = []
    for i, bag in enumerate(bags):
        ep_name = f"episode_{i:04d}_{bag.stem}"
        tasks.append((
            bag, separate_dir, ep_name,
            args.custom_processor, args.mapping_file,
            f"{args.repo_id}/{ep_name}",
            args.robot_type, args.task_description,
            args.fps, args.vcodec, args.crf
        ))
    
    # Parallel processing
    success_datasets = []
    failed_episodes = []
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(convert_bag_worker, t): t for t in tasks}
        
        with tqdm(total=len(bags), desc="Converting Episodes") as pbar:
            for future in as_completed(futures):
                res = future.result()
                if res['success']:
                    success_datasets.append(Path(res['dataset_path']))
                else:
                    failed_episodes.append(f"{res['bag']}: {res['error']}")
                    tqdm.write(f"FAILED: {res['bag']}")
                pbar.update(1)
    
    elapsed = time.time() - start_time
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Conversion completed in {elapsed:.1f}s")
    print(f"Success: {len(success_datasets)} / {len(bags)}")
    print(f"Failed: {len(failed_episodes)}")
    print(f"{'='*70}")
    
    if failed_episodes:
        print("\nFailed episodes:")
        for f in failed_episodes[:10]:
            print(f"  - {f[:200]}...")
    
    if not success_datasets:
        print("No valid datasets. Exiting.")
        sys.exit(1)
    
    # Merge datasets
    if output_root.exists():
        shutil.rmtree(output_root)
    print("\nMerging into final dataset...")
    datasets_to_merge = []
    for dpath in sorted(success_datasets):
        try:
            ds = LeRobotDataset(root=dpath, repo_id=dpath.name)
            datasets_to_merge.append(ds)
        except Exception as e:
            print(f"Failed to load {dpath}: {e}")
    
    if datasets_to_merge:
        merged = merge_datasets(
            datasets=datasets_to_merge,
            output_dir=output_root,
            output_repo_id=args.repo_id,
        )
        
        print(f"\nMerge Complete!")
        print(f"Output: {output_root}")
        print(f"Total Episodes: {merged.meta.total_episodes}")
        print(f"Total Frames: {merged.meta.total_frames}")
        
        # Cleanup temporary separate datasets
        if separate_dir.exists():
            shutil.rmtree(separate_dir)
            print("Cleaned up temporary episode datasets.")


if __name__ == "__main__":
    main()