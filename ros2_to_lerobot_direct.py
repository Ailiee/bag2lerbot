#!/usr/bin/env python3
"""
Direct ROS2 to LeRobot Converter (Separate + Merge Mode)

This script performs a one-step conversion from ROS2 bags to a unified LeRobot dataset.
It uses multiprocessing to convert each bag into a temporary separate dataset,
then merges them all into the final output efficiently.

Key Features:
- Parallel processing of bags (Separate Mode).
- Direct memory-to-video piping (No intermediate image files).
- Auto-merge of episodes.
- "LeRobot calling architecture" compliant (uses LeRobotDataset class).
"""

import argparse
import importlib.util
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from rosbags.rosbag2 import Reader as RosbagReader
from rosbags.serde import deserialize_cdr
from rosbags.typesys import get_typestore, Stores
from tqdm import tqdm

# LeRobot imports
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.utils import (
        validate_frame,
        update_chunk_file_indices,
        get_file_size_in_mb,
        write_info,
    )
    from lerobot.datasets.video_utils import (
        get_video_duration_in_s,
        concatenate_video_files,
        get_video_info,
    )
    # Optional: stats computation if manually needed, though LeRobotDataset handles it sometimes
    from lerobot.datasets.compute_stats import compute_episode_stats
except ImportError:
    print("Error: 'lerobot' package not found. Please install it first.")
    sys.exit(1)

# Import base classes from the original converter to maintain compatibility
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from ros2_to_lerobot_converter import (
        ConverterConfig,
        MessageProcessor,
        TimestampSynchronizer,
        TopicConfig,
        CameraConfig
    )
except ImportError:
    # Fallback/Mock classes if file missing (though it should be there)
    pass

class TimestampSynchronizer:
    """
    Synchronizes multiple streams using Nearest Neighbor to the Reference stream.
    Strictly follows reference stream frames. If a stream is missing data, uses nearest available.
    """
    def __init__(self, tolerance_ms=None):
        self.tolerance_ms = tolerance_ms # Ignored now in favor of pure nearest

    def synchronize_streams(self, ref_timestamps, other_streams_timestamps):
        """
        Match every timestamp in ref_timestamps with the NEAREST timestamp in each other stream.
        No dropout. Always finds a match if data exists.
        """
        sync_indices = {}
        
        for topic, stream_ts in other_streams_timestamps.items():
            if len(stream_ts) == 0:
                # If a stream is empty, we can't match anything.
                sync_indices[topic] = [None] * len(ref_timestamps)
                continue
                
            # Efficient Nearest Neighbor Search using searchsorted
            # Find indices in stream_ts that are just right of ref_ts
            idx_right = np.searchsorted(stream_ts, ref_timestamps, side='right')
            idx_left = idx_right - 1
            
            # Clip indices to boundaries
            idx_right = np.clip(idx_right, 0, len(stream_ts) - 1)
            idx_left = np.clip(idx_left, 0, len(stream_ts) - 1)
            
            # Calculate diffs
            diff_left = np.abs(stream_ts[idx_left] - ref_timestamps)
            diff_right = np.abs(stream_ts[idx_right] - ref_timestamps)
            
            # Choose nearest
            nearest_indices = np.where(diff_left < diff_right, idx_left, idx_right)
            
            # Unconditionally use the nearest index
            sync_indices[topic] = nearest_indices.tolist()
            
        return sync_indices

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class StateActionMapping:
    """Define how to map data to LeRobot state and action tensors."""
    state_components: List[str] = field(default_factory=list)
    action_components: List[str] = field(default_factory=list)
    state_combine_fn: Optional[Any] = None
    action_combine_fn: Optional[Any] = None
    normalize: bool = True


def get_message_timestamp(msg, default_ts):
    """Extract timestamp from message header or timestamp field if available."""
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
        raise ValueError("Failed to decode compressed image data")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def encode_video_from_entries(
    entries: List[Dict[str, Any]],
    video_path: Path,
    fps: int,
    vcodec: str,
    pix_fmt: str,
    gop: int,
    crf: int,
    fast_decode: bool = False,
):
    """Encode video from a list of per-frame image entries."""
    if not entries:
        raise ValueError("No frames provided for video encoding")

    # Decode first frame to determine resolution
    first_entry = entries[0]
    if first_entry.get("storage") == "compressed":
        first_frame = decode_compressed_rgb(first_entry["data"])
    else:
        first_frame = np.asarray(first_entry["data"], dtype=np.uint8)

    if first_frame.ndim != 3 or first_frame.shape[2] not in (3, 4):
        raise ValueError("Video frames must be HxWx3 or HxWx4 arrays")

    if first_frame.shape[2] == 4:
        first_frame = first_frame[:, :, :3]

    height, width, _ = first_frame.shape
    video_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        vcodec,
        "-pix_fmt",
        pix_fmt,
    ]

    if gop is not None:
        cmd.extend(["-g", str(gop)])

    if "libsvtav1" in vcodec:
        cmd.extend(["-crf", str(crf), "-preset", "8"])
        if fast_decode:
            cmd.extend(["-svtav1-params", "tune=0"])
    elif "nvenc" in vcodec:
        cmd.extend(["-cq", str(crf), "-preset", "p4", "-bf", "0"])
    else:
        cmd.extend(["-crf", str(crf), "-preset", "fast"])
        if fast_decode and vcodec in {"libx264", "libx265"}:
            cmd.extend(["-tune", "fastdecode"])

    cmd.append(str(video_path))

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10 ** 8,
    )

    try:
        # Write first frame and reuse for iteration
        process.stdin.write(first_frame.astype(np.uint8).tobytes())
        for entry in entries[1:]:
            if entry.get("storage") == "compressed":
                frame = decode_compressed_rgb(entry["data"])
            else:
                frame = np.asarray(entry["data"], dtype=np.uint8)
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]
            process.stdin.write(frame.astype(np.uint8).tobytes())
        process.stdin.flush()
    except Exception as exc:
        stdout, stderr = process.communicate()
        err_msg = stderr.decode("utf-8", errors="ignore") if stderr else str(exc)
        raise RuntimeError(
            f"FFmpeg encoding failed for {video_path}.\nCommand: {' '.join(cmd)}\nError Output:\n{err_msg}"
        ) from exc
    finally:
        try:
            process.stdin.close()
        except Exception:
            pass

    process.wait()
    if process.returncode != 0:
        stderr = process.stderr.read().decode("utf-8", errors="ignore") if process.stderr else ""
        raise RuntimeError(
            f"FFmpeg exited with code {process.returncode} for {video_path}.\n{stderr}"
        )

class OptimizedLeRobotDataset(LeRobotDataset):
    """LeRobotDataset variant that bypasses per-frame image writes and encodes videos on save."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_custom_attributes()

    def _init_custom_attributes(self):
        self.current_image_buffers: Dict[str, List[Dict[str, Any]]] = {}
        self.vcodec = "libsvtav1"
        self.crf = 30
        self.pix_fmt = "yuv420p"
        self.gop = 2
        self.fast_decode = True
        self.encoding_threads = 4
        self.video_chunk_state: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def create(cls, *args, **kwargs):
        obj = super().create(*args, **kwargs)
        obj._init_custom_attributes()
        return obj

    def set_current_image_buffers(self, buffers: Dict[str, List[Dict[str, Any]]]):
        self.current_image_buffers = buffers

    def add_frame(self, frame: dict) -> None:
        frame_copy = {}
        for key, value in frame.items():
            if hasattr(value, "numpy"):
                frame_copy[key] = value.numpy()
            else:
                frame_copy[key] = value

        # Build placeholder frame for validation
        validate_frame_dict = {}
        for key, value in frame_copy.items():
            if key in ("timestamp", "task"):
                continue
            feature = self.features.get(key)
            if feature and feature["dtype"] in ["image", "video"]:
                shape = feature["shape"]
                validate_frame_dict[key] = np.zeros(shape, dtype=np.uint8)
            else:
                validate_frame_dict[key] = value

        task_value = frame_copy.get("task")
        if task_value is None:
            raise ValueError("Frame missing 'task' field")
        validate_frame_dict["task"] = task_value

        validate_frame(validate_frame_dict, self.features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()
            self.episode_buffer["episode_index"] = self.meta.total_episodes
            self.current_image_buffers = {k: [] for k in self.meta.video_keys}

        frame_index = self.episode_buffer["size"]
        timestamp = frame_copy.pop("timestamp", frame_index / self.fps)
        frame_copy.pop("task", None)

        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["task"].append(task_value)

        for key, value in frame_copy.items():
            feature = self.features.get(key)
            if feature and feature["dtype"] in ["image", "video"]:
                self.episode_buffer[key].append("placeholder")
                self.current_image_buffers.setdefault(key, []).append(value)
            else:
                self.episode_buffer[key].append(value)

        self.episode_buffer["size"] += 1

    def save_episode(self, episode_data: dict | None = None) -> None:
        episode_buffer = episode_data if episode_data is not None else self.episode_buffer
        if episode_buffer is None:
            raise ValueError("No episode data to save")

        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(
            self.meta.total_frames,
            self.meta.total_frames + episode_length,
        )
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        self.meta.save_episode_tasks(episode_tasks)
        episode_buffer["task_index"] = np.array(
            [self.meta.get_task_index(task) for task in tasks]
        )

        for key, ft in self.features.items():
            if key in ["index", "episode_index", "task_index"]:
                continue
            if ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        vector_features = {
            k: v for k, v in self.features.items() if v["dtype"] not in ["image", "video"]
        }
        vector_buffer = {k: episode_buffer[k] for k in vector_features if k in episode_buffer}
        ep_stats = compute_episode_stats(vector_buffer, vector_features)

        ep_metadata = self._save_episode_data(episode_buffer)

        if self.meta.video_keys:
            with ThreadPoolExecutor(max_workers=self.encoding_threads) as executor:
                futures = {
                    executor.submit(self._save_episode_video, video_key, episode_index): video_key
                    for video_key in self.meta.video_keys
                }
                for future in futures:
                    video_key = futures[future]
                    metadata = future.result()
                    ep_metadata.update(metadata)

        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats, ep_metadata)

        if episode_data is None:
            self.clear_episode_buffer(delete_images=False)
        self.current_image_buffers = {k: [] for k in self.meta.video_keys}

    def clear_episode_buffer(self, delete_images: bool = True) -> None:
        super().clear_episode_buffer(delete_images=delete_images)
        self.episode_buffer = None
        self.current_image_buffers = {k: [] for k in self.meta.video_keys}

    def _init_video_state(self, video_key: str, episode_index: int):
        if video_key in self.video_chunk_state:
            return

        if (
            episode_index == 0
            or self.meta.latest_episode is None
            or f"videos/{video_key}/chunk_index" not in self.meta.latest_episode
        ):
            chunk_idx, file_idx = 0, 0
            if self.meta.episodes is not None and len(self.meta.episodes) > 0:
                old_chunk_idx = self.meta.episodes[-1][f"videos/{video_key}/chunk_index"]
                old_file_idx = self.meta.episodes[-1][f"videos/{video_key}/file_index"]
                chunk_idx, file_idx = update_chunk_file_indices(
                    old_chunk_idx, old_file_idx, self.meta.chunks_size
                )
            latest_duration_in_s = 0.0
        else:
            latest_ep = self.meta.latest_episode
            chunk_idx = latest_ep[f"videos/{video_key}/chunk_index"][0]
            file_idx = latest_ep[f"videos/{video_key}/file_index"][0]
            latest_duration_in_s = latest_ep[f"videos/{video_key}/to_timestamp"][0]

        video_path = self.root / self.meta.video_path.format(
            video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
        )

        pending_files: List[Path] = []
        current_chunk_size = 0.0

        if video_path.exists():
            pending_files.append(video_path)
            current_chunk_size = get_file_size_in_mb(video_path)

        self.video_chunk_state[video_key] = {
            "chunk_idx": chunk_idx,
            "file_idx": file_idx,
            "pending_files": pending_files,
            "current_chunk_size": current_chunk_size,
            "current_chunk_duration": latest_duration_in_s,
        }

    def _flush_chunk(self, video_key: str):
        state = self.video_chunk_state[video_key]
        if not state["pending_files"]:
            return

        chunk_idx, file_idx = state["chunk_idx"], state["file_idx"]
        video_path = self.root / self.meta.video_path.format(
            video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
        )

        if len(state["pending_files"]) == 1 and state["pending_files"][0] == video_path:
            state["pending_files"] = []
            return

        valid_files = [p for p in state["pending_files"] if p.exists()]
        if not valid_files:
            state["pending_files"] = []
            return

        concatenate_video_files(valid_files, video_path)

        for p in valid_files:
            if p != video_path:
                try:
                    p.unlink()
                except OSError:
                    pass

        state["pending_files"] = []

    def flush(self):
        for video_key in self.video_chunk_state:
            self._flush_chunk(video_key)

    def _save_episode_video(self, video_key: str, episode_index: int) -> dict:
        self._init_video_state(video_key, episode_index)
        state = self.video_chunk_state[video_key]

        temp_dir = self.root / "videos" / video_key / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_video_path = temp_dir / f"ep_{episode_index}.mp4"

        if video_key not in self.current_image_buffers:
            raise ValueError(f"No image buffers registered for {video_key}")
        frame_entries = self.current_image_buffers[video_key]

        encode_video_from_entries(
            frame_entries,
            temp_video_path,
            self.fps,
            self.vcodec,
            self.pix_fmt,
            self.gop,
            self.crf,
            fast_decode=self.fast_decode,
        )

        ep_duration = get_video_duration_in_s(temp_video_path)
        ep_size = get_file_size_in_mb(temp_video_path)

        if state["current_chunk_size"] + ep_size >= self.meta.video_files_size_in_mb:
            self._flush_chunk(video_key)
            state["chunk_idx"], state["file_idx"] = update_chunk_file_indices(
                state["chunk_idx"], state["file_idx"], self.meta.chunks_size
            )
            state["current_chunk_size"] = 0.0
            state["current_chunk_duration"] = 0.0

        state["pending_files"].append(temp_video_path)
        state["current_chunk_size"] += ep_size

        from_timestamp = state["current_chunk_duration"]
        state["current_chunk_duration"] += ep_duration

        metadata = {
            "episode_index": episode_index,
            f"videos/{video_key}/chunk_index": state["chunk_idx"],
            f"videos/{video_key}/file_index": state["file_idx"],
            f"videos/{video_key}/from_timestamp": from_timestamp,
            f"videos/{video_key}/to_timestamp": state["current_chunk_duration"],
        }

        if episode_index == 0:
            self.meta.info["features"][video_key]["info"] = get_video_info(temp_video_path)
            write_info(self.meta.info, self.meta.root)

        return metadata
def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def convert_bag_worker(args):
    """
    Worker function to process a SINGLE bag into a SINGLE episode dataset.
    Strictly follows LeRobot standards using add_frame.
    """
    (bag_path, output_dir, dataset_name, processor_path, mapping_path, 
     repo_id, robot_type, task_desc, fps, vcodec, crf) = args
    
    result = {
        'bag': str(bag_path),
        'success': False,
        'frames': 0,
        'error': None,
        'dataset_path': None
    }
    
    try:
        # 1. Setup Environment
        typestore = get_typestore(Stores.ROS2_HUMBLE)
        proc_mod = load_module_from_path("custom_processor", processor_path)
        message_processors = proc_mod.get_message_processors()
        config = message_processors['ConfigProvider'].get_converter_config()
        map_mod = load_module_from_path("custom_mapping", mapping_path)
        mapping = map_mod.get_state_action_mapping()
        
        # 2. Pass 1: Scan Timestamps
        topic_timestamps = {}
        target_topics = [t.name for t in config.robot_state.topics]
        for cam in config.cameras:
            target_topics.extend([t.name for t in cam.topics])
        for t in target_topics: topic_timestamps[t] = []
        
        topic_to_config = {}
        for t in config.robot_state.topics: topic_to_config[t.name] = t
        for c in config.cameras:
           for t in c.topics: topic_to_config[t.name] = t
           
        with RosbagReader(bag_path) as reader:
            for processor in message_processors.values():
                processor.register_custom_types(reader, typestore)
            connections = [x for x in reader.connections if x.topic in topic_timestamps]
            for conn, ts, rawdata in reader.messages(connections=connections):
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                real_ts = get_message_timestamp(msg, ts)
                topic_timestamps[conn.topic].append(real_ts)
                
        for t in topic_timestamps:
            topic_timestamps[t] = np.array(sorted(topic_timestamps[t]))
            
        # 3. Synchronize
        synchronizer = TimestampSynchronizer(None) # Use Nearest Neighbor strict
        
        # Select reference topic (min frames)
        if config.sync_reference:
             ref_topic = config.sync_reference
        else:
             candidate_topics = {t: len(ts) for t, ts in topic_timestamps.items() if len(ts) > 0}
             if not candidate_topics:
                 result['error'] = "No data found in any monitored topics"
                 return result
             ref_topic = min(candidate_topics, key=candidate_topics.get)

        ref_ts = topic_timestamps[ref_topic]
        other_streams = {k: v for k, v in topic_timestamps.items() if k != ref_topic}
        sync_indices = synchronizer.synchronize_streams(ref_ts, other_streams)
        sync_indices[ref_topic] = list(range(len(ref_ts)))
        
        num_frames = len(ref_ts)
        if num_frames == 0:
            raise ValueError("No frames found after synchronization")

        # Build timestamp -> [frame_indices] lookup (One-to-Many due to NN)
        timestamp_to_frames = {}
        for topic, indices in sync_indices.items():
            ts_arr = topic_timestamps[topic]
            for frame_idx, time_idx in enumerate(indices):
                if time_idx is not None:
                    t_val = ts_arr[time_idx]
                    key = (topic, t_val)
                    if key not in timestamp_to_frames:
                        timestamp_to_frames[key] = []
                    timestamp_to_frames[key].append(frame_idx)

        # 4. Collect Data into RAM
        frame_buffer = [{'state': {}, 'action': {}, 'images': {}} for _ in range(num_frames)]
        image_shape_cache: Dict[str, Tuple[int, int, int]] = {}
        
        with RosbagReader(bag_path) as reader:
            connections = [x for x in reader.connections if x.topic in topic_timestamps]
            for conn, ts, rawdata in reader.messages(connections=connections):
                msg = typestore.deserialize_cdr(rawdata, conn.msgtype)
                real_ts = get_message_timestamp(msg, ts)
                
                key = (conn.topic, real_ts)
                target_frames = timestamp_to_frames.get(key, [])
                if not target_frames: continue
                    
                image_entry = None
                payload_proc = None
                
                cam_id = next((c.camera_id for c in config.cameras for t in c.topics if t.name == conn.topic), None)
                
                if cam_id:
                    t_cfg = topic_to_config[conn.topic]
                    if "CompressedImage" in t_cfg.type:
                        img_bytes = bytes(msg.data)
                        if cam_id not in image_shape_cache:
                            sample_rgb = decode_compressed_rgb(img_bytes)
                            image_shape_cache[cam_id] = sample_rgb.shape
                        image_entry = {"storage": "compressed", "data": img_bytes}

                    elif hasattr(msg, 'encoding'):
                        if msg.encoding == 'rgb8':
                            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                            image_shape_cache.setdefault(cam_id, img.shape)
                            image_entry = {"storage": "array", "data": img}
                        elif msg.encoding == 'bgr8':
                            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            image_shape_cache.setdefault(cam_id, img.shape)
                            image_entry = {"storage": "array", "data": img}

                else:
                    proc = message_processors.get(conn.msgtype.split('/')[-1]) or message_processors.get(conn.msgtype)
                    if proc:
                        payload_proc = proc.process(msg, ts)

                for f_idx in target_frames:
                    if cam_id and image_entry is not None:
                        frame_buffer[f_idx]['images'][cam_id] = image_entry
                    elif payload_proc:
                        if 'state' in payload_proc:
                            frame_buffer[f_idx]['state'].update(payload_proc['state'])
                        if 'action' in payload_proc:
                            frame_buffer[f_idx]['action'].update(payload_proc['action'])

        # 5. Define Features 
        valid_idx = 0
        s0_dict = frame_buffer[valid_idx]['state']
        a0_dict = frame_buffer[valid_idx]['action']
        
        # Aliasing
        if 'q_pos' in s0_dict: s0_dict['driver/q_pos'] = s0_dict['q_pos']
        if 'eef' in s0_dict: s0_dict['end/eef'] = s0_dict['eef']
        if 'q_pos' in a0_dict: a0_dict['driver/q_pos'] = a0_dict['q_pos']
        if 'eef' in a0_dict: a0_dict['end/eef'] = a0_dict['eef']

        try:
            s0 = mapping.state_combine_fn(s0_dict)
            a0 = mapping.action_combine_fn(a0_dict)
        except Exception as e:
            raise ValueError(f"Mapping failed on reference frame {valid_idx}: {e}")
            
        s0 = np.asarray(s0, dtype=np.float32)
        a0 = np.asarray(a0, dtype=np.float32)
        
        features = {
            "observation.state": {"dtype": "float32", "shape": s0.shape, "names": None},
            "action": {"dtype": "float32", "shape": a0.shape, "names": None},
        }
        # Note: task is NOT a feature - it's handled as metadata via save_episode
        # This avoids schema conflicts in multiprocessing context (following Astribot pattern)
            
        for cid in config.cameras:
            c_name = cid.camera_id
            if c_name in image_shape_cache:
                h, w, c = image_shape_cache[c_name]
                features[f"observation.images.{c_name}"] = {
                    "dtype": "video",
                    "shape": (h, w, c),
                    "names": ["height", "width", "channels"]
                }

        # Create Episode Dataset
        episode_dir = output_dir / dataset_name
        if episode_dir.exists(): shutil.rmtree(episode_dir)
        
        dataset = OptimizedLeRobotDataset.create(
            repo_id=repo_id,
            root=episode_dir,
            robot_type=robot_type,
            fps=fps,
            features=features,
            use_videos=True,
            image_writer_threads=0
        )
        dataset.vcodec = vcodec
        dataset.crf = crf
        dataset.fast_decode = True
        dataset.pix_fmt = "yuv420p"
        dataset.gop = 2
        dataset.encoding_threads = max(1, len(config.cameras))

        # 6. Add Frames Strict
        for i, frame_data in enumerate(frame_buffer):
            # State
            s_dict = frame_data['state']
            if 'q_pos' in s_dict: s_dict['driver/q_pos'] = s_dict['q_pos']
            if 'eef' in s_dict: s_dict['end/eef'] = s_dict['eef']
            s = np.asarray(mapping.state_combine_fn(s_dict), dtype=np.float32)
            
            # Action
            a_dict = frame_data['action']
            if 'q_pos' in a_dict: a_dict['driver/q_pos'] = a_dict['q_pos']
            if 'eef' in a_dict: a_dict['end/eef'] = a_dict['eef']
            a = np.asarray(mapping.action_combine_fn(a_dict), dtype=np.float32)
            
            frame_dict = {
                "observation.state": s,
                "action": a,
                "task": task_desc if task_desc else "Manipulation Task",  # Task as metadata, not schema feature
            }
            image_store = frame_data.get('images', {})

            for cid in config.cameras:
                c_id = cid.camera_id
                feature_key = f"observation.images.{c_id}"
                if feature_key not in features:
                    continue
                if c_id in image_store:
                    img_entry = image_store[c_id]
                    frame_dict[feature_key] = img_entry
                else:
                    raise ValueError(f"Frame {i}: Missing image for {c_id}")

            dataset.add_frame(frame_dict)
            frame_buffer[i] = None  # Release memory as we go

        # 7. Save Episode
        dataset.save_episode()
        dataset.flush()
        dataset.finalize()
        
        result['success'] = True
        result['frames'] = num_frames
        result['dataset_path'] = str(episode_dir)
        
    except Exception as e:
        result['error'] = str(e)
        # Immediate logging for visibility
        sys.stderr.write(f"\n[ERROR] Processing {bag_path.name} failed: {e}\n")
        sys.stderr.flush()
        
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direct ROS2 to LeRobot Converter (Standard Mode)")
    
    # Input/Output
    parser.add_argument("--bags-dir", help="Directory containing .mcap files")
    parser.add_argument("--bag", help="Single .mcap file")
    parser.add_argument("--output-dir", required=True, help="Root output directory for merged dataset")
    
    # Configs
    parser.add_argument("--custom-processor", required=True, help="Path to python file defining MessageProcessors and Config")
    parser.add_argument("--mapping-file", required=True, help="Path to custom_state_action_mapping.py")
    
    # Dataset Meta
    parser.add_argument("--repo-id", required=True, help="HuggingFace Repo ID")
    parser.add_argument("--robot-type", required=True, help="Robot type name")
    parser.add_argument("--task-description", help="Task description")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel processes")
    
    # Video Encoding
    parser.add_argument("--vcodec", default="libsvtav1", help="Video codec (default: libsvtav1 for CPU)")
    parser.add_argument("--crf", type=int, default=30)
    
    args = parser.parse_args()
    
    # 1. Collect Bags
    bags = []
    if args.bag:
        bags.append(Path(args.bag))
    elif args.bags_dir:
        bd = Path(args.bags_dir)
        metas = sorted(list(bd.rglob("metadata.yaml")))
        if metas:
            bags = [p.parent for p in metas]
        else:
            bags = sorted(list(bd.rglob("*.mcap"))) + sorted(list(bd.rglob("*.db3")))
            
    if not bags:
        print("No bags found!")
        sys.exit(1)
        
    print(f"Found {len(bags)} bags. Using {args.workers} workers.")
    
    # 2. Parallel Processing (Separate Mode)
    output_root = Path(args.output_dir)
    separate_dir = output_root / "_separate_episodes"
    separate_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for i, bag in enumerate(bags):
        ep_name = f"episode_{i}_{bag.stem}"
        tasks.append((
            bag, separate_dir, ep_name, 
            args.custom_processor, args.mapping_file,
            f"{args.repo_id}/{ep_name}", 
            args.robot_type, args.task_description,
            args.fps, args.vcodec, args.crf
        ))
        
    success_datasets = []
    failed_episodes = [] 
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(convert_bag_worker, task): task for task in tasks}
        
        with tqdm(total=len(bags), desc="Converting Episodes") as pbar:
            for future in as_completed(futures):
                res = future.result()
                if res['success']:
                    success_datasets.append(Path(res['dataset_path']))
                else:
                    err_msg = f"{res['bag']} (Error: {res['error']})"
                    failed_episodes.append(err_msg)
                    tqdm.write(f"FAILED: {err_msg}")
                pbar.update(1)
                
    # 3. Summary & Merge
    print(f"\nConversion Finished.")
    print(f"Success: {len(success_datasets)}")
    print(f"Failed: {len(failed_episodes)}")
    
    if failed_episodes:
        print("\n=== Failed Episodes ===")
        for fail_msg in failed_episodes:
            print(f"- {fail_msg}")
        print("=======================\n")
    
    if not success_datasets:
        print("No valid datasets generated. Exiting.")
        sys.exit(1)
        
    print("Merging into final dataset...")
    datasets_to_merge = []
    for dpath in success_datasets:
        try:
            ds = LeRobotDataset(root=dpath, repo_id=dpath.name)
            datasets_to_merge.append(ds)
        except Exception as e:
            print(f"Failed to load {dpath} for merging: {e}")

    if datasets_to_merge:
        merge_target = output_root / "_merged_tmp"
        if merge_target.exists():
            shutil.rmtree(merge_target)

        merged = merge_datasets(
            datasets=datasets_to_merge,
            output_dir=merge_target,
            output_repo_id=args.repo_id
        )
        # Clean previous merged outputs except separate episodes
        for existing in list(output_root.iterdir()):
            if existing.name == "_separate_episodes":
                continue
            if existing == merge_target:
                continue
            if existing.is_dir():
                shutil.rmtree(existing)
            else:
                existing.unlink()

        # Move merged dataset into final location
        for item in merge_target.iterdir():
            destination = output_root / item.name
            if destination.exists():
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink()
            shutil.move(str(item), destination)

        if merge_target.exists():
            shutil.rmtree(merge_target)

        print(f"Merge Complete! Final dataset at: {output_root}")
        print(f"Total Frames: {merged.meta.total_frames}")

        # Cleanup temporary episodes once final dataset is ready
        if separate_dir.exists():
            shutil.rmtree(separate_dir)
            print("Removed temporary _separate_episodes directory.")

    # Cleanup separate (Optional: ask user)
    # shutil.rmtree(separate_dir)

