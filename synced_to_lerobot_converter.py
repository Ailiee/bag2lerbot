#!/usr/bin/env python3
"""
Optimized HDF5 to LeRobot Dataset Converter

优化点：
1. 自定义 Dataset 类，彻底移除 add_frame 中的图片 I/O 操作 (解决频繁 IO 问题)。
2. 集成 FFmpeg 硬件编码到 save_episode 流程中，直接从源读取 (直通编码)。
3. 修复了 'action'/'state' 形状校验错误。
4. 显式添加图像元数据以支持训练。
5. [新增] 多线程并行编码多相机视频，恢复并提升并发效率。
"""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional
import importlib.util
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import (
    hw_to_dataset_features,
    validate_frame,
    update_chunk_file_indices,
    get_file_size_in_mb,
    write_info
)
# 修复 Import 错误：从 video_utils 导入视频相关函数
from lerobot.datasets.video_utils import (
    get_video_duration_in_s,
    concatenate_video_files,
    get_video_info
)
from lerobot.datasets.compute_stats import compute_episode_stats


def load_custom_mapping(mapping_file: str):
    """加载自定义映射文件"""
    spec = importlib.util.spec_from_file_location("custom_mapping", mapping_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if hasattr(module, 'get_state_action_mapping'):
        return module.get_state_action_mapping()
    raise ValueError("Mapping file must have get_state_action_mapping() function")


def extract_components(h5_group: h5py.Group, frame_idx: int, 
                       component_paths: List[str]) -> Dict[str, np.ndarray]:
    """从 HDF5 提取指定组件"""
    components = {}
    for path in component_paths:
        try:
            dataset = h5_group[path]
            data = dataset[frame_idx] if len(dataset) > frame_idx else dataset[0]
            components[path] = np.asarray(data, dtype=np.float32)
        except KeyError:
            continue
    return components


def find_camera_dirs(episode_dir: Path) -> Dict[str, Path]:
    """查找 episode 目录下的相机图像目录"""
    images_dir = episode_dir / "images"
    if not images_dir.exists():
        return {}
    
    camera_dirs = {}
    for cam_dir in images_dir.iterdir():
        if cam_dir.is_dir():
            rgb_dir = cam_dir / "rgb"
            if rgb_dir.exists() and rgb_dir.is_dir():
                camera_dirs[cam_dir.name] = rgb_dir
            else:
                camera_dirs[cam_dir.name] = cam_dir
    
    return camera_dirs


def load_image(image_path: Path) -> np.ndarray:
    """加载图像并转换为 numpy 数组"""
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)


def get_image_shape_from_dir(camera_dir: Path) -> tuple:
    """从相机目录中读取第一张图像获取尺寸"""
    image_files = sorted(camera_dir.glob("frame_*.png"))
    if not image_files:
        image_files = sorted(camera_dir.glob("*.png"))
    if not image_files:
        raise ValueError(f"No images found in {camera_dir}")
    
    first_img = load_image(image_files[0])
    return first_img.shape  # (H, W, C)


def list_frame_files(camera_dir: Path) -> List[Path]:
    """获取按顺序排列的 PNG 帧列表。"""
    frame_files = sorted(camera_dir.glob("frame_*.png"))
    if not frame_files:
        frame_files = sorted(camera_dir.glob("*.png"))
    if not frame_files:
        raise ValueError(f"No images found in {camera_dir}")
    return frame_files


def encode_video_from_image_files(
    frame_files: List[Path],
    video_path: Path,
    fps: int,
    vcodec: str,
    pix_fmt: str,
    crf: int,
    g: int,
    fast_decode: bool,
):
    """使用 ffmpeg 将磁盘上的图像帧直接编码为视频，避免加载到 Python 内存。"""
    if not frame_files:
        raise ValueError("No frames provided for video encoding")
    
    # 移除 PIL 读取第一帧获取尺寸的逻辑，image2pipe 不需要预先知道尺寸，ffmpeg 会从流中读取
    video_path.parent.mkdir(parents=True, exist_ok=True)

    # 构建 FFmpeg 命令
    # 优化：使用 image2pipe 模式，直接传输 png 文件流，让 ffmpeg 解码，速度远快于 Python PIL 解码
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "image2pipe",
        "-vcodec", "png",   # 告诉 ffmpeg 输入流是 png 格式
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-c:v", vcodec,
        "-pix_fmt", pix_fmt,
        "-g", str(g),
    ]
    
    # 根据编码器调整参数
    if "libsvtav1" in vcodec:
        cmd.extend(["-crf", str(crf), "-preset", "10"])
        if fast_decode:
            cmd.extend(["-tune", "fastdecode"])
    elif "nvenc" in vcodec:
        # NVENC 使用 -cq 控制质量 (0-51)，-preset p1-p7 (p4=medium)
        # 注意：av1_nvenc 不支持 -crf
        # 修复: Gop Length should be greater than number of B frames + 1.
        # 当 -g 2 时，必须设置 -bf 0 以避免 "InitializeEncoder failed: invalid param (8)"
        cmd.extend(["-cq", str(crf), "-preset", "p4", "-bf", "0"])
        if fast_decode:
            # NVENC 没有 fastdecode tune，忽略
            pass
    else:
        # libx264, libx265 等
        cmd.extend(["-crf", str(crf), "-preset", "fast"])
        if fast_decode:
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
        # 优化：直接读取文件二进制数据写入管道，不进行 PIL 解码
        for frame_path in frame_files:
            with open(frame_path, "rb") as f:
                # 直接传输文件字节，极快
                process.stdin.write(f.read())
        process.stdin.flush()
    except (BrokenPipeError, IOError) as exc:
        # FFmpeg process likely died
        try:
            # Try to capture error output
            stdout, stderr = process.communicate()
            error_msg = stderr.decode('utf-8', errors='ignore')
        except Exception:
            error_msg = "Unknown error (could not capture stderr)"
            
        raise RuntimeError(
            f"FFmpeg encoding failed for {video_path}.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Error Output:\n{error_msg}"
        ) from exc

    try:
        process.stdin.close()
    except (BrokenPipeError, IOError):
        pass
        
    process.wait()
    
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg encoding failed for {video_path}")


def compute_stats(all_data: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """计算统计信息"""
    if not all_data:
        return {"mean": np.array([]), "std": np.array([]), 
                "min": np.array([]), "max": np.array([])}
    
    data_array = np.stack(all_data)
    return {
        "mean": data_array.mean(axis=0),
        "std": data_array.std(axis=0),
        "min": data_array.min(axis=0),
        "max": data_array.max(axis=0)
    }


class LeRobotDatasetOptimized(LeRobotDataset):
    """
    优化后的 Dataset 类：
    1. add_frame 时完全跳过视频特征的磁盘写入，避免 I/O。
    2. save_episode 时使用多线程并行编码所有相机的视频。
    3. 视频写入采用延迟合并策略，避免 O(N^2) 的文件复制开销。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_custom_attributes()

    def _init_custom_attributes(self):
        self.current_image_paths = {}  # 存储当前 episode 的源图片路径
        # 编码参数默认值
        self.vcodec = "libsvtav1"
        self.crf = 30
        self.pix_fmt = "yuv420p"
        self.gop = 2
        self.encoding_threads = 4 # 默认并行编码线程数
        
        # 视频分块状态: {video_key: {'chunk_idx': int, 'file_idx': int, 'pending_files': List[Path], 'current_chunk_size': float, 'current_chunk_duration': float}}
        self.video_chunk_state = {}

    @classmethod
    def create(cls, *args, **kwargs):
        obj = super().create(*args, **kwargs)
        obj._init_custom_attributes()
        return obj

    def set_current_image_paths(self, paths: Dict[str, List[Path]]):
        """设置当前 episode 的源图片路径映射"""
        self.current_image_paths = paths

    def add_frame(self, frame: dict) -> None:
        """重写 add_frame，跳过视频数据的磁盘写入"""
        # 转换 torch tensor
        for name in frame:
            if isinstance(frame[name], (np.ndarray, list)):
                pass # keep as is
            elif hasattr(frame[name], "numpy"):
                frame[name] = frame[name].numpy()

        validate_frame(frame, self.features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        frame_index = self.episode_buffer["size"]
        timestamp = frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["task"].append(frame.pop("task"))

        for key in frame:
            if self.features[key]["dtype"] in ["image", "video"]:
                # 关键优化：不保存图片，只存占位符
                self.episode_buffer[key].append("placeholder")
            else:
                self.episode_buffer[key].append(frame[key])
        
        self.episode_buffer["size"] += 1

    def save_episode(self, episode_data: dict | None = None) -> None:
        """
        重写 save_episode 以支持多线程并行编码视频。
        """
        episode_buffer = episode_data if episode_data is not None else self.episode_buffer
        
        # 1. 基础处理 (复制自父类逻辑)
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        self.meta.save_episode_tasks(episode_tasks)
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        # 2. 计算统计信息
        # 优化：仅计算向量特征的统计信息，跳过图像统计
        vector_features = {k: v for k, v in self.features.items() if v["dtype"] not in ["image", "video"]}
        vector_buffer = {k: episode_buffer[k] for k in vector_features if k in episode_buffer}
        
        ep_stats = compute_episode_stats(vector_buffer, vector_features)
        
        ep_metadata = self._save_episode_data(episode_buffer)

        # 3. 并行视频编码 (核心优化)
        has_video_keys = len(self.meta.video_keys) > 0
        
        if has_video_keys:
            # 使用线程池并行编码不同视角的视频
            with ThreadPoolExecutor(max_workers=self.encoding_threads) as executor:
                futures = {
                    executor.submit(self._save_episode_video, video_key, episode_index): video_key
                    for video_key in self.meta.video_keys
                }
                for future in futures:
                    video_key = futures[future]
                    try:
                        metadata = future.result()
                        ep_metadata.update(metadata)
                    except Exception as e:
                        print(f"Error encoding video {video_key}: {e}")
                        raise e

        # 4. 保存元数据
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats, ep_metadata)

        if not episode_data:
            self.clear_episode_buffer(delete_images=False)

    def _init_video_state(self, video_key: str, episode_index: int):
        """初始化视频分块状态"""
        if video_key in self.video_chunk_state:
            return

        # Initialize state from metadata or defaults
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

        # Check existing file
        video_path = self.root / self.meta.video_path.format(
            video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
        )
        
        pending_files = []
        current_chunk_size = 0.0
        
        if video_path.exists():
            # 如果存在现有文件，将其作为第一个 pending file
            pending_files.append(video_path)
            current_chunk_size = get_file_size_in_mb(video_path)
        
        self.video_chunk_state[video_key] = {
            'chunk_idx': chunk_idx,
            'file_idx': file_idx,
            'pending_files': pending_files,
            'current_chunk_size': current_chunk_size,
            'current_chunk_duration': latest_duration_in_s
        }

    def _flush_chunk(self, video_key: str):
        """将 pending 的视频文件合并写入磁盘"""
        state = self.video_chunk_state[video_key]
        if not state['pending_files']:
            return

        chunk_idx, file_idx = state['chunk_idx'], state['file_idx']
        video_path = self.root / self.meta.video_path.format(
            video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
        )
        
        # 如果只有一个文件且就是目标文件，无需操作
        if len(state['pending_files']) == 1 and state['pending_files'][0] == video_path:
            state['pending_files'] = []
            return

        valid_files = [p for p in state['pending_files'] if p.exists()]
        if not valid_files:
            state['pending_files'] = []
            return

        # print(f"Flushing chunk {chunk_idx}/{file_idx} for {video_key} with {len(valid_files)} files...")
        concatenate_video_files(valid_files, video_path)
        
        # 删除临时文件 (除了目标文件本身)
        for p in valid_files:
            if p != video_path:
                try:
                    p.unlink()
                except OSError:
                    pass
        
        state['pending_files'] = []

    def flush(self):
        """Flush all pending videos to disk."""
        for video_key in self.video_chunk_state:
            self._flush_chunk(video_key)

    def _save_episode_video(self, video_key: str, episode_index: int) -> dict:
        """重写视频保存逻辑，使用延迟合并策略"""
        self._init_video_state(video_key, episode_index)
        state = self.video_chunk_state[video_key]
        
        # 1. 编码到临时文件
        # 使用 videos/{video_key}/temp/ 目录
        temp_dir = self.root / "videos" / video_key / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_video_path = temp_dir / f"ep_{episode_index}.mp4"
        
        if video_key not in self.current_image_paths:
            raise ValueError(f"No source images found for {video_key}")
        source_files = self.current_image_paths[video_key]
        
        encode_video_from_image_files(
            source_files,
            temp_video_path,
            self.fps,
            self.vcodec,
            self.pix_fmt,
            self.crf,
            self.gop,
            fast_decode=True
        )
        
        ep_duration = get_video_duration_in_s(temp_video_path)
        ep_size = get_file_size_in_mb(temp_video_path)
        
        # 2. 检查是否需要切换 Chunk
        if state['current_chunk_size'] + ep_size >= self.meta.video_files_size_in_mb:
            # 先 Flush 当前 Chunk
            self._flush_chunk(video_key)
            
            # 切换到下一个 Chunk
            state['chunk_idx'], state['file_idx'] = update_chunk_file_indices(
                state['chunk_idx'], state['file_idx'], self.meta.chunks_size
            )
            state['current_chunk_size'] = 0.0
            state['current_chunk_duration'] = 0.0
            # pending_files 已清空
        
        # 3. 添加到 Pending
        state['pending_files'].append(temp_video_path)
        state['current_chunk_size'] += ep_size
        
        from_timestamp = state['current_chunk_duration']
        state['current_chunk_duration'] += ep_duration
        
        metadata = {
            "episode_index": episode_index,
            f"videos/{video_key}/chunk_index": state['chunk_idx'],
            f"videos/{video_key}/file_index": state['file_idx'],
            f"videos/{video_key}/from_timestamp": from_timestamp,
            f"videos/{video_key}/to_timestamp": state['current_chunk_duration'],
        }
        
        # 4. 更新 Info (如果是第一个 episode)
        if episode_index == 0:
            # self.meta.update_video_info(video_key)
            # FIX: 由于延迟写入，此时 chunk 文件还不存在，使用临时文件获取元数据
            self.meta.info["features"][video_key]["info"] = get_video_info(temp_video_path)
            write_info(self.meta.info, self.meta.root)
            
        return metadata


def convert_hdf5_to_lerobot_optimized(
    input_dir: str,
    output_dir: str,
    repo_id: str,
    fps: int = 30,
    robot_type: str = "custom_robot",
    mapping_file: Optional[str] = None,
    camera_keys: Optional[List[str]] = None,
    use_hardware_encoding: bool = True,
    vcodec: str = "libsvtav1",
    crf: int = 30,
    batch_size: int = 4,
):
    # 加载映射
    if mapping_file:
        print(f"Loading custom mapping from {mapping_file}")
        mapping = load_custom_mapping(mapping_file)
    else:
        from ur_state_action_mapping import get_state_action_mapping
        mapping = get_state_action_mapping()

    print(f"State components: {mapping.state_components}")
    print(f"Action components: {mapping.action_components}")

    # 查找所有 episode
    input_path = Path(input_dir)
    episode_dirs = sorted(
        [d for d in input_path.iterdir() if d.is_dir() and (d / "data.h5").exists()]
    )
    if not episode_dirs:
        raise ValueError(f"No episodes found in {input_dir}")
    print(f"Found {len(episode_dirs)} episodes")

    # 读取第一个 episode 确定维度
    first_ep_dir = episode_dirs[0]
    with h5py.File(first_ep_dir / "data.h5", "r") as f:
        state_components = extract_components(f["state"], 0, mapping.state_components)
        action_components = extract_components(f["action"], 0, mapping.action_components)
        state_dim = mapping.state_combine_fn(state_components).shape[0]
        action_dim = mapping.action_combine_fn(action_components).shape[0]

    camera_dirs = find_camera_dirs(first_ep_dir)
    if not camera_dirs:
        raise ValueError(f"No camera images found in {first_ep_dir / 'images'}")

    if camera_keys is None:
        camera_keys = list(camera_dirs.keys())

    camera_shapes = {}
    for cam_key in camera_keys:
        if cam_key not in camera_dirs:
            raise ValueError(f"Camera '{cam_key}' not found")
        camera_shapes[cam_key] = get_image_shape_from_dir(camera_dirs[cam_key])

    print("\nDataset dimensions:")
    print(f"  State: {state_dim}")
    print(f"  Action: {action_dim}")
    for cam_key, shape in camera_shapes.items():
        print(f"  {cam_key}: {shape}")

    # 构建特征定义
    obs_hw_features = {f"state_{i}": float for i in range(state_dim)}
    obs_hw_features.update({cam_key: shape for cam_key, shape in camera_shapes.items()})
    
    obs_features = hw_to_dataset_features(
        obs_hw_features, "observation", use_video=True
    )

    action_hw_features = {f"action_{i}": float for i in range(action_dim)}
    action_features = hw_to_dataset_features(action_hw_features, "action", use_video=False)
    dataset_features = {**obs_features, **action_features}

    # 显式定义向量特征
    dataset_features["observation.state"] = {
        "dtype": "float32",
        "shape": (state_dim,),
        "names": None,
    }
    dataset_features["action"] = {
        "dtype": "float32",
        "shape": (action_dim,),
        "names": None,
    }

    # 清理输出目录
    output_path = Path(output_dir)
    if output_path.exists():
        print(f"\nRemoving existing output directory: {output_path}")
        shutil.rmtree(output_path)
        time.sleep(0.1)

    # 使用自定义的优化 Dataset 类
    # image_writer_threads=0 因为我们不写图片，但我们开启内部 encoding_threads
    dataset = LeRobotDatasetOptimized.create(
        repo_id=repo_id,
        fps=fps,
        root=output_path,
        robot_type=robot_type,
        features=dataset_features,
        use_videos=True,
        image_writer_threads=0, 
    )
    
    # 设置编码参数
    dataset.vcodec = vcodec
    dataset.crf = crf
    dataset.pix_fmt = "yuv420p"  # 显式设置 pix_fmt
    dataset.gop = 2              # 显式设置 gop
    # 设置并行编码线程数 (对应 batch_size 参数，或者固定为 4)
    dataset.encoding_threads = batch_size if batch_size > 0 else 4

    all_states = []
    all_actions = []
    total_frames = 0

    # 循环处理
    for ep_idx, ep_dir in enumerate(tqdm(episode_dirs, desc="Converting episodes")):
        h5_file = ep_dir / "data.h5"
        ep_camera_dirs = find_camera_dirs(ep_dir)
        
        # 获取本 episode 的图像文件列表
        frame_files_per_cam = {}
        frame_counts = []
        # 构建映射：feature_key -> file_list
        image_paths_map = {}
        
        for cam_key in camera_keys:
            files = list_frame_files(ep_camera_dirs[cam_key])
            frame_files_per_cam[cam_key] = files
            frame_counts.append(len(files))
            # 对应 dataset 中的 key
            image_paths_map[f"observation.images.{cam_key}"] = files
        
        num_frames = min(frame_counts)
        
        # 将源文件路径传递给 dataset，供 _save_episode_video 使用
        dataset.set_current_image_paths(image_paths_map)

        with h5py.File(h5_file, "r") as f:
            task = f.attrs.get("task", "default_task")
            if isinstance(task, bytes):
                task = task.decode("utf-8")

            for frame_idx in range(num_frames):
                state = mapping.state_combine_fn(
                    extract_components(f["state"], frame_idx, mapping.state_components)
                )
                action = mapping.action_combine_fn(
                    extract_components(f["action"], frame_idx, mapping.action_components)
                )
                
                all_states.append(state)
                all_actions.append(action)

                # 构建帧数据，不包含图像数据
                frame = {
                    "observation.state": np.asarray(state, dtype=np.float32),
                    "action": np.asarray(action, dtype=np.float32),
                    "task": task,
                }
                
                # 填充占位符
                for cam_key in camera_keys:
                    # FIX: 使用正确尺寸的占位符以通过 validate_frame 校验
                    # 虽然我们不写入磁盘，但 validate_frame 需要检查形状
                    shape = camera_shapes[cam_key]
                    # 使用 uint8 类型的零矩阵，开销很小
                    frame[f"observation.images.{cam_key}"] = np.zeros(shape, dtype=np.uint8)

                dataset.add_frame(frame)

        # 保存 episode，此时会调用我们重写的 save_episode 并行编码
        dataset.save_episode()
        total_frames += num_frames

    # Flush pending videos
    print("\nFlushing pending video chunks...")
    dataset.flush()

    # 计算统计信息
    print("\nComputing statistics...")
    state_stats = compute_stats(all_states)
    action_stats = compute_stats(all_actions)

    stats = {
        "observation.state": {
            "mean": state_stats["mean"].tolist(),
            "std": state_stats["std"].tolist(),
            "min": state_stats["min"].tolist(),
            "max": state_stats["max"].tolist(),
        },
        "action": {
            "mean": action_stats["mean"].tolist(),
            "std": action_stats["std"].tolist(),
            "min": action_stats["min"].tolist(),
            "max": action_stats["max"].tolist(),
        },
    }

    stats_path = output_path / "meta" / "stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print("Conversion Complete!")
    print(f"{'='*60}")
    print(f"Dataset saved to: {output_path}")
    print(f"Total episodes: {len(episode_dirs)}")
    print(f"Total frames: {total_frames}")
    print(f"Video codec: {vcodec}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Optimized HDF5 + external images to LeRobot dataset converter"
    )
    parser.add_argument("--input-dir", required=True, help="Input directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--repo-id", required=True, help="Dataset repo ID")
    parser.add_argument("--fps", type=int, default=30, help="FPS")
    parser.add_argument("--robot-type", default="custom_robot", help="Robot type")
    parser.add_argument("--mapping-file", help="Custom mapping file")
    parser.add_argument("--camera-keys", nargs="+", help="Camera names")
    parser.add_argument("--use-hardware-encoding", action="store_true", default=True, help="Use hardware encoding")
    parser.add_argument("--vcodec", default="libsvtav1", help="Video codec")
    parser.add_argument("--crf", type=int, default=30, help="CRF")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of threads for parallel video encoding")
    
    args = parser.parse_args()
    
    convert_hdf5_to_lerobot_optimized(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        fps=args.fps,
        robot_type=args.robot_type,
        mapping_file=args.mapping_file,
        camera_keys=args.camera_keys,
        use_hardware_encoding=args.use_hardware_encoding,
        vcodec=args.vcodec,
        crf=args.crf,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()