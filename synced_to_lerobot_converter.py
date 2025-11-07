#!/usr/bin/env python3
"""
HDF5 to LeRobot Dataset Converter

转换 HDF5 数据集到 LeRobot 格式，图像从外部目录读取。

数据结构:
input_dir/
├── episode_0000/
│   ├── data.h5              # 状态、动作、时间戳
│   └── images/              # 图像目录
│       ├── top_cam/
│       │   └── rgb/         # RGB 图像子目录
│       │       ├── frame_000000.png
│       │       └── ...
│       └── left_cam/
│           └── rgb/
│               ├── frame_000000.png
│               └── ...
│
注意: 支持两种结构:
  1. images/camera_name/rgb/frame_*.png  (推荐)
  2. images/camera_name/frame_*.png      (兼容)
"""

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional
import importlib.util

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features


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
    """查找 episode 目录下的相机图像目录
    
    支持两种结构：
    1. images/top_cam/frame_*.png
    2. images/top_cam/rgb/frame_*.png  (推荐)
    """
    images_dir = episode_dir / "images"
    if not images_dir.exists():
        return {}
    
    camera_dirs = {}
    for cam_dir in images_dir.iterdir():
        if cam_dir.is_dir():
            # 检查是否有 rgb 子目录
            rgb_dir = cam_dir / "rgb"
            if rgb_dir.exists() and rgb_dir.is_dir():
                camera_dirs[cam_dir.name] = rgb_dir
            else:
                camera_dirs[cam_dir.name] = cam_dir
    
    return camera_dirs


def load_image(image_path: Path) -> np.ndarray:
    """加载图像并转换为 numpy 数组"""
    img = Image.open(image_path)
    # 转换为 RGB（如果是 RGBA 或其他格式）
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)


def get_image_shape_from_dir(camera_dir: Path) -> tuple:
    """从相机目录中读取第一张图像获取尺寸
    
    Args:
        camera_dir: 相机图像目录（已经是 rgb 子目录或直接的图像目录）
    
    Returns:
        图像尺寸 (H, W, C)
    """
    image_files = sorted(camera_dir.glob("frame_*.png"))
    if not image_files:
        image_files = sorted(camera_dir.glob("*.png"))
    if not image_files:
        raise ValueError(f"No images found in {camera_dir}")
    
    first_img = load_image(image_files[0])
    return first_img.shape  # (H, W, C)


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


def convert_hdf5_to_lerobot(
    input_dir: str,
    output_dir: str,
    repo_id: str,
    fps: int = 30,
    robot_type: str = "custom_robot",
    mapping_file: Optional[str] = None,
    camera_keys: Optional[List[str]] = None
):
    """
    转换 HDF5 数据集到 LeRobot 格式
    
    Args:
        input_dir: 包含 episode 目录的输入路径
        output_dir: LeRobot 数据集输出目录
        repo_id: 数据集仓库 ID（例如：username/dataset-name）
        fps: 帧率
        robot_type: 机器人类型
        mapping_file: 自定义映射文件路径
        camera_keys: 相机名称列表（如 ["top_cam", "left_cam"]），None 则自动检测
    """
    
    # 加载自定义映射
    if mapping_file:
        print(f"Loading custom mapping from {mapping_file}")
        mapping = load_custom_mapping(mapping_file)
    else:
        # 默认映射
        from custom_state_action_mapping import get_state_action_mapping
        mapping = get_state_action_mapping()
    
    print(f"State components: {mapping.state_components}")
    print(f"Action components: {mapping.action_components}")
    
    # 查找所有 episode 目录
    input_path = Path(input_dir)
    episode_dirs = sorted([d for d in input_path.iterdir() 
                          if d.is_dir() and (d / 'data.h5').exists()])
    
    if not episode_dirs:
        raise ValueError(f"No episodes found in {input_dir}")
    
    print(f"Found {len(episode_dirs)} episodes")
    
    # 读取第一个 episode 以确定特征维度
    first_ep_dir = episode_dirs[0]
    with h5py.File(first_ep_dir / 'data.h5', 'r') as f:
        # 读取状态/动作维度
        state_components = extract_components(f['state'], 0, mapping.state_components)
        action_components = extract_components(f['action'], 0, mapping.action_components)
        
        state_dim = mapping.state_combine_fn(state_components).shape[0]
        action_dim = mapping.action_combine_fn(action_components).shape[0]
    
    # 查找相机图像目录
    camera_dirs = find_camera_dirs(first_ep_dir)
    if not camera_dirs:
        raise ValueError(f"No camera images found in {first_ep_dir / 'images'}")
    
    # 确定要使用的相机
    if camera_keys is None:
        camera_keys = list(camera_dirs.keys())
    
    # 获取每个相机的图像尺寸
    camera_shapes = {}
    for cam_key in camera_keys:
        if cam_key not in camera_dirs:
            raise ValueError(f"Camera '{cam_key}' not found in {first_ep_dir / 'images'}")
        camera_shapes[cam_key] = get_image_shape_from_dir(camera_dirs[cam_key])
    
    print(f"\nDataset dimensions:")
    print(f"  State: {state_dim}")
    print(f"  Action: {action_dim}")
    for cam_key, shape in camera_shapes.items():
        print(f"  {cam_key}: {shape}")
    
    # 使用 hw_to_dataset_features 构建特征
    # 1. 观测特征（状态 + 图像）
    obs_hw_features = {
        **{f"state_{i}": float for i in range(state_dim)},
        **{cam_key: shape for cam_key, shape in camera_shapes.items()}
    }
    obs_features = hw_to_dataset_features(obs_hw_features, "observation", use_video=True)
    
    # 2. 动作特征
    action_hw_features = {f"action_{i}": float for i in range(action_dim)}
    action_features = hw_to_dataset_features(action_hw_features, "action", use_video=False)
    
    # 3. 合并特征 - 这是 LeRobot 格式的 features 字典
    dataset_features = {**obs_features, **action_features}
    
    # 注意：不要转换为 HuggingFace Features！
    # LeRobotDataset.create() 内部会自动处理转换
    
    print(f"\nDataset features:")
    for key, value in dataset_features.items():
        print(f"  {key}: {value}")
    
    # 清理输出目录
    output_path = Path(output_dir)
    if output_path.exists():
        print(f"\nRemoving existing output directory: {output_path}")
        shutil.rmtree(output_path)
        time.sleep(0.1)
    
    # 创建数据集
    print("\nCreating LeRobot dataset...")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=output_path,
        robot_type=robot_type,
        features=dataset_features,  # 直接传递 LeRobot 格式的 features 字典
        use_videos=True,
        image_writer_threads=4,
    )
    
    # 收集统计数据
    all_states = []
    all_actions = []
    
    # 处理每个 episode
    for ep_idx, ep_dir in enumerate(tqdm(episode_dirs, desc="Converting episodes")):
        h5_file = ep_dir / 'data.h5'
        camera_dirs = find_camera_dirs(ep_dir)
        
        with h5py.File(h5_file, 'r') as f:
            frame_counts = []
            for cam_key in camera_keys:
                if cam_key not in camera_dirs:
                    raise ValueError(f"Camera '{cam_key}' not found in {ep_dir / 'images'}")
                cam_dir = camera_dirs[cam_key]
                frame_files = sorted(cam_dir.glob("frame_*.png"))
                if not frame_files:
                    frame_files = sorted(cam_dir.glob("*.png"))
                if not frame_files:
                    raise ValueError(f"No images found for camera '{cam_key}' in {cam_dir}")
                frame_counts.append(len(frame_files))

            if not frame_counts:
                raise ValueError(f"No camera images available in {ep_dir / 'images'}")

            num_frames = min(frame_counts)
            
            # 读取任务描述
            task = f.attrs.get('task', 'default_task')
            if isinstance(task, bytes):
                task = task.decode('utf-8')
            
            for frame_idx in range(num_frames):
                # 提取状态
                state_components = extract_components(
                    f['state'], frame_idx, mapping.state_components)
                state = mapping.state_combine_fn(state_components)
                all_states.append(state)
                
                # 提取动作
                action_components = extract_components(
                    f['action'], frame_idx, mapping.action_components)
                action = mapping.action_combine_fn(action_components)
                all_actions.append(action)
                
                # 读取图像
                images = {}
                for cam_key in camera_keys:
                    img_file = camera_dirs[cam_key] / f"frame_{frame_idx:06d}.png"
                    if not img_file.exists():
                        # 尝试其他可能的命名格式
                        img_files = sorted(camera_dirs[cam_key].glob("*.png"))
                        if frame_idx < len(img_files):
                            img_file = img_files[frame_idx]
                        else:
                            raise FileNotFoundError(f"Image not found: {img_file}")
                    
                    images[f"observation.images.{cam_key}"] = load_image(img_file)

                
                # 构建帧数据
                frame = {
                    "observation.state": state,
                    **images,  # 展开所有相机图像
                    "action": action,
                    "task": task,
                }
                
                dataset.add_frame(frame)
        
        # 保存 episode
        dataset.save_episode()
    
    # 计算并保存统计信息
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
        }
    }
    
    # 保存统计信息
    stats_path = output_path / "meta" / "stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Conversion Complete!")
    print(f"{'='*60}")
    print(f"Dataset saved to: {output_path}")
    print(f"Total episodes: {len(episode_dirs)}")
    print(f"Total frames: {len(all_states)}")
    print(f"\nState stats - shape: {state_stats['mean'].shape}")
    print(f"  Mean: {state_stats['mean'][:3]}...")
    print(f"  Std:  {state_stats['std'][:3]}...")
    print(f"\nAction stats - shape: {action_stats['mean'].shape}")
    print(f"  Mean: {action_stats['mean'][:3]}...")
    print(f"  Std:  {action_stats['std'][:3]}...")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HDF5 + external images to LeRobot dataset"
    )
    parser.add_argument(
        "--input-dir", 
        required=True, 
        help="Input directory with episode folders"
    )
    parser.add_argument(
        "--output-dir", 
        required=True, 
        help="Output directory for LeRobot dataset"
    )
    parser.add_argument(
        "--repo-id", 
        required=True, 
        help="Dataset repo ID (e.g., user/dataset-name)"
    )
    parser.add_argument(
        "--fps", 
        type=int, 
        default=30, 
        help="Frames per second (default: 30)"
    )
    parser.add_argument(
        "--robot-type", 
        default="custom_robot", 
        help="Robot type name"
    )
    parser.add_argument(
        "--mapping-file", 
        help="Path to custom mapping file (custom_state_action_mapping.py)"
    )
    parser.add_argument(
        "--camera-keys",
        nargs="+",
        help="Camera names to include (e.g., top_cam left_cam). If not specified, all cameras are used."
    )
    
    args = parser.parse_args()
    
    convert_hdf5_to_lerobot(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        fps=args.fps,
        robot_type=args.robot_type,
        mapping_file=args.mapping_file,
        camera_keys=args.camera_keys
    )


if __name__ == "__main__":
    main()