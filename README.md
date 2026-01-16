# OpenLoong to LeRobot Converter

这是一个用于将 ROS2 数据转换为 LeRobot 数据集格式的通用工具包。本工具包经过深度优化，显著提升了数据处理效率。

## 主要特性

### 1. 单步直出 LeRobot 数据集（`ros2_to_lerobot_direct.py`）
- 单遍读取 ROS2 bag，同步 + 编码 + 写数据一次完成。
- 即时视频编码（PyAV 优先，缺失时回落 FFmpeg），释放内存更快。
- 预编码视频直接落盘，跳过中间图片写入，I/O 最小化。
- 临时 episode 放在独立目录，最终合并时不会因目标目录已存在而报错。

### 2. 传统双步流程（保留）
- **同步阶段**: `ros2_to_lerobot_converter.py`，76 个数据集约 **13 分钟**。
- **转换阶段**: `synced_to_lerobot_converter.py`，76 个数据集约 **11 分钟**（GPU）。
- 核心优化：并行、最小化 I/O、硬件/软件编码任选。

## 安装

需要安装的关键依赖：

```bash
# rosbag 读取
pip install rosbags==0.10.4 --index-url https://pypi.org/simple

# 建议安装 PyAV（可选，缺失则自动用 FFmpeg 子进程）
pip install av
```

确保系统已装 FFmpeg（用于回退编码和探测）。

## 使用指南

### 方案 A：单步从 ROS2 Bag 直接生成 LeRobot 数据集（推荐）

```bash
python ros2_to_lerobot_direct.py \
    --bags-dir /path/to/bags \
    --output-dir /path/to/output_dataset \
    --repo-id your_repo_id \
    --robot-type your_robot \
    --task-description "Task description" \
    --custom-processor /workspace/openloong2lerobot/processors_qingloongROS2.py \
    --mapping-file /workspace/openloong2lerobot/custom_state_action_mapping_qingloongROS2.py \
    --fps 30 \
    --workers 4 \
    --vcodec libsvtav1 \
    --crf 30
```

说明：
- 输出目录若已存在会被清理，请提前备份。
- 预编码视频放在临时目录，合并完成后自动清理。
- `--workers` 控制并行处理 bag 数量，依机器 I/O 适当调节。

### 方案 B：ROS2 Bag 同步 (ROS2 to Synced Format)

使用 `ros2_to_lerobot_converter.py` 将 ROS2 bag 数据提取并同步。

**UR 机械臂示例:**

```bash
python ros2_to_lerobot_converter.py batch \
    --bags-dir=/media/aliee/data/UR/20251029 \
    --output-dir=/workspace/testDataOut \
    --custom-processor=/workspace/openloong2lerobot/processors_ur.py
```

**青龙 (QingLoong) 机器人示例:**

```bash
python ros2_to_lerobot_converter.py batch \
    --bags-dir=/media/aliee5/QL_ros2/20251121 \
    --output-dir=/media/aliee5/QL_ros2/20251121Out \
    --custom-processor=/workspace/openloong2lerobot/processors_qingloongROS2.py
```

### 第二步：转换为 LeRobot 数据集 (Synced to LeRobot Dataset)

使用 `synced_to_lerobot_converter.py` 将同步后的数据转换为最终的 LeRobot 数据集格式。

**UR 机械臂示例:**

```bash
python synced_to_lerobot_converter.py \
    --input-dir /workspace/testDataOut \
    --output-dir /workspace/testLerobot/UR20251029 \
    --repo-id UR20251029 \
    --fps=60 \
    --robot-type=ur_dual_arm \
    --mapping-file=/workspace/openloong2lerobot/custom_state_action_mapping_ur.py \
    --use-hardware-encoding \
    --vcodec libsvtav1 \
    --crf 30 \
    --batch-size 4
```

**青龙 (QingLoong) 机器人示例 (CPU):**

```bash
python synced_to_lerobot_converter.py \
    --input-dir /media/aliee5/QL_ros2/20251121OUT \
    --output-dir /media/aliee5/QL_ros2/qingloong_Foldingclothes_20251121 \
    --repo-id qingloong_Foldingclothes_20251121 \
    --fps=30 \
    --robot-type=qingloongROS2  \
    --mapping-file=/workspace/openloong2lerobot/custom_state_action_mapping_qingloongROS2.py \
    --use-hardware-encoding \
    --vcodec libsvtav1 \
    --crf 30 \
    --batch-size 4
```

**青龙 (QingLoong) 机器人示例 (GPU 加速):**

适用于上传 GitHub 或需要高性能转换的场景。

```bash
python synced_to_lerobot_converter.py \
    --input-dir /media/aliee5/QL_ros2/20251121Out \
    --output-dir /media/aliee5/QL_ros2/qingloong_Foldingclothes_20251121 \
    --repo-id qingloong_Foldingclothes_20251121 \
    --fps=30 \
    --robot-type=qingloongROS2  \
    --mapping-file=/workspace/openloong2lerobot/custom_state_action_mapping_qingloongROS2.py \
    --use-hardware-encoding \
    --vcodec av1_nvenc \
    --crf 25 \
    --batch-size 6
```
