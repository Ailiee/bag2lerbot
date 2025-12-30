# OpenLoong to LeRobot Converter

这是一个用于将 ROS2 数据转换为 LeRobot 数据集格式的通用工具包。本工具包经过深度优化，显著提升了数据处理效率。

## 主要特性

### 1. 优化同步代码
解决了数据同步过程中的性能瓶颈。
- **性能表现**: 76个数据集仅需 **13分钟** 即可完成同步。

### 2. 优化转换代码
针对视频编码和数据处理流程进行了深度优化。
- **性能表现**: 76个数据集仅需 **11分钟** 即可完成转换（使用 GPU 资源）。
- **对比**: 相比之前的硬件编码方案（60条数据需2小时），效率大幅提升。
- **技术改进**:
    1.  **并行化运行**: 开启多线程并行编译，充分利用计算资源。
    2.  **减少 I/O**: 优化了数据流，直接基于源图像目录进行视频编码，避免了将图像复制到数据集目录的冗余 I/O 操作。

## 安装

需要安装特定版本的 `rosbags` 库：

```bash
# 特定版本 rosbag 包
pip install rosbags==0.10.4 --index-url https://pypi.org/simple
```

## 使用指南

### 第一步：ROS2 Bag 同步 (ROS2 to Synced Format)

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
