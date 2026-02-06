#!/bin/bash


# 1. 从华为云拉取原始数据
/workspace/rclone/rclone copy -P --config="/workspace/rclone/rclone.conf"\
  "huawei-cloud:/openloong-apps-prod-private/data-collector-svc/raw/${task_id}" \
  "/qinglong_datasets/qinglong/raw/${task_id}"

# # 2. 执行预处理脚本
/workspace/code/bag2lerbot/script-extract.sh \
  /qinglong_datasets/qinglong/raw/${task_id}

# 3. 将 ROS2 bag 直接转换 LeRobot 
python /workspace/code/bag2lerbot/ros2_to_lerobot_direct.py \
    --bags-dir /qinglong_datasets/qinglong/raw/${task_id} \
    --output-dir /qinglong_datasets/qinglong/lerobot_v30/${task_id} \
    --repo-id ${repo_id} \
    --robot-type qingloongROS2 \
    --task-description "${task_description}" \
    --custom-processor /workspace/code/bag2lerbot/processors_qingloongROS2.py \
    --mapping-file /workspace/code/bag2lerbot/custom_state_action_mapping_qingloongROS2.py \
    --fps 30 \
    --workers 8  \
    --vcodec libsvtav1 \
    --crf 30
    
# 4. 转化数据
pip install "datasets<4.0.0"
cd /workspace/code/any4lerobot/ds_version_convert/v30_to_v21/
python convert_dataset_v30_to_v21.py \
    --repo-id=${repo_id} \
    --root=/qinglong_datasets/qinglong/lerobot_v30/${task_id} \
    --output-root=/qinglong_datasets/qinglong/pretrain_lerobot_v21/${task_id}
    
cp /workspace/code/bag2lerbot/qingloong_modality.json /qinglong_datasets/qinglong/pretrain_lerobot_v21/${task_id}/meta/modality.json

if [ $? -eq 0 ]; then
    echo "===========转换完成=============="
else
    echo "===========转换失败=============="
    # 可选：失败时退出脚本，返回非0状态码
    exit 1
fi
