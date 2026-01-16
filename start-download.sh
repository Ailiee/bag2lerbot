#!/bin/bash


# 1. 从华为云拉取原始数据
/opt/rclone-v1.68.2-linux-amd64/rclone copy -P --transfers=4 \
  "huawei-cloud:/openloong-apps-prod-private/data-collector-svc/raw/${task_id}" \
  "/qinglong_datasets/qinglong/raw/${task_id}"
