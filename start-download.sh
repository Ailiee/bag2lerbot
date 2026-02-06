#!/bin/bash


# 1. 从华为云拉取原始数据
/workspace/rclone/rclone copy -P --transfers=4 --config="/workspace/rclone/rclone.conf"\
  "huawei-cloud:/openloong-apps-prod-private/data-collector-svc/raw/${task_id}" \
  "/qinglong_datasets/qinglong/raw/${task_id}"
