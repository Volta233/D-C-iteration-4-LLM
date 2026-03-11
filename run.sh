#!/bin/bash
# 在项目根目录运行评测任务
# 用法: ./run.sh <start_task_id> <end_task_id>
# 示例: ./run.sh 0 66
cd "$(dirname "$0")"
python dc_iteration/CodeCore/my_evalpref.py "$@"
