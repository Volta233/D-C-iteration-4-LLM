import os
from pathlib import Path

# 文件路径
HUMANEVAL_OVERRIDE_PATH = None
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_PATH = os.path.join(BASE_DIR, "my_data", "result", "humaneval")
PROBLEM_PATH = os.path.join(BASE_DIR,"my_data","problem")
SCORE_PATH = os.path.join(BASE_DIR,"my_data","score")
REPORT_PATH = os.path.join(BASE_DIR,"my_data","report.jsonl")

def get_task_score_path(task_id: str) -> str:
    """获取任务特定的score路径"""
    task_id_safe = task_id.replace("/", "_")
    return os.path.join(SCORE_PATH, task_id_safe)

def get_task_result_path(task_id: str) -> str:
    """获取任务特定的result路径"""
    task_id_safe = task_id.replace("/", "_")
    return os.path.join(RESULT_PATH, task_id_safe)
# 确保路径存在
for path in [RESULT_PATH, PROBLEM_PATH, SCORE_PATH]:
    Path(path).mkdir(parents=True, exist_ok=True)  # 递归创建目录
    print(f"Verified directory: {path}")

# 检查报告文件的父目录
report_dir = os.path.dirname(REPORT_PATH)
if not os.path.exists(report_dir):
    os.makedirs(report_dir)
    print(f"Created report directory: {report_dir}")

# 模型配置
GPT_BASE_URL = "https://svip.xty.app/v1"
MODEL_NAME = "gpt-4o-mini"
API_KEY = "sk-viODA4qbv0KsnroB8950Df3602F84d1fB1Bb2b53803151E1"
BASE_URL = "https://svip.xty.app/v1"

# 迭代参数
NUM_SAMPLES_PER_TASK = 3
NUM_ITERATION = 3

def extract_task_number(task_id):
    """从task_id中提取数字部分进行排序"""
    parts = task_id.split('/')
    if len(parts) < 2:
        return 0
    try:
        return int(parts[-1])
    except ValueError:
        return 0