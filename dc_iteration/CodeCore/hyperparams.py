"""CodeCore 配置：路径与模型参数"""
import os
from pathlib import Path

# 文件路径（BASE_DIR = dc_iteration 根目录）
HUMANEVAL_OVERRIDE_PATH = None
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_PATH = os.path.join(BASE_DIR, "store_data", "result", "humaneval")
PROBLEM_PATH = os.path.join(BASE_DIR, "store_data", "problem")
SCORE_PATH = os.path.join(BASE_DIR, "store_data", "score")
REPORT_PATH = os.path.join(BASE_DIR, "store_data", "report.jsonl")


def get_task_score_path(task_id: str) -> str:
    """获取任务特定的 score 路径"""
    task_id_safe = task_id.replace("/", "_")
    return os.path.join(SCORE_PATH, task_id_safe)


def get_task_result_path(task_id: str) -> str:
    """获取任务特定的 result 路径"""
    task_id_safe = task_id.replace("/", "_")
    return os.path.join(RESULT_PATH, task_id_safe)


# 确保路径存在
for path in [RESULT_PATH, PROBLEM_PATH, SCORE_PATH]:
    Path(path).mkdir(parents=True, exist_ok=True)
    print(f"Verified directory: {path}")

report_dir = os.path.dirname(REPORT_PATH)
if not os.path.exists(report_dir):
    os.makedirs(report_dir)
    print(f"Created report directory: {report_dir}")

# 迭代参数
NUM_SAMPLES_PER_TASK = 3
NUM_ITERATION = 2

# 运行时间限制
DEFAULT_GT_TIME_LIMIT_FACTOR = 4.0
DEFAULT_MIN_TIME_LIMIT = 1.0


def extract_task_number(task_id):
    """从 task_id 中提取数字部分进行排序"""
    parts = task_id.split("/")
    if len(parts) < 2:
        return 0
    try:
        return int(parts[-1])
    except ValueError:
        return 0

# ============ 模型配置 (从环境变量读取) ============
# 重要：以下敏感信息不再硬编码。请通过环境变量设置。
# 示例：export OPENAI_API_KEY=“your_api_key_here”

# OpenAI API 密钥 (必需)
# 从环境变量 `OPENAI_API_KEY` 读取
API_KEY = os.getenv("OPENAI_API_KEY")
if API_KEY is None:
    print("警告: OPENAI_API_KEY 环境变量未设置。程序将无法调用 OpenAI API。")

# OpenAI 模型名称 (默认值: gpt-4o-mini)
# 可通过设置 `OPENAI_MODEL_NAME` 环境变量覆盖
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

# OpenAI API 基础URL (默认值: https://api.openai.com/v1，官方标准端点)
# 可通过设置 `OPENAI_BASE_URL` 环境变量覆盖
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://svip.xty.app/v1")

# GPT 基础URL (用于代码生成任务，默认值与 BASE_URL 保持一致)
# 可通过设置 `GPT_BASE_URL` 环境变量覆盖
GPT_BASE_URL = os.getenv("GPT_BASE_URL", BASE_URL)
