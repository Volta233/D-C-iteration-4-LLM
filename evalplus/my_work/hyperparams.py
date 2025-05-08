import os

# 文件路径
HUMANEVAL_OVERRIDE_PATH = None
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_PATH = os.path.join(BASE_DIR, "my_data", "result", "humaneval")
PROBLEM_PATH = os.path.join(BASE_DIR,"my_data","problem")
SCORE_PATH = os.path.join(BASE_DIR,"my_data","score")
REPORT_PATH = os.path.join(BASE_DIR,"my_data","report.jsonl")

# 模型配置
GPT_BASE_URL = "https://svip.xty.app/v1"
MODEL_NAME = "gpt-4o-mini"
API_KEY = "sk-viODA4qbv0KsnroB8950Df3602F84d1fB1Bb2b53803151E1"
BASE_URL = "https://svip.xty.app/v1"

# 迭代参数
NUM_SAMPLES_PER_TASK = 40
NUM_ITERATION = 10