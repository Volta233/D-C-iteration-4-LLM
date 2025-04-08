
import sys
import os
import shutil
import json
import numpy as np
import re
import ast
import io
from typing import Dict, List, Generator
from evalplus.data.utils import stream_jsonl,write_jsonl
from evalplus.my_work.hyperparams import *

def clean_humaneval_dir():
    """安全清理目录函数，保留符合正则模式的文件"""
    pattern = re.compile(r'^gpt-4o-mini_openai_temp_0\..+\.raw\.jsonl$')
    
    for filename in os.listdir(RESULT_PATH):
        if pattern.match(filename):  # 匹配目标文件名模式则跳过
            continue
        file_path = os.path.join(RESULT_PATH, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {str(e)}")

    for filename in os.listdir(SCORE_PATH) :
        file_path = os.path.join(SCORE_PATH, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {str(e)}")


def read_problems(evalset_file: str = PROBLEM_PATH) -> Dict[str, Dict]:
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}

def stream_eval_results(file_path: str) -> Generator[Dict, None, None]:
    """流式读取评估结果文件并展平数据结构"""
    with open(file_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            # 展开eval字段中的任务数据
            for task_id, samples in record["eval"].items():
                for sample in samples:
                    yield {
                        **sample,
                        "task_id": task_id,
                        "pass_at_k": record["pass_at_k"],
                        "base_status": sample.get("base_status", "unknown"),
                        "plus_status": sample.get("plus_status", "unknown")
                    }

def read_evaluated_samples(eval_file: str) -> Dict[str, List[Dict]]:
    """读取评估结果并转换为按task_id分组的样本字典"""
    samples = {}
    for record in stream_eval_results(eval_file):
        task_id = record["task_id"]
        if task_id not in samples:
            samples[task_id] = []
        samples[task_id].append({
            "solution": record["solution"],
            "pass_at_k": record["pass_at_k"],
            "base_status": record["base_status"],
            "plus_status": record["plus_status"],
            "base_fail_tests": record["base_fail_tests"],
            "plus_fail_tests": record["plus_fail_tests"]
        })
    return samples
