
import sys
import os
import shutil
import json
import numpy as np
import re
import ast
import io
from typing import Dict, List, Generator, Any
from evalplus.data.utils import stream_jsonl,write_jsonl
from evalplus.my_work.hyperparams import *

def clean_humaneval_dir(task_id: str = None):
    """安全清理目录函数，只删除文件，保留文件夹结构，支持任务特定清理"""
    pattern = re.compile(r'^gpt-4o-mini_openai_temp_0\..+\.raw\.jsonl$')  # 保留的文件模式
    
    if task_id:
        # 清理特定任务的目录
        task_score_dir = get_task_score_path(task_id)
        task_result_dir = get_task_result_path(task_id)
        
        # 清理score目录：只删除文件，不删除目录
        if os.path.exists(task_score_dir):
            for filename in os.listdir(task_score_dir):
                file_path = os.path.join(task_score_dir, filename)
                try:
                    if os.path.isfile(file_path):  # 只删除文件，忽略目录
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {str(e)}")
        
        # 清理result目录：只删除文件，保留模式匹配的文件
        if os.path.exists(task_result_dir):
            for filename in os.listdir(task_result_dir):
                file_path = os.path.join(task_result_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        if pattern.match(filename):  # 匹配目标文件名模式则跳过
                            continue
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {str(e)}")
    else:
        # 清理整个RESULT_PATH和SCORE_PATH，但只删除文件，不删除文件夹
        for path in [RESULT_PATH, SCORE_PATH]:
            if os.path.exists(path):
                for filename in os.listdir(path):
                    file_path = os.path.join(path, filename)
                    try:
                        if os.path.isfile(file_path):  # 只删除文件
                            if pattern.match(filename):  # 保留模式匹配的文件
                                continue
                            os.remove(file_path)
                        # 如果是目录，跳过不删除
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

def ensure_paths_exist() -> None:
    """检查并创建所有必要的目录结构"""
    for path in [PROBLEM_PATH, SCORE_PATH, os.path.dirname(REPORT_PATH)]:
        os.makedirs(path, exist_ok=True)

# 加载问题数据集
def load_problems(iteration: int) -> Dict[str, Any]:
    """加载指定迭代轮次的问题集"""
    problem_path = os.path.join(PROBLEM_PATH, f"problems{iteration}.jsonl")
    return read_problems(problem_path)

def load_humaneval_plus_tasks(start_id: int, end_id: int) -> List[Dict]:
    """从多个HumanEvalPlus文件加载指定范围内的任务"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    tasks = []
    
    # 定义可能的文件列表
    humaneval_files = [
        os.path.join(project_root, "HumanEvalPlus1.jsonl"),
        os.path.join(project_root, "HumanEvalPlus2.jsonl"),
    ]
    
    # original_file = os.path.join(project_root, "HumanEvalPlus.jsonl")
    # if os.path.exists(original_file):
    #     humaneval_files.insert(0, original_file)
    
    for file_path in humaneval_files:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}, skipping...")
            continue
            
        print(f"Loading tasks from: {file_path}")
        tasks_from_file = _load_tasks_from_file(file_path, start_id, end_id)
        tasks.extend(tasks_from_file)
        
        # 如果已经收集到足够任务，提前退出
        if len(tasks) >= (end_id - start_id + 1):
            break
    
    # 去重（按task_id）
    seen = set()
    unique_tasks = []
    for task in tasks:
        task_id = task["task_id"]
        if task_id not in seen:
            seen.add(task_id)
            unique_tasks.append(task)
    
    # 按任务ID排序
    unique_tasks.sort(key=lambda x: int(x["task_id"].split("/")[1]))
    
    return unique_tasks

def _load_tasks_from_file(file_path: str, start_id: int, end_id: int) -> List[Dict]:
    """从单个文件加载任务，带有错误处理"""
    tasks = []
    line_count = 0
    error_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line_count += 1
            line = line.strip()
            if not line:  # 跳过空行
                continue
                
            try:
                task = json.loads(line)
                # 提取任务编号
                task_number_str = task["task_id"].split("/")[-1]
                task_number = int(task_number_str)
                
                # 检查是否在目标范围内
                if start_id <= task_number <= end_id:
                    tasks.append(task)
                    
            except json.JSONDecodeError as e:
                error_count += 1
                print(f"  JSON decode error at line {line_num}: {e}")
                print(f"  Problematic line (first 200 chars): {line[:200]}...")
                continue
            except (KeyError, ValueError, IndexError) as e:
                error_count += 1
                print(f"  Data format error at line {line_num}: {e}")
                continue
    
    print(f"  Processed {line_count} lines, found {len(tasks)} valid tasks, {error_count} errors")
    return tasks

def create_task_problems_file(tasks: List[Dict], iteration: int = 0):
    """创建指定任务的问题文件"""
    problem_path = os.path.join(PROBLEM_PATH, f"problems{iteration}.jsonl")
    
    with open(problem_path, 'w') as f:
        for task in tasks:
            f.write(json.dumps(task) + '\n')
    
    print(f"Created {problem_path} with {len(tasks)} tasks")

def setup_task_directories(tasks: List[Dict]):
    """为每个任务创建独立的目录结构"""
    for task in tasks:
        task_id = task["task_id"]
        task_score_dir = get_task_score_path(task_id)
        task_result_dir = get_task_result_path(task_id)
        
        os.makedirs(task_score_dir, exist_ok=True)
        os.makedirs(task_result_dir, exist_ok=True)
        
        print(f"Created directories for {task_id}")
