import sys
import os
import json
import re
from typing import Dict, List, Generator, Any

from dc_iteration.data.utils import stream_jsonl, write_jsonl
from dc_iteration.CodeCore.hyperparams import *
from collections import defaultdict


def clean_humaneval_dir(task_id: str = None):
    """安全清理目录函数，保留必要的文件结构"""
    preserve_patterns = [
        r'^gpt-4o-mini_openai_temp_0\..+\.raw\.jsonl$',
        r'^global_report\.ndjson$',
        r'^problems\d+\.jsonl$',
        r'^samples\d+.*\.jsonl$',
        r'^samples\d+.*_eval_results\.json$',
        r'^\.gitkeep$',
    ]
    preserve_regexes = [re.compile(p) for p in preserve_patterns]

    def should_preserve(filename: str) -> bool:
        for regex in preserve_regexes:
            if regex.match(filename):
                return True
        return False

    def delete_bak_files(directory: str):
        if not os.path.exists(directory):
            return
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.bak') or '.bak.' in file:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        print(f"Deleted .bak file: {file_path}")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {str(e)}")

    def clean_residual_files(directory: str):
        if not os.path.exists(directory):
            return
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path):
                    if not should_preserve(filename):
                        os.remove(file_path)
                        print(f"Deleted residual file: {file_path}")
                elif os.path.isdir(file_path):
                    if not os.listdir(file_path):
                        os.rmdir(file_path)
                        print(f"Deleted empty directory: {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

    if task_id:
        task_score_dir = get_task_score_path(task_id)
        task_result_dir = get_task_result_path(task_id)
        delete_bak_files(RESULT_PATH)
        delete_bak_files(task_result_dir)
        delete_bak_files(task_score_dir)
        clean_residual_files(task_score_dir)
        clean_residual_files(task_result_dir)
        clean_residual_files(PROBLEM_PATH)
    else:
        for directory in [RESULT_PATH, SCORE_PATH, PROBLEM_PATH]:
            delete_bak_files(directory)
            clean_residual_files(directory)
        print("Cleaned all directories, preserving essential files and removing .bak files")


def delete_empty_task_directories():
    """删除 RESULT_PATH 下所有空的任务目录"""
    if not os.path.exists(RESULT_PATH):
        return
    for item in os.listdir(RESULT_PATH):
        item_path = os.path.join(RESULT_PATH, item)
        if os.path.isdir(item_path) and "HumanEval_" in item:
            try:
                if not os.listdir(item_path):
                    os.rmdir(item_path)
                    print(f"Deleted empty task directory: {item_path}")
            except OSError as e:
                print(f"Error deleting directory {item_path}: {e}")


def read_problems(evalset_file: str = PROBLEM_PATH) -> Dict[str, Dict]:
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}


def stream_eval_results(file_path: str) -> Generator[Dict, None, None]:
    """流式读取评估结果文件并展平数据结构"""
    with open(file_path, 'r') as f:
        for line in f:
            record = json.loads(line)
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
    """读取评估结果并转换为按 task_id 分组的样本字典"""
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
    for path in [PROBLEM_PATH, SCORE_PATH, os.path.dirname(REPORT_PATH)]:
        os.makedirs(path, exist_ok=True)


def load_problems(iteration: int) -> Dict[str, Any]:
    """加载指定迭代轮次的问题集"""
    problem_path = os.path.join(PROBLEM_PATH, f"problems{iteration}.jsonl")
    return read_problems(problem_path)


def load_humaneval_plus_tasks(start_id: int, end_id: int) -> List[Dict]:
    """从多个 HumanEvalPlus 文件加载指定范围内的任务"""
    # project_root = dc_iteration 的父目录（HumanEvalPlus 文件所在位置）
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    tasks = []
    humaneval_files = [
        os.path.join(project_root, "HumanEvalPlus1.jsonl"),
        os.path.join(project_root, "HumanEvalPlus2.jsonl"),
    ]

    for file_path in humaneval_files:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}, skipping...")
            continue
        print(f"Loading tasks from: {file_path}")
        tasks_from_file = _load_tasks_from_file(file_path, start_id, end_id)
        tasks.extend(tasks_from_file)
        if len(tasks) >= (end_id - start_id + 1):
            break

    seen = set()
    unique_tasks = []
    for task in tasks:
        task_id = task["task_id"]
        if task_id not in seen:
            seen.add(task_id)
            unique_tasks.append(task)
    unique_tasks.sort(key=lambda x: int(x["task_id"].split("/")[1]))
    return unique_tasks


def _load_tasks_from_file(file_path: str, start_id: int, end_id: int) -> List[Dict]:
    tasks = []
    line_count = 0
    error_count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line_count += 1
            line = line.strip()
            if not line:
                continue
            try:
                task = json.loads(line)
                task_number = int(task["task_id"].split("/")[-1])
                if start_id <= task_number <= end_id:
                    tasks.append(task)
            except json.JSONDecodeError as e:
                error_count += 1
                print(f"  JSON decode error at line {line_num}: {e}")
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


def organize_eval_results(result_path: str) -> None:
    """整理评估结果目录（占位实现）"""
    if not os.path.exists(result_path):
        return


def setup_task_directories(tasks: List[Dict]):
    """为每个任务创建独立的目录结构"""
    for task in tasks:
        task_id = task["task_id"]
        task_score_dir = get_task_score_path(task_id)
        task_result_dir = get_task_result_path(task_id)
        os.makedirs(task_score_dir, exist_ok=True)
        os.makedirs(task_result_dir, exist_ok=True)
        print(f"Created directories for {task_id}")
