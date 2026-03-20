import os
import json
from collections import defaultdict
from typing import Any, List, Optional, Tuple, Dict
from datetime import datetime

import numpy as np

from dc_iteration.evaluate import evaluate
from dc_iteration.CodeCore.hyperparams import *
from dc_iteration.data.utils import write_jsonl, stream_jsonl
from dc_iteration.CodeCore.IO_process import *


def select_sample(samples: List[Dict], problem: Dict) -> Optional[Dict]:
    """基于评估结果的筛选策略"""
    perfect_pass = [s for s in samples
                   if s["base_status"] == "pass" and s["plus_status"] == "pass"]
    if perfect_pass:
        return perfect_pass[0]

    scored_samples = []
    for s in samples:
        total_fails = len(s["base_fail_tests"]) + len(s["plus_fail_tests"])
        total_tests = len(problem["base_input"]) + len(problem["plus_input"])
        fail_ratio = total_fails / total_tests if total_tests else 1.0
        pass_ratio = 1 - fail_ratio
        composite_score = pass_ratio * (1 - fail_ratio**2)
        scored_samples.append((composite_score, s))

    return max(scored_samples, key=lambda x: x[0])[1] if scored_samples else None


def calculate_final_score(B_list: List[float]) -> float:
    """计算归一化的最终 C 评分"""
    N = len(B_list)
    lambda_val = 0.05
    weighted_score_sum = 0.0
    weight_sum = 0.0  # 用于归一化的权重总和

    for i in range(N):
        linear_decay = (N - (i + 1) + 1) / N
        exp_decay = np.exp(-lambda_val * i)
        weight_i = linear_decay * exp_decay  # 当前轮次的权重

        weighted_score_sum += B_list[i] * weight_i
        weight_sum += weight_i  # 累积理论最大权重

    # 归一化：除以理论最大可能得分（即所有权重之和）
    if weight_sum > 0:
        return weighted_score_sum / weight_sum
    else:
        return 0.0


def filter_frequent_fails(fail_stats, num_iterations, all_task_ids):
    threshold = 0
    frequent_cases = {}
    for task_id in all_task_ids:
        task_fail_counts = fail_stats.get(task_id, {})
        frequent = [{"test_input": t} for t, cnt in task_fail_counts.items() if cnt > threshold]
        frequent_cases[task_id] = frequent
    return frequent_cases


def evaluate_samples_single_task(samples_path: str, iteration: int, task_id: str) -> str:
    """评估单个任务的样本"""
    problems = load_problems(iteration)
    if task_id in problems:
        temp_problem_path = os.path.join(PROBLEM_PATH, f"temp_problems{iteration}.jsonl")
        write_jsonl(temp_problem_path, [problems[task_id]])

        evaluate(
            dataset="humaneval",
            samples=samples_path,
            i_just_wanna_run=True,
            HUMANEVAL_OVERRIDE_PATH=temp_problem_path
        )

        if os.path.exists(temp_problem_path):
            os.remove(temp_problem_path)

    return samples_path.replace(".jsonl", "_eval_results.json")


def calculate_and_log_scores_single_task(task_id: str, task_samples: List[Dict],
                                         problem: Dict, iteration: int) -> Tuple[float, Dict]:
    """为单个任务计算分数"""
    task_score_dir = get_task_score_path(task_id)
    os.makedirs(task_score_dir, exist_ok=True)

    score_log_path = os.path.join(task_score_dir, f"score_{iteration}.ndjson")
    total_base = len(problem["base_input"])
    total_plus = len(problem["plus_input"])
    total_tests = total_base + total_plus
    epsilon = 1e-8

    A_list = []
    get_passk = False
    passk = {}

    with open(score_log_path, 'a') as f:
        for sample in task_samples:
            if not get_passk:
                passk = sample.get("pass_at_k", {})
                get_passk = True

            W_base = len(sample["base_fail_tests"])
            W_plus = len(sample["plus_fail_tests"])
            pass_rate = 1 - (W_base + W_plus) / total_tests if total_tests else 0.0
            pass_rate = max(0.0, min(1.0, pass_rate))

            denominator = 1 - pass_rate + epsilon
            A_j = pass_rate
            A_list.append(A_j)

            log_record = {
                "task_id": task_id,
                "iteration": iteration,
                "composite_score": round(A_j, 6),
                "base_fail_details": [{"test_input": t} for t in sample["base_fail_tests"]],
                "plus_fail_details": [{"test_input": t} for t in sample["plus_fail_tests"]]
            }
            f.write(json.dumps(log_record) + '\n')

    if len(A_list) == 0:
        B_i = 0.0
    else:
        A_arr = np.array(A_list)
        mu = A_arr.mean()
        sigma = A_arr.std(ddof=0)
        cv = sigma / mu if mu != 0 else float('inf')
        penalty_factor = np.exp(-cv) if cv != float('inf') else 0.0
        B_i = mu * penalty_factor

    return B_i, passk


def collect_fail_cases_single_task(task_id: str, num_iterations: int) -> Tuple[Dict[str, Dict[str, int]], List[str]]:
    """收集单个任务的失败案例"""
    fail_stats = defaultdict(lambda: defaultdict(int))
    task_score_dir = get_task_score_path(task_id)

    for i in range(num_iterations):
        score_path = os.path.join(task_score_dir, f"score_{i}.ndjson")
        if not os.path.exists(score_path):
            continue
        with open(score_path, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    for fail in record.get("base_fail_details", []):
                        test_input = fail.get("test_input")
                        if test_input:
                            fail_stats[task_id][test_input] += 1
                    for fail in record.get("plus_fail_details", []):
                        test_input = fail.get("test_input")
                        if test_input:
                            fail_stats[task_id][test_input] += 1
                except (json.JSONDecodeError, Exception):
                    continue

    return fail_stats, [task_id]


def generate_task_report(task_id: str, final_score: float, frequent_cases: Dict,
                         task_metrics: List[Dict]) -> None:
    """生成单个任务的报告"""
    task_report_dir = os.path.join(SCORE_PATH, task_id.replace("/", "_"))
    os.makedirs(task_report_dir, exist_ok=True)

    report_path = os.path.join(task_report_dir, "task_report.json")
    report_data = {
        "task_id": task_id,
        "final_score": final_score,
        "frequent_fail_cases": frequent_cases.get(task_id, []),
        "iterations": NUM_ITERATION,
        "metrics_per_iteration": task_metrics
    }
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)

    global_report_path = os.path.join(SCORE_PATH, "global_report.ndjson")
    with open(global_report_path, 'a') as f:
        f.write(json.dumps({
            "task_id": task_id,
            "final_score": final_score,
            "timestamp": str(datetime.now())
        }) + '\n')
