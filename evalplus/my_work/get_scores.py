import sys
import os
import shutil
import json
from collections import defaultdict
from evalplus.evaluate import evaluate
import numpy as np
from typing import Any, List, Optional, Tuple, Dict
from evalplus.my_work.hyperparams import *
from itertools import islice

def select_sample(samples: List[Dict], problem: Dict) -> Optional[Dict]:
    """基于评估结果的四维筛选策略, 同时记录每个sample的评分, 用简单评分最高的作为下次迭代的输入"""
    # 维度1：完全通过测试的样本
    perfect_pass = [s for s in samples 
                   if s["base_status"] == "pass" 
                   and s["plus_status"] == "pass"]
    if perfect_pass:
        return perfect_pass[0]
    
    # 维度2：失败测试数量统计
    scored_samples = []
    for s in samples:
        # 计算失败测试比例
        total_fails = len(s["base_fail_tests"]) + len(s["plus_fail_tests"])
        total_tests = len(problem["base_input"]) + len(problem["plus_input"])
        fail_ratio = total_fails / total_tests if total_tests else 1.0
        
        # 通过率逆向计算
        pass_ratio = 1 - fail_ratio
        
        # 综合评分（应用平方惩罚）
        composite_score = pass_ratio * (1 - fail_ratio**2)
        scored_samples.append((composite_score, s))
    
    return max(scored_samples, key=lambda x: x[0])[1] if scored_samples else None


def calculate_and_log_scores(task_id: str,
                            task_samples: List[Dict], 
                            problem: Dict, 
                            iteration: int,
                            folder_path: str) -> float:
    """动态计算评分并写入日志,返回本轮B_i评分"""
    score_log_path = os.path.join(folder_path, f"score_{iteration}.ndjson")
    
    # 预计算总测试数
    total_base = len(problem["base_input"])
    total_plus = len(problem["plus_input"])
    total_tests = total_base + total_plus
    epsilon = 1e-8  # 公式定义的小量
    
    A_list = []
    get_passk = False
    passk = {}

    with open(score_log_path, 'w') as f:
        for sample in task_samples:
            if get_passk == False:
                passk = sample.get("pass_at_k", {})
                get_passk = True
            # 计算基础指标
            W_base = len(sample["base_fail_tests"])
            W_plus = len(sample["plus_fail_tests"])
            
            # 通过率计算
            pass_rate = 1 - (W_base + W_plus)/total_tests if total_tests else 0.0
            pass_rate = np.clip(pass_rate, 0.0, 1.0)  # 确保在[0,1]区间
            
            # A_j评分计算
            denominator = 1 - pass_rate + epsilon
            A_j = np.log(pass_rate / denominator) if denominator != 0 else 0.0
            A_list.append(A_j)
            
            # 构建日志记录
            log_record = {
                "task_id": task_id,
                "iteration": iteration,
                "composite_score": round(A_j, 6),
                "base_fail_details": [{"test_input": t} for t in sample["base_fail_tests"]],
                "plus_fail_details": [{"test_input": t} for t in sample["plus_fail_tests"]]
            }
            f.write(json.dumps(log_record) + '\n')
    
    # 计算B_i评分
    if len(A_list) == 0:
        B_i = 0.0
    else:
        A_arr = np.array(A_list)
        mu = A_arr.mean()
        sigma = A_arr.std(ddof=0)  # 总体标准差
        cv = sigma / mu  # 变异系数
        penalty_factor = np.exp(-cv)  # 惩罚因子
        B_i = mu * penalty_factor
    
    # 追加B_i记录
    with open(score_log_path, 'a') as f:
        f.write(json.dumps({
            "task_id": task_id,
            "iteration": iteration,
            "pass@k": passk,
            "B_score": round(B_i, 6),
            "type": "B_scores_summary"
        }) + '\n')
    
    return B_i

def calculate_final_score(B_list: List[float]) -> float:
    """计算最终C评分"""
    N = len(B_list)
    lambda_val = 0.05
    
    total = 0.0
    for i in range(N):
        # 混合衰减权重（网页10的指数衰减实现）
        linear_decay = (N - (i+1) + 1) / N  # 线性衰减分量
        exp_decay = np.exp(-lambda_val * i)  # 指数衰减分量
        total += B_list[i] * linear_decay * exp_decay
    
    return total

def collect_fail_cases(score_path, num_iterations, num_samples_per_task):
    # 数据结构：{task_id: {"base": Counter, "plus": Counter}}
    fail_stats = defaultdict(lambda: {"base": defaultdict(int), "plus": defaultdict(int)})
    all_task_ids = set()  # 新增：存储所有遇到的task_id

    for i in range(num_iterations):
        score_file = os.path.join(score_path, f"score_{i}.ndjson")
        with open(score_file, 'r') as f:
            for line in islice(f, num_samples_per_task):
                entry = json.loads(line.strip())
                task_id = entry["task_id"]
                all_task_ids.add(task_id)  # 记录所有task_id

                # 统计失败用例
                for case in entry.get("base_fail_details", []):
                    test_input = tuple(case["test_input"])
                    fail_stats[task_id]["base"][test_input] += 1
                for case in entry.get("plus_fail_details", []):
                    test_input = tuple(case["test_input"])
                    fail_stats[task_id]["plus"][test_input] += 1
                    
    return fail_stats, all_task_ids  # 返回新增的all_task_ids


def filter_frequent_fails(fail_stats, num_iterations, all_task_ids):
    threshold = 0
    frequent_cases = {}
    
    # 遍历所有记录过的task_id（关键修改）
    for task_id in all_task_ids:
        # 获取该task_id的计数器（可能不存在）
        base_counts = fail_stats.get(task_id, {}).get("base", {})
        plus_counts = fail_stats.get(task_id, {}).get("plus", {})
        
        frequent = []
        # 合并两种失败用例
        for test_input in set(base_counts) | set(plus_counts):
            total = base_counts.get(test_input, 0) + plus_counts.get(test_input, 0)
            if total > threshold:
                frequent.append({"test_input": list(test_input)})
                
        frequent_cases[task_id] = frequent
        
    return frequent_cases

def extract_task_number(task_id):
    """从task_id中提取数字部分进行排序"""
    parts = task_id.split('/')
    if len(parts) < 2:
        return 0
    try:
        return int(parts[-1])
    except ValueError:
        return 0


def generate_report(final_score, frequent_cases, report_path):
    existing_entries = {}
    try:
        with open(report_path, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                existing_entries[entry["task_id"]] = entry
    except FileNotFoundError:
        pass

    # 合并更新当前生成的cases到现有数据
    for task_id, cases in frequent_cases.items():
        existing_entries[task_id] = {
            "task_id": task_id,
            "frequent_fail_cases": cases,
            "final_score_C": final_score
        }

    # 按任务编号排序
    sorted_entries = sorted(
        existing_entries.values(),
        key=lambda x: extract_task_number(x["task_id"])
    )

    # 原子化写入：先写临时文件再重命名
    tmp_path = report_path + ".tmp"
    with open(tmp_path, 'w') as f:
        for entry in sorted_entries:
            f.write(json.dumps(entry) + '\n')
    os.replace(tmp_path, report_path)  # 原子操作避免写入中断


def test_report():
    B_list = [18.420681 for _ in range(10)]
    # 获取失败统计和所有task_id集合
    all_correct_score = calculate_final_score(B_list)
    fail_stats, all_task_ids = collect_fail_cases(SCORE_PATH, NUM_ITERATION, NUM_SAMPLES_PER_TASK)
    # 传入所有task_id集合
    frequent_cases = filter_frequent_fails(fail_stats, NUM_ITERATION, all_task_ids)
    generate_report(all_correct_score, frequent_cases, REPORT_PATH)


# 评估样本并返回结果路径
def evaluate_samples(samples_path: str, iteration: int) -> str:
    """执行样本评估并返回结果文件路径"""
    evaluate(
        dataset="humaneval",
        samples=samples_path,
        i_just_wanna_run=True,
        HUMANEVAL_OVERRIDE_PATH=os.path.join(PROBLEM_PATH, f"problems{iteration}.jsonl")
    )
    return samples_path.replace(".jsonl", "_eval_results.json")

if __name__ == "__main__":
    print("try to get a report line.")
    test_report()
    print("finish getting report line.")
