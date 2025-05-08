import sys
import os
import shutil
import json
from collections import defaultdict
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
    
    # 遍历每个迭代的score文件 
    for i in range(num_iterations):
        score_file = os.path.join(score_path, f"score_{i}.ndjson")
        with open(score_file, 'r') as f:
            for line in islice(f, num_samples_per_task):
                entry = json.loads(line.strip())
                task_id = entry["task_id"]
                
                # 统计base失败用例 
                for case in entry.get("base_fail_details", []):
                    test_input = tuple(case["test_input"])  # 转换为可哈希类型
                    fail_stats[task_id]["base"][test_input] += 1
                
                # 统计plus失败用例 
                for case in entry.get("plus_fail_details", []):
                    test_input = tuple(case["test_input"])
                    fail_stats[task_id]["plus"][test_input] += 1
    return fail_stats

def filter_frequent_fails(fail_stats, num_iterations):
    threshold = num_iterations // 2  # 超过半数迭代失败
    frequent_cases = {}
    
    for task_id, counters in fail_stats.items():
        frequent = []
        # 合并base和plus用例 
        for test_input in set(counters["base"]) | set(counters["plus"]):
            total = counters["base"].get(test_input, 0) + counters["plus"].get(test_input, 0)
            if total > threshold:
                frequent.append({"test_input": list(test_input)})  # 转换回列表
        frequent_cases[task_id] = frequent
    return frequent_cases

def generate_report(final_score, frequent_cases, report_path):
    report = []
    for task_id, cases in frequent_cases.items():
        report_entry = {
            "task_id": task_id,
            "frequent_fail_cases": cases,
            "final_score_C": final_score  
        }
        report.append(report_entry)
    
    # 写入JSONL文件 
    with open(report_path, 'w') as f:
        for entry in report:
            f.write(json.dumps(entry) + '\n')