import sys
import os
import shutil
import json
import numpy as np
from typing import Any, List, Optional, Tuple, Dict
from evalplus.my_work.hyperparams import *

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
    
    with open(score_log_path, 'w') as f:
        for sample in task_samples:
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
                "pass@k": sample.get("pass_at_k", {}),
                "composite_score": round(A_j, 6),
                "base_fail_details": [{"test_input": t, "time_weight": 1.0} for t in sample["base_fail_tests"]],
                "plus_fail_details": [{"test_input": t, "time_weight": 1.0} for t in sample["plus_fail_tests"]]
            }
            f.write(json.dumps(log_record) + '\n')
    
    # 计算B_i评分（网页1的统计量计算方法）
    if len(A_list) == 0:
        B_i = 0.0
    else:
        A_arr = np.array(A_list)
        mu = A_arr.mean()
        sigma = A_arr.std(ddof=0)  # 总体标准差
        B_i = (mu + sigma)  # 公式等价转换后的简化计算
    
    # 追加B_i记录（网页2的扩展日志格式）
    with open(score_log_path, 'a') as f:
        f.write(json.dumps({
            "task_id": task_id,
            "iteration": iteration,
            "B_score": round(B_i, 6),
            "type": "iteration_summary"
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

# 老版本评分公式，带有时间权重
# def calculate_and_log_scores(task_id:str,
#                             task_samples: List[Dict], 
#                             problem: Dict, 
#                             iteration: int,
#                             folder_path: str,
#                             decay_rate: float = 0.1) -> None:
#     """动态计算样本评分并写入迭代日志文件"""
#     # 生成带迭代次数的文件名
#     score_log_path = os.path.join(folder_path, f"score_{iteration}.ndjson")
    
#     # 预计算总测试数量
#     total_base_tests = len(problem["base_input"])
#     total_plus_tests = len(problem["plus_input"])
#     total_tests = total_base_tests + total_plus_tests
    
#     with open(score_log_path, 'w') as f:
#         for sample in task_samples:
#             # 关键字段提取
#             task_id = task_id
#             pass_at_k = sample.get("pass_at_k", {})
            
#             # 动态权重计算
#             time_weight = 1 + decay_rate * iteration
#             weighted_base = len(sample["base_fail_tests"]) * time_weight
#             weighted_plus = len(sample["plus_fail_tests"]) * time_weight
#             total_fails = weighted_base + weighted_plus
            
#             # 评分公式实现
#             fail_ratio = total_fails / total_tests if total_tests else 1.0
#             pass_ratio = 1 - fail_ratio
#             composite_score = pass_ratio * (1 - (fail_ratio * time_weight)**2)
#             # 动态ε修正策略
#             if composite_score >= 0.99:
#                 # 当评分≥0.99时，ε随接近1的程度线性增大（系数0.1可调）
#                 epsilon = max(1e-8, 0.1 * (1 - composite_score))
#             else:
#                 # 低分区保持固定极小量（避免浮点误差）
#                 epsilon = 1e-8
    
#             # 对数变换的统一处理（消除条件分支）
#             adjusted_score = np.log(1 / (1 - composite_score + epsilon))
#             # 构建带权重的失败记录
#             base_details = [
#                 {"test_input": test, "time_weight": time_weight}
#                 for test in sample["base_fail_tests"]
#             ]
#             plus_details = [
#                 {"test_input": test, "time_weight": time_weight}
#                 for test in sample["plus_fail_tests"]
#             ]
            
#             # 生成符合格式要求的记录
#             log_record = {
#                 "task_id": task_id,
#                 "iteration": iteration,
#                 "pass@k": pass_at_k,
#                 "composite_score": round(adjusted_score, 6),
#                 "base_fail_details": base_details,
#                 "plus_fail_details": plus_details
#             }
            
#             # 写入NDJSON格式
#             f.write(json.dumps(log_record) + '\n')