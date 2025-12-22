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

    with open(score_log_path, 'a') as f:
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
    
    return B_i, passk

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

def collect_fail_cases(
    score_dir: str,
    num_iterations: int,
    num_samples_per_task: int
) -> Tuple[Dict[str, Dict[str, int]], List[str]]:
    """
    收集所有任务的失败案例统计，确保不遗漏任何任务
    返回: (fail_stats, all_task_ids)
        fail_stats: {task_id: {test_input: 失败次数}}
        all_task_ids: 所有唯一任务ID列表（包含无失败案例的任务）
    """
    fail_stats = defaultdict(lambda: defaultdict(int))
    # 收集所有出现过的任务ID（确保无遗漏）
    all_task_ids = set()

    # 遍历所有迭代的score日志文件
    for i in range(num_iterations):
        score_path = os.path.join(score_dir, f"score_{i}.ndjson")
        if not os.path.exists(score_path):
            continue
        
        # 读取当前迭代的所有日志记录
        with open(score_path, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    task_id = record.get("task_id")
                    if not task_id:
                        continue
                    
                    # 将任务ID加入全局集合（确保不遗漏）
                    all_task_ids.add(task_id)
                    
                    # 统计base失败案例
                    for fail in record.get("base_fail_details", []):
                        test_input = fail.get("test_input")
                        if test_input:
                            fail_stats[task_id][test_input] += 1
                    
                    # 统计plus失败案例
                    for fail in record.get("plus_fail_details", []):
                        test_input = fail.get("test_input")
                        if test_input:
                            fail_stats[task_id][test_input] += 1
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    continue

    # 转换为列表返回（保持顺序一致）
    return fail_stats, sorted(list(all_task_ids))


def filter_frequent_fails(fail_stats, num_iterations, all_task_ids):
    threshold = 0  # 只要失败过就记录（次数>0）
    frequent_cases = {}
    
    for task_id in all_task_ids:
        task_fail_counts = fail_stats.get(task_id, {})
        
        frequent = []
        # 遍历所有发生过失败的测试用例
        for test_input, total_failures in task_fail_counts.items():
            if total_failures > threshold:
                frequent.append({"test_input": test_input}) 
                
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

    with open(report_path, 'w') as f:
        for entry in sorted_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


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

def evaluate_samples_single_task(samples_path: str, iteration: int, task_id: str) -> str:
    """评估单个任务的样本"""
    # 创建临时问题文件（只包含当前任务）
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
        
        # 清理临时文件
        if os.path.exists(temp_problem_path):
            os.remove(temp_problem_path)
    
    return samples_path.replace(".jsonl", "_eval_results.json")

def calculate_and_log_scores_single_task(task_id: str, task_samples: List[Dict], 
                                       problem: Dict, iteration: int) -> Tuple[float, Dict]:
    """为单个任务计算分数"""
    task_score_dir = get_task_score_path(task_id)
    os.makedirs(task_score_dir, exist_ok=True)
    
    score_log_path = os.path.join(task_score_dir, f"score_{iteration}.ndjson")
    
    # 预计算总测试数
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
            
            # 计算基础指标
            W_base = len(sample["base_fail_tests"])
            W_plus = len(sample["plus_fail_tests"])
            
            # 通过率计算
            pass_rate = 1 - (W_base + W_plus) / total_tests if total_tests else 0.0
            pass_rate = max(0.0, min(1.0, pass_rate))  # 确保在[0,1]区间
            
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
        cv = sigma / mu if mu != 0 else float('inf')  # 变异系数
        penalty_factor = np.exp(-cv) if cv != float('inf') else 0.0  # 惩罚因子
        B_i = mu * penalty_factor
    
    return B_i, passk

def collect_fail_cases_single_task(task_id: str, num_iterations: int) -> Tuple[Dict[str, Dict[str, int]], List[str]]:
    """收集单个任务的失败案例"""
    from collections import defaultdict
    
    fail_stats = defaultdict(lambda: defaultdict(int))
    task_score_dir = get_task_score_path(task_id)
    
    # 遍历所有迭代的score日志文件
    for i in range(num_iterations):
        score_path = os.path.join(task_score_dir, f"score_{i}.ndjson")
        if not os.path.exists(score_path):
            continue
        
        # 读取当前迭代的所有日志记录
        with open(score_path, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    
                    # 统计base失败案例
                    for fail in record.get("base_fail_details", []):
                        test_input = fail.get("test_input")
                        if test_input:
                            fail_stats[task_id][test_input] += 1
                    
                    # 统计plus失败案例
                    for fail in record.get("plus_fail_details", []):
                        test_input = fail.get("test_input")
                        if test_input:
                            fail_stats[task_id][test_input] += 1
                except json.JSONDecodeError:
                    continue
                except Exception:
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
    
    # 同时写入全局报告文件
    global_report_path = os.path.join(SCORE_PATH, "global_report.ndjson")
    with open(global_report_path, 'a') as f:
        f.write(json.dumps({
            "task_id": task_id,
            "final_score": final_score,
            "timestamp": str(datetime.now())
        }) + '\n')


if __name__ == "__main__":
    print("try to get a report line.")
    test_report()
    print("finish getting report line.")
