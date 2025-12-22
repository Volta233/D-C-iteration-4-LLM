import sys
import os
import json
from typing import Dict, List
from evalplus.my_work.hyperparams import *
from evalplus.my_work.IO_process import *
from evalplus.my_work.generate_samples import *
from evalplus.my_work.get_scores import *

def run_single_task_evaluation(task_id: str, problems: Dict[str, Any], iteration: int):
    """为单个任务运行评测流程"""
    # 设置任务特定的路径
    task_score_dir = get_task_score_path(task_id)
    task_result_dir = get_task_result_path(task_id)
    
    # 清理任务目录
    clean_humaneval_dir(task_id)
    
    # 生成并评估样本（需要修改generate_and_rename_samples来支持任务特定路径）
    samples_path = generate_and_rename_samples(iteration, problems, task_result_dir)
    eval_result_path = evaluate_samples(samples_path, iteration)
    evaluated_samples = read_evaluated_samples(eval_result_path)
    
    # 筛选样本生成新问题集
    task_samples = evaluated_samples.get(task_id, [])
    problem = problems[task_id]
    
    B, passk = calculate_and_log_scores(
        task_id=task_id,
        task_samples=task_samples,
        problem=problem,
        iteration=iteration,
        folder_path=task_score_dir  # 使用任务特定的score目录
    )
    
    candidate = select_sample(task_samples, problem)
    new_problem = None
    if candidate:
        new_problem = {
            "task_id": task_id,
            "prompt": generate_one_problem(candidate["solution"], problem["entry_point"]),
            **{k: v for k, v in problem.items() if k != "prompt"}
        }
    
    return B, passk, new_problem

def main_process(start_id: int, end_id: int):
    """主处理流程"""
    from my_evalpref import load_humaneval_plus_tasks, create_task_problems_file
    
    # 加载任务
    tasks = load_humaneval_plus_tasks(start_id, end_id)
    task_ids = [task["task_id"] for task in tasks]
    
    # 创建初始问题文件
    create_task_problems_file(tasks, 0)
    
    # 初始化结果存储
    all_task_metrics = {}
    
    for iteration in range(NUM_ITERATION):
        print(f"Starting iteration {iteration}")
        
        # 加载当前问题集
        problems = load_problems(iteration)
        
        new_problems = []
        iteration_metrics = {}
        
        for task_id in task_ids:
            print(f"Processing {task_id}, iteration {iteration}")
            
            B, passk, new_problem = run_single_task_evaluation(task_id, problems, iteration)
            
            # 存储指标
            iteration_metrics[task_id] = {
                "iteration": iteration,
                "B_score": B,
                "pass@k": passk
            }
            
            if new_problem:
                new_problems.append(new_problem)
        
        # 保存新问题集
        if new_problems:
            new_problems_path = os.path.join(PROBLEM_PATH, f"problems{iteration+1}.jsonl")
            write_jsonl(new_problems_path, new_problems)
        
        # 保存迭代指标
        all_task_metrics[iteration] = iteration_metrics
    
    return all_task_metrics

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main_process.py <start_task_id> <end_task_id>")
        sys.exit(1)
    
    start_id = int(sys.argv[1])
    end_id = int(sys.argv[2])
    
    metrics = main_process(start_id, end_id)
    print(f"Completed evaluation for tasks {start_id}-{end_id}")