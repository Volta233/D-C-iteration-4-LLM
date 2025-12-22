import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from evalplus.my_work.IO_process import *
from evalplus.my_work.hyperparams import *
from evalplus.my_work.generate_samples import *
from evalplus.my_work.get_scores import *

def main():
    if len(sys.argv) != 3:
        print("Usage: python my_evalpref.py <start_task_id> <end_task_id>")
        print("Example: python my_evalpref.py 0 66")
        sys.exit(1)
    
    try:
        start_id = int(sys.argv[1])
        end_id = int(sys.argv[2])
    except ValueError:
        print("Error: Task IDs must be integers")
        sys.exit(1)
    
    # 验证参数范围
    if start_id < 0 or end_id < start_id:
        print("Error: Invalid task ID range")
        sys.exit(1)

    clean_humaneval_dir()

    success_count = 0
    failed_tasks = []

    print(f"Searching for tasks from HumanEval/{start_id} to HumanEval/{end_id}")
    # 加载指定范围内的任务
    tasks = load_humaneval_plus_tasks(start_id, end_id)
    if not tasks:
        print(f"No tasks found in range {start_id}-{end_id}")
        print("Please check:")
        print("1. HumanEvalPlus.jsonl or HumanEvalPlus1.jsonl/HumanEvalPlus2.jsonl files exist")
        print("2. Task IDs in the files follow the pattern 'HumanEval/X'")
        sys.exit(1)
    
    # 检查是否找到了所有请求的任务
    found_ids = [int(task["task_id"].split("/")[1]) for task in tasks]
    expected_ids = list(range(start_id, end_id + 1))
    missing_ids = set(expected_ids) - set(found_ids)
    
    if missing_ids:
        print(f"Warning: Missing tasks: {sorted(missing_ids)}")
        print(f"Found {len(tasks)} out of {len(expected_ids)} requested tasks")
        response = input("Continue with available tasks? (y/n): ")
        if response.lower() != 'y':
            print("Execution cancelled")
            sys.exit(0)
    
    print(f"Loaded {len(tasks)} tasks")
    
    # 创建任务目录
    setup_task_directories(tasks)
    
    # 按顺序执行每个任务
    for i, task in enumerate(tasks):
        task_id = task["task_id"]
        print(f"\n{'='*60}")
        print(f"Starting evaluation for Task {task_id} ({i+1}/{len(tasks)})")
        print(f"{'='*60}")
        
        try:
            # 在开始处理任务前，先清理该任务的特定目录
            clean_humaneval_dir(task_id)
            
            # 为当前任务创建初始问题文件
            create_task_problems_file([task], 0)
            
            # 执行当前任务的完整评测流程
            run_single_task_evaluation(task_id)
            success_count += 1
            print(f"✓ {task_id} completed successfully")
            
        except Exception as e:
            print(f"✗ Error processing {task_id}: {e}")
            failed_tasks.append(task_id)
            # 记录详细错误信息
            import traceback
            traceback.print_exc()
            
            # 询问是否继续
            response = input("Continue with next task? (y/n): ")
            if response.lower() != 'y':
                print("Execution stopped by user")
                break
    
    # 生成最终总结报告
    print(f"\n{'='*60}")
    print("EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total tasks processed: {len(tasks)}")
    print(f"Successfully completed: {success_count}")
    print(f"Failed tasks: {len(failed_tasks)}")
    
    if failed_tasks:
        print(f"Failed task IDs: {failed_tasks}")
    
    if success_count > 0:
        print("Evaluation partially completed")
    else:
        print("Evaluation failed for all tasks")
    
    print(f"{'='*60}")

def run_single_task_evaluation(task_id: str):
    """为单个任务运行完整的评测流程"""
    # 存储当前任务的B分数
    task_B_scores = []
    task_metrics = []
    
    # 执行多次迭代
    for iteration in range(NUM_ITERATION):
        print(f"  Iteration {iteration + 1}/{NUM_ITERATION}")
        
        # 1. 加载当前问题集
        problems = load_problems(iteration)
        if not problems or task_id not in problems:
            print(f"  Warning: No problem found for {task_id} at iteration {iteration}")
            break
        
        problem = problems[task_id]
        
        # 2. 生成并评估样本
        samples_path = generate_and_rename_samples_single_task(iteration, task_id, problem)
        eval_result_path = evaluate_samples_single_task(samples_path, iteration, task_id)
        evaluated_samples = read_evaluated_samples(eval_result_path)
        
        # 3. 筛选样本并计算分数
        task_samples = evaluated_samples.get(task_id, [])
        
        B, passk = calculate_and_log_scores_single_task(
            task_id=task_id,
            task_samples=task_samples,
            problem=problem,
            iteration=iteration
        )
        
        task_B_scores.append(B)
        task_metrics.append({
            "task_id": task_id,
            "iteration": iteration,
            "B_score": round(B, 6),
            "pass@k": passk
        })
        
        # 4. 生成新问题集
        candidate = select_sample(task_samples, problem)
        new_problem = None
        if candidate:
            new_problem = {
                "task_id": task_id,
                "prompt": generate_one_problem(candidate["solution"], problem["entry_point"]),
                **{k: v for k, v in problem.items() if k != "prompt"}
            }
        
        # 5. 保存新问题集用于下一次迭代
        if new_problem:
            new_problems_path = os.path.join(PROBLEM_PATH, f"problems{iteration+1}.jsonl")
            write_jsonl(new_problems_path, [new_problem])
        else:
            print(f"  Warning: No candidate found for {task_id} at iteration {iteration}")
            # 如果没有候选，复制当前问题到下一次迭代
            current_problems_path = os.path.join(PROBLEM_PATH, f"problems{iteration}.jsonl")
            next_problems_path = os.path.join(PROBLEM_PATH, f"problems{iteration+1}.jsonl")
            if os.path.exists(current_problems_path):
                with open(current_problems_path, 'r') as f:
                    problems_data = [json.loads(line) for line in f]
                write_jsonl(next_problems_path, problems_data)
    
    # 6. 计算最终分数和生成报告
    if task_B_scores:
        final_score = calculate_final_score(task_B_scores)
        
        # 收集失败案例
        fail_stats, all_task_ids = collect_fail_cases_single_task(task_id, NUM_ITERATION)
        frequent_cases = filter_frequent_fails(fail_stats, NUM_ITERATION, [task_id])
        
        # 生成本任务报告
        generate_task_report(task_id, final_score, frequent_cases, task_metrics)
        
        print(f"  Task {task_id} completed. Final score: {final_score:.6f}")
    else:
        print(f"  Warning: No evaluation results for {task_id}")


if __name__ == "__main__":
    main()