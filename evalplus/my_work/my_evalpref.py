import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from evalplus.my_work.IO_process import *
from evalplus.my_work.hyperparams import *
from evalplus.my_work.generate_samples import *
from evalplus.my_work.get_scores import *
from evalplus.my_work.comment_similarity import *

def load_humaneval_plus_tasks(start_id: int, end_id: int) -> List[Dict]:
    """从HumanEvalPlus.jsonl加载指定范围内的任务"""
    humaneval_plus_path = os.path.join(project_root, "HumanEvalPlus.jsonl")
    tasks = []
    
    with open(humaneval_plus_path, 'r') as f:
        for line in f:
            task = json.loads(line.strip())
            task_number = int(task["task_id"].split("/")[1])
            if start_id <= task_number <= end_id:
                tasks.append(task)
    
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
        task_id = task["task_id"].replace("/", "_")  # 将斜杠替换为下划线
        task_score_dir = os.path.join(SCORE_PATH, task_id)
        task_result_dir = os.path.join(RESULT_PATH, task_id)
        
        os.makedirs(task_score_dir, exist_ok=True)
        os.makedirs(task_result_dir, exist_ok=True)
        
        print(f"Created directories for {task_id}")

def run_evaluation_loop(tasks: List[Dict], start_iteration: int = 0):
    """运行完整的评测循环"""
    task_ids = [task["task_id"] for task in tasks]
    # 存储所有任务的B分数
    all_B_scores = []
    task_metrics = []

    for i in range(start_iteration, NUM_ITERATION):
        print(f"\n=== Starting Iteration {i} ===")
        
        # 1. 加载当前问题集
        problems = load_problems(i)
        if not problems:
            print(f"No problems found for iteration {i}")
            break
            
        # 2. 生成并评估样本
        print("Generating samples...")
        samples_path = generate_and_rename_samples(i, problems)
        
        print("Evaluating samples...")
        eval_result_path = evaluate_samples(samples_path, i)
        evaluated_samples = read_evaluated_samples(eval_result_path)
        
        # 3. 筛选样本生成新问题集
        new_problems = []
        current_iter_metrics = []
        
        for task_id in task_ids:
            if task_id not in problems:
                continue
                
            task_samples = evaluated_samples.get(task_id, [])
            problem = problems[task_id]
            
            # 计算B分数和pass@k
            B, passk = calculate_and_log_scores(
                task_id=task_id,
                task_samples=task_samples,
                problem=problem,
                iteration=i,
                folder_path=SCORE_PATH
            )
            
            all_B_scores.append(B)
            current_iter_metrics.append({
                "task_id": task_id,
                "iteration": i,
                "B_score": round(B, 6),
                "pass@k": passk
            })
            
            # 筛选最佳样本生成新问题
            candidate = select_sample(task_samples, problem)
            if candidate:
                new_problems.append({
                    "task_id": task_id,
                    "prompt": generate_one_problem(candidate["solution"], problem["entry_point"]),
                    **{k: v for k, v in problem.items() if k != "prompt"}
                })
        
        # 保存迭代指标
        iter_metrics_path = os.path.join(SCORE_PATH, f"iteration_{i}_metrics.ndjson")
        with open(iter_metrics_path, 'w') as f:
            for metric in current_iter_metrics:
                f.write(json.dumps(metric) + '\n')
        task_metrics.extend(current_iter_metrics)
        
        # 4. 保存新问题集（用于下一次迭代）
        if new_problems:
            new_problems_path = os.path.join(PROBLEM_PATH, f"problems{i+1}.jsonl")
            write_jsonl(new_problems_path, new_problems)
            print(f"Saved new problems for iteration {i+1}")
        else:
            print(f"No new problems generated for iteration {i+1}")
            break

    # 5. 计算最终分数和生成报告
    if all_B_scores:
        final_score = calculate_final_score(all_B_scores)
        fail_stats, all_task_ids = collect_fail_cases(SCORE_PATH, NUM_ITERATION, NUM_SAMPLES_PER_TASK)
        frequent_cases = filter_frequent_fails(fail_stats, NUM_ITERATION, all_task_ids)
        
        # 生成报告（暂时简化处理）
        generate_simple_report(final_score, frequent_cases, tasks)
        
        print(f"\n=== Evaluation Complete ===")
        print(f"Final Score: {final_score:.6f}")
        print(f"Processed {len(tasks)} tasks over {NUM_ITERATION} iterations")
    else:
        print("No evaluation results to report")

def generate_simple_report(final_score, frequent_cases, tasks):

    """生成简化版报告"""

    report_data = {
    "final_score": final_score,
    "total_tasks": len(tasks),
    "iterations": NUM_ITERATION,
    "frequent_fail_cases": frequent_cases}

    report_path = os.path.join(SCORE_PATH, "evaluation_report.json")
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)

    print(f"Report saved to {report_path}")

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
    
    # 加载指定范围内的任务
    tasks = load_humaneval_plus_tasks(start_id, end_id)
    if not tasks:
        print(f"No tasks found in range {start_id}-{end_id}")
        sys.exit(1)
    
    print(f"Loaded {len(tasks)} tasks from HumanEval/{start_id} to HumanEval/{end_id}")
    
    # 创建任务目录
    setup_task_directories(tasks)
    
    # 创建初始问题文件
    create_task_problems_file(tasks, 0)
    
    # 执行主评测流程（这里需要修改原有的主流程来适应新的目录结构）
    # 由于时间关系，这里先打印信息，具体实现需要修改原有流程
    run_evaluation_loop(tasks)

if __name__ == "__main__":
    main()