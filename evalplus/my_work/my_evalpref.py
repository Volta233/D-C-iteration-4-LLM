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

def generate_and_rename_samples_single_task(iteration: int, task_id: str, problem: Dict) -> str:
    """为单个任务生成样本"""
    # 创建临时问题文件（只包含当前任务）
    temp_problem_path = os.path.join(PROBLEM_PATH, f"temp_problems{iteration}.jsonl")
    write_jsonl(temp_problem_path, [problem])
    
    samples_path = my_run_codegen(
        model="gpt-4o-mini",
        root=os.path.join(BASE_DIR, "my_data", "result"),
        n_samples=NUM_SAMPLES_PER_TASK,
        temperature=0.6,
        greedy=False,
        dataset="humaneval",
        base_url=GPT_BASE_URL,
        backend="openai",
        HUMANEVAL_OVERRIDE_PATH=temp_problem_path,
        resume=False
    )
    
    # 重命名文件以包含任务ID
    task_safe_id = task_id.replace("/", "_")
    new_path = samples_path.replace("samples.jsonl", f"samples{iteration}_{task_safe_id}.jsonl")
    os.rename(samples_path, new_path)
    
    # 清理临时文件
    if os.path.exists(temp_problem_path):
        os.remove(temp_problem_path)
    
    return new_path

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
    main()