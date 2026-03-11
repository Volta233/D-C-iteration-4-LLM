#!/usr/bin/env python3
"""主入口：在 dc_iteration/CodeCore 中运行以执行评测任务"""
import sys
import os
import json

# 确保项目根目录在路径中（dc_iteration 的父目录）
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from dc_iteration.CodeCore.IO_process import *
from dc_iteration.CodeCore.hyperparams import *
from dc_iteration.CodeCore.generate_samples import *
from dc_iteration.CodeCore.get_scores import *


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

    if start_id < 0 or end_id < start_id:
        print("Error: Invalid task ID range")
        sys.exit(1)

    clean_humaneval_dir()
    delete_empty_task_directories()

    success_count = 0
    failed_tasks = []

    print(f"Searching for tasks from HumanEval/{start_id} to HumanEval/{end_id}")
    tasks = load_humaneval_plus_tasks(start_id, end_id)
    if not tasks:
        print(f"No tasks found in range {start_id}-{end_id}")
        print("Please check HumanEvalPlus1.jsonl / HumanEvalPlus2.jsonl exist at project root")
        sys.exit(1)

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
    setup_task_directories(tasks)

    for i, task in enumerate(tasks):
        task_id = task["task_id"]
        print(f"\n{'='*60}")
        print(f"Starting evaluation for Task {task_id} ({i+1}/{len(tasks)})")
        print(f"{'='*60}")

        try:
            clean_humaneval_dir(task_id)
            create_task_problems_file([task], 0)
            run_single_task_evaluation(task_id)
            success_count += 1
            print(f"✓ {task_id} completed successfully")

            try:
                organize_eval_results(RESULT_PATH)
                print(f"  Organized result files for {task_id}")
            except Exception as e:
                print(f"  Warning: Failed to organize result files: {e}")

        except Exception as e:
            print(f"✗ Error processing {task_id}: {e}")
            failed_tasks.append(task_id)
            import traceback
            traceback.print_exc()

            response = input("Continue with next task? (y/n): ")
            if response.lower() != 'y':
                print("Execution stopped by user")
                break

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
    task_B_scores = []
    task_metrics = []

    for iteration in range(NUM_ITERATION):
        print(f"  Iteration {iteration + 1}/{NUM_ITERATION}")

        problems = load_problems(iteration)
        if not problems or task_id not in problems:
            print(f"  Warning: No problem found for {task_id} at iteration {iteration}")
            break

        problem = problems[task_id]

        samples_path = generate_and_rename_samples_single_task(iteration, task_id, problem)
        eval_result_path = evaluate_samples_single_task(samples_path, iteration, task_id)
        evaluated_samples = read_evaluated_samples(eval_result_path)

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

        candidate = select_sample(task_samples, problem)
        new_problem = None
        if candidate:
            new_problem = {
                "task_id": task_id,
                "prompt": generate_one_problem(candidate["solution"], problem["entry_point"]),
                **{k: v for k, v in problem.items() if k != "prompt"}
            }

        if new_problem:
            new_problems_path = os.path.join(PROBLEM_PATH, f"problems{iteration+1}.jsonl")
            write_jsonl(new_problems_path, [new_problem])
        else:
            print(f"  Warning: No candidate found for {task_id} at iteration {iteration}")
            current_problems_path = os.path.join(PROBLEM_PATH, f"problems{iteration}.jsonl")
            next_problems_path = os.path.join(PROBLEM_PATH, f"problems{iteration+1}.jsonl")
            if os.path.exists(current_problems_path):
                with open(current_problems_path, 'r') as f:
                    problems_data = [json.loads(line) for line in f]
                write_jsonl(next_problems_path, problems_data)

    if task_B_scores:
        final_score = calculate_final_score(task_B_scores)
        fail_stats, all_task_ids = collect_fail_cases_single_task(task_id, NUM_ITERATION)
        frequent_cases = filter_frequent_fails(fail_stats, NUM_ITERATION, [task_id])
        generate_task_report(task_id, final_score, frequent_cases, task_metrics)

        print(f"  Task {task_id} completed. Final score: {final_score:.6f}")

        try:
            organize_eval_results(RESULT_PATH)
            print(f"  Organized result files for {task_id}")
        except Exception as e:
            print(f"  Warning: Failed to organize result files: {e}")
    else:
        print(f"  Warning: No evaluation results for {task_id}")


if __name__ == "__main__":
    main()
