import sys
import os
import shutil
import json
import numpy as np
from typing import Dict, List, Generator

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from evalplus.my_work.generate_samples import *
from evalplus.my_work.get_scores import *
from evalplus.my_work.IO_process import *
from evalplus.my_work.hyperparams import *
from evalplus.my_work.comment_similarity import *

# 主流程
def main():
    # 初始化阶段
    ensure_paths_exist()  # 确保路径存在
    clean_humaneval_dir()  # 执行目录清理
    
    # 加载初始数据集
    original_problems = read_problems(os.path.join(PROBLEM_PATH, "problems0.jsonl"))
    original_docs = {
        task_id: extract_docstring(problem["prompt"])
        for task_id, problem in original_problems.items()
    }
    
    # 迭代处理
    B_scores = []
    task_metrics = []
    for i in range(NUM_ITERATION):
        # 1. 加载当前问题集
        problems = load_problems(i)
        
        # 2. 生成并评估样本
        samples_path = generate_and_rename_samples(i, problems)
        eval_result_path = evaluate_samples(samples_path, i)
        evaluated_samples = read_evaluated_samples(eval_result_path)
        
        # 3. 筛选样本生成新问题集
        new_problems = []
        current_iter_metrics = []
        for task_id, problem in problems.items():
            task_samples = evaluated_samples.get(task_id, [])
            B, passk = calculate_and_log_scores(  # 原有函数不变
                task_id=task_id,
                task_samples=task_samples,
                problem=problem,
                iteration=i,
                folder_path=SCORE_PATH
            )
            B_scores.append(B)
            current_iter_metrics.append({
                "task_id": task_id,
                "iteration": i,
                "B_score": round(B, 6),
                "pass@k": passk
            })

            candidate = select_sample(task_samples, problem)  # 原有函数不变
            if candidate:
                new_problems.append({
                    "task_id": task_id,
                    "prompt": generate_one_problem(candidate["solution"], problem["entry_point"]),
                    **{k: v for k, v in problem.items() if k != "prompt"}
                })

        iter_metrics_path = os.path.join(SCORE_PATH, f"iteration_{i}_metrics.ndjson")
        with open(iter_metrics_path, 'w') as f:
            for metric in current_iter_metrics:
                f.write(json.dumps(metric) + '\n')
        task_metrics.extend(current_iter_metrics)
        
        # 4. 保存新问题集
        new_problems_path = os.path.join(PROBLEM_PATH, f"problems{i+1}.jsonl")
        write_jsonl(new_problems_path, new_problems)
        
        # # 5. 计算语义相似度
        # avg_semantic = compute_semantic_similarity(
        #     read_problems(new_problems_path),
        #     original_docs
        # )
        # # 记录语义相似度
        # with open(os.path.join(SCORE_PATH, "semantic_score.ndjson"), 'a') as f:
        #     f.write(json.dumps({
        #         "iteration": i,
        #         "semantic_similarity": avg_semantic,
        #         "type": "text_similarity_summary"
        #     }) + '\n')
    
    # 最终处理
    final_score = calculate_final_score(B_scores)
    fail_stats, all_task_ids = collect_fail_cases(SCORE_PATH, NUM_ITERATION, NUM_SAMPLES_PER_TASK)
    frequent_cases = filter_frequent_fails(fail_stats, NUM_ITERATION, all_task_ids)
    generate_report(final_score, frequent_cases, REPORT_PATH)

if __name__ == "__main__":
    sys.exit(main())
