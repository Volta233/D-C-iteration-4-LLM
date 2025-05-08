import sys
import os
import shutil
import json
import numpy as np
from typing import Dict, List, Generator

from evalplus.my_work.generate_samples import *
from evalplus.my_work.get_scores import *
from evalplus.my_work.IO_process import *
from evalplus.evaluate import evaluate
from evalplus.codegen import my_run_codegen
from evalplus.my_work.hyperparams import *
from evalplus.my_work.comment_similarity import *

def main():
    clean_humaneval_dir()  # 执行目录清理
    # 定义每个任务生成的样本数量
    num_samples_per_task = NUM_SAMPLES_PER_TASK
    # 流程迭代次数
    num_iteration = NUM_ITERATION

    # data文件夹绝对路径
    folder_path = PROBLEM_PATH
    score_log_path = SCORE_PATH
    # 循环体,初始提供problems0.jsonl文件
    # 加载原始问题描述
    original_problems = read_problems(os.path.join(PROBLEM_PATH, "problems0.jsonl"))
    original_docs = {
        task_id: extract_docstring(problem["prompt"])
        for task_id, problem in original_problems.items()
    }
    i = 0
    B_scores = []
    while i < num_iteration :
        # 获取problems
        problems_name = "problems" + str(i) + ".jsonl"
        problems_path = os.path.join(folder_path, problems_name)
        HUMANEVAL_OVERRIDE_PATH = problems_path
        problems = read_problems(problems_path)

        # 对每个任务生成样本
        samples_path = my_run_codegen(model="gpt-4o-mini",
                              root=os.path.join(BASE_DIR, "my_data", "result"),
                              n_samples = num_samples_per_task,
                              temperature=0.6,
                              greedy=False,
                              dataset="humaneval",
                              base_url=GPT_BASE_URL,
                              backend="openai",
                              HUMANEVAL_OVERRIDE_PATH=HUMANEVAL_OVERRIDE_PATH
                              )

        samples_name = "samples" + str(i) 
        original_file_path = samples_path
        _, file_extension = os.path.splitext(original_file_path)
        new_file_name = samples_name + file_extension  # 新的文件名
        new_file_path = os.path.join(os.path.dirname(original_file_path), new_file_name)
        os.rename(original_file_path, new_file_path)
        # 对样本进行检测
        evaluate(dataset="humaneval",
                 samples=new_file_path,
                 i_just_wanna_run=True,
                 HUMANEVAL_OVERRIDE_PATH=HUMANEVAL_OVERRIDE_PATH)
        # 根据样本生成新的problem文件
        # 读取 JSONL 文件并解析每一行
        seen_ids = set()
        new_problems = []

        eval_result_name = samples_name + "_eval_results" + ".json"
        eval_result_path = os.path.join(os.path.dirname(new_file_path), eval_result_name)
        evaluated_samples = read_evaluated_samples(eval_result_path)
        
        for task_id, problem in problems.items():
            # 获取当前task_id对应的所有样本
            task_samples = evaluated_samples.get(task_id, [])
            assert len(task_samples) > 0, "解析评估结果失败"

            # 验证字段完整性
            assert "base_fail_tests" in task_samples[0], "缺失关键评估字段"
            B = calculate_and_log_scores(
                task_id=task_id,
                task_samples=task_samples,
                problem=problem,
                iteration=i,
                folder_path=score_log_path
            )
            B_scores.append(B)
            # 执行双重筛选
            candidate = select_sample(task_samples, problem) 
            
            # 确保唯一性
            if task_id not in seen_ids and candidate:
                seen_ids.add(task_id)
                new_problems.append(
                    dict(
                        task_id=task_id,
                        prompt=generate_one_problem(
                            candidate["solution"],
                            problem["entry_point"]
                        ),
                        # 保留原有字段
                        entry_point=problem["entry_point"],
                        canonical_solution=problem["canonical_solution"],
                        contract=problem["contract"],
                        base_input=problem["base_input"],
                        atol=problem["atol"],
                        plus_input=problem["plus_input"]
                    )
                )

        new_problems_name = "problems" + str(i+1) + ".jsonl"
        new_problems_path = os.path.join(folder_path,new_problems_name)
        write_jsonl(new_problems_path, new_problems)

        new_problems = read_problems(new_problems_path)
    
        # 计算语义相似度
        semantic_scores = []
        for task_id, problem in new_problems.items():
            new_doc = extract_docstring(problem["prompt"])
            original_doc = original_docs.get(task_id, "")
            similarity = get_similarity_score(original_doc, new_doc)
            semantic_scores.append(similarity)
        
        avg_semantic = sum(semantic_scores)/len(semantic_scores) if semantic_scores else 0.0

        # 写入到score文件 
        summary_entry = {
            "iteration": i,
            "semantic_similarity": avg_semantic,
            "type": "text_similarity_summary"
        }
        score_file = os.path.join(SCORE_PATH, f"semantic_score.ndjson")
        with open(score_file, 'a') as f:
            f.write(json.dumps(summary_entry) + '\n')

        i += 1
    
    final_score = calculate_final_score(B_scores)
    fail_stats = collect_fail_cases(SCORE_PATH, num_iteration)
    frequent_cases = filter_frequent_fails(fail_stats, num_iteration)
    generate_report(final_score, frequent_cases, REPORT_PATH)
    
sys.exit(main())
