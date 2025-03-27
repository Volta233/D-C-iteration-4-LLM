import sys
import os
import shutil
import json
import numpy as np
from typing import Dict, List, Generator

from evalplus.my_work.generate_samples import *
from evalplus.my_work.get_final_output import generate_final_report

from evalplus.data.utils import stream_jsonl,write_jsonl
from evalplus.evaluate import evaluate
from evalplus.codegen import my_run_codegen

HUMANEVAL_OVERRIDE_PATH = None
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_PATH = os.path.join(BASE_DIR, "my_data", "result", "humaneval")
PROBLEM_PATH = os.path.join(BASE_DIR,"my_data","problem")
SCORE_PATH = os.path.join(BASE_DIR,"my_data","score")
GPT_BASE_URL = "https://svip.xty.app/v1"

def clean_humaneval_dir():
    """安全清理目录函数，保留符合正则模式的文件"""
    pattern = re.compile(r'^gpt-4o-mini_openai_temp_0\..+\.raw\.jsonl$')
    
    for filename in os.listdir(RESULT_PATH):
        if pattern.match(filename):  # 匹配目标文件名模式则跳过
            continue
        file_path = os.path.join(RESULT_PATH, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {str(e)}")

    for filename in os.listdir(SCORE_PATH) :
        file_path = os.path.join(SCORE_PATH, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {str(e)}")

    # for filename in os.listdir(PROBLEM_PATH) :
    #     file_path = os.path.join(PROBLEM_PATH, filename)
    #     try:
    #         if os.path.isfile(file_path):
    #             os.remove(file_path)
    #         elif os.path.isdir(file_path):
    #             shutil.rmtree(file_path)
    #     except Exception as e:
    #         print(f"Error deleting {file_path}: {str(e)}")


def read_problems(evalset_file: str = PROBLEM_PATH) -> Dict[str, Dict]:
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}

def stream_eval_results(file_path: str) -> Generator[Dict, None, None]:
    """流式读取评估结果文件并展平数据结构"""
    with open(file_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            # 展开eval字段中的任务数据
            for task_id, samples in record["eval"].items():
                for sample in samples:
                    yield {
                        **sample,
                        "task_id": task_id,
                        "pass_at_k": record["pass_at_k"],
                        "base_status": sample.get("base_status", "unknown"),
                        "plus_status": sample.get("plus_status", "unknown")
                    }

def read_evaluated_samples(eval_file: str) -> Dict[str, List[Dict]]:
    """读取评估结果并转换为按task_id分组的样本字典"""
    samples = {}
    for record in stream_eval_results(eval_file):
        task_id = record["task_id"]
        if task_id not in samples:
            samples[task_id] = []
        samples[task_id].append({
            "solution": record["solution"],
            "pass_at_k": record["pass_at_k"],
            "base_status": record["base_status"],
            "plus_status": record["plus_status"],
            "base_fail_tests": record["base_fail_tests"],
            "plus_fail_tests": record["plus_fail_tests"]
        })
    return samples

def calculate_and_log_scores(task_id:str,
                            task_samples: List[Dict], 
                            problem: Dict, 
                            iteration: int,
                            folder_path: str,
                            decay_rate: float = 0.1) -> None:
    """动态计算样本评分并写入迭代日志文件"""
    # 生成带迭代次数的文件名
    score_log_path = os.path.join(folder_path, f"score_{iteration}.ndjson")
    
    # 预计算总测试数量
    total_base_tests = len(problem["base_input"])
    total_plus_tests = len(problem["plus_input"])
    total_tests = total_base_tests + total_plus_tests
    
    with open(score_log_path, 'w') as f:
        for sample in task_samples:
            # 关键字段提取
            task_id = task_id
            pass_at_k = sample.get("pass_at_k", {})
            
            # 动态权重计算
            time_weight = 1 + decay_rate * iteration
            weighted_base = len(sample["base_fail_tests"]) * time_weight
            weighted_plus = len(sample["plus_fail_tests"]) * time_weight
            total_fails = weighted_base + weighted_plus
            
            # 评分公式实现
            fail_ratio = total_fails / total_tests if total_tests else 1.0
            pass_ratio = 1 - fail_ratio
            composite_score = pass_ratio * (1 - (fail_ratio * time_weight)**2)
            # 动态ε修正策略
            if composite_score >= 0.99:
                # 当评分≥0.99时，ε随接近1的程度线性增大（系数0.1可调）
                epsilon = max(1e-8, 0.1 * (1 - composite_score))
            else:
                # 低分区保持固定极小量（避免浮点误差）
                epsilon = 1e-8
    
            # 对数变换的统一处理（消除条件分支）
            adjusted_score = np.log(1 / (1 - composite_score + epsilon))
            # 构建带权重的失败记录
            base_details = [
                {"test_input": test, "time_weight": time_weight}
                for test in sample["base_fail_tests"]
            ]
            plus_details = [
                {"test_input": test, "time_weight": time_weight}
                for test in sample["plus_fail_tests"]
            ]
            
            # 生成符合格式要求的记录
            log_record = {
                "task_id": task_id,
                "iteration": iteration,
                "pass@k": pass_at_k,
                "composite_score": round(adjusted_score, 6),
                "base_fail_details": base_details,
                "plus_fail_details": plus_details
            }
            
            # 写入NDJSON格式
            f.write(json.dumps(log_record) + '\n')

def select_sample(samples: List[Dict], problem: Dict) -> Optional[Dict]:
    """基于评估结果的四维筛选策略, 同时记录每个sample的评分"""
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

def main():
    clean_humaneval_dir()  # 执行目录清理
    # 定义每个任务生成的样本数量
    num_samples_per_task = 10
    # 流程迭代次数
    num_iteration = 10

    # data文件夹绝对路径
    folder_path = PROBLEM_PATH
    score_log_path = SCORE_PATH
    #循环体,初始提供problems0.jsonl文件
    i = 0
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
            calculate_and_log_scores(
                task_id=task_id,
                task_samples=task_samples,
                problem=problem,
                iteration=i,
                folder_path=score_log_path,
                decay_rate=0.1  # 可配置参数
            )
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

        i += 1
    
generate_final_report()
sys.exit(main())
