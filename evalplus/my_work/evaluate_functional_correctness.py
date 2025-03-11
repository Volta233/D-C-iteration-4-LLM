import sys
import os
import shutil
import json
from typing import Dict, List, Generator

from evalplus.my_work.generate_samples import *

from evalplus.data.utils import stream_jsonl,write_jsonl
from evalplus.evaluate import evaluate, my_test_results
from evalplus.codegen import my_run_codegen

HUMANEVAL_OVERRIDE_PATH = None
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_PATH = os.path.join(BASE_DIR, "my_data", "result", "humaneval")
PROBLEM_PATH = os.path.join(BASE_DIR,"my_data","problem")

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
                        "base_status": record.get("base_status", "unknown"),
                        "plus_status": record.get("plus_status", "unknown")
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
            "base_status": record["base_status"],
            "plus_status": record["plus_status"],
            "base_fail_tests": record["base_fail_tests"],
            "plus_fail_tests": record["plus_fail_tests"]
        })
    return samples

def select_sample(samples: List[Dict], problem: Dict) -> Optional[Dict]:
    """基于评估结果的四维筛选策略"""
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
    num_samples_per_task = 2
    num_of_tasks = 1

    # 流程迭代次数
    num_iteration = 2

    # data文件夹绝对路径
    folder_path = PROBLEM_PATH

    #循环体,初始提供problems0.jsonl文件
    i = 0
    while i < num_iteration :
        # 获取problems
        # print("=============start getting problems=================")
        problems_name = "problems" + str(i) + ".jsonl"
        problems_path = os.path.join(folder_path, problems_name)
        HUMANEVAL_OVERRIDE_PATH = problems_path
        # print("HUMANEVAL_OVERRIDE_PATH = " + HUMANEVAL_OVERRIDE_PATH)
        problems = read_problems(problems_path)

        # 对每个任务生成样本
        # print("=============start getting samples===================")
        samples_path = my_run_codegen(model="gpt-4o-mini",
                              root=os.path.join(BASE_DIR, "my_data", "result"),
                              n_samples = num_samples_per_task,
                              greedy=False,
                              dataset="humaneval",
                              base_url="https://svip.xty.app/v1",
                              backend="openai",
                              HUMANEVAL_OVERRIDE_PATH=HUMANEVAL_OVERRIDE_PATH
                              )

        samples_name = "samples" + str(i) 
        # 获取原文件路径
        original_file_path = samples_path

        # 设置新的文件名
        _, file_extension = os.path.splitext(original_file_path)
        new_file_name = samples_name + file_extension  # 新的文件名
        new_file_path = os.path.join(os.path.dirname(original_file_path), new_file_name)

        # 使用 os.rename() 重命名文件
        os.rename(original_file_path, new_file_path)
        # 对样本进行检测
        # print("=============start evaluating samples===================")

        evaluate(dataset="humaneval",
                 samples=new_file_path,
                 i_just_wanna_run=True,
                 HUMANEVAL_OVERRIDE_PATH=HUMANEVAL_OVERRIDE_PATH)
        
        # 根据样本生成新的problem文件
        # 读取 JSONL 文件并解析每一行

        # 修改后的new_problems生成逻辑
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


sys.exit(main())
