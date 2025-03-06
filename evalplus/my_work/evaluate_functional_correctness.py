import sys
import os
from typing import Dict
from evalplus.my_work.generate_samples import *

from evalplus.data.utils import stream_jsonl,write_jsonl
from evalplus.evaluate import evaluate, my_test_results
from evalplus.codegen import my_run_codegen

ABSOLUTE_PATH = "/home/lyw/Documents/evalplus/evalplus/my_data/"
HUMANEVAL_OVERRIDE_PATH = None

def read_problems(evalset_file: str = ABSOLUTE_PATH) -> Dict[str, Dict]:
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}

def main():
    # 定义每个任务生成的样本数量
    num_samples_per_task = 1
    num_of_tasks = 30

    # 流程迭代次数
    num_iteration = 10

    # data文件夹绝对路径
    folder_path = ABSOLUTE_PATH

    #循环体,初始提供problems0.jsonl文件
    i = 0
    my_test_results.normal = 0
    my_test_results.plus = 0
    my_test_results.first_no_pass = -1
    my_test_results.flag = False
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
                              root="/home/lyw/Documents/evalplus/evalplus/my_data/result",
                              greedy=True,
                              dataset="humaneval",
                              base_url="https://svip.xty.app/v1",
                              backend="openai",
                              HUMANEVAL_OVERRIDE_PATH=HUMANEVAL_OVERRIDE_PATH
                              )
        samples = read_problems(samples_path)
        # 将生成的样本保存到jsonl文件中,文件名形如“samples0.jsonl"
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
        # print("=============start generating new problems===================")
        new_problems = [
            dict(task_id=task_id,
                 prompt=generate_one_problem(
            next((sample["solution"] for sample in samples.values() if sample["task_id"] == task_id), None),
            problem["entry_point"]
        ),
                 entry_point=problem["entry_point"],
                 canonical_solution=problem["canonical_solution"],
                 contract=problem["contract"],
                 base_input=problem["base_input"],
                 atol=problem["atol"],
                 plus_input=problem["plus_input"])
            for task_id,problem in problems.items()
        ]

        new_problems_name = "problems" + str(i+1) + ".jsonl"
        new_problems_path = os.path.join(folder_path,new_problems_name)
        write_jsonl(new_problems_path, new_problems)

        i += 1
    
    print(f"\033[31mbase_status: {my_test_results.normal} / {num_iteration * num_of_tasks}\033[0m")  # 红色
    print(f"\033[32mplus_status: {my_test_results.plus} / {num_iteration * num_of_tasks}\033[0m")    # 绿色
    if my_test_results.flag and num_of_tasks == 1:
        print(f"\033[33mfirst error occur at {my_test_results.first_no_pass}\033[0m")  # 黄色


sys.exit(main())
