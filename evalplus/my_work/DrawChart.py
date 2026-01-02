#!/usr/bin/env python3
import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from evalplus.my_work.hyperparams import *

class ChartDrawer:
    def __init__(self):
        self.task_reports = {}
        self.all_tasks_data = {}
    
    def load_single_task_report(self, task_id: int) -> Dict:
        """加载单个任务的报告数据"""
        task_id_str = f"HumanEval/{task_id}"
        task_safe_id = task_id_str.replace("/", "_")
        report_path = os.path.join(SCORE_PATH, task_safe_id, "task_report.json")
        
        if not os.path.exists(report_path):
            print(f"Error: Task report not found at {report_path}")
            return None
        
        try:
            with open(report_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading task report: {e}")
            return None
    
    def load_all_tasks_reports(self, start_id: int = 0, end_id: int = 163) -> Dict:
        """加载所有任务的报告数据"""
        all_data = {}
        valid_count = 0
        
        for task_id in range(start_id, end_id + 1):
            task_report = self.load_single_task_report(task_id)
            if task_report:
                all_data[task_id] = task_report
                valid_count += 1
                if valid_count % 20 == 0:
                    print(f"Loaded {valid_count} task reports...")
        
        print(f"Successfully loaded {valid_count} out of {end_id - start_id + 1} task reports")
        return all_data
    
    def draw_single_task_chart(self, task_id: int):
        """绘制单个任务的pass@1和B_score变化趋势"""
        task_report = self.load_single_task_report(task_id)
        if not task_report:
            return
        
        metrics = task_report.get("metrics_per_iteration", [])
        if not metrics:
            print(f"No metrics found for task {task_id}")
            return
        
        # 提取数据
        iterations = []
        base_pass_at_1 = []
        plus_pass_at_1 = []
        b_scores = []
        
        for metric in metrics:
            iterations.append(metric["iteration"])
            b_scores.append(metric["B_score"])
            
            # 提取pass@1数据
            pass_at_k = metric.get("pass@k", {})
            base_pass = pass_at_k.get("base", {}).get("pass@1", 0)
            plus_pass = pass_at_k.get("plus", {}).get("pass@1", 0)
            
            base_pass_at_1.append(base_pass)
            plus_pass_at_1.append(plus_pass)
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 绘制pass@1图表
        ax1.plot(iterations, base_pass_at_1, 'b-o', label='Base pass@1', linewidth=2, markersize=8)
        ax1.plot(iterations, plus_pass_at_1, 'r-o', label='Plus pass@1', linewidth=2, markersize=8)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Pass@1 Rate')
        ax1.set_title(f'Task HumanEval/{task_id} - Pass@1 Trend')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(iterations)
        
        # 绘制B_score图表
        ax2.plot(iterations, b_scores, 'g-o', label='B Score', linewidth=2, markersize=8)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('B Score')
        ax2.set_title(f'Task HumanEval/{task_id} - B Score Trend')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(iterations)
        
        plt.tight_layout()
        
        # 保存图表
        output_dir = os.path.join(SCORE_PATH, "charts")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"task_{task_id}_trend.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Chart saved to: {output_path}")
        
        # 打印数据摘要
        print(f"\nTask HumanEval/{task_id} Summary:")
        print(f"Iterations: {len(iterations)}")
        print(f"Final B Score: {b_scores[-1]:.6f}")
        print(f"Final Base pass@1: {base_pass_at_1[-1]:.4f}")
        print(f"Final Plus pass@1: {plus_pass_at_1[-1]:.4f}")
    
    def calculate_overall_pass_at_1(self, start_id: int = 0, end_id: int = 163) -> Tuple[List[float], List[float]]:
        """计算所有任务的总体pass@1"""
        all_data = self.load_all_tasks_reports(start_id, end_id)
        
        # 初始化存储结构
        max_iterations = NUM_ITERATION
        total_base_pass_at_1 = [0.0] * max_iterations
        total_plus_pass_at_1 = [0.0] * max_iterations
        task_count_per_iteration = [0] * max_iterations
        
        # 汇总数据
        for task_id, report in all_data.items():
            metrics = report.get("metrics_per_iteration", [])
            for metric in metrics:
                iteration = metric["iteration"]
                if iteration < max_iterations:
                    pass_at_k = metric.get("pass@k", {})
                    base_pass = pass_at_k.get("base", {}).get("pass@1", 0)
                    plus_pass = pass_at_k.get("plus", {}).get("pass@1", 0)
                    
                    # 乘以样本数量并累加
                    total_base_pass_at_1[iteration] += base_pass * NUM_SAMPLES_PER_TASK
                    total_plus_pass_at_1[iteration] += plus_pass * NUM_SAMPLES_PER_TASK
                    task_count_per_iteration[iteration] += 1
        
        # 计算平均值
        overall_base_pass_at_1 = []
        overall_plus_pass_at_1 = []
        
        for i in range(max_iterations):
            if task_count_per_iteration[i] > 0:
                total_samples = task_count_per_iteration[i] * NUM_SAMPLES_PER_TASK
                base_avg = total_base_pass_at_1[i] / total_samples if total_samples > 0 else 0
                plus_avg = total_plus_pass_at_1[i] / total_samples if total_samples > 0 else 0
            else:
                base_avg = 0
                plus_avg = 0
            
            overall_base_pass_at_1.append(base_avg)
            overall_plus_pass_at_1.append(plus_avg)
        
        return overall_base_pass_at_1, overall_plus_pass_at_1
    
    def draw_overall_chart(self, start_id: int = 0, end_id: int = 163):
        """绘制所有任务的总体pass@1趋势图"""
        base_pass_at_1, plus_pass_at_1 = self.calculate_overall_pass_at_1(start_id, end_id)
        
        iterations = list(range(len(base_pass_at_1)))
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, base_pass_at_1, 'b-o', label='Overall Base pass@1', linewidth=2, markersize=8)
        plt.plot(iterations, plus_pass_at_1, 'r-o', label='Overall Plus pass@1', linewidth=2, markersize=8)
        plt.xlabel('Iteration')
        plt.ylabel('Overall Pass@1 Rate')
        plt.title(f'Overall Pass@1 Trend (Tasks {start_id}-{end_id})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(iterations)
        
        # 保存图表
        output_dir = os.path.join(SCORE_PATH, "charts")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"overall_pass_at_1_trend.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Overall chart saved to: {output_path}")
        
        # 打印统计数据
        print(f"\nOverall Pass@1 Statistics (Tasks {start_id}-{end_id}):")
        for i in range(len(iterations)):
            print(f"Iteration {i}: Base={base_pass_at_1[i]:.4f}, Plus={plus_pass_at_1[i]:.4f}")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 DrawChart.py <task_id>          # Draw chart for single task")
        print("  python3 DrawChart.py -1                 # Draw overall chart for all tasks")
        print("  python3 DrawChart.py -1 <start> <end>   # Draw overall chart for tasks in range")
        sys.exit(1)
    
    drawer = ChartDrawer()
    
    if sys.argv[1] == "-1":
        # 绘制总体图表
        start_id = 0
        end_id = 163
        
        if len(sys.argv) >= 4:
            try:
                start_id = int(sys.argv[2])
                end_id = int(sys.argv[3])
            except ValueError:
                print("Error: Start and end IDs must be integers")
                sys.exit(1)
        
        drawer.draw_overall_chart(start_id, end_id)
    else:
        # 绘制单个任务图表
        try:
            task_id = int(sys.argv[1])
            drawer.draw_single_task_chart(task_id)
        except ValueError:
            print("Error: Task ID must be an integer")
            sys.exit(1)

if __name__ == "__main__":
    main()