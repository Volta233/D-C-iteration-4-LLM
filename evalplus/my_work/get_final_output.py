import glob
import json
import os
from collections import defaultdict

def calculate_average_score(file_path):
    """计算单个文件的平均分"""
    total_score = 0.0
    count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                if 'composite_score' in data:
                    total_score += data['composite_score']
                    count += 1
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error processing {file_path}: {str(e)}")
        return 0.0
    
    return total_score / count if count > 0 else 0.0

def generate_final_report():
    """生成最终统计报告"""
    # 收集所有score文件
    score_files = glob.glob('score_*.ndjson')
    
    # 创建按迭代次数排序的字典
    results = defaultdict(list)
    for file_path in score_files:
        # 从文件名提取迭代次数
        base_name = os.path.basename(file_path)
        iteration = int(base_name.split('_')[1].split('.')[0])
        
        # 计算平均分
        avg_score = calculate_average_score(file_path)
        results[iteration].append(avg_score)
    
    # 处理最终结果
    final_report = []
    for iteration in sorted(results.keys()):
        # 计算每个迭代次数的平均分
        scores = results[iteration]
        final_avg = sum(scores) / len(scores) if scores else 0.0
        final_report.append((iteration, final_avg))
    
    # 格式化表格输出
    header =        "┌───────────────┬───────────────┐\n  \
                     │ Iteration     │ Average Score │\n   \
                     ├───────────────┼───────────────┤"
    row_template =  "│ {iter:^13}    │ {score:^13.4f}│"
    footer =        "└───────────────┴───────────────┘"
    
    print("\n\033[1mFINAL EVALUATION REPORT\033[0m")  # 加粗标题
    print(header)
    for iter_num, avg_score in final_report:
        print(row_template.format(iter=iter_num, score=avg_score))
    print(footer)

if __name__ == "__main__":
    generate_final_report()