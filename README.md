# D-C Iteration

基于 HumanEval 的 LLM 语义偏移检测与迭代评测框架。

## 项目结构

```
.
├── dc_iteration/           # 主模块
│   ├── CodeCore/           # 主驱动（入口与核心逻辑）
│   ├── store_data/         # 运行数据（problem, result, score）
│   ├── data/               # 数据加载
│   ├── eval/               # 评估逻辑
│   ├── provider/           # 模型调用
│   └── ...
├── HumanEvalPlus1.jsonl     # 任务数据
├── HumanEvalPlus2.jsonl
├── run.sh                  # Linux 运行脚本
└── requirements.txt
```

## 运行方式

在项目根目录执行：

```bash
python dc_iteration/CodeCore/my_evalpref.py <start_task_id> <end_task_id>
```

示例：`python dc_iteration/CodeCore/my_evalpref.py 0 66`

Linux 下可使用：`./run.sh 0 66`

## 依赖

```bash
pip install -r requirements.txt
```

## 配置

模型与路径配置见 `dc_iteration/CodeCore/hyperparams.py`。
