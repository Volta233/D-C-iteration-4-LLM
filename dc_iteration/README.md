# dc_iteration（D-C_iteration）

项目主模块，包含 CodeCore 主驱动与 store_data 数据目录。

## 目录结构

```
dc_iteration/
├── CodeCore/           # 主驱动
│   ├── my_evalpref.py  # 入口脚本
│   ├── IO_process.py
│   ├── generate_samples.py
│   ├── get_scores.py
│   └── hyperparams.py
├── store_data/         # 运行数据（problem, result, score）
├── data/               # 数据加载（humaneval）
├── eval/               # 评估逻辑
├── provider/           # 模型调用（OpenAI）
├── gen/util/           # 可信执行与 API 请求
├── config.py
├── utils.py
├── sanitize.py
├── syncheck.py
├── codegen.py
└── evaluate.py
```

## 运行方式

在**项目根目录**执行：

```bash
python dc_iteration/CodeCore/my_evalpref.py <start_task_id> <end_task_id>
```

或使用 run.sh（Linux）：

```bash
chmod +x run.sh
./run.sh 0 66
```

## 前置条件

- 项目根目录需有 `HumanEvalPlus1.jsonl`、`HumanEvalPlus2.jsonl`
- 依赖见 `requirements.txt`
