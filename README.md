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
注意需要先使用 `chmod +x run.sh`

## 依赖

python版本推荐为3.11

```bash
pip install -r requirements.txt
```

## 配置

模型与路径配置见 `dc_iteration/CodeCore/hyperparams.py`。

本项目的模型相关配置（如API密钥、服务地址、模型名称）已改为从环境变量读取，以提升安全性并便于在不同部署环境中切换。

1. 配置非敏感参数

在 `dc_iteration/CodeCore/hyperparams.py`中，您可以修改以下非敏感配置的默认值（如非必要，可保持默认）：

基础路径：RESULT_PATH, PROBLEM_PATH等。

迭代参数：NUM_SAMPLES_PER_TASK, NUM_ITERATION。

模型相关配置的占位符也在此文件中，但实际值由环境变量控制：

`MODEL_NAME`: 默认值为 "gpt-4o-mini"。可通过环境变量 OPENAI_MODEL_NAME覆盖。

`BASE_URL`: 通过环境变量 OPENAI_BASE_URL覆盖。

`GPT_BASE_URL`: 默认值与 BASE_URL一致。可通过环境变量 GPT_BASE_URL覆盖。

2. 设置敏感的环境变量 (在运行脚本前)

在终端中执行以下命令，设置必需和可选的API密钥与端点：

```
export OPENAI_API_KEY="sk-your-actual-openai-api-key-here"

# 可选：如果您使用其他兼容 OpenAI API 的服务，可自定义模型名称和基础URL
export OPENAI_MODEL_NAME="gpt-4o"  # 覆盖默认模型
export OPENAI_BASE_URL="https://your-custom-endpoint.com/v1"  # 覆盖默认API端点
export GPT_BASE_URL="https://your-custom-endpoint.com/v1"  # 覆盖代码生成专用端点
```

重要：

将 `sk-your-actual-openai-api-key-here`替换为您真实的API密钥。

上述 export命令仅在当前终端会话中有效。若需永久设置，请将命令添加到您的 shell 配置文件（如 ~/.bashrc, ~/.zshrc）中。

3. 验证与运行

设置好环境变量后，即可运行主脚本：

Linux 下可使用：`./run.sh 0 66`
注意需要先使用 `chmod +x run.sh`
