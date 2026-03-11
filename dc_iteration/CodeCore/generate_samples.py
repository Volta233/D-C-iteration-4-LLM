import openai
import re
import ast
import io
import os
import json
import tokenize
from typing import Any, List, Optional, Tuple, Dict

from dc_iteration.sanitize import sanitize
from dc_iteration.codegen import my_run_codegen
from dc_iteration.CodeCore.hyperparams import *
from dc_iteration.data.utils import stream_jsonl, write_jsonl
from dc_iteration.evaluate import evaluate
from dc_iteration.CodeCore.IO_process import *


def generate_one_problem(prompt, entry_point, max_retries: int = 3):
    client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
    system_msg = "You are an expert in writing structured Python docstrings with examples."
    prompt = remove_comments_ast(prompt)
    user_prompt = f"""
        Generate a Markdown code block for the function below, ONLY including:
        1. Function header
        2. Docstring describing logic and examples.
        3. Examples with input/output pairs.

        Example Input:
        def example_func(a, b):
            if a > b:
                return a
            else:
                return b

        Example Output:
        ```markdown
        def example_func(a, b):
            \"\"\"Return the larger of two integers a and b.
            
            Examples:
            example_func(5, 3) == 5
            example_func(2, 4) == 4
            \"\"\"
        ```

        Target Function:
        {prompt}
        """
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=512,
        temperature=0
    )
    ret = completion.choices[0].message.content
    ret = post_process(ret, prompt, entry_point, max_retries)
    return ret


def post_process(text: str, prompt: str, entry_point: str, max_retries: int = 3) -> Optional[str]:
    code_pattern = r"```(?:python3?|markdown)[^\S\n]*\n(.*?)[^\S\n]*```"
    matches = re.findall(code_pattern, text, re.DOTALL)
    if matches:
        combined_code = "\n\n".join([m.strip() for m in matches])
        return combined_code
    if max_retries > 0:
        return generate_one_problem(prompt, entry_point, max_retries - 1)
    return None


def remove_comments_ast(code: str) -> str:
    class DocstringRemover(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return None
            return node

    try:
        tree = ast.parse(code)
        tree = DocstringRemover().visit(tree)
        ast.fix_missing_locations(tree)
        processed_code = ast.unparse(tree)
    except Exception:
        processed_code = code

    tokens = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(processed_code).readline):
            if tok.type != tokenize.COMMENT:
                tokens.append((tok.type, tok.string, tok.start, tok.end, tok.line))
    except tokenize.TokenError:
        pass

    reconstructed = tokenize.untokenize(tokens)
    if isinstance(reconstructed, bytes):
        return reconstructed.decode("utf-8")
    return reconstructed


def generate_and_rename_samples_single_task(iteration: int, task_id: str, problem: Dict) -> str:
    """为单个任务生成样本"""
    temp_problem_path = os.path.join(PROBLEM_PATH, f"temp_problems{iteration}.jsonl")
    write_jsonl(temp_problem_path, [problem])

    samples_path = my_run_codegen(
        model="gpt-4o-mini",
        root=os.path.join(BASE_DIR, "store_data", "result"),
        n_samples=NUM_SAMPLES_PER_TASK,
        temperature=0.6,
        greedy=False,
        dataset="humaneval",
        base_url=GPT_BASE_URL,
        backend="openai",
        HUMANEVAL_OVERRIDE_PATH=temp_problem_path,
        resume=False
    )

    task_safe_id = task_id.replace("/", "_")
    new_path = samples_path.replace(".jsonl", f"_samples{iteration}_{task_safe_id}.jsonl")
    if os.path.exists(samples_path):
        os.rename(samples_path, new_path)

    if os.path.exists(temp_problem_path):
        os.remove(temp_problem_path)

    return new_path
