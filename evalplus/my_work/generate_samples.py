import openai
import re
import ast
import io
import tokenize
from typing import Any, List, Optional, Tuple
from evalplus.sanitize import sanitize
from evalplus.my_work.hyperparams import *

def generate_one_problem(prompt,entry_point,max_retries: int = 3):
    client = openai.OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL
        )
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
    ret = post_process(ret, prompt, entry_point,max_retries)
    # print(ret)
    return ret

def post_process(text: str, prompt: str, entry_point: str, max_retries: int = 3) -> Optional[str]:
    """增强版代码块提取函数"""
    # 优化后的正则表达式（支持宽松的代码块边界）
    code_pattern = r"```(?:python3?|markdown)[^\S\n]*\n(.*?)[^\S\n]*```"
    matches = re.findall(code_pattern, text, re.DOTALL)
    
    if matches:
        # 清理每段代码的首尾空白并合并
        combined_code = "\n\n".join([m.strip() for m in matches])
        return combined_code
    
    # 无代码块时的递归生成控制
    if max_retries > 0:
        return generate_one_problem(prompt, entry_point, max_retries-1)
    
    return None  # 达到最大重试次数仍未生成有效代码

def remove_comments_ast(code: str) -> str:
    """使用 AST + Tokenize 移除所有注释（含三引号文档字符串）"""
    class DocstringRemover(ast.NodeTransformer):
        def visit_Expr(self, node):
            # 过滤文档字符串（位于模块/函数/类的第一个表达式）
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return None
            return node

    try:
        # Phase 1: 使用AST移除文档字符串
        tree = ast.parse(code)
        tree = DocstringRemover().visit(tree)
        ast.fix_missing_locations(tree)  # 修复节点位置
        processed_code = ast.unparse(tree)
    except Exception as e:
        processed_code = code  # 解析失败时保留原始代码

    # Phase 2: 使用Tokenize移除其他注释
    tokens = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(processed_code).readline):
            # 保留非注释token，并携带完整元数据
            if tok.type != tokenize.COMMENT:
                tokens.append((
                    tok.type,
                    tok.string,
                    tok.start,
                    tok.end,
                    tok.line
                ))
    except tokenize.TokenError:
        pass

    # 重构代码并统一编码
    reconstructed = tokenize.untokenize(tokens)
    if isinstance(reconstructed, bytes):
        return reconstructed.decode("utf-8")
    return reconstructed