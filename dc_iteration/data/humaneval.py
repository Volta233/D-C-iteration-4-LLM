import hashlib
import json
import os
from typing import Dict

from dc_iteration.data.utils import (
    CACHE_DIR,
    completeness_check,
    get_dataset_metadata,
    make_cache,
    stream_jsonl,
)

HUMANEVAL_PLUS_VERSION = "v0.1.10"


def _ready_human_eval_plus_path(mini=False, noextreme=False, version="default", HUMANEVAL_OVERRIDE_PATH=None) -> str:
    if HUMANEVAL_OVERRIDE_PATH:
        return HUMANEVAL_OVERRIDE_PATH

    version = HUMANEVAL_PLUS_VERSION if version == "default" else version
    url, plus_path = get_dataset_metadata(
        "HumanEvalPlus", HUMANEVAL_PLUS_VERSION, mini, noextreme
    )
    make_cache(url, plus_path)

    return plus_path


def get_human_eval_plus_hash(mini=False, noextreme=False, version="default", HUMANEVAL_OVERRIDE_PATH=None) -> str:
    plus_path = _ready_human_eval_plus_path(mini, noextreme, version="default", HUMANEVAL_OVERRIDE_PATH=HUMANEVAL_OVERRIDE_PATH)
    with open(plus_path, "rb") as f:
        plus = f.read()
    return hashlib.md5(plus).hexdigest()


def get_human_eval_plus(
    err_incomplete=True, mini=False, noextreme=False, version="default", HUMANEVAL_OVERRIDE_PATH=None
) -> Dict[str, Dict]:
    plus_path = _ready_human_eval_plus_path(
        mini=mini, noextreme=noextreme, version=version, HUMANEVAL_OVERRIDE_PATH=HUMANEVAL_OVERRIDE_PATH
    )
    plus = {task["task_id"]: task for task in stream_jsonl(plus_path)}
    if err_incomplete:
        completeness_check("HumanEval+", plus)
    return plus


def get_original_human_eval_plus(
    err_incomplete=True, mini=False, noextreme=False, version="default",
    HUMANEVAL_OVERRIDE_PATH=None, HUMANEVAL_PROBLEM_DIR=None
) -> Dict[str, Dict]:
    """从 problems0.jsonl 加载原始问题（用于 ground truth）。
    HUMANEVAL_PROBLEM_DIR 或 dirname(HUMANEVAL_OVERRIDE_PATH) 指定问题目录。
    """
    if HUMANEVAL_PROBLEM_DIR:
        plus_path = os.path.join(HUMANEVAL_PROBLEM_DIR, "problems0.jsonl")
    elif HUMANEVAL_OVERRIDE_PATH:
        plus_path = os.path.join(os.path.dirname(HUMANEVAL_OVERRIDE_PATH), "problems0.jsonl")
    else:
        raise ValueError("Need HUMANEVAL_PROBLEM_DIR or HUMANEVAL_OVERRIDE_PATH to locate problems0.jsonl")
    if not os.path.exists(plus_path):
        raise FileNotFoundError(f"Original problems file not found: {plus_path}")
    plus = {task["task_id"]: task for task in stream_jsonl(plus_path)}
    if err_incomplete:
        completeness_check("HumanEval+", plus)
    return plus


def get_original_human_eval_plus_hash(mini=False, noextreme=False, version="default",
                                     HUMANEVAL_OVERRIDE_PATH=None, HUMANEVAL_PROBLEM_DIR=None) -> str:
    if HUMANEVAL_PROBLEM_DIR:
        plus_path = os.path.join(HUMANEVAL_PROBLEM_DIR, "problems0.jsonl")
    elif HUMANEVAL_OVERRIDE_PATH:
        plus_path = os.path.join(os.path.dirname(HUMANEVAL_OVERRIDE_PATH), "problems0.jsonl")
    else:
        raise ValueError("Need HUMANEVAL_PROBLEM_DIR or HUMANEVAL_OVERRIDE_PATH")
    with open(plus_path, "rb") as f:
        plus = f.read()
    return hashlib.md5(plus).hexdigest()


def get_human_eval() -> Dict[str, Dict]:
    human_eval_path = os.path.join(CACHE_DIR, "HumanEval.jsonl")
    make_cache(
        "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz",
        human_eval_path,
    )

    human_eval = open(human_eval_path, "r").read().split("\n")
    human_eval = [json.loads(line) for line in human_eval if line]

    human_eval[115]["prompt"] = "import math\n" + human_eval[115]["prompt"].replace(
        "import math\n", ""
    )

    return {task["task_id"]: task for task in human_eval}
