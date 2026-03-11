# The MIT License - Copyright (c) OpenAI

import itertools
import multiprocessing
import os
import time
from multiprocessing import Array, Value
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil

from dc_iteration.config import *
from dc_iteration.eval._special_oracle import _poly
from dc_iteration.eval.utils import (
    create_tempdir,
    reliability_guard,
    swallow_io,
    time_limit,
)


def compatible_eval_result(results: Dict) -> Dict:
    for task_results in results["eval"].values():
        if "files" in task_results and "nfiles" not in task_results:
            task_results["nfiles"] = len(task_results.pop("files"))
    return results


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    def estimator(n: int, c: int, k: int) -> float:
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"

_SUCCESS = 0
_FAILED = 1
_TIMEOUT = 2
_UNKNOWN = 3

_mapping = {_SUCCESS: PASS, _FAILED: FAIL, _TIMEOUT: TIMEOUT, _UNKNOWN: None}


def query_maximum_memory_bytes() -> Optional[int]:
    maximum_memory_bytes = os.getenv(
        "EVALPLUS_MAX_MEMORY_BYTES", 4 * 1024 * 1024 * 1024
    )
    maximum_memory_bytes = min(int(maximum_memory_bytes), psutil.virtual_memory().total)
    if maximum_memory_bytes == -1:
        return None
    return maximum_memory_bytes


def is_floats(x) -> bool:
    if isinstance(x, float):
        return True
    if isinstance(x, (list, tuple)) and x:
        return all(isinstance(i, float) for i in x)
    if isinstance(x, np.ndarray):
        return x.dtype == np.float64 or x.dtype == np.float32
    return False


def unsafe_execute(
    dataset: str,
    entry_point: str,
    code: str,
    inputs,
    expected: List,
    time_limits,
    atol,
    fast_check,
    stat,
    details,
    progress,
):
    with create_tempdir():
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        reliability_guard(maximum_memory_bytes=query_maximum_memory_bytes())
        exec_globals = {}
        try:
            with swallow_io():
                exec(code, exec_globals)
                fn = exec_globals[entry_point]

            for i, inp in enumerate(inputs):
                try:
                    with time_limit(time_limits[i]):
                        with swallow_io():
                            out = fn(*inp)

                    exp = expected[i]
                    exact_match = out == exp

                    if dataset == "humaneval":
                        if "find_zero" == entry_point:
                            assert abs(_poly(*inp, out)) <= atol
                            details[i] = True
                            progress.value += 1
                            continue

                    if atol == 0 and is_floats(exp):
                        atol = 1e-6
                    if not exact_match and atol != 0:
                        assert type(out) == type(exp)
                        if isinstance(exp, (list, tuple)):
                            assert len(out) == len(exp)
                        assert np.allclose(out, exp, rtol=1e-07, atol=atol)
                    else:
                        assert exact_match
                except BaseException:
                    details[i] = False
                    progress.value += 1
                    if fast_check:
                        raise
                    continue

                details[i] = True
                progress.value += 1

            stat.value = _SUCCESS
        except BaseException:
            stat.value = _FAILED
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


def untrusted_check(
    dataset: str,
    code: str,
    inputs: List[Any],
    entry_point: str,
    expected,
    atol,
    ref_time: List[float],
    fast_check: bool = False,
    min_time_limit: float = DEFAULT_MIN_TIME_LIMIT,
    gt_time_limit_factor: float = DEFAULT_GT_TIME_LIMIT_FACTOR,
) -> Tuple[str, np.ndarray]:
    time_limits = [max(min_time_limit, gt_time_limit_factor * t) for t in ref_time]
    timeout = min(int(os.getenv("EVALPLUS_TIMEOUT_PER_TASK", 60)), sum(time_limits)) + 1
    if not fast_check:
        timeout += 1

    progress = Value("i", 0)
    stat = Value("i", _UNKNOWN)
    details = Array("b", [False for _ in range(len(inputs))])

    p = multiprocessing.Process(
        target=unsafe_execute,
        args=(
            dataset,
            entry_point,
            code,
            inputs,
            expected,
            time_limits,
            atol,
            fast_check,
            stat,
            details,
            progress,
        ),
    )
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
    if p.is_alive():
        p.kill()
        time.sleep(0.1)

    stat = _mapping[stat.value]
    details = details[: progress.value]

    if not stat:
        stat = TIMEOUT

    if stat == PASS:
        if len(details) != len(inputs) or not all(details):
            stat = FAIL

    return stat, details
