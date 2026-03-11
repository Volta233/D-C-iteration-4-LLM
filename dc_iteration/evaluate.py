import json
import multiprocessing
import os
import pickle
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from warnings import warn

import numpy as np
from termcolor import cprint
from tqdm import tqdm

from dc_iteration.config import *
from dc_iteration.data import (
    get_human_eval_plus,
    get_human_eval_plus_hash,
    load_solutions,
)
from dc_iteration.data.utils import CACHE_DIR
from dc_iteration.data.humaneval import get_original_human_eval_plus, get_original_human_eval_plus_hash
from dc_iteration.eval import (
    PASS,
    compatible_eval_result,
    estimate_pass_at_k,
    untrusted_check,
)
from dc_iteration.gen.util import trusted_exec

Result = Tuple[str, List[bool]]


def get_groundtruth(problems, hashcode, tasks_only_output_not_none):
    cache_file = os.path.join(CACHE_DIR, f"{hashcode}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    os.makedirs(CACHE_DIR, exist_ok=True)
    tbegin = time.time()
    expected_output = {}
    for task_id, problem in problems.items():
        oracle = {}
        oracle["base"], oracle["base_time"] = trusted_exec(
            problem["prompt"] + problem["canonical_solution"],
            problem["base_input"],
            problem["entry_point"],
            record_time=True,
            output_not_none=problem["entry_point"] in tasks_only_output_not_none,
        )
        oracle["plus"], oracle["plus_time"] = trusted_exec(
            problem["prompt"] + problem["canonical_solution"],
            problem["plus_input"],
            problem["entry_point"],
            record_time=True,
            output_not_none=problem["entry_point"] in tasks_only_output_not_none,
        )
        expected_output[task_id] = oracle
    print(f"Expected outputs computed in {time.time() - tbegin:.2f}s")

    with open(cache_file, "wb") as f:
        pickle.dump(expected_output, f)

    return expected_output


def check_correctness(
    dataset: str,
    completion_id: int,
    problem: Dict[str, Any],
    solution: str,
    expected_output: Dict[str, List],
    base_only=False,
    fast_check=False,
    identifier=None,
    min_time_limit: float = DEFAULT_MIN_TIME_LIMIT,
    gt_time_limit_factor: float = DEFAULT_GT_TIME_LIMIT_FACTOR,
) -> Dict[str, Result]:
    ret = {
        "completion_id": completion_id,
        "task_id": problem["task_id"],
        "_identifier": identifier,
        "solution": solution,
    }
    ret["base"] = untrusted_check(
        dataset,
        solution,
        problem["base_input"],
        problem["entry_point"],
        expected=expected_output["base"],
        atol=problem["atol"],
        ref_time=expected_output["base_time"],
        fast_check=fast_check,
        min_time_limit=min_time_limit,
        gt_time_limit_factor=gt_time_limit_factor,
    )

    if not base_only:
        ret["plus"] = untrusted_check(
            dataset,
            solution,
            problem["plus_input"],
            problem["entry_point"],
            expected=expected_output["plus"],
            atol=problem["atol"],
            ref_time=expected_output["plus_time"],
            fast_check=fast_check,
            min_time_limit=min_time_limit,
            gt_time_limit_factor=gt_time_limit_factor,
        )

    return ret


def evaluate(
    dataset: str,
    samples: Optional[str] = None,
    base_only: bool = False,
    parallel: Optional[int] = None,
    i_just_wanna_run: bool = False,
    test_details: bool = True,
    min_time_limit: float = DEFAULT_MIN_TIME_LIMIT,
    gt_time_limit_factor: float = DEFAULT_GT_TIME_LIMIT_FACTOR,
    mini: bool = False,
    noextreme: bool = False,
    version: str = "default",
    HUMANEVAL_OVERRIDE_PATH: str = None,
    **model_kwargs,
):
    if model_kwargs:
        raise ValueError("dc_iteration.evaluate does not support run_codegen. Provide samples path directly.")

    assert samples is not None, "No samples provided"

    n_workers = parallel or max(1, multiprocessing.cpu_count() // 2)

    if os.path.isdir(samples):
        result_path = os.path.join(samples, "eval_results.json")
    else:
        assert samples.endswith(".jsonl")
        result_path = samples.replace(".jsonl", "_eval_results.json")

    if os.path.isfile(result_path) and not i_just_wanna_run:
        with open(result_path, "r") as f:
            results = json.load(f)
        results = compatible_eval_result(results)
    else:
        if dataset == "humaneval":
            problems = get_human_eval_plus(
                mini=mini, noextreme=noextreme, version=version, HUMANEVAL_OVERRIDE_PATH=HUMANEVAL_OVERRIDE_PATH
            )
            dataset_hash = get_human_eval_plus_hash(
                mini=mini, noextreme=noextreme, version=version, HUMANEVAL_OVERRIDE_PATH=HUMANEVAL_OVERRIDE_PATH
            )
            original_problems = get_original_human_eval_plus(
                HUMANEVAL_OVERRIDE_PATH=HUMANEVAL_OVERRIDE_PATH
            )
            original_hashcode = get_original_human_eval_plus_hash(
                HUMANEVAL_OVERRIDE_PATH=HUMANEVAL_OVERRIDE_PATH
            )
            expected_output = get_groundtruth(original_problems, original_hashcode, [])
        else:
            raise ValueError(f"Unsupported dataset: {dataset}. Only 'humaneval' is supported.")

        results = {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "hash": dataset_hash,
            "eval": {},
        }

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            completion_id = Counter()
            n_samples = 0
            eval_results = defaultdict(list)
            remainings = set()

            for sample in tqdm(load_solutions(samples)):
                task_id = sample["task_id"]
                if task_id not in problems:
                    warn(
                        f"Task {task_id} is found in the samples but not found in the dataset"
                    )
                    continue
                solution = (
                    sample["solution"]
                    if "solution" in sample
                    else problems[task_id]["prompt"] + sample["completion"]
                )
                remainings.add(sample["_identifier"])
                args = (
                    dataset,
                    completion_id[task_id],
                    problems[task_id],
                    solution,
                    expected_output[task_id],
                    base_only,
                    not test_details,
                    sample["_identifier"],
                    min_time_limit,
                    gt_time_limit_factor,
                )
                futures.append(executor.submit(check_correctness, *args))
                completion_id[task_id] += 1
                n_samples += 1

            assert n_samples == len(remainings), "Missing problems in unfinished"
            assert len(completion_id) == len(problems), "Missing problems in samples"

            def stucking_checker():
                while remainings:
                    last_size = len(remainings)
                    time.sleep(20)
                    if last_size != len(remainings) or len(remainings) == 0:
                        continue
                    warn("No samples had finished testing in the last 20s")
                    warn(f"{len(remainings)} samples to be tested: {remainings}")

            threading.Thread(target=stucking_checker).start()

            for future in tqdm(as_completed(futures), total=n_samples):
                result = future.result()
                remainings.remove(result["_identifier"])
                eval_results[result["task_id"]].append(result)

        for task_id, task_results in eval_results.items():
            task_results.sort(key=lambda x: x["completion_id"])
            results["eval"][task_id] = []
            for res in task_results:

                def get_failed_tests(stat, details, inputs) -> List[Any]:
                    if stat == PASS or not details:
                        if stat != PASS and not details:
                            return inputs[:]
                        return []
                    if test_details:
                        return [
                            inputs[i] for i in range(len(details)) if not details[i]
                        ]
                    return [inputs[len(details) - 1]]

                base_stat, base_details = res["base"]
                if len(base_details) != len(problems[task_id]["base_input"]):
                    print(f"WARNING: Details length mismatch for {task_id}")
                    base_details = [False] * len(problems[task_id]["base_input"])

                base_fail_tests = get_failed_tests(
                    base_stat, base_details, problems[task_id]["base_input"]
                )
                plus_stat = None
                plus_fail_tests = []

                if not base_only:
                    plus_stat, plus_details = res["plus"]
                    plus_fail_tests = get_failed_tests(
                        plus_stat, plus_details, problems[task_id]["plus_input"]
                    )

                results["eval"][task_id].append(
                    {
                        "task_id": task_id,
                        "solution": res["solution"],
                        "base_status": base_stat,
                        "plus_status": plus_stat,
                        "base_fail_tests": base_fail_tests,
                        "plus_fail_tests": plus_fail_tests,
                    }
                )

    total = np.array([len(r) for r in results["eval"].values()])
    base_correct = []
    new_correct = []

    for res in results["eval"].values():
        bc = sum([r["base_status"] == PASS for r in res])
        base_correct.append(bc)
        if not base_only:
            new_correct.append(
                sum(
                    [
                        res[i]["base_status"] == res[i]["plus_status"] == PASS
                        for i in range(len(res))
                    ]
                )
            )

    base_correct = np.array(base_correct)

    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, base_correct, k).mean()
        for k in [1, 10, 100]
        if total.min() >= k
    }
    cprint(f"{dataset} (base tests)", "red")
    for k, v in pass_at_k.items():
        cprint(f"{k}:\t{v:.3f}", "red")
    results["pass_at_k"] = {"base": pass_at_k}

    if new_correct:
        cprint(f"{dataset}+ (base + extra tests)", "green")
        pass_at_k = {
            f"pass@{k}": estimate_pass_at_k(total, np.array(new_correct), k).mean()
            for k in [1, 10, 100]
            if (total >= k).all()
        }
        for k, v in pass_at_k.items():
            cprint(f"{k}:\t{v:.3f}", "green")
        results["pass_at_k"]["plus"] = pass_at_k

    if os.path.isfile(result_path) and i_just_wanna_run:
        new_path = result_path + ".bak"
        while os.path.isfile(new_path):
            new_path += ".bak"
        os.rename(result_path, new_path)

    if not os.path.isfile(result_path):
        with open(result_path, "w") as f:
            json.dump(results, f)
