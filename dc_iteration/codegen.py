import json
import os
from typing import Dict, List, Optional

from dc_iteration.data import get_human_eval_plus
from dc_iteration.provider import DecoderBase, make_model
from dc_iteration.sanitize import sanitize
from dc_iteration.utils import progress


def codegen(
    target_path: str,
    model: DecoderBase,
    dataset: Dict,
    greedy=False,
    n_samples=1,
    id_range=None,
    resume=True,
):
    task2nexist = {}
    if resume and target_path.endswith(".jsonl") and os.path.isfile(target_path):
        with open(target_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                task_id = json.loads(line)["task_id"]
                task2nexist[task_id] = task2nexist.get(task_id, 0) + 1

    if target_path.endswith(".jsonl"):
        raw_target_path = target_path.replace(".jsonl", ".raw.jsonl")
    else:
        raw_target_path = target_path + ".raw"
        os.makedirs(target_path, exist_ok=True)

    backend_type: str = type(model).__name__
    with progress(backend_type) as p:
        for task_id, task in p.track(dataset.items()):
            if id_range is not None:
                id_num = int(task_id.split("/")[1])
                low, high = id_range
                if id_num < low or id_num >= high:
                    p.console.print(f"Skipping {task_id} as it is not in {id_range}")
                    continue

            if not target_path.endswith(".jsonl"):
                p_name = task_id.replace("/", "_")
                os.makedirs(os.path.join(target_path, p_name), exist_ok=True)
                task2nexist[task_id] = len(
                    [f for f in os.listdir(os.path.join(target_path, p_name)) if f.endswith(".py")]
                )

            n_more_samples = n_samples
            log = f"Codegen: {task_id} @ {model}"
            if resume and task2nexist.get(task_id, 0) > 0:
                log += f" (resuming from {task2nexist[task_id]})"
                n_more_samples -= task2nexist[task_id]

            p.console.print(log)

            sidx = n_samples - n_more_samples
            while sidx < n_samples:
                prompt = task["prompt"].strip() + "\n"
                outputs = model.codegen(
                    prompt,
                    do_sample=not greedy,
                    num_samples=n_samples - sidx,
                )
                assert outputs, "No outputs from model!"
                for impl in outputs:
                    solution = prompt + impl if model.is_direct_completion() else impl
                    sanitized_solution = sanitize(
                        solution, entrypoint=task["entry_point"]
                    )
                    if target_path.endswith(".jsonl"):
                        with open(target_path, "a") as f:
                            f.write(
                                json.dumps(
                                    {"task_id": task_id, "solution": sanitized_solution}
                                )
                                + "\n"
                            )
                        with open(raw_target_path, "a") as f:
                            f.write(
                                json.dumps({"task_id": task_id, "solution": solution})
                                + "\n"
                            )
                    else:
                        with open(
                            os.path.join(target_path, p_name, f"{sidx}.py"),
                            "w",
                            encoding="utf-8",
                        ) as f:
                            f.write(sanitized_solution)
                        with open(
                            os.path.join(raw_target_path, p_name, f"{sidx}.py"),
                            "w",
                            encoding="utf-8",
                        ) as f:
                            f.write(solution)
                    sidx += 1

    return target_path


def my_run_codegen(
    model: str,
    dataset: str,
    root: str = "evalplus_results",
    bs: Optional[int] = None,
    n_samples: int = 1,
    temperature: float = 0.2,
    resume: bool = False,
    greedy: bool = False,
    id_range: List = None,
    version: str = "default",
    backend: str = "vllm",
    force_base_prompt: bool = False,
    base_url: str = None,
    tp: int = 1,
    jsonl_fmt: bool = True,
    HUMANEVAL_OVERRIDE_PATH=None
):
    assert dataset == "humaneval", f"Invalid dataset {dataset}. Only humaneval is supported."

    identifier = model.strip("./").replace("/", "--") + f"_{backend}_temp_{temperature}"

    target_path = os.path.join(root, dataset, identifier)

    if jsonl_fmt:
        target_path += ".jsonl"
    else:
        os.makedirs(target_path, exist_ok=True)

    if not resume and os.path.exists(target_path):
        os.remove(target_path)
        raw_target_path = target_path.replace(".jsonl", ".raw.jsonl")
        if os.path.exists(raw_target_path):
            os.remove(raw_target_path)

    dataset_dict = get_human_eval_plus(version=version, HUMANEVAL_OVERRIDE_PATH=HUMANEVAL_OVERRIDE_PATH)

    if greedy and (temperature != 0 or bs != 1 or n_samples != 1):
        temperature = 0.0
        bs = 1
        n_samples = 1

    if id_range is not None:
        assert len(id_range) == 2, "id_range must be a list of length 2"
        assert id_range[0] < id_range[1], "id_range must be increasing"
        id_range = tuple(id_range)

    if bs is None:
        bs = min(n_samples, 32)
        print(f"Setting batch size to {bs}")

    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(root, dataset), exist_ok=True)

    instruction_prefix = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
    response_prefix = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"

    model_runner = make_model(
        model=model,
        backend=backend,
        batch_size=bs,
        temperature=temperature,
        force_base_prompt=force_base_prompt,
        dataset=dataset,
        base_url=base_url,
        tp=tp,
        instruction_prefix=instruction_prefix,
        response_prefix=response_prefix,
    )

    codegen(
        target_path=target_path,
        dataset=dataset_dict,
        greedy=greedy,
        model=model_runner,
        n_samples=n_samples,
        resume=resume,
        id_range=id_range,
    )

    del model_runner
    import gc
    gc.collect()

    return target_path
