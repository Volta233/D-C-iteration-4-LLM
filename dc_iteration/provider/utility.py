from concurrent.futures import ThreadPoolExecutor
from typing import List

EOS = [
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
]


def concurrent_call(n, callback, /, *args, **kwargs):
    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = [executor.submit(callback, *args, **kwargs) for _ in range(n)]
        return [future.result() for future in futures]
