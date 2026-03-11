"""Syntax check for LLM-generated code."""

import ast

from dc_iteration.data import load_solutions


def syntax_check(code, verbose=False):
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        if verbose:
            import traceback
            traceback.print_exc()
        return False
