"""Special oracle for HumanEval/032 find_zero."""

import math


def _poly(xs: list, x: float):
    return sum([coeff * math.pow(x, i) for i, coeff in enumerate(xs)])
