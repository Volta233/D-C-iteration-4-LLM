import time
from copy import deepcopy

from dc_iteration.eval.utils import time_limit


def trusted_exec(code, inputs, entry_point, record_time=False, output_not_none=False):
    """Execute trusted code in place."""
    exec_globals = {}
    exec(code, exec_globals)
    fn = exec_globals[entry_point]

    rtime = []
    ret = []
    for inp in inputs:
        inp = deepcopy(inp)
        if record_time:
            start = time.time()
            ret.append(fn(*inp))
            rtime.append(time.time() - start)
        else:
            ret.append(fn(*inp))

    if output_not_none:
        ret = [i is not None for i in ret]

    if record_time:
        return ret, rtime
    else:
        return ret
