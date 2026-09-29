"""Fit WARP's native context with the installed runtime's own memory planner.

Executed in an isolated process: importing an upstream ``serve`` package or
loading its native library must not pollute the gateway's interpreter.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable


def fit_native_context(native: int, budget: int, required: Callable[[int], int]) -> int:
    """Largest context within the budget, rounded down to 4096-token blocks."""
    if not 0 < native <= 2**31 - 1 or budget <= 0:
        raise ValueError("native context and memory budget must be positive")
    if required(native) <= budget:
        return native
    minimum = min(native, 4096)
    if required(minimum) > budget:
        raise ValueError("WARP's minimum context does not fit the available memory budget")
    low, high = 1, native // 4096
    while low < high:
        middle = (low + high + 1) // 2
        if required(middle * 4096) <= budget:
            low = middle
        else:
            high = middle - 1
    return low * 4096


def main() -> None:
    root, container, native_text, extra_json = sys.argv[1:]
    sys.path.insert(0, root)
    from serve.engine import plan_memory, usable_ram
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--budget")
    parser.add_argument("--vision", action="store_true")
    options, _ = parser.parse_known_args(json.loads(extra_json))
    ram = usable_ram()
    # Match WARP's automatic 75% usable-RAM ceiling, including cgroup limits.
    budget = ram * 3 // 4
    if options.budget is not None:
        from serve.__main__ import parse_size
        explicit_budget = parse_size(options.budget)
        if explicit_budget:
            budget = min(budget, explicit_budget) if budget else explicit_budget
    if budget <= 0:
        raise ValueError("cannot determine WARP memory budget; set --budget or a fixed n_ctx")

    def required(ctx: int) -> int:
        plan = plan_memory(container, ctx)
        return max(plan.floor_bytes, plan.recommended_bytes) + (
            plan.vision_bytes if options.vision else 0
        )

    native = int(native_text)
    context = fit_native_context(native, budget, required)
    print(json.dumps({"n_ctx": context, "budget_bytes": budget,
                      "required_bytes": required(context)}))


if __name__ == "__main__":
    main()
