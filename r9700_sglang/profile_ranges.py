# SPDX-License-Identifier: Apache-2.0
"""Opt-in host ranges for a separate GLM prefill diagnostic image.

Disabled decorators return the original callable, without a wrapper on the
serving path. Enabled ranges annotate CPU execution and GPU launches for
Kineto correlation; their durations are not GPU execution times. No device
synchronization, tensor inspection, arithmetic or cache policy changes occur.
Do not enable these annotations in the unprofiled performance comparison.
"""

from functools import wraps
import os


def profile_range(name, *, attributes=()):
    enabled = os.environ.get("R9K_SGLANG_PROFILE_RANGES", "0")
    if enabled not in ("0", "1"):
        raise ValueError("R9K_SGLANG_PROFILE_RANGES must be 0 or 1")

    def decorate(function):
        if enabled == "0":
            return function
        from torch.profiler import record_function

        @wraps(function)
        def annotated(*args, **kwargs):
            label = "r9700::" + name
            if attributes:
                # Only known Python scalar identifiers: formatting a tensor
                # could synchronize a device and invalidate the diagnostic.
                values = [getattr(args[0], key) for key in attributes]
                if not all(isinstance(value, (int, bool)) for value in values):
                    raise ValueError("Profile identifiers must be Python int/bool values")
                label += "[" + ",".join(f"{key}={value}" for key,value in zip(attributes,values)) + "]"
            with record_function(label):
                return function(*args, **kwargs)

        return annotated

    return decorate
