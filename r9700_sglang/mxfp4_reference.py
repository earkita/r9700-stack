"""Small CPU oracle for the GLM routed MXFP4 expert arithmetic.

This deliberately expands every selected weight before multiplication. It is
independent of the tiled GPU GEMV and accepts activations *after* Quark QDQ;
the caller must supply a separate QDQ oracle for end-to-end quantization tests.
It is intended for tiny fixtures, never for serving or benchmarking.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Callable, Sequence


E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def bf16(value: float) -> float:
    """Round a finite FP32 value to BF16, ties to even."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFF0000))[0]


def unpack_row(packed: Sequence[int], scales: Sequence[int]) -> list[float]:
    """Decode checkpoint-order E2M1 bytes with one E8M0 scale per 32 values."""
    if not packed or len(packed) % 16 or len(scales) != len(packed) // 16:
        raise ValueError("Expected 16 packed bytes per E8M0 scale")
    result = []
    for group, scale_byte in enumerate(scales):
        if not 0 <= scale_byte <= 254:
            raise ValueError("E8M0 code 255 is reserved")
        scale = math.ldexp(1.0, scale_byte - 127)
        for byte in packed[group * 16:(group + 1) * 16]:
            if not 0 <= byte <= 255:
                raise ValueError("Packed MXFP4 byte out of range")
            result.extend((E2M1[byte & 15] * scale,
                           E2M1[byte >> 4] * scale))
    return result


def project(x: Sequence[float], packed: Sequence[Sequence[int]],
            scales: Sequence[Sequence[int]], *,
            router_weight: float = 1.0) -> list[float]:
    """Round once, after the optional FP32 down-projection route weight."""
    if len(packed) != len(scales):
        raise ValueError("Weight rows and scale rows differ")
    result = []
    for weight, scale in zip(packed, scales):
        decoded = unpack_row(weight, scale)
        if len(x) != len(decoded):
            raise ValueError("Activation width differs from weight K")
        result.append(bf16(math.fsum(float(a) * b for a, b in zip(x, decoded))
                           * float(router_weight)))
    return result


def routed_moe(
    hidden: Sequence[Sequence[float]],
    w13: Sequence[Sequence[Sequence[int]]],
    s13: Sequence[Sequence[Sequence[int]]],
    w2: Sequence[Sequence[Sequence[int]]],
    s2: Sequence[Sequence[Sequence[int]]],
    expert_ids: Sequence[Sequence[int]],
    router_weights: Sequence[Sequence[float]],
    qdq: Callable[[Sequence[float]], Sequence[float]],
    clamp: float = 10.0,
) -> list[list[float]]:
    """Reference gate/up → clamped SiLU → down → route-weighted sum.

    The explicit ``qdq`` callback must match the checkpoint's activation
    quantizer. For fixtures whose activations are already MXFP4-QDQ values,
    use ``lambda row: row``. Gate/up outputs are rounded to BF16; down folds
    the route weight into its accumulator before the single BF16 store.
    """
    if not (len(hidden) == len(expert_ids) == len(router_weights)):
        raise ValueError("Request row counts differ")
    outputs = []
    for x, ids, route_weights in zip(hidden, expert_ids, router_weights):
        if len(ids) != len(route_weights):
            raise ValueError("Expert IDs and routing weights differ")
        combined = None
        for expert, route_weight in zip(ids, route_weights):
            if not 0 <= expert < len(w13) or expert >= len(w2):
                raise ValueError("Expert ID out of range")
            gate_up = project(qdq(x), w13[expert], s13[expert])
            if len(gate_up) % 2:
                raise ValueError("Gate/up projection width must be even")
            middle = len(gate_up) // 2
            gate, up = gate_up[:middle], gate_up[middle:]
            activated = [bf16((min(g, clamp) /
                               (1.0 + math.exp(-min(g, clamp)))) *
                              max(-clamp, min(u, clamp)))
                         for g, u in zip(gate, up)]
            weighted = project(qdq(activated), w2[expert], s2[expert],
                               router_weight=route_weight)
            if combined is None:
                combined = [0.0] * len(weighted)
            if len(combined) != len(weighted):
                raise ValueError("Experts have different output widths")
            combined = [a + b for a, b in zip(combined, weighted)]
        if combined is None:
            raise ValueError("Each row needs at least one routed expert")
        outputs.append([bf16(v) for v in combined])
    return outputs
