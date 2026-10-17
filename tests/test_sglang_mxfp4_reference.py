"""CPU oracle tests: no GPU or running model required."""

import math

import pytest

from r9700_sglang.mxfp4_reference import bf16, project, routed_moe, unpack_row


def test_unpack_checkpoint_order_and_distinct_group_scales():
    packed = [0x21] * 16 + [0xA4] * 16
    values = unpack_row(packed, [127, 128])
    assert values[:4] == [0.5, 1.0, 0.5, 1.0]
    assert values[32:36] == [4.0, -2.0, 4.0, -2.0]
    assert len(values) == 64


def test_weight_decode_and_projection_include_partial_nonzero_column():
    # Two groups with different scales; both contribute to the projection.
    x = [1.0] * 64
    row = [0x11] * 16 + [0x11] * 16
    assert project(x, [row], [[127, 128]]) == [48.0]
    assert project(x, [row], [[128, 127]]) == [48.0]
    assert project(x, [row], [[127, 127]]) == [32.0]


def test_routed_moe_keeps_weights_after_activation_and_rows_independent():
    # Every expert has a distinct gate, up and down. Repeated expert IDs in
    # separate rows exercise per-row activation and routing weights.
    gate_up = [
        [[0x22] * 16 for _ in range(32)] + [[0x22] * 16 for _ in range(32)],
        [[0x44] * 16 for _ in range(32)] + [[0x22] * 16 for _ in range(32)],
    ]
    down = [[[0x11] * 16 for _ in range(32)] for _ in range(2)]
    s13 = [[[122] for _ in range(64)] for _ in range(2)]
    s2 = [[[127] for _ in range(32)] for _ in range(2)]
    actual = routed_moe(
        hidden=[[0.5] * 32, [1.0] * 32], w13=gate_up, s13=s13,
        w2=down, s2=s2, expert_ids=[[0, 1], [1, 0]],
        router_weights=[[0.25, 0.75], [0.9, 0.1]],
        qdq=lambda x: x,
    )
    assert len(actual) == 2 and all(len(row) == 32 for row in actual)
    assert all(math.isfinite(v) for row in actual for v in row)
    assert actual[0] != actual[1]
    # With identical experts and equal weights, routing sum must equal one
    # expert's output (up to BF16 rounding), rather than weight the gate.
    equal = routed_moe(
        hidden=[[0.5] * 32], w13=gate_up, s13=s13, w2=down, s2=s2,
        expert_ids=[[0, 0]], router_weights=[[0.5, 0.5]],
        qdq=lambda x: x,
    )
    single = routed_moe(
        hidden=[[0.5] * 32], w13=gate_up, s13=s13, w2=down, s2=s2,
        expert_ids=[[0]], router_weights=[[1.0]], qdq=lambda x: x,
    )
    assert equal == single


@pytest.mark.parametrize("packed,scales", [([0] * 15, [127]),
                                            ([0] * 16, [127, 127]),
                                            ([0] * 16, [255])])
def test_invalid_scale_geometry_or_code_is_rejected(packed, scales):
    with pytest.raises(ValueError):
        unpack_row(packed, scales)


def test_bf16_rounds_ties_to_even():
    assert bf16(1.0 + 2 ** -8) == 1.0
    assert bf16(1.0 + 3 * 2 ** -8) == 1.0 + 2 ** -6


def test_down_weights_accumulator_before_single_bf16_store():
    # Exactly representable BF16 inputs and MXFP4 weights. The unweighted
    # dot lands at a BF16 tie; rounding it before routing changes the answer.
    x = [2.0, 2 ** -7] + [0.0] * 30
    weights, scales = [[0x11] * 16], [[127]]
    unweighted = project(x, weights, scales)[0]
    actual = project(x, weights, scales, router_weight=0.75)[0]
    assert unweighted == 1.0
    assert actual == 0.75390625
    assert actual != bf16(unweighted * 0.75)
