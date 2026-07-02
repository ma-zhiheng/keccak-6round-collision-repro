"""Lift 5-bit S-box equations to Keccak state-bit equations."""

from __future__ import annotations

from dataclasses import dataclass

from sbox_constraints import AffineSBoxModel, linearizable_models_for_transition


LANE_SIZE = 64
STATE_BITS_1600 = 1600
SBOXES_1600 = 320


def lane_index(x: int, y: int) -> int:
    return x + 5 * y


def bit_position(x: int, y: int, z: int) -> int:
    return LANE_SIZE * lane_index(x, y) + z


def sbox_index_to_yz(sbox_index: int) -> tuple[int, int]:
    if not 0 <= sbox_index < SBOXES_1600:
        raise ValueError("sbox_index must be in 0..319 for Keccak-f[1600]")
    return sbox_index // LANE_SIZE, sbox_index % LANE_SIZE


def state_positions_for_sbox(sbox_index: int) -> list[int]:
    y, z = sbox_index_to_yz(sbox_index)
    return [bit_position(x, y, z) for x in range(5)]


def lift_local_mask(sbox_index: int, local_mask: int) -> int:
    """Turn a 5-bit local equation mask into a 1600-bit state mask."""
    coeff = 0
    positions = state_positions_for_sbox(sbox_index)
    for local_bit, state_bit in enumerate(positions):
        if (local_mask >> local_bit) & 1:
            coeff |= 1 << state_bit
    return coeff


@dataclass(frozen=True)
class LiftedSBoxModel:
    sbox_index: int
    input_equations: tuple[tuple[int, int], ...]
    output_masks: tuple[int, ...]
    output_constants: tuple[int, ...]


def lift_model(sbox_index: int, model: AffineSBoxModel) -> LiftedSBoxModel:
    """Lift local first-round linearization data to state coordinates."""
    return LiftedSBoxModel(
        sbox_index=sbox_index,
        input_equations=tuple(
            (lift_local_mask(sbox_index, mask), rhs)
            for mask, rhs in model.input_equations
        ),
        output_masks=tuple(
            lift_local_mask(sbox_index, mask)
            for mask in model.output_masks
        ),
        output_constants=model.output_constants,
    )


def self_test() -> None:
    model = linearizable_models_for_transition(0x03, 0x02)[0]
    lifted = lift_model(0, model)
    positions = state_positions_for_sbox(0)

    assert positions == [0, 64, 128, 192, 256]
    assert len(lifted.input_equations) == 3
    assert lifted.input_equations[0][0] == ((1 << positions[0]) | (1 << positions[1]))


if __name__ == "__main__":
    self_test()
    print("state lifting checks passed")
