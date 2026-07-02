"""Verify Keccak differential trail-core compatibility."""

from __future__ import annotations

from dataclasses import dataclass

from linear_layer import apply_l_int
from sbox_linearization import DDT
from state_lift import SBOXES_1600, bit_position, sbox_index_to_yz
from trail_parser import active_sboxes


@dataclass(frozen=True)
class TransitionStats:
    compatible: bool
    active_input_sboxes: int
    active_output_sboxes: int
    weight: int
    impossible_sboxes: tuple[int, ...]


def sbox_value(state: int, sbox_index: int) -> int:
    y, z = sbox_index_to_yz(sbox_index)
    value = 0
    for x in range(5):
        value |= ((state >> bit_position(x, y, z)) & 1) << x
    return value


def sbox_transition_values(input_difference: int, output_difference: int) -> list[tuple[int, int, int]]:
    return [
        (
            sbox_index,
            sbox_value(input_difference, sbox_index),
            sbox_value(output_difference, sbox_index),
        )
        for sbox_index in range(SBOXES_1600)
    ]


def transition_stats(input_difference: int, output_difference: int) -> TransitionStats:
    impossible: list[int] = []
    weight = 0
    for sbox_index, delta_in, delta_out in sbox_transition_values(
        input_difference,
        output_difference,
    ):
        entry = DDT[delta_in][delta_out]
        if entry == 0:
            impossible.append(sbox_index)
        elif delta_in or delta_out:
            # Differential probability for this S-box is entry/32.
            weight += 5 - (entry.bit_length() - 1)

    return TransitionStats(
        compatible=not impossible,
        active_input_sboxes=active_sboxes(input_difference),
        active_output_sboxes=active_sboxes(output_difference),
        weight=weight,
        impossible_sboxes=tuple(impossible),
    )


def verify_l_relation(alpha: int, beta: int) -> bool:
    return apply_l_int(alpha) == beta


def demo() -> None:
    # Trivial all-zero transition: useful as a smoke test for the DDT plumbing.
    stats = transition_stats(0, 0)
    assert stats.compatible
    assert stats.weight == 0
    assert verify_l_relation(0, 0)
    print("trail verification checks passed")


if __name__ == "__main__":
    demo()
