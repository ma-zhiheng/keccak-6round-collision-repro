"""Keccak-f[1600] state execution helpers."""

from __future__ import annotations

from linear_layer import (
    MASK64,
    ROUND_CONSTANTS,
    apply_l_lanes,
    int_to_lanes,
    lanes_to_int,
)


def chi_lanes(state: list[int]) -> None:
    for y in range(5):
        row = [state[x + 5 * y] for x in range(5)]
        for x in range(5):
            state[x + 5 * y] = (
                row[x] ^ ((~row[(x + 1) % 5]) & row[(x + 2) % 5])
            ) & MASK64


def round_int(state: int, round_number: int) -> int:
    lanes = apply_l_lanes(int_to_lanes(state))
    chi_lanes(lanes)
    lanes[0] ^= ROUND_CONSTANTS[round_number]
    return lanes_to_int(lanes)


def rounds_int(state: int, rounds: int) -> int:
    out = state
    for round_number in range(rounds):
        out = round_int(out, round_number)
    return out


def squeeze_digest(state: int, digest_bits: int) -> int:
    return state & ((1 << digest_bits) - 1)


def demo() -> None:
    assert rounds_int(0, 0) == 0
    assert rounds_int(0, 1) != 0
    print("keccak state checks passed")


if __name__ == "__main__":
    demo()
