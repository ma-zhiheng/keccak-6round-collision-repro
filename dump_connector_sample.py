"""Dump a connector sample in paper-style lane matrices."""

from __future__ import annotations

import random

from connector_runner import build_connector_from_alpha2
from sample_connector import sample_and_verify
from state_lift import bit_position
from trail_parser import format_state_matrix


def main() -> None:
    alpha2 = 1 << bit_position(0, 0, 0)
    attempt = build_connector_from_alpha2(alpha2, rate=1440, padding_bits=1, seed=2)
    sample = sample_and_verify(attempt, random.Random(2), max_basis=16)

    print("M1 initial state:")
    print(format_state_matrix(sample.message1_state, zero="0"))
    print("\nM2 initial state:")
    print(format_state_matrix(sample.message2_state, zero="0"))
    print("\nR2 output difference:")
    print(format_state_matrix(sample.output_difference, zero="-"))
    print(f"\nverified: {sample.verifies}")


if __name__ == "__main__":
    main()
