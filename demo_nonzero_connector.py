"""Demonstrate a non-zero two-round connector target."""

from __future__ import annotations

import random

from connector_runner import build_connector_from_alpha2
from sample_connector import sample_and_verify
from state_lift import bit_position
from trail_parser import active_sboxes


def main() -> None:
    alpha2 = 1 << bit_position(0, 0, 0)
    attempt = build_connector_from_alpha2(alpha2, rate=1440, padding_bits=1, seed=2)
    sample = sample_and_verify(attempt, random.Random(2), max_basis=64)

    print("non-zero connector demo")
    print(f"  target alpha2 active S-boxes: {active_sboxes(alpha2)}")
    print(f"  beta1 transition weight: {attempt.beta1_weight}")
    print(f"  system rank: {attempt.system_rank}")
    print(f"  system dimension: {attempt.system_dimension}")
    print(f"  verifies R^2(M1)+R^2(M2)=alpha2: {sample.verifies}")
    print(f"  output difference low lane: {sample.output_difference & ((1 << 64) - 1):016X}")

    if not sample.verifies:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
