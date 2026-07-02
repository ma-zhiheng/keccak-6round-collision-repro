"""Sparse non-zero connector stress demo."""

from __future__ import annotations

import random

from beta_selector import set_sbox_value
from connector_runner import build_connector_from_alpha2
from sample_connector import sample_and_verify
from trail_parser import active_sboxes


def main() -> None:
    rng = random.Random(7)
    alpha2 = 0
    for sbox_index in rng.sample(range(320), 5):
        alpha2 = set_sbox_value(alpha2, sbox_index, rng.randrange(1, 32))

    attempt = build_connector_from_alpha2(alpha2, rate=1440, padding_bits=1, seed=7)
    sample = sample_and_verify(attempt, rng, max_basis=96)

    print("sparse connector demo")
    print(f"  target alpha2 active S-boxes: {active_sboxes(alpha2)}")
    print(f"  beta1 transition weight: {attempt.beta1_weight}")
    print(f"  system rank: {attempt.system_rank}")
    print(f"  system dimension: {attempt.system_dimension}")
    print(f"  verifies R^2(M1)+R^2(M2)=alpha2: {sample.verifies}")

    if not sample.verifies:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
