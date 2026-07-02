"""Try the two-round connector on Table 7 trail core No. 2."""

from __future__ import annotations

import random
import time

from connector_runner import build_connector_from_alpha2
from linear_layer import apply_matrix_columns, load_or_build_matrices
from sample_connector import sample_and_verify
from trail_data import TRAIL_CORE_2_PARTIAL, state_from_matrix
from trail_parser import active_sboxes


def main() -> None:
    _, linv = load_or_build_matrices()
    beta2 = state_from_matrix(TRAIL_CORE_2_PARTIAL.beta2)
    alpha2 = apply_matrix_columns(linv, beta2)

    print("Table 7 core No. 2 connector search")
    print(f"  target alpha2 active S-boxes: {active_sboxes(alpha2)}")
    rng = random.Random(2026)
    best = None
    start = time.time()
    attempts = 100
    for index in range(1, attempts + 1):
        seed = rng.randrange(1 << 60)
        attempt = build_connector_from_alpha2(
            alpha2,
            rate=1440,
            padding_bits=1,
            seed=seed,
            randomize_model_choices=True,
            sample_beta0=True,
            beta0_max_basis=512,
        )
        if best is None or attempt.system_rank < best.system_rank:
            best = attempt
        if index % 10 == 0 or attempt.consistent:
            elapsed = time.time() - start
            print(
                f"  {index} attempts, elapsed={elapsed:.1f}s, "
                f"best_rank={best.system_rank}, best_dim={best.system_dimension}, "
                f"last_consistent={attempt.consistent}"
            )
        if attempt.consistent:
            sample = sample_and_verify(attempt, random.Random(seed), max_basis=256)
            print(f"  FOUND consistent connector")
            print(f"  rank: {attempt.system_rank}")
            print(f"  dimension: {attempt.system_dimension}")
            print(f"  beta1 weight: {attempt.beta1_weight}")
            print(f"  beta0 transitions: {attempt.beta0_transitions}")
            print(f"  verifies R^2 target: {sample.verifies}")
            return

    assert best is not None
    print("  no consistent connector found within limit")
    print(f"  best rank: {best.system_rank}")
    print(f"  best dimension: {best.system_dimension}")
    print(f"  best beta1 weight: {best.beta1_weight}")
    print(f"  best beta0 transitions: {best.beta0_transitions}")


if __name__ == "__main__":
    main()
