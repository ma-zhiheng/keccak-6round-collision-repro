"""Backtracking connector search for Table 7 trail core No. 2."""

from __future__ import annotations

import random
import time

from beta0_selector import choose_concrete_beta0, run_beta0_difference_phase
from beta_selector import search_beta_for_alpha
from connector_runner import transitions_from_pair
from incremental_connector import build_backtracking_connector
from keccak_state import rounds_int
from linear_layer import apply_matrix_columns, load_or_build_matrices
from sample_connector import sample_solution
from trail_data import TRAIL_CORE_2_PARTIAL, state_from_matrix
from trail_parser import active_sboxes


def main() -> None:
    _, linv = load_or_build_matrices()
    beta2 = state_from_matrix(TRAIL_CORE_2_PARTIAL.beta2)
    alpha2 = apply_matrix_columns(linv, beta2)
    rng = random.Random(2029)

    print("backtracking search for Table 7 core No. 2")
    print(f"  target alpha2 active S-boxes: {active_sboxes(alpha2)}")
    start = time.time()
    best = None
    attempts = 40
    orders = ["mixed", "random", "large", "small"]

    for index in range(1, attempts + 1):
        seed = rng.randrange(1 << 60)
        local_rng = random.Random(seed)
        beta1_choice = search_beta_for_alpha(
            alpha2,
            min_active_alpha_sboxes=320,
            attempts=200,
            seed=seed,
        )
        beta0_diff = run_beta0_difference_phase(
            beta1_choice.alpha,
            rate=1440,
            padding_bits=1,
            seed=seed,
        )
        beta0_choice = choose_concrete_beta0(beta0_diff, rng=local_rng, max_basis=768)
        connector = build_backtracking_connector(
            transitions_from_pair(beta0_choice.beta0, beta1_choice.alpha),
            transitions_from_pair(beta1_choice.beta, alpha2),
            rate=1440,
            padding_bits=1,
            seed=seed,
            row_order=orders[index % len(orders)],
            max_nodes=8000,
            options_per_row=60,
        )
        if best is None or connector.added_g_rows > best[0]:
            best = (connector.added_g_rows, seed, connector.system.rank, connector.system.dimension)
            print(
                f"  new best at {index}: added={connector.added_g_rows}/{connector.total_g_rows}, "
                f"rank={connector.system.rank}, dim={connector.system.dimension}, "
                f"order={orders[index % len(orders)]}, seed={seed}"
            )
        if not connector.system.inconsistent:
            x = sample_solution(connector.system, random.Random(seed), max_basis=512)
            m1 = apply_matrix_columns(linv, x)
            m2 = m1 ^ beta0_choice.alpha0
            verifies = (rounds_int(m1, 2) ^ rounds_int(m2, 2)) == alpha2
            print("  FOUND connector")
            print(f"  elapsed={time.time()-start:.1f}s")
            print(f"  rank={connector.system.rank}, dim={connector.system.dimension}")
            print(f"  assigned_sboxes={connector.assigned_sboxes}")
            print(f"  verifies R^2 target: {verifies}")
            return
        if index % 5 == 0:
            print(f"  {index} attempts, elapsed={time.time()-start:.1f}s, best={best}")

    print("  no connector found")
    print(f"  best={best}")


if __name__ == "__main__":
    main()
