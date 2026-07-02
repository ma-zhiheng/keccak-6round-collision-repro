"""Incremental connector search for Table 7 trail core No. 2."""

from __future__ import annotations

import random
import time

from beta0_selector import choose_concrete_beta0, run_beta0_difference_phase
from beta_selector import search_beta_for_alpha
from connector_runner import transitions_from_pair
from incremental_connector import build_incremental_connector
from keccak_state import rounds_int
from linear_layer import apply_matrix_columns, load_or_build_matrices
from sample_connector import sample_solution
from trail_data import TRAIL_CORE_2_PARTIAL, state_from_matrix
from trail_parser import active_sboxes


def main() -> None:
    _, linv = load_or_build_matrices()
    beta2 = state_from_matrix(TRAIL_CORE_2_PARTIAL.beta2)
    alpha2 = apply_matrix_columns(linv, beta2)
    rng = random.Random(2027)

    print("incremental Table 7 core No. 2 connector search", flush=True)
    print(f"  target alpha2 active S-boxes: {active_sboxes(alpha2)}", flush=True)
    start = time.time()
    attempts = 40
    best_added = -1
    row_orders = ["mixed", "random", "small", "large"]

    for index in range(1, attempts + 1):
        seed = rng.randrange(1 << 60)
        local_rng = random.Random(seed)
        beta1_choice = search_beta_for_alpha(
            alpha2,
            min_active_alpha_sboxes=320,
            attempts=80,
            seed=seed,
        )
        beta0_diff = run_beta0_difference_phase(
            beta1_choice.alpha,
            rate=1440,
            padding_bits=1,
            seed=seed,
        )
        beta0_choice = choose_concrete_beta0(
            beta0_diff,
            rng=local_rng,
            max_basis=512,
        )
        connector = build_incremental_connector(
            transitions_from_pair(beta0_choice.beta0, beta1_choice.alpha),
            transitions_from_pair(beta1_choice.beta, alpha2),
            rate=1440,
            padding_bits=1,
            seed=seed,
            row_retries=200,
            row_order=row_orders[index % len(row_orders)],
        )
        best_added = max(best_added, connector.added_g_rows)
        if index % 5 == 0 or not connector.system.inconsistent:
            print(
                f"  {index} attempts, elapsed={time.time()-start:.1f}s, "
                f"best_added_g={best_added}, last_added_g={connector.added_g_rows}, "
                f"last_inconsistent={connector.system.inconsistent}, "
                f"order={row_orders[index % len(row_orders)]}"
            , flush=True)
        if not connector.system.inconsistent:
            x = sample_solution(connector.system, random.Random(seed), max_basis=256)
            m1 = apply_matrix_columns(linv, x)
            m2 = m1 ^ beta0_choice.alpha0
            verifies = (rounds_int(m1, 2) ^ rounds_int(m2, 2)) == alpha2
            print("  FOUND incremental connector", flush=True)
            print(f"  rank={connector.system.rank}, dimension={connector.system.dimension}", flush=True)
            print(f"  assigned_sboxes={connector.assigned_sboxes}, added_g_rows={connector.added_g_rows}/{connector.total_g_rows}", flush=True)
            print(f"  verifies R^2 target: {verifies}", flush=True)
            return

    print("  no incremental connector found within limit", flush=True)
    print(f"  best added G rows before conflict: {best_added}", flush=True)


if __name__ == "__main__":
    main()
