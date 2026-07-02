"""Greedy-prefix plus model-level repair search for Table 7 core No. 2."""

from __future__ import annotations

import random
import time

from beta0_selector import choose_concrete_beta0, run_beta0_difference_phase
from beta_selector import search_beta_for_alpha
from connector_runner import transitions_from_pair
from incremental_connector import build_bitwise_prefix_connector, build_model_backtracking_connector
from keccak_state import rounds_int
from linear_layer import apply_matrix_columns, load_or_build_matrices
from sample_connector import sample_solution
from trail_data import TRAIL_CORE_2_PARTIAL, state_from_matrix
from trail_parser import active_sboxes


def main() -> None:
    _, linv = load_or_build_matrices()
    beta2 = state_from_matrix(TRAIL_CORE_2_PARTIAL.beta2)
    alpha2 = apply_matrix_columns(linv, beta2)
    rng = random.Random(2032)

    print("repair search for Table 7 core No. 2", flush=True)
    print(f"  target alpha2 active S-boxes: {active_sboxes(alpha2)}", flush=True)
    start = time.time()
    best = None
    attempts = 30
    orders = ["small", "mixed", "random", "large"]
    skipped_beta1 = 0

    for index in range(1, attempts + 1):
        seed = rng.randrange(1 << 60)
        local_rng = random.Random(seed)
        try:
            beta1_choice = search_beta_for_alpha(
                alpha2,
                min_active_alpha_sboxes=320,
                attempts=300,
                seed=seed,
                strict=True,
            )
        except RuntimeError:
            skipped_beta1 += 1
            continue

        beta0_diff = run_beta0_difference_phase(
            beta1_choice.alpha,
            rate=1440,
            padding_bits=1,
            seed=seed,
        )
        beta0_choice = choose_concrete_beta0(beta0_diff, rng=local_rng, max_basis=768)
        beta0_alpha1 = transitions_from_pair(beta0_choice.beta0, beta1_choice.alpha)
        beta1_alpha2 = transitions_from_pair(beta1_choice.beta, alpha2)
        order = orders[index % len(orders)]
        prefix, assigned, rows = build_bitwise_prefix_connector(
            beta0_alpha1,
            beta1_alpha2,
            rate=1440,
            padding_bits=1,
            seed=seed,
            row_order=order,
            row_retries=200,
        )
        repair_start = max(0, prefix.added_g_rows - 8)
        prefix, assigned, rows = build_bitwise_prefix_connector(
            beta0_alpha1,
            beta1_alpha2,
            rate=1440,
            padding_bits=1,
            seed=seed,
            row_order=order,
            row_retries=200,
            stop_after_rows=repair_start,
        )
        connector = build_model_backtracking_connector(
            beta0_alpha1,
            beta1_alpha2,
            rate=1440,
            padding_bits=1,
            seed=seed ^ 0x5EED,
            row_order=order,
            max_nodes=60000,
            max_candidates_per_sbox=16,
            start_system=prefix.system,
            start_assigned=assigned,
            start_row_index=repair_start,
            prepared_rows=rows,
        )

        if best is None or connector.added_g_rows > best[0]:
            best = (
                connector.added_g_rows,
                seed,
                prefix.added_g_rows,
                connector.system.rank,
                connector.system.dimension,
                connector.nodes,
                connector.failed_row,
                len(connector.failed_sboxes),
                connector.failure_reason,
            )
            print(
                f"  new best at {index}: added={connector.added_g_rows}/{connector.total_g_rows}, "
                f"repair_start={repair_start}, rank={connector.system.rank}, "
                f"dim={connector.system.dimension}, assigned={connector.assigned_sboxes}, "
                f"nodes={connector.nodes}, failed_row={connector.failed_row}, "
                f"failed_deps={len(connector.failed_sboxes)}, reason={connector.failure_reason}, "
                f"order={order}, seed={seed}",
                flush=True,
            )

        if not connector.system.inconsistent:
            x = sample_solution(connector.system, random.Random(seed), max_basis=512)
            m1 = apply_matrix_columns(linv, x)
            m2 = m1 ^ beta0_choice.alpha0
            verifies = (rounds_int(m1, 2) ^ rounds_int(m2, 2)) == alpha2
            print("  FOUND connector", flush=True)
            print(f"  elapsed={time.time()-start:.1f}s", flush=True)
            print(f"  rank={connector.system.rank}, dim={connector.system.dimension}", flush=True)
            print(f"  assigned_sboxes={connector.assigned_sboxes}, nodes={connector.nodes}", flush=True)
            print(f"  verifies R^2 target: {verifies}", flush=True)
            return

        if index % 5 == 0:
            print(
                f"  {index} attempts, elapsed={time.time()-start:.1f}s, "
                f"skipped_beta1={skipped_beta1}, best={best}",
                flush=True,
            )

    print("  no connector found", flush=True)
    print(f"  skipped beta1 candidates: {skipped_beta1}", flush=True)
    print(f"  best={best}", flush=True)


if __name__ == "__main__":
    main()
