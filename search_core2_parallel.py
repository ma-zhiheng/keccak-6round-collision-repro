"""Parallel connector search for Table 7 trail core No. 2.

The individual connector attempts are independent, so CPU parallelism is the
most useful accelerator for the current reproduction stage.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from beta0_selector import choose_best_concrete_beta0, choose_concrete_beta0, run_beta0_difference_phase
from beta_selector import search_beta_for_alpha
from connector_runner import transitions_from_pair
from incremental_connector import build_bitwise_prefix_connector
from keccak_state import rounds_int
from linear_layer import apply_matrix_columns, load_or_build_matrices
from sample_connector import sample_solution
from trail_data import TRAIL_CORE_2_PARTIAL, state_from_matrix
from trail_parser import active_sboxes


@dataclass(frozen=True)
class AttemptResult:
    seed: int
    order: str
    added_g_rows: int
    total_g_rows: int
    rank: int
    dimension: int
    assigned_sboxes: int
    failed_row: int | None
    failed_deps: int
    consistent: bool
    verifies: bool
    first_round_equations: int = 0
    ddt2_transitions: int = 0
    ddt8_transitions: int = 0
    skipped_beta1: bool = False
    error: str = ""


def target_alpha2() -> int:
    _, linv = load_or_build_matrices()
    beta2 = state_from_matrix(TRAIL_CORE_2_PARTIAL.beta2)
    return apply_matrix_columns(linv, beta2)


def run_attempt(
    seed: int,
    order: str,
    row_retries: int,
    beta0_basis: int,
    beta0_samples: int,
) -> AttemptResult:
    try:
        _, linv = load_or_build_matrices()
        alpha2 = target_alpha2()
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
            return AttemptResult(
                seed=seed,
                order=order,
                added_g_rows=0,
                total_g_rows=0,
                rank=0,
                dimension=0,
                assigned_sboxes=0,
                failed_row=None,
                failed_deps=0,
                consistent=False,
                verifies=False,
                skipped_beta1=True,
            )

        beta0_diff = run_beta0_difference_phase(
            beta1_choice.alpha,
            rate=1440,
            padding_bits=1,
            seed=seed,
        )
        if beta0_samples > 0:
            beta0_choice = choose_best_concrete_beta0(
                beta0_diff,
                rng=local_rng,
                samples=beta0_samples,
                max_basis=beta0_basis,
            )
        else:
            beta0_choice = choose_concrete_beta0(beta0_diff, rng=local_rng, max_basis=beta0_basis)
        connector, _assigned, _rows = build_bitwise_prefix_connector(
            transitions_from_pair(beta0_choice.beta0, beta1_choice.alpha),
            transitions_from_pair(beta1_choice.beta, alpha2),
            rate=1440,
            padding_bits=1,
            seed=seed,
            row_order=order,
            row_retries=row_retries,
        )

        verifies = False
        if not connector.system.inconsistent:
            x = sample_solution(connector.system, random.Random(seed), max_basis=512)
            m1 = apply_matrix_columns(linv, x)
            m2 = m1 ^ beta0_choice.alpha0
            verifies = (rounds_int(m1, 2) ^ rounds_int(m2, 2)) == alpha2

        return AttemptResult(
            seed=seed,
            order=order,
            added_g_rows=connector.added_g_rows,
            total_g_rows=connector.total_g_rows,
            rank=connector.system.rank,
            dimension=connector.system.dimension,
            assigned_sboxes=connector.assigned_sboxes,
            failed_row=connector.failed_row,
            failed_deps=len(connector.failed_sboxes),
            consistent=not connector.system.inconsistent,
            verifies=verifies,
            first_round_equations=beta0_choice.first_round_equations,
            ddt2_transitions=beta0_choice.ddt2_transitions,
            ddt8_transitions=beta0_choice.ddt8_transitions,
        )
    except Exception as exc:  # pragma: no cover - diagnostic search runner
        return AttemptResult(
            seed=seed,
            order=order,
            added_g_rows=0,
            total_g_rows=0,
            rank=0,
            dimension=0,
            assigned_sboxes=0,
            failed_row=None,
            failed_deps=0,
            consistent=False,
            verifies=False,
            error=repr(exc),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempts", type=int, default=256)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=2040)
    parser.add_argument("--row-retries", type=int, default=800)
    parser.add_argument("--beta0-basis", type=int, default=768)
    parser.add_argument("--beta0-samples", type=int, default=0)
    parser.add_argument("--orders", default="large,small,mixed,random")
    args = parser.parse_args()

    alpha2 = target_alpha2()
    print("parallel Table 7 core No. 2 connector search", flush=True)
    print(f"  target alpha2 active S-boxes: {active_sboxes(alpha2)}", flush=True)
    print(
        f"  attempts={args.attempts}, workers={args.workers}, "
        f"row_retries={args.row_retries}, beta0_samples={args.beta0_samples}",
        flush=True,
    )

    rng = random.Random(args.seed)
    orders = [item.strip() for item in args.orders.split(",") if item.strip()]
    jobs = [
        (
            rng.randrange(1 << 60),
            orders[index % len(orders)],
            args.row_retries,
            args.beta0_basis,
            args.beta0_samples,
        )
        for index in range(args.attempts)
    ]

    start = time.time()
    best: AttemptResult | None = None
    skipped_beta1 = 0
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_attempt, *job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            if result.skipped_beta1:
                skipped_beta1 += 1
                continue
            if result.error:
                print(f"  error seed={result.seed}: {result.error}", flush=True)
                continue
            if best is None or result.added_g_rows > best.added_g_rows:
                best = result
                print(
                    f"  new best after {completed}: added={result.added_g_rows}/{result.total_g_rows}, "
                    f"rank={result.rank}, dim={result.dimension}, assigned={result.assigned_sboxes}, "
                    f"failed_row={result.failed_row}, failed_deps={result.failed_deps}, "
                    f"first_eq={result.first_round_equations}, ddt2={result.ddt2_transitions}, "
                    f"order={result.order}, seed={result.seed}",
                    flush=True,
                )
            if result.consistent:
                print("  FOUND connector", flush=True)
                print(f"  elapsed={time.time()-start:.1f}s", flush=True)
                print(f"  seed={result.seed}, order={result.order}", flush=True)
                print(f"  rank={result.rank}, dim={result.dimension}", flush=True)
                print(
                    f"  first_eq={result.first_round_equations}, "
                    f"ddt2={result.ddt2_transitions}, ddt8={result.ddt8_transitions}",
                    flush=True,
                )
                print(f"  verifies R^2 target: {result.verifies}", flush=True)
                pool.shutdown(cancel_futures=True)
                return
            if completed % max(1, args.workers) == 0:
                print(
                    f"  completed={completed}, elapsed={time.time()-start:.1f}s, "
                    f"skipped_beta1={skipped_beta1}, best={best}",
                    flush=True,
                )

    print("  no connector found", flush=True)
    print(f"  skipped beta1 candidates: {skipped_beta1}", flush=True)
    print(f"  best={best}", flush=True)


if __name__ == "__main__":
    main()
