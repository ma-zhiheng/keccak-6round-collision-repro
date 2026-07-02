"""Parallel search for the first 2-round connector of the 6-round attack."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from connector_runner import transitions_from_pair
from core3_connector import CORE3_CONNECTOR_ROW_ORDER, choose_beta1_beta0_pair, trail_states
from incremental_connector import build_bitwise_prefix_connector
from keccak_state import rounds_int
from linear_layer import apply_matrix_columns, load_or_build_matrices
from sample_connector import sample_solution
from trail_data_6round import TRAIL_CORE_5_KECCAK_1440_160_6_160
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
    consistent: bool
    verifies: bool
    first_round_equations: int = 0
    ddt2_transitions: int = 0
    ddt8_transitions: int = 0
    beta1_beta_hex: str = ""
    beta1_alpha_hex: str = ""
    beta0_hex: str = ""
    alpha0_hex: str = ""
    skipped_beta1: bool = False
    error: str = ""


def target_alpha2() -> int:
    return trail_states(TRAIL_CORE_5_KECCAK_1440_160_6_160).alpha2


def run_attempt(
    seed: int,
    order: str,
    row_retries: int,
    beta_attempts: int,
    beta1_candidates: int,
    beta0_basis: int,
    beta0_samples: int,
) -> AttemptResult:
    try:
        _l_columns, linv_columns = load_or_build_matrices()
        alpha2 = target_alpha2()
        try:
            beta1_choice, beta0_choice = choose_beta1_beta0_pair(
                alpha2,
                seed=seed,
                beta_attempts=beta_attempts,
                beta1_candidates=beta1_candidates,
                beta0_samples=beta0_samples,
                beta0_basis=beta0_basis,
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
                consistent=False,
                verifies=False,
                skipped_beta1=True,
            )

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
            m1 = apply_matrix_columns(linv_columns, x)
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
            consistent=not connector.system.inconsistent,
            verifies=verifies,
            first_round_equations=beta0_choice.first_round_equations,
            ddt2_transitions=beta0_choice.ddt2_transitions,
            ddt8_transitions=beta0_choice.ddt8_transitions,
            beta1_beta_hex=f"{beta1_choice.beta:x}",
            beta1_alpha_hex=f"{beta1_choice.alpha:x}",
            beta0_hex=f"{beta0_choice.beta0:x}",
            alpha0_hex=f"{beta0_choice.alpha0:x}",
        )
    except Exception as exc:  # pragma: no cover - diagnostic runner
        return AttemptResult(
            seed=seed,
            order=order,
            added_g_rows=0,
            total_g_rows=0,
            rank=0,
            dimension=0,
            assigned_sboxes=0,
            consistent=False,
            verifies=False,
            error=repr(exc),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempts", type=int, default=128)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=6103)
    parser.add_argument("--row-retries", type=int, default=2400)
    parser.add_argument("--beta-attempts", type=int, default=1000)
    parser.add_argument("--beta1-candidates", type=int, default=1)
    parser.add_argument("--beta0-basis", type=int, default=768)
    parser.add_argument("--beta0-samples", type=int, default=512)
    parser.add_argument("--orders", default=CORE3_CONNECTOR_ROW_ORDER)
    parser.add_argument("--result-file", default="results/core3_first_two_search.jsonl")
    args = parser.parse_args()

    alpha2 = target_alpha2()
    print("parallel first 2-round connector search for Keccak[1440,160,6,160]", flush=True)
    print(f"  target alpha2 active S-boxes: {active_sboxes(alpha2)}", flush=True)
    print(
        f"  attempts={args.attempts}, workers={args.workers}, row_retries={args.row_retries}, "
        f"beta_attempts={args.beta_attempts}, beta1_candidates={args.beta1_candidates}, "
        f"beta0_samples={args.beta0_samples}",
        flush=True,
    )
    result_path = Path(args.result_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    orders = [item.strip() for item in args.orders.split(",") if item.strip()]
    jobs = [
        (
            rng.randrange(1 << 60),
            orders[index % len(orders)],
            args.row_retries,
            args.beta_attempts,
            args.beta1_candidates,
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
                with result_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"event": "new_best", **asdict(result)}, sort_keys=True) + "\n")
                print(
                    f"  new best after {completed}: added={result.added_g_rows}/{result.total_g_rows}, "
                    f"rank={result.rank}, dim={result.dimension}, assigned={result.assigned_sboxes}, "
                    f"first_eq={result.first_round_equations}, ddt2={result.ddt2_transitions}, "
                    f"ddt8={result.ddt8_transitions}, order={result.order}, seed={result.seed}",
                    flush=True,
                )
            if result.consistent:
                with result_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"event": "found", **asdict(result)}, sort_keys=True) + "\n")
                print("  FOUND first 2-round connector", flush=True)
                print(f"  elapsed={time.time() - start:.1f}s", flush=True)
                print(f"  seed={result.seed}, order={result.order}", flush=True)
                print(f"  rank={result.rank}, dim={result.dimension}", flush=True)
                print(f"  verifies R^2 target: {result.verifies}", flush=True)
                return

    print("  no connector found", flush=True)
    print(f"  elapsed={time.time() - start:.1f}s", flush=True)
    print(f"  skipped beta1 candidates: {skipped_beta1}", flush=True)
    print(f"  best={best}", flush=True)


if __name__ == "__main__":
    main()
