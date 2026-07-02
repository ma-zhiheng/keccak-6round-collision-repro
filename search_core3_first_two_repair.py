"""Repair-style search for the 6-round first 2-round connector.

The greedy connector often reaches a useful prefix and then fails on one dense
row. This runner rebuilds a prefix, backs up by a small window, then lets the
model-level backtracker search locally from there.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from connector_runner import transitions_from_pair
from core3_connector import CORE3_CONNECTOR_ROW_ORDER, choose_beta1_beta0_pair, trail_states
from incremental_connector import build_bitwise_prefix_connector, build_model_backtracking_connector
from keccak_state import rounds_int
from linear_layer import apply_matrix_columns, load_or_build_matrices
from sample_connector import sample_solution
from trail_data_6round import TRAIL_CORE_5_KECCAK_1440_160_6_160
from trail_parser import active_sboxes


@dataclass(frozen=True)
class RepairResult:
    seed: int
    order: str
    greedy_rows: int
    repair_start: int
    added_g_rows: int
    total_g_rows: int
    rank: int
    dimension: int
    assigned_sboxes: int
    nodes: int
    failed_row: int | None
    failed_deps: int
    consistent: bool
    verifies: bool
    first_round_equations: int
    ddt2_transitions: int
    ddt8_transitions: int
    beta1_beta_hex: str = ""
    beta1_alpha_hex: str = ""
    beta0_hex: str = ""
    alpha0_hex: str = ""
    error: str = ""


def target_alpha2() -> int:
    return trail_states(TRAIL_CORE_5_KECCAK_1440_160_6_160).alpha2


def run_attempt(
    seed: int,
    order: str,
    beta_attempts: int,
    beta1_candidates: int,
    beta0_samples: int,
    beta0_basis: int,
    greedy_retries: int,
    repair_window: int,
    repair_nodes: int,
    repair_candidates: int,
) -> RepairResult:
    try:
        _l_columns, linv_columns = load_or_build_matrices()
        alpha2 = target_alpha2()
        beta1_choice, beta0_choice = choose_beta1_beta0_pair(
            alpha2,
            seed=seed,
            beta_attempts=beta_attempts,
            beta1_candidates=beta1_candidates,
            beta0_samples=beta0_samples,
            beta0_basis=beta0_basis,
        )

        beta0_alpha1 = transitions_from_pair(beta0_choice.beta0, beta1_choice.alpha)
        beta1_alpha2 = transitions_from_pair(beta1_choice.beta, alpha2)
        greedy, _assigned, _rows = build_bitwise_prefix_connector(
            beta0_alpha1,
            beta1_alpha2,
            rate=1440,
            padding_bits=1,
            seed=seed,
            row_order=order,
            row_retries=greedy_retries,
        )
        repair_start = max(0, greedy.added_g_rows - repair_window)
        prefix, assigned, rows = build_bitwise_prefix_connector(
            beta0_alpha1,
            beta1_alpha2,
            rate=1440,
            padding_bits=1,
            seed=seed,
            row_order=order,
            row_retries=greedy_retries,
            stop_after_rows=repair_start,
        )
        connector = build_model_backtracking_connector(
            beta0_alpha1,
            beta1_alpha2,
            rate=1440,
            padding_bits=1,
            seed=seed ^ 0xC03E,
            row_order=order,
            max_nodes=repair_nodes,
            max_candidates_per_sbox=repair_candidates,
            start_system=prefix.system,
            start_assigned=assigned,
            start_row_index=repair_start,
            prepared_rows=rows,
        )

        verifies = False
        if not connector.system.inconsistent:
            x = sample_solution(connector.system, random.Random(seed), max_basis=512)
            m1 = apply_matrix_columns(linv_columns, x)
            m2 = m1 ^ beta0_choice.alpha0
            verifies = (rounds_int(m1, 2) ^ rounds_int(m2, 2)) == alpha2

        return RepairResult(
            seed=seed,
            order=order,
            greedy_rows=greedy.added_g_rows,
            repair_start=repair_start,
            added_g_rows=connector.added_g_rows,
            total_g_rows=connector.total_g_rows,
            rank=connector.system.rank,
            dimension=connector.system.dimension,
            assigned_sboxes=connector.assigned_sboxes,
            nodes=connector.nodes,
            failed_row=connector.failed_row,
            failed_deps=len(connector.failed_sboxes),
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
        return RepairResult(
            seed=seed,
            order=order,
            greedy_rows=0,
            repair_start=0,
            added_g_rows=0,
            total_g_rows=0,
            rank=0,
            dimension=0,
            assigned_sboxes=0,
            nodes=0,
            failed_row=None,
            failed_deps=0,
            consistent=False,
            verifies=False,
            first_round_equations=0,
            ddt2_transitions=0,
            ddt8_transitions=0,
            error=repr(exc),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7409)
    parser.add_argument("--single-seed", type=int, default=None)
    parser.add_argument("--orders", default=CORE3_CONNECTOR_ROW_ORDER)
    parser.add_argument("--beta-attempts", type=int, default=1200)
    parser.add_argument("--beta1-candidates", type=int, default=1)
    parser.add_argument("--beta0-samples", type=int, default=512)
    parser.add_argument("--beta0-basis", type=int, default=768)
    parser.add_argument("--greedy-retries", type=int, default=1200)
    parser.add_argument("--repair-window", type=int, default=10)
    parser.add_argument("--repair-nodes", type=int, default=120000)
    parser.add_argument("--repair-candidates", type=int, default=20)
    parser.add_argument("--result-file", default="results/core3_first_two_repair.jsonl")
    args = parser.parse_args()

    alpha2 = target_alpha2()
    print("repair search for Keccak[1440,160,6,160] first 2-round connector", flush=True)
    print(f"  target alpha2 active S-boxes: {active_sboxes(alpha2)}", flush=True)
    print(
        f"  attempts={args.attempts}, greedy_retries={args.greedy_retries}, "
        f"repair_window={args.repair_window}, repair_nodes={args.repair_nodes}, "
        f"beta1_candidates={args.beta1_candidates}",
        flush=True,
    )
    result_path = Path(args.result_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    orders = [item.strip() for item in args.orders.split(",") if item.strip()]
    if args.single_seed is not None:
        seeds = [args.single_seed]
        args.attempts = 1
    else:
        rng = random.Random(args.seed)
        seeds = [rng.randrange(1 << 60) for _ in range(args.attempts)]
    best: RepairResult | None = None
    start = time.time()
    for index, seed in enumerate(seeds, start=1):
        order = orders[(index - 1) % len(orders)]
        result = run_attempt(
            seed,
            order,
            args.beta_attempts,
            args.beta1_candidates,
            args.beta0_samples,
            args.beta0_basis,
            args.greedy_retries,
            args.repair_window,
            args.repair_nodes,
            args.repair_candidates,
        )
        if result.error:
            print(f"  error attempt={index} seed={seed}: {result.error}", flush=True)
            continue
        if best is None or result.added_g_rows > best.added_g_rows:
            best = result
            with result_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": "new_best", **asdict(result)}, sort_keys=True) + "\n")
            print(
                f"  new best at {index}: greedy={result.greedy_rows}, "
                f"repair_start={result.repair_start}, added={result.added_g_rows}/{result.total_g_rows}, "
                f"rank={result.rank}, dim={result.dimension}, assigned={result.assigned_sboxes}, "
                f"nodes={result.nodes}, failed_row={result.failed_row}, failed_deps={result.failed_deps}, "
                f"first_eq={result.first_round_equations}, ddt2={result.ddt2_transitions}, "
                f"order={result.order}, seed={result.seed}",
                flush=True,
            )
        if result.consistent:
            with result_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": "found", **asdict(result)}, sort_keys=True) + "\n")
            print("  FOUND first 2-round connector", flush=True)
            print(f"  elapsed={time.time() - start:.1f}s", flush=True)
            print(f"  verifies R^2 target: {result.verifies}", flush=True)
            return
        if index % 5 == 0:
            print(f"  {index} attempts done, elapsed={time.time() - start:.1f}s, best={best}", flush=True)

    print("  no connector found", flush=True)
    print(f"  elapsed={time.time() - start:.1f}s", flush=True)
    print(f"  best={best}", flush=True)


if __name__ == "__main__":
    main()
