"""Repair a first 2-round connector from a saved beta pair and row seed."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from connector_runner import transitions_from_pair
from core3_connector import CORE3_CONNECTOR_ROW_ORDER, trail_states
from incremental_connector import build_bitwise_prefix_connector, build_model_backtracking_connector
from keccak_state import rounds_int
from linear_layer import apply_matrix_columns, load_or_build_matrices
from sample_connector import sample_solution
from search_core3_with_beta_pair import SavedBetaPair, load_pairs
from search_core3_row_reorder import apply_chain, base_rows, load_best_reorder
from trail_data_6round import TRAIL_CORE_5_KECCAK_1440_160_6_160


@dataclass(frozen=True)
class SearchBest:
    pair_index: int
    row_seed: int
    order: str
    added_g_rows: int


@dataclass(frozen=True)
class FixedPairRepairResult:
    pair_index: int
    row_seed: int
    repair_seed: int
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
    error: str = ""


def target_alpha2() -> int:
    return trail_states(TRAIL_CORE_5_KECCAK_1440_160_6_160).alpha2


def load_best_search(path: Path) -> SearchBest:
    best: SearchBest | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("event") not in ("new_best", "found"):
            continue
        if not all(key in entry for key in ("pair_index", "row_seed", "order", "added_g_rows")):
            continue
        current = SearchBest(
            pair_index=int(entry["pair_index"]),
            row_seed=int(entry["row_seed"]),
            order=str(entry["order"]),
            added_g_rows=int(entry["added_g_rows"]),
        )
        if best is None or current.added_g_rows > best.added_g_rows:
            best = current
    if best is None:
        raise ValueError(f"no fixed-pair search result in {path}")
    return best


def run_repair(
    pair: SavedBetaPair,
    pair_index: int,
    row_seed: int,
    repair_seed: int,
    order: str,
    greedy_retries: int,
    repair_window: int,
    repair_nodes: int,
    repair_candidates: int | None,
    prepared_rows=None,
) -> FixedPairRepairResult:
    try:
        _l_columns, linv_columns = load_or_build_matrices()
        alpha2 = target_alpha2()
        beta0_alpha1 = transitions_from_pair(pair.beta0, pair.beta1_alpha)
        beta1_alpha2 = transitions_from_pair(pair.beta1_beta, alpha2)
        greedy, _assigned, _rows = build_bitwise_prefix_connector(
            beta0_alpha1,
            beta1_alpha2,
            rate=1440,
            padding_bits=1,
            seed=row_seed,
            row_order=order,
            row_retries=greedy_retries,
            prepared_rows=prepared_rows,
        )
        repair_start = max(0, greedy.added_g_rows - repair_window)
        prefix, assigned, rows = build_bitwise_prefix_connector(
            beta0_alpha1,
            beta1_alpha2,
            rate=1440,
            padding_bits=1,
            seed=row_seed,
            row_order=order,
            row_retries=greedy_retries,
            stop_after_rows=repair_start,
            prepared_rows=prepared_rows,
        )
        connector = build_model_backtracking_connector(
            beta0_alpha1,
            beta1_alpha2,
            rate=1440,
            padding_bits=1,
            seed=repair_seed,
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
            x = sample_solution(connector.system, random.Random(row_seed ^ repair_seed), max_basis=512)
            m1 = apply_matrix_columns(linv_columns, x)
            m2 = m1 ^ pair.alpha0
            verifies = (rounds_int(m1, 2) ^ rounds_int(m2, 2)) == alpha2

        return FixedPairRepairResult(
            pair_index=pair_index,
            row_seed=row_seed,
            repair_seed=repair_seed,
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
        )
    except Exception as exc:  # pragma: no cover - diagnostic runner
        return FixedPairRepairResult(
            pair_index=pair_index,
            row_seed=row_seed,
            repair_seed=repair_seed,
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
            error=repr(exc),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-file", default="results/core3_beta_pairs.jsonl")
    parser.add_argument("--search-file", default="results/core3_pair_connector_search.jsonl")
    parser.add_argument("--reorder-file", default="results/core3_row_reorder.jsonl")
    parser.add_argument("--top-pairs", type=int, default=8)
    parser.add_argument("--from-search-best", action="store_true")
    parser.add_argument("--from-reorder-best", action="store_true")
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--row-seed", type=int, default=143651641172717653)
    parser.add_argument("--parent-chain", default="")
    parser.add_argument("--best-prefix", type=int, default=0)
    parser.add_argument("--order", default=CORE3_CONNECTOR_ROW_ORDER)
    parser.add_argument("--greedy-retries", type=int, default=2400)
    parser.add_argument("--repair-windows", default="8,16,32")
    parser.add_argument("--repair-attempts", type=int, default=1)
    parser.add_argument("--repair-seed", type=int, default=9001)
    parser.add_argument("--repair-nodes", type=int, default=120000)
    parser.add_argument("--repair-candidates", type=int, default=20)
    parser.add_argument("--result-file", default="results/core3_pair_repair.jsonl")
    args = parser.parse_args()

    pairs = load_pairs(Path(args.pair_file), args.top_pairs)
    if args.from_search_best:
        best = load_best_search(Path(args.search_file))
        args.pair_index = best.pair_index
        args.row_seed = best.row_seed
        args.order = best.order
    if args.from_reorder_best:
        best_reorder = load_best_reorder(Path(args.reorder_file))
        args.pair_index = best_reorder.pair_index
        args.row_seed = best_reorder.base_seed
        args.parent_chain = best_reorder.chain
        args.best_prefix = best_reorder.added_g_rows
        args.order = "random"
    if args.pair_index >= len(pairs):
        raise ValueError(f"pair index {args.pair_index} unavailable; loaded {len(pairs)} pairs")
    pair = pairs[args.pair_index]
    prepared_rows = None
    if args.parent_chain:
        prepared_rows = apply_chain(
            base_rows(pair, args.row_seed, args.order),
            args.parent_chain,
            args.best_prefix,
        )
    windows = [int(item.strip()) for item in args.repair_windows.split(",") if item.strip()]
    rng = random.Random(args.repair_seed)
    result_path = Path(args.result_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    print("fixed-pair repair for Keccak[1440,160,6,160] first 2-round connector", flush=True)
    print(
        f"  pair={args.pair_index}, first_eq={pair.first_round_equations}, "
        f"ddt2={pair.ddt2_transitions}, row_seed={args.row_seed}, order={args.order}",
        flush=True,
    )
    if args.parent_chain:
        print(f"  parent_chain={args.parent_chain}", flush=True)
    print(
        f"  windows={windows}, attempts/window={args.repair_attempts}, "
        f"nodes={args.repair_nodes}, candidates={args.repair_candidates}",
        flush=True,
    )

    best_result: FixedPairRepairResult | None = None
    start = time.time()
    for window in windows:
        for attempt in range(1, args.repair_attempts + 1):
            repair_seed = rng.randrange(1 << 60)
            result = run_repair(
                pair,
                args.pair_index,
                args.row_seed,
                repair_seed,
                args.order,
                args.greedy_retries,
                window,
                args.repair_nodes,
                None if args.repair_candidates <= 0 else args.repair_candidates,
                prepared_rows=prepared_rows,
            )
            if result.error:
                print(f"  error window={window} attempt={attempt}: {result.error}", flush=True)
                continue
            if best_result is None or result.added_g_rows > best_result.added_g_rows:
                best_result = result
                with result_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"event": "new_best", **asdict(result)}, sort_keys=True) + "\n")
                print(
                    f"  new best window={window} attempt={attempt}: greedy={result.greedy_rows}, "
                    f"repair_start={result.repair_start}, added={result.added_g_rows}/{result.total_g_rows}, "
                    f"rank={result.rank}, dim={result.dimension}, assigned={result.assigned_sboxes}, "
                    f"nodes={result.nodes}, failed_row={result.failed_row}, failed_deps={result.failed_deps}",
                    flush=True,
                )
            if result.consistent:
                with result_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"event": "found", **asdict(result)}, sort_keys=True) + "\n")
                print("  FOUND first 2-round connector", flush=True)
                print(f"  elapsed={time.time() - start:.1f}s", flush=True)
                print(f"  verifies R^2 target: {result.verifies}", flush=True)
                return
    print("  no connector found", flush=True)
    print(f"  elapsed={time.time() - start:.1f}s", flush=True)
    print(f"  best={best_result}", flush=True)


if __name__ == "__main__":
    main()
