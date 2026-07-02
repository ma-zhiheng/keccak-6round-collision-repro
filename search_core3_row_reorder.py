"""Targeted row-reordering search for the 6-round first 2-round connector."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from connector_equations import second_round_g_equations
from connector_runner import transitions_from_pair
from core3_connector import trail_states
from incremental_connector import PreparedBitRow, build_bitwise_prefix_connector, prepared_bit_rows
from keccak_state import rounds_int
from linear_layer import apply_matrix_columns, load_or_build_matrices
from sample_connector import sample_solution
from search_core3_with_beta_pair import SavedBetaPair, load_pairs
from trail_data_6round import TRAIL_CORE_5_KECCAK_1440_160_6_160


@dataclass(frozen=True)
class SearchBest:
    pair_index: int
    row_seed: int
    order: str
    added_g_rows: int


@dataclass(frozen=True)
class ReorderAttempt:
    pair_index: int
    base_seed: int
    parent_chain: str
    parent_strategy: str
    parent_seed: int
    model_seed: int
    strategy: str
    chain: str
    added_g_rows: int
    total_g_rows: int
    rank: int
    dimension: int
    assigned_sboxes: int
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
        current_key = (current.added_g_rows, current.rank, -current.dimension)
        best_key = (-1, -1, 0) if best is None else (best.added_g_rows, best.rank, -best.dimension)
        if current_key >= best_key:
            best = current
    if best is None:
        raise ValueError(f"no fixed-pair search result in {path}")
    return best


def load_best_reorder(path: Path) -> ReorderAttempt:
    best: ReorderAttempt | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("event") not in ("new_best", "found"):
            continue
        if not all(key in entry for key in ("pair_index", "base_seed", "model_seed", "strategy", "added_g_rows")):
            continue
        parent_chain = str(entry.get("parent_chain", ""))
        chain = str(entry.get("chain", ""))
        if not chain:
            parts = []
            if entry.get("parent_strategy"):
                parts.append(f"{entry['parent_strategy']}:{entry.get('parent_seed', 0)}")
            parts.append(f"{entry['strategy']}:{entry['model_seed']}")
            chain = ";".join(parts)
        current = ReorderAttempt(
            pair_index=int(entry["pair_index"]),
            base_seed=int(entry["base_seed"]),
            parent_chain=parent_chain,
            parent_strategy=str(entry.get("parent_strategy", "")),
            parent_seed=int(entry.get("parent_seed", 0)),
            model_seed=int(entry["model_seed"]),
            strategy=str(entry["strategy"]),
            chain=chain,
            added_g_rows=int(entry["added_g_rows"]),
            total_g_rows=int(entry.get("total_g_rows", 292)),
            rank=int(entry.get("rank", 0)),
            dimension=int(entry.get("dimension", 0)),
            assigned_sboxes=int(entry.get("assigned_sboxes", 0)),
            failed_row=entry.get("failed_row"),
            failed_deps=int(entry.get("failed_deps", 0)),
            consistent=bool(entry.get("consistent", False)),
            verifies=bool(entry.get("verifies", False)),
            error=str(entry.get("error", "")),
        )
        if best is None or current.added_g_rows > best.added_g_rows:
            best = current
    if best is None:
        raise ValueError(f"no row-reorder result in {path}")
    return best


def apply_chain(rows: list[PreparedBitRow], chain: str, best_prefix: int) -> list[PreparedBitRow]:
    result = rows
    for item in chain.split(";"):
        if not item.strip():
            continue
        strategy, seed_text = item.rsplit(":", 1)
        result = make_variant(result, random.Random(int(seed_text)), strategy, best_prefix)
    return result


def base_rows(pair: SavedBetaPair, base_seed: int, order: str) -> list[PreparedBitRow]:
    l_columns, _linv_columns = load_or_build_matrices()
    alpha2 = target_alpha2()
    beta0_alpha1 = transitions_from_pair(pair.beta0, pair.beta1_alpha)
    # Build candidate counts through the public builder helpers only where the
    # row ordering needs them; random order ignores the counts.
    rng = random.Random(base_seed)
    rows = prepared_bit_rows(
        second_round_g_equations(transitions_from_pair(pair.beta1_beta, alpha2)),
        l_columns,
        order,
        rng,
    )
    # Touch beta0_alpha1 so accidental stale pair/order bugs are easier to spot
    # under static checkers; the connector run uses it directly.
    assert beta0_alpha1 or pair.beta0 == 0
    return rows


def move_index(rows: list[PreparedBitRow], source_index: int, target_index: int) -> list[PreparedBitRow]:
    source_index = max(0, min(source_index, len(rows) - 1))
    target_index = max(0, min(target_index, len(rows) - 1))
    reordered = list(rows)
    row = reordered.pop(source_index)
    reordered.insert(target_index, row)
    return reordered


def make_variant(
    rows: list[PreparedBitRow],
    rng: random.Random,
    strategy: str,
    best_prefix: int,
) -> list[PreparedBitRow]:
    if strategy.startswith("move-index-"):
        rest = strategy.removeprefix("move-index-")
        source_text, target_text = rest.split("-to-", 1)
        return move_index(rows, int(source_text), int(target_text))
    if strategy == "identity":
        return list(rows)
    if strategy.startswith("shuffle-slice-"):
        rest = strategy.removeprefix("shuffle-slice-")
        start_text, end_text = rest.split("-", 1)
        start = int(start_text)
        end = int(end_text)
        reordered = list(rows)
        window_rows = reordered[start:end]
        rng.shuffle(window_rows)
        return reordered[:start] + window_rows + reordered[end:]
    if strategy.startswith("move-fail-"):
        target = int(strategy.rsplit("-", 1)[1])
        return move_index(rows, best_prefix, target)
    if strategy.startswith("move-tail-"):
        target = int(strategy.rsplit("-", 1)[1])
        source = rng.randrange(best_prefix, len(rows))
        return move_index(rows, source, target)
    if strategy.startswith("shuffle-tail-"):
        cut = int(strategy.rsplit("-", 1)[1])
        reordered = list(rows)
        tail = reordered[cut:]
        rng.shuffle(tail)
        return reordered[:cut] + tail
    if strategy.startswith("frontier-window-"):
        window = int(strategy.rsplit("-", 1)[1])
        start = max(0, best_prefix - window)
        end = min(len(rows), best_prefix + window)
        reordered = list(rows)
        window_rows = reordered[start:end]
        rng.shuffle(window_rows)
        return reordered[:start] + window_rows + reordered[end:]
    if strategy == "random-all":
        reordered = list(rows)
        rng.shuffle(reordered)
        return reordered
    raise ValueError(f"unknown strategy: {strategy}")


def run_attempt(
    pair: SavedBetaPair,
    pair_index: int,
    base_seed: int,
    order: str,
    parent_chain: str,
    parent_strategy: str,
    parent_seed: int,
    model_seed: int,
    strategy: str,
    best_prefix: int,
    row_retries: int,
) -> ReorderAttempt:
    try:
        _l_columns, linv_columns = load_or_build_matrices()
        alpha2 = target_alpha2()
        rows = base_rows(pair, base_seed, order)
        if parent_chain:
            rows = apply_chain(rows, parent_chain, best_prefix)
        if parent_strategy:
            rows = make_variant(rows, random.Random(parent_seed), parent_strategy, best_prefix)
        concrete_strategy = strategy
        if strategy.startswith("move-fail-"):
            target = int(strategy.rsplit("-", 1)[1])
            concrete_strategy = f"move-index-{best_prefix}-to-{target}"
            variant_rows = move_index(rows, best_prefix, target)
        elif strategy.startswith("move-tail-"):
            target = int(strategy.rsplit("-", 1)[1])
            source = random.Random(model_seed).randrange(best_prefix, len(rows))
            concrete_strategy = f"move-index-{source}-to-{target}"
            variant_rows = move_index(rows, source, target)
        elif strategy.startswith("frontier-window-"):
            window = int(strategy.rsplit("-", 1)[1])
            start = max(0, best_prefix - window)
            end = min(len(rows), best_prefix + window)
            concrete_strategy = f"shuffle-slice-{start}-{end}"
            variant_rows = make_variant(rows, random.Random(model_seed), concrete_strategy, best_prefix)
        else:
            variant_rows = make_variant(rows, random.Random(model_seed), strategy, best_prefix)
        chain_parts = []
        if parent_chain:
            chain_parts.append(parent_chain)
        if parent_strategy:
            chain_parts.append(f"{parent_strategy}:{parent_seed}")
        chain_parts.append(f"{concrete_strategy}:{model_seed}")
        chain = ";".join(chain_parts)
        connector, _assigned, _rows = build_bitwise_prefix_connector(
            transitions_from_pair(pair.beta0, pair.beta1_alpha),
            transitions_from_pair(pair.beta1_beta, alpha2),
            rate=1440,
            padding_bits=1,
            seed=model_seed,
            row_order=order,
            row_retries=row_retries,
            prepared_rows=variant_rows,
        )
        verifies = False
        if not connector.system.inconsistent:
            x = sample_solution(connector.system, random.Random(model_seed), max_basis=512)
            m1 = apply_matrix_columns(linv_columns, x)
            m2 = m1 ^ pair.alpha0
            verifies = (rounds_int(m1, 2) ^ rounds_int(m2, 2)) == alpha2
        return ReorderAttempt(
            pair_index=pair_index,
            base_seed=base_seed,
            parent_chain=parent_chain,
            parent_strategy=parent_strategy,
            parent_seed=parent_seed,
            model_seed=model_seed,
            strategy=concrete_strategy,
            chain=chain,
            added_g_rows=connector.added_g_rows,
            total_g_rows=connector.total_g_rows,
            rank=connector.system.rank,
            dimension=connector.system.dimension,
            assigned_sboxes=connector.assigned_sboxes,
            failed_row=connector.failed_row,
            failed_deps=len(connector.failed_sboxes),
            consistent=not connector.system.inconsistent,
            verifies=verifies,
        )
    except Exception as exc:  # pragma: no cover - diagnostic runner
        return ReorderAttempt(
            pair_index=pair_index,
            base_seed=base_seed,
            parent_chain=parent_chain,
            parent_strategy=parent_strategy,
            parent_seed=parent_seed,
            model_seed=model_seed,
            strategy=strategy,
            chain="",
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
    parser.add_argument("--pair-file", default="results/core3_beta_pairs.jsonl")
    parser.add_argument("--search-file", default="results/core3_pair_connector_search.jsonl")
    parser.add_argument("--reorder-file", default="results/core3_row_reorder.jsonl")
    parser.add_argument("--result-file", default="results/core3_row_reorder.jsonl")
    parser.add_argument("--top-pairs", type=int, default=12)
    parser.add_argument("--from-search-best", action="store_true")
    parser.add_argument("--from-reorder-best", action="store_true")
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=1014163185563850369)
    parser.add_argument("--parent-chain", default="")
    parser.add_argument("--parent-strategy", default="")
    parser.add_argument("--parent-seed", type=int, default=0)
    parser.add_argument("--order", default="random")
    parser.add_argument("--best-prefix", type=int, default=233)
    parser.add_argument("--attempts", type=int, default=512)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=9901)
    parser.add_argument("--include-base-model-seed", action="store_true")
    parser.add_argument("--model-seeds", default="")
    parser.add_argument("--row-retries", type=int, default=2400)
    parser.add_argument(
        "--strategies",
        default="move-fail-0,move-fail-32,move-fail-64,move-fail-96,move-fail-128,move-fail-160,move-fail-200,move-tail-64,move-tail-128,frontier-window-16,frontier-window-32,shuffle-tail-128,random-all",
    )
    args = parser.parse_args()

    pairs = load_pairs(Path(args.pair_file), args.top_pairs)
    if args.from_search_best:
        best = load_best_search(Path(args.search_file))
        args.pair_index = best.pair_index
        args.base_seed = best.row_seed
        args.order = best.order
        args.best_prefix = best.added_g_rows
    if args.from_reorder_best:
        best_reorder = load_best_reorder(Path(args.reorder_file))
        args.pair_index = best_reorder.pair_index
        args.base_seed = best_reorder.base_seed
        args.parent_chain = best_reorder.chain
        args.parent_strategy = ""
        args.parent_seed = 0
        args.best_prefix = best_reorder.added_g_rows
    if args.pair_index >= len(pairs):
        raise ValueError(f"pair index {args.pair_index} unavailable; loaded {len(pairs)} pairs")
    pair = pairs[args.pair_index]
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    rng = random.Random(args.seed)
    jobs = []
    fixed_model_seeds = [int(item.strip()) for item in args.model_seeds.split(",") if item.strip()]
    for model_seed in fixed_model_seeds:
        for strategy in strategies:
            jobs.append(
                (
                    pair,
                    args.pair_index,
                    args.base_seed,
                    args.order,
                    args.parent_chain,
                    args.parent_strategy,
                    args.parent_seed,
                    model_seed,
                    strategy,
                    args.best_prefix,
                    args.row_retries,
                )
            )
    if args.include_base_model_seed:
        for strategy in strategies:
            jobs.append(
                (
                    pair,
                args.pair_index,
                args.base_seed,
                args.order,
                args.parent_chain,
                args.parent_strategy,
                args.parent_seed,
                args.base_seed,
                strategy,
                    args.best_prefix,
                    args.row_retries,
                )
            )
    while len(jobs) < args.attempts:
        index = len(jobs)
        jobs.append(
            (
                pair,
                args.pair_index,
                args.base_seed,
                args.order,
                args.parent_chain,
                args.parent_strategy,
                args.parent_seed,
                rng.randrange(1 << 60),
                strategies[index % len(strategies)],
                args.best_prefix,
                args.row_retries,
            )
        )

    result_path = Path(args.result_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    print("targeted row reorder search for Keccak[1440,160,6,160]", flush=True)
    print(
        f"  pair={args.pair_index}, first_eq={pair.first_round_equations}, ddt2={pair.ddt2_transitions}, "
        f"base_seed={args.base_seed}, order={args.order}, best_prefix={args.best_prefix}",
        flush=True,
    )
    if args.parent_chain:
        print(f"  parent_chain={args.parent_chain}", flush=True)
    if args.parent_strategy:
        print(f"  parent={args.parent_strategy}, parent_seed={args.parent_seed}", flush=True)
    print(f"  attempts={args.attempts}, workers={args.workers}, row_retries={args.row_retries}", flush=True)
    best_result: ReorderAttempt | None = None
    start = time.time()
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_attempt, *job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            if result.error:
                print(f"  error strategy={result.strategy}: {result.error}", flush=True)
                continue
            if best_result is None or result.added_g_rows > best_result.added_g_rows:
                best_result = result
                with result_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"event": "new_best", **asdict(result)}, sort_keys=True) + "\n")
                print(
                    f"  new best after {completed}: added={result.added_g_rows}/{result.total_g_rows}, "
                    f"rank={result.rank}, dim={result.dimension}, assigned={result.assigned_sboxes}, "
                    f"failed_row={result.failed_row}, failed_deps={result.failed_deps}, "
                    f"strategy={result.strategy}, model_seed={result.model_seed}",
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
