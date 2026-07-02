"""Try first 2-round connectors from a saved beta1/beta0 pair."""

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
from core3_connector import CORE3_CONNECTOR_ROW_ORDER, trail_states
from incremental_connector import build_bitwise_prefix_connector
from keccak_state import rounds_int
from linear_layer import apply_matrix_columns, load_or_build_matrices
from sample_connector import sample_solution
from trail_data_6round import TRAIL_CORE_5_KECCAK_1440_160_6_160


@dataclass(frozen=True)
class SavedBetaPair:
    beta1_beta: int
    beta1_alpha: int
    beta0: int
    alpha0: int
    first_round_equations: int
    ddt2_transitions: int
    ddt8_transitions: int
    seed: int
    source: str


@dataclass(frozen=True)
class PairAttempt:
    pair_index: int
    pair_first_round_equations: int
    pair_ddt2_transitions: int
    row_seed: int
    order: str
    added_g_rows: int
    total_g_rows: int
    rank: int
    dimension: int
    assigned_sboxes: int
    consistent: bool
    verifies: bool
    error: str = ""


def load_pairs(path: Path, limit: int) -> list[SavedBetaPair]:
    entries: list[dict] = []
    seen: set[tuple[str, str, str, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("event") != "new_best" and entry.get("event") != "found":
            continue
        if not all(key in entry for key in ("beta1_beta_hex", "beta1_alpha_hex", "beta0_hex", "alpha0_hex")):
            continue
        fingerprint = (
            entry["beta1_beta_hex"],
            entry["beta1_alpha_hex"],
            entry["beta0_hex"],
            entry["alpha0_hex"],
        )
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        entries.append(entry)
    if not entries:
        raise ValueError(f"no saved beta pair in {path}")
    entries.sort(
        key=lambda entry: (
            entry.get("first_round_equations", 10**9),
            entry.get("ddt2_transitions", 10**9),
            -entry.get("ddt8_transitions", 0),
        )
    )
    selected = entries[:limit]
    return [
        SavedBetaPair(
            beta1_beta=int(entry["beta1_beta_hex"], 16),
            beta1_alpha=int(entry["beta1_alpha_hex"], 16),
            beta0=int(entry["beta0_hex"], 16),
            alpha0=int(entry["alpha0_hex"], 16),
            first_round_equations=entry.get("first_round_equations", 0),
            ddt2_transitions=entry.get("ddt2_transitions", 0),
            ddt8_transitions=entry.get("ddt8_transitions", 0),
            seed=entry.get("seed", 0),
            source=json.dumps(entry, sort_keys=True),
        )
        for entry in selected
    ]


def run_attempt(
    pair: SavedBetaPair,
    pair_index: int,
    row_seed: int,
    order: str,
    row_retries: int,
) -> PairAttempt:
    try:
        _l_columns, linv_columns = load_or_build_matrices()
        alpha2 = trail_states(TRAIL_CORE_5_KECCAK_1440_160_6_160).alpha2
        connector, _assigned, _rows = build_bitwise_prefix_connector(
            transitions_from_pair(pair.beta0, pair.beta1_alpha),
            transitions_from_pair(pair.beta1_beta, alpha2),
            rate=1440,
            padding_bits=1,
            seed=row_seed,
            row_order=order,
            row_retries=row_retries,
        )
        verifies = False
        if not connector.system.inconsistent:
            x = sample_solution(connector.system, random.Random(row_seed), max_basis=512)
            m1 = apply_matrix_columns(linv_columns, x)
            m2 = m1 ^ pair.alpha0
            verifies = (rounds_int(m1, 2) ^ rounds_int(m2, 2)) == alpha2
        return PairAttempt(
            pair_index=pair_index,
            pair_first_round_equations=pair.first_round_equations,
            pair_ddt2_transitions=pair.ddt2_transitions,
            row_seed=row_seed,
            order=order,
            added_g_rows=connector.added_g_rows,
            total_g_rows=connector.total_g_rows,
            rank=connector.system.rank,
            dimension=connector.system.dimension,
            assigned_sboxes=connector.assigned_sboxes,
            consistent=not connector.system.inconsistent,
            verifies=verifies,
        )
    except Exception as exc:  # pragma: no cover - diagnostic runner
        return PairAttempt(
            pair_index=pair_index,
            pair_first_round_equations=pair.first_round_equations,
            pair_ddt2_transitions=pair.ddt2_transitions,
            row_seed=row_seed,
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
    parser.add_argument("--pair-file", default="results/core3_beta_pairs.jsonl")
    parser.add_argument("--top-pairs", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=256)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=8601)
    parser.add_argument("--row-retries", type=int, default=2400)
    parser.add_argument("--orders", default=CORE3_CONNECTOR_ROW_ORDER)
    parser.add_argument("--result-file", default="results/core3_pair_connector_search.jsonl")
    args = parser.parse_args()

    pairs = load_pairs(Path(args.pair_file), args.top_pairs)
    result_path = Path(args.result_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    print("connector row/model search from saved beta pair", flush=True)
    print(f"  pair_file={args.pair_file}", flush=True)
    print(
        f"  attempts={args.attempts}, workers={args.workers}, row_retries={args.row_retries}, "
        f"top_pairs={len(pairs)}",
        flush=True,
    )
    for index, pair in enumerate(pairs):
        print(
            f"  pair[{index}]: first_eq={pair.first_round_equations}, ddt2={pair.ddt2_transitions}, "
            f"ddt8={pair.ddt8_transitions}, seed={pair.seed}",
            flush=True,
        )

    rng = random.Random(args.seed)
    orders = [item.strip() for item in args.orders.split(",") if item.strip()]
    jobs = [
        (
            pairs[index % len(pairs)],
            index % len(pairs),
            rng.randrange(1 << 60),
            orders[index % len(orders)],
            args.row_retries,
        )
        for index in range(args.attempts)
    ]
    best: PairAttempt | None = None
    start = time.time()
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_attempt, *job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            if result.error:
                print(f"  error row_seed={result.row_seed}: {result.error}", flush=True)
                continue
            if best is None or result.added_g_rows > best.added_g_rows:
                best = result
                with result_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"event": "new_best", **asdict(result)}, sort_keys=True) + "\n")
                print(
                    f"  new best after {completed}: added={result.added_g_rows}/{result.total_g_rows}, "
                    f"rank={result.rank}, dim={result.dimension}, assigned={result.assigned_sboxes}, "
                    f"pair={result.pair_index}, first_eq={result.pair_first_round_equations}, "
                    f"ddt2={result.pair_ddt2_transitions}, order={result.order}, row_seed={result.row_seed}",
                    flush=True,
                )
            if result.consistent:
                with result_path.open("a", encoding="utf-8") as handle:
                    found_pair = pairs[result.pair_index]
                    handle.write(
                        json.dumps({"event": "found", "pair": found_pair.source, **asdict(result)}, sort_keys=True)
                        + "\n"
                    )
                print("  FOUND first 2-round connector", flush=True)
                print(f"  elapsed={time.time() - start:.1f}s", flush=True)
                print(f"  verifies R^2 target: {result.verifies}", flush=True)
                return
    print("  no connector found", flush=True)
    print(f"  elapsed={time.time() - start:.1f}s", flush=True)
    print(f"  best={best}", flush=True)


if __name__ == "__main__":
    main()
