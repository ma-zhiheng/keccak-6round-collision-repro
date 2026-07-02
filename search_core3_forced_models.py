"""Local forced-model repair for the 6-round first 2-round connector."""

from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from connector_runner import transitions_from_pair
from core3_connector import trail_states
from incremental_connector import (
    build_bitwise_prefix_connector,
    lifted_candidates_by_sbox,
)
from keccak_state import rounds_int
from linear_layer import apply_matrix_columns, load_or_build_matrices
from sample_connector import sample_solution
from search_core3_row_reorder import apply_chain, base_rows, load_best_reorder
from search_core3_with_beta_pair import load_pairs
from trail_data_6round import TRAIL_CORE_5_KECCAK_1440_160_6_160


@dataclass(frozen=True)
class ForcedModelAttempt:
    forced_models: dict[int, int]
    seed: int
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


def parse_forced(text: str) -> dict[int, int]:
    result: dict[int, int] = {}
    if not text:
        return result
    for item in text.split(","):
        if not item.strip():
            continue
        sbox_text, choice_text = item.split(":", 1)
        result[int(sbox_text)] = int(choice_text)
    return result


def run_attempt(
    pair_index: int,
    chain: str,
    row_seed: int,
    model_seed: int,
    row_retries: int,
    forced_models: dict[int, int],
) -> ForcedModelAttempt:
    try:
        _l_columns, linv_columns = load_or_build_matrices()
        pair = load_pairs(Path("results/core3_beta_pairs.jsonl"), max(16, pair_index + 1))[pair_index]
        alpha2 = target_alpha2()
        beta0_alpha1 = transitions_from_pair(pair.beta0, pair.beta1_alpha)
        beta1_alpha2 = transitions_from_pair(pair.beta1_beta, alpha2)
        rows = apply_chain(base_rows(pair, row_seed, "random"), chain, 292)
        connector, _assigned, _rows = build_bitwise_prefix_connector(
            beta0_alpha1,
            beta1_alpha2,
            rate=1440,
            padding_bits=1,
            seed=model_seed,
            row_order="random",
            row_retries=row_retries,
            prepared_rows=rows,
            forced_model_choices=forced_models,
        )
        verifies = False
        if not connector.system.inconsistent:
            x = sample_solution(connector.system, random.Random(model_seed), max_basis=512)
            m1 = apply_matrix_columns(linv_columns, x)
            m2 = m1 ^ pair.alpha0
            verifies = (rounds_int(m1, 2) ^ rounds_int(m2, 2)) == alpha2
        return ForcedModelAttempt(
            forced_models=forced_models,
            seed=model_seed,
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
        return ForcedModelAttempt(
            forced_models=forced_models,
            seed=model_seed,
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
    parser.add_argument("--reorder-file", default="results/core3_row_reorder.jsonl")
    parser.add_argument("--result-file", default="results/core3_forced_models.jsonl")
    parser.add_argument("--row-retries", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--extra-forced", default="")
    parser.add_argument("--max-variable-sboxes", type=int, default=4)
    args = parser.parse_args()

    best = load_best_reorder(Path(args.reorder_file))
    pair = load_pairs(Path("results/core3_beta_pairs.jsonl"), max(16, best.pair_index + 1))[best.pair_index]
    alpha2 = target_alpha2()
    beta0_alpha1 = transitions_from_pair(pair.beta0, pair.beta1_alpha)
    beta1_alpha2 = transitions_from_pair(pair.beta1_beta, alpha2)
    rows = apply_chain(base_rows(pair, best.base_seed, "random"), best.chain, best.added_g_rows)
    baseline, _assigned, _rows = build_bitwise_prefix_connector(
        beta0_alpha1,
        beta1_alpha2,
        rate=1440,
        padding_bits=1,
        seed=best.model_seed,
        row_order="random",
        row_retries=args.row_retries,
        prepared_rows=rows,
    )
    candidates = lifted_candidates_by_sbox(beta0_alpha1)
    failed_deps = list(baseline.failed_sboxes)
    variable_sboxes = [sbox for sbox in failed_deps if len(candidates[sbox]) > 1]
    variable_sboxes = variable_sboxes[: args.max_variable_sboxes]
    fixed = parse_forced(args.extra_forced)
    seed = best.model_seed if args.seed is None else args.seed
    result_path = Path(args.result_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    print("forced-model local repair for Keccak[1440,160,6,160]", flush=True)
    print(
        f"  best={best.added_g_rows}/{best.total_g_rows}, pair={best.pair_index}, "
        f"seed={seed}, failed_row={baseline.failed_row}, failed_deps={failed_deps}",
        flush=True,
    )
    print(f"  variable_sboxes={[(s, len(candidates[s])) for s in variable_sboxes]}", flush=True)
    print(f"  fixed={fixed}", flush=True)

    domains = [range(len(candidates[sbox])) for sbox in variable_sboxes]
    best_result: ForcedModelAttempt | None = None
    start = time.time()
    for indexes in itertools.product(*domains):
        forced = dict(fixed)
        forced.update(dict(zip(variable_sboxes, indexes)))
        result = run_attempt(
            best.pair_index,
            best.chain,
            best.base_seed,
            seed,
            args.row_retries,
            forced,
        )
        if result.error:
            print(f"  error forced={forced}: {result.error}", flush=True)
            continue
        if best_result is None or (result.added_g_rows, result.rank) > (best_result.added_g_rows, best_result.rank):
            best_result = result
            with result_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": "new_best", **asdict(result)}, sort_keys=True) + "\n")
            print(
                f"  new best: added={result.added_g_rows}/{result.total_g_rows}, "
                f"rank={result.rank}, dim={result.dimension}, failed_row={result.failed_row}, "
                f"forced={forced}",
                flush=True,
            )
        if result.consistent:
            with result_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": "found", **asdict(result)}, sort_keys=True) + "\n")
            print("  FOUND first 2-round connector", flush=True)
            print(f"  elapsed={time.time() - start:.1f}s verifies={result.verifies}", flush=True)
            return
    print("  no connector found", flush=True)
    print(f"  elapsed={time.time() - start:.1f}s best={best_result}", flush=True)


if __name__ == "__main__":
    main()
