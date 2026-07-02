"""Search beta1/beta0 pairs before attempting the full connector."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from core3_connector import choose_beta1_beta0_pair, trail_states
from trail_data_6round import TRAIL_CORE_5_KECCAK_1440_160_6_160


@dataclass(frozen=True)
class BetaPairResult:
    seed: int
    first_round_equations: int
    ddt2_transitions: int
    ddt8_transitions: int
    beta1_weight: int
    alpha1_active_sboxes: int
    beta1_beta_hex: str
    beta1_alpha_hex: str
    beta0_hex: str
    alpha0_hex: str
    error: str = ""


def run_seed(
    seed: int,
    beta_attempts: int,
    beta1_candidates: int,
    beta0_samples: int,
    beta0_basis: int,
) -> BetaPairResult:
    try:
        alpha2 = trail_states(TRAIL_CORE_5_KECCAK_1440_160_6_160).alpha2
        beta1_choice, beta0_choice = choose_beta1_beta0_pair(
            alpha2,
            seed=seed,
            beta_attempts=beta_attempts,
            beta1_candidates=beta1_candidates,
            beta0_samples=beta0_samples,
            beta0_basis=beta0_basis,
        )
        return BetaPairResult(
            seed=seed,
            first_round_equations=beta0_choice.first_round_equations,
            ddt2_transitions=beta0_choice.ddt2_transitions,
            ddt8_transitions=beta0_choice.ddt8_transitions,
            beta1_weight=beta1_choice.transition_weight,
            alpha1_active_sboxes=beta1_choice.active_alpha_sboxes,
            beta1_beta_hex=f"{beta1_choice.beta:x}",
            beta1_alpha_hex=f"{beta1_choice.alpha:x}",
            beta0_hex=f"{beta0_choice.beta0:x}",
            alpha0_hex=f"{beta0_choice.alpha0:x}",
        )
    except Exception as exc:  # pragma: no cover - diagnostic runner
        return BetaPairResult(
            seed=seed,
            first_round_equations=0,
            ddt2_transitions=0,
            ddt8_transitions=0,
            beta1_weight=0,
            alpha1_active_sboxes=0,
            beta1_beta_hex="",
            beta1_alpha_hex="",
            beta0_hex="",
            alpha0_hex="",
            error=repr(exc),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempts", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=5103)
    parser.add_argument("--beta-attempts", type=int, default=1500)
    parser.add_argument("--beta1-candidates", type=int, default=8)
    parser.add_argument("--beta0-samples", type=int, default=512)
    parser.add_argument("--beta0-basis", type=int, default=768)
    parser.add_argument("--result-file", default="results/core3_beta_pairs.jsonl")
    args = parser.parse_args()

    result_path = Path(args.result_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    best: BetaPairResult | None = None
    start = time.time()
    print("beta1/beta0 pair search for Keccak[1440,160,6,160]", flush=True)
    print(
        f"  attempts={args.attempts}, workers={args.workers}, beta_attempts={args.beta_attempts}, "
        f"beta1_candidates={args.beta1_candidates}, beta0_samples={args.beta0_samples}",
        flush=True,
    )

    seeds = [rng.randrange(1 << 60) for _ in range(args.attempts)]

    def handle_result(index: int, result: BetaPairResult) -> None:
        nonlocal best
        result = run_seed(
            result.seed,
            args.beta_attempts,
            args.beta1_candidates,
            args.beta0_samples,
            args.beta0_basis,
        )
        if result.error:
            print(f"  error seed={result.seed}: {result.error}", flush=True)
            return
        if (
            best is None
            or result.first_round_equations < best.first_round_equations
            or (
                result.first_round_equations == best.first_round_equations
                and result.ddt2_transitions < best.ddt2_transitions
            )
        ):
            best = result
            with result_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"event": "new_best", **asdict(result)}, sort_keys=True) + "\n")
            print(
                f"  new best at {index}: first_eq={result.first_round_equations}, "
                f"ddt2={result.ddt2_transitions}, ddt8={result.ddt8_transitions}, "
                f"beta1_weight={result.beta1_weight}, seed={result.seed}",
                flush=True,
            )
        if index % 25 == 0:
            print(f"  {index} attempts, elapsed={time.time() - start:.1f}s, best={best}", flush=True)

    if args.workers <= 1:
        for index, seed in enumerate(seeds, start=1):
            handle_result(
                index,
                BetaPairResult(
                    seed=seed,
                    first_round_equations=0,
                    ddt2_transitions=0,
                    ddt8_transitions=0,
                    beta1_weight=0,
                    alpha1_active_sboxes=0,
                    beta1_beta_hex="",
                    beta1_alpha_hex="",
                    beta0_hex="",
                    alpha0_hex="",
                ),
            )
    else:
        workers = min(args.workers, os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    run_seed,
                    seed,
                    args.beta_attempts,
                    args.beta1_candidates,
                    args.beta0_samples,
                    args.beta0_basis,
                ): seed
                for seed in seeds
            }
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                if result.error:
                    print(f"  error seed={futures[future]}: {result.error}", flush=True)
                    continue
                if (
                    best is None
                    or result.first_round_equations < best.first_round_equations
                    or (
                        result.first_round_equations == best.first_round_equations
                        and result.ddt2_transitions < best.ddt2_transitions
                    )
                ):
                    best = result
                    with result_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"event": "new_best", **asdict(result)}, sort_keys=True) + "\n")
                    print(
                        f"  new best after {index}: first_eq={result.first_round_equations}, "
                        f"ddt2={result.ddt2_transitions}, ddt8={result.ddt8_transitions}, "
                        f"beta1_weight={result.beta1_weight}, seed={result.seed}",
                        flush=True,
                    )
                if index % 25 == 0:
                    print(f"  {index} attempts, elapsed={time.time() - start:.1f}s, best={best}", flush=True)
    print(f"done elapsed={time.time() - start:.1f}s best={best}", flush=True)


if __name__ == "__main__":
    main()
