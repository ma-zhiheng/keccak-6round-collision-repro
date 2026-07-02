"""Entry point for staged 6-round connector reproduction."""

from __future__ import annotations

import argparse

from core3_connector import (
    CORE3_CONNECTOR_BETA0_SAMPLES,
    CORE3_CONNECTOR_ROW_ORDER,
    CORE3_CONNECTOR_ROW_RETRIES,
    CORE3_CONNECTOR_SEED,
    build_first_two_round_connector,
    build_full_linearized_third_round_connector,
    expected_seconds,
    verify_first_two_round_sample,
    verify_third_round_sample,
)
from trail_data_6round import TRAIL_CORE_5_KECCAK_1440_160_6_160, summarize_core


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-first-two", action="store_true")
    parser.add_argument("--build-third", action="store_true")
    parser.add_argument("--seed", type=int, default=CORE3_CONNECTOR_SEED)
    parser.add_argument("--beta-attempts", type=int, default=1000)
    parser.add_argument("--beta1-candidates", type=int, default=1)
    parser.add_argument("--beta0-samples", type=int, default=CORE3_CONNECTOR_BETA0_SAMPLES)
    parser.add_argument("--row-retries", type=int, default=CORE3_CONNECTOR_ROW_RETRIES)
    parser.add_argument("--pairs-per-second", type=float, default=1.0e9)
    args = parser.parse_args()

    print("Keccak[1440,160,6,160] staged reproduction")
    print(f"  seed={args.seed}")
    print(f"  row_order={CORE3_CONNECTOR_ROW_ORDER}, row_retries={args.row_retries}")
    print(f"  beta0_samples={args.beta0_samples}")
    print(f"  expected 2^47.81 at {args.pairs_per_second:.3g}/s: {expected_seconds(47.81, args.pairs_per_second) / 3600:.2f} h")
    print(f"  paper actual 2^49.07 at {args.pairs_per_second:.3g}/s: {expected_seconds(49.07, args.pairs_per_second) / 3600:.2f} h")

    trail_ok = summarize_core(TRAIL_CORE_5_KECCAK_1440_160_6_160)
    if not trail_ok:
        print("trail transcription is not verified yet; skip connector build")
        raise SystemExit(1)

    if not args.build_first_two:
        print("trail verified; pass --build-first-two to construct the first connector")
        return

    reproduction = build_first_two_round_connector(
        TRAIL_CORE_5_KECCAK_1440_160_6_160,
        seed=args.seed,
        beta_attempts=args.beta_attempts,
        beta1_candidates=args.beta1_candidates,
        beta0_samples=args.beta0_samples,
        row_retries=args.row_retries,
    )
    connector = reproduction.connector
    print("first 2-round connector")
    print(f"  beta1 weight: {reproduction.beta1_choice.transition_weight}")
    print(f"  alpha1 active S-boxes: {reproduction.beta1_choice.active_alpha_sboxes}")
    print(f"  beta0 transitions: {len(reproduction.beta0_choice.transitions)}")
    print(f"  added G rows: {connector.added_g_rows}/{connector.total_g_rows}")
    print(f"  assigned first-round S-boxes: {connector.assigned_sboxes}")
    print(f"  rank: {connector.system.rank}")
    print(f"  dimension: {connector.system.dimension}")
    print(f"  inconsistent: {connector.system.inconsistent}")
    if connector.system.inconsistent:
        raise SystemExit(1)
    print(f"  direct R^2 sample verifies alpha2: {verify_first_two_round_sample(reproduction)}")

    if args.build-third:
        third = build_full_linearized_third_round_connector(reproduction, seed=args.seed)
        print("conservative third-round connector")
        print(f"  linearized chi1 S-boxes: {third.linearized_sboxes}")
        print(f"  third-round rows: {third.third_round_rows}")
        print(f"  rank: {third.system.rank}")
        print(f"  dimension: {third.system.dimension}")
        print(f"  consistent: {third.consistent}")
        if third.consistent:
            print(f"  direct R^3 sample verifies alpha3: {verify_third_round_sample(reproduction, third)}")


if __name__ == "__main__":
    main()
