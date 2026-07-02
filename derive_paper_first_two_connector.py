"""Derive a first 2-round connector directly from the printed 6-round collision."""

from __future__ import annotations

import random
import argparse

from connector_runner import transitions_from_pair
from incremental_connector import build_bitwise_prefix_connector, model_candidates_by_sbox
from keccak_state import round_int, rounds_int
from linear_layer import apply_l_int, apply_matrix_columns, load_or_build_matrices
from paper_collisions import TABLE18_KECCAK_1440_160_6_160
from sample_connector import sample_solution
from trail_data_6round import TRAIL_CORE_5_KECCAK_1440_160_6_160
from trail_parser import active_sboxes, parse_state_matrix
from trail_verify import sbox_value
from trail_verify import transition_stats
from beta_selector import BetaChoice
from beta0_selector import Beta0ConcreteChoice
from core3_connector import (
    FirstTwoRoundConnector,
    build_full_linearized_third_round_connector,
    build_known_value_bitwise_third_round_connector,
    trail_states,
    verify_third_round_sample,
)


def forced_models_from_message(beta0: int, alpha1: int, chi0_input: int) -> dict[int, int]:
    """Pick first-round linearization planes containing the known message value."""
    transitions = transitions_from_pair(beta0, alpha1)
    candidates = model_candidates_by_sbox(transitions)
    forced: dict[int, int] = {}
    for sbox_index, models in enumerate(candidates):
        value = sbox_value(chi0_input, sbox_index)
        for index, model in enumerate(models):
            if value in model.subset:
                forced[sbox_index] = index
                break
        else:
            raise ValueError(f"known value not covered by any model at S-box {sbox_index}")
    return forced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-third", action="store_true")
    args = parser.parse_args()

    m1 = parse_state_matrix(TABLE18_KECCAK_1440_160_6_160.m1)
    m2 = parse_state_matrix(TABLE18_KECCAK_1440_160_6_160.m2)
    alpha0 = m1 ^ m2
    beta0 = apply_l_int(alpha0)
    alpha1 = round_int(m1, 0) ^ round_int(m2, 0)
    beta1 = apply_l_int(alpha1)
    alpha2 = rounds_int(m1, 2) ^ rounds_int(m2, 2)
    expected = trail_states(TRAIL_CORE_5_KECCAK_1440_160_6_160).alpha2
    if alpha2 != expected:
        raise AssertionError("printed collision does not hit Table 11 alpha2")

    forced = forced_models_from_message(beta0, alpha1, apply_l_int(m1))
    connector, _assigned, _rows = build_bitwise_prefix_connector(
        transitions_from_pair(beta0, alpha1),
        transitions_from_pair(beta1, alpha2),
        rate=1440,
        padding_bits=1,
        seed=1,
        row_order="large",
        row_retries=1,
        forced_model_choices=forced,
    )

    _l_columns, linv_columns = load_or_build_matrices()
    verifies = False
    if not connector.system.inconsistent:
        x = sample_solution(connector.system, random.Random(1), max_basis=512)
        sample_m1 = apply_matrix_columns(linv_columns, x)
        sample_m2 = sample_m1 ^ alpha0
        verifies = (rounds_int(sample_m1, 2) ^ rounds_int(sample_m2, 2)) == alpha2

    print("paper-derived first 2-round connector")
    print(f"  alpha0 active S-boxes: {active_sboxes(alpha0)}")
    print(f"  alpha1 active S-boxes: {active_sboxes(alpha1)}")
    print(f"  alpha2 active S-boxes: {active_sboxes(alpha2)}")
    print(f"  forced first-round models: {len(forced)}")
    print(f"  added G rows: {connector.added_g_rows}/{connector.total_g_rows}")
    print(f"  rank/dimension: {connector.system.rank}/{connector.system.dimension}")
    print(f"  consistent: {not connector.system.inconsistent}")
    print(f"  sample verifies R^2 target: {verifies}")
    if connector.system.inconsistent:
        raise SystemExit(1)
    if not verifies:
        raise SystemExit(2)

    if args.build_third:
        states = trail_states(TRAIL_CORE_5_KECCAK_1440_160_6_160)
        beta1_stats = transition_stats(beta1, alpha2)
        beta0_stats = transition_stats(beta0, alpha1)
        reproduction = FirstTwoRoundConnector(
            trail=states,
            beta1_choice=BetaChoice(
                beta=beta1,
                alpha=alpha1,
                transition_weight=beta1_stats.weight,
                active_alpha_sboxes=active_sboxes(alpha1),
                compatible=beta1_stats.compatible,
            ),
            beta0_choice=Beta0ConcreteChoice(
                beta0=beta0,
                alpha0=alpha0,
                transitions=tuple(
                    (t.sbox_index, t.delta_in, t.delta_out)
                    for t in transitions_from_pair(beta0, alpha1)
                ),
                compatible=beta0_stats.compatible,
            ),
            connector=connector,
        )
        chi1_input = apply_l_int(round_int(m1, 0))
        forced_chi1 = forced_models_from_message(beta1, alpha2, chi1_input)
        third = build_full_linearized_third_round_connector(
            reproduction,
            seed=1,
            forced_model_choices=forced_chi1,
        )
        third_verifies = third.consistent and verify_third_round_sample(
            reproduction,
            third,
            seed=1,
            max_basis=512,
        )
        print("paper-derived conservative third-round connector")
        print(f"  forced chi1 models: {len(forced_chi1)}")
        print(f"  linearized chi1 S-boxes: {third.linearized_sboxes}")
        print(f"  third-round rows: {third.third_round_rows}")
        print(f"  rank/dimension: {third.system.rank}/{third.system.dimension}")
        print(f"  consistent: {third.consistent}")
        print(f"  sample verifies R^3 target: {third_verifies}")
        if not third.consistent:
            raise SystemExit(3)
        if not third_verifies:
            raise SystemExit(4)

        bitwise_third = build_known_value_bitwise_third_round_connector(
            reproduction,
            chi1_input=chi1_input,
        )
        bitwise_verifies = bitwise_third.consistent and verify_third_round_sample(
            reproduction,
            bitwise_third,
            seed=1,
            max_basis=512,
        )
        print("paper-derived bitwise third-round connector")
        print(f"  linearized chi1 S-boxes: {bitwise_third.linearized_sboxes}")
        print(f"  third-round rows: {bitwise_third.third_round_rows}")
        print(f"  rank/dimension: {bitwise_third.system.rank}/{bitwise_third.system.dimension}")
        print(f"  consistent: {bitwise_third.consistent}")
        print(f"  sample verifies R^3 target: {bitwise_verifies}")
        if not bitwise_third.consistent:
            raise SystemExit(5)
        if not bitwise_verifies:
            raise SystemExit(6)


if __name__ == "__main__":
    main()
