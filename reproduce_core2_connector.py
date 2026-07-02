"""Reproduce the first successful Table 7 trail core No. 2 connector."""

from __future__ import annotations

import random

from connector_runner import transitions_from_pair
from core2_connector import (
    CORE2_CONNECTOR_BETA0_SAMPLES,
    CORE2_CONNECTOR_ROW_ORDER,
    CORE2_CONNECTOR_ROW_RETRIES,
    CORE2_CONNECTOR_SEED,
    build_reproduced_core2_connector,
)
from keccak_state import rounds_int
from linear_layer import apply_matrix_columns, load_or_build_matrices
from sample_connector import sample_solution
from trail_parser import active_sboxes


def main() -> None:
    _, linv = load_or_build_matrices()
    reproduction = build_reproduced_core2_connector()
    alpha2 = reproduction.alpha2
    beta1_choice = reproduction.beta1_choice
    beta0_choice = reproduction.beta0_choice
    connector = reproduction.connector

    x = sample_solution(connector.system, random.Random(CORE2_CONNECTOR_SEED), max_basis=512)
    message1 = apply_matrix_columns(linv, x)
    message2 = message1 ^ beta0_choice.alpha0
    output_difference = rounds_int(message1, 2) ^ rounds_int(message2, 2)
    verifies = output_difference == alpha2
    if not verifies:
        raise RuntimeError("connector solution does not verify against two Keccak rounds")

    print("Table 7 core No. 2 connector reproduced")
    print(f"  seed: {CORE2_CONNECTOR_SEED}")
    print(f"  row order: {CORE2_CONNECTOR_ROW_ORDER}, row retries: {CORE2_CONNECTOR_ROW_RETRIES}")
    print(f"  beta0 samples: {CORE2_CONNECTOR_BETA0_SAMPLES}")
    print(f"  target alpha2 active S-boxes: {active_sboxes(alpha2)}")
    print(f"  beta1 weight: {beta1_choice.transition_weight}")
    print(f"  alpha1 active S-boxes: {beta1_choice.active_alpha_sboxes}")
    print(f"  beta0 transitions: {len(transitions_from_pair(beta0_choice.beta0, beta1_choice.alpha))}")
    print(f"  first-round equations: {beta0_choice.first_round_equations}")
    print(f"  beta0 DDT 2/4/8 counts: {beta0_choice.ddt2_transitions}/{beta0_choice.ddt4_transitions}/{beta0_choice.ddt8_transitions}")
    print(f"  added G rows: {connector.added_g_rows}/{connector.total_g_rows}")
    print(f"  assigned first-round S-boxes: {connector.assigned_sboxes}")
    print(f"  rank: {connector.system.rank}")
    print(f"  dimension: {connector.system.dimension}")
    print(f"  verifies R^2(M1)+R^2(M2)=alpha2: {verifies}")
    print(f"  M1: {message1:0400x}")
    print(f"  M2: {message2:0400x}")


if __name__ == "__main__":
    main()
