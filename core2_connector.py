"""Reusable reproduction entry point for Table 7 trail core No. 2."""

from __future__ import annotations

import random
from dataclasses import dataclass

from beta0_selector import Beta0ConcreteChoice, choose_best_concrete_beta0, run_beta0_difference_phase
from beta_selector import BetaChoice, search_beta_for_alpha
from connector_runner import transitions_from_pair
from incremental_connector import IncrementalConnector, build_bitwise_prefix_connector
from linear_layer import apply_matrix_columns, load_or_build_matrices
from trail_data import TRAIL_CORE_2_PARTIAL, state_from_matrix


CORE2_CONNECTOR_SEED = 665828876142873857
CORE2_CONNECTOR_ROW_ORDER = "large"
CORE2_CONNECTOR_ROW_RETRIES = 1600
CORE2_CONNECTOR_BETA0_SAMPLES = 512


@dataclass(frozen=True)
class Core2ConnectorReproduction:
    alpha2: int
    beta1_choice: BetaChoice
    beta0_choice: Beta0ConcreteChoice
    connector: IncrementalConnector


def table7_core2_alpha2() -> int:
    _, linv = load_or_build_matrices()
    beta2 = state_from_matrix(TRAIL_CORE_2_PARTIAL.beta2)
    return apply_matrix_columns(linv, beta2)


def build_reproduced_core2_connector() -> Core2ConnectorReproduction:
    alpha2 = table7_core2_alpha2()
    beta1_choice = search_beta_for_alpha(
        alpha2,
        min_active_alpha_sboxes=320,
        attempts=300,
        seed=CORE2_CONNECTOR_SEED,
        strict=True,
    )
    beta0_diff = run_beta0_difference_phase(
        beta1_choice.alpha,
        rate=1440,
        padding_bits=1,
        seed=CORE2_CONNECTOR_SEED,
    )
    beta0_choice = choose_best_concrete_beta0(
        beta0_diff,
        rng=random.Random(CORE2_CONNECTOR_SEED),
        samples=CORE2_CONNECTOR_BETA0_SAMPLES,
        max_basis=768,
    )
    connector, _assigned, _rows = build_bitwise_prefix_connector(
        transitions_from_pair(beta0_choice.beta0, beta1_choice.alpha),
        transitions_from_pair(beta1_choice.beta, alpha2),
        rate=1440,
        padding_bits=1,
        seed=CORE2_CONNECTOR_SEED,
        row_order=CORE2_CONNECTOR_ROW_ORDER,
        row_retries=CORE2_CONNECTOR_ROW_RETRIES,
    )
    if connector.system.inconsistent:
        raise RuntimeError(
            f"expected complete connector, got {connector.added_g_rows}/{connector.total_g_rows}"
        )
    return Core2ConnectorReproduction(
        alpha2=alpha2,
        beta1_choice=beta1_choice,
        beta0_choice=beta0_choice,
        connector=connector,
    )
