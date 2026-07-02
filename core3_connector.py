"""Connector scaffolding for the 6-round Keccak[1440,160,6,160] attack."""

from __future__ import annotations

import random
from dataclasses import dataclass

from beta0_selector import Beta0ConcreteChoice, choose_best_concrete_beta0, run_beta0_difference_phase
from beta_selector import BetaChoice, choose_beta_for_alpha, search_beta_for_alpha
from connector_runner import transitions_from_pair
from connector_equations import second_round_g_equations, substitute_second_round_equation
from gf2 import GF2System
from incremental_connector import IncrementalConnector, build_bitwise_prefix_connector
from keccak_state import rounds_int
from linear_layer import apply_matrix_columns, load_or_build_matrices, round_constant_state, transpose_apply
from sample_connector import sample_solution
from sbox_constraints import all_two_dim_linearizable_models, linearizable_models_for_transition
from state_lift import SBOXES_1600, bit_position, lift_model, sbox_index_to_yz
from trail_data_6round import SixRoundTrailCoreData, state_from_matrix


CORE3_CONNECTOR_SEED = 917306210421
CORE3_CONNECTOR_ROW_ORDER = "large"
CORE3_CONNECTOR_ROW_RETRIES = 2400
CORE3_CONNECTOR_BETA0_SAMPLES = 1024


@dataclass(frozen=True)
class SixRoundTrailStates:
    beta2: int
    beta3: int
    beta4: int
    alpha2: int
    alpha3: int
    alpha4: int


@dataclass(frozen=True)
class FirstTwoRoundConnector:
    trail: SixRoundTrailStates
    beta1_choice: BetaChoice
    beta0_choice: Beta0ConcreteChoice
    connector: IncrementalConnector


@dataclass(frozen=True)
class ThirdRoundConnector:
    system: GF2System
    linearized_sboxes: int
    third_round_rows: int
    consistent: bool


def trail_states(core: SixRoundTrailCoreData) -> SixRoundTrailStates:
    _l_columns, linv_columns = load_or_build_matrices()
    beta2 = state_from_matrix(core.beta2)
    beta3 = state_from_matrix(core.beta3)
    beta4 = state_from_matrix(core.beta4)
    return SixRoundTrailStates(
        beta2=beta2,
        beta3=beta3,
        beta4=beta4,
        alpha2=apply_matrix_columns(linv_columns, beta2),
        alpha3=apply_matrix_columns(linv_columns, beta3),
        alpha4=apply_matrix_columns(linv_columns, beta4),
    )


def build_first_two_round_connector(
    core: SixRoundTrailCoreData,
    seed: int = CORE3_CONNECTOR_SEED,
    beta_attempts: int = 1000,
    beta0_samples: int = CORE3_CONNECTOR_BETA0_SAMPLES,
    row_retries: int = CORE3_CONNECTOR_ROW_RETRIES,
    beta1_candidates: int = 1,
) -> FirstTwoRoundConnector:
    """Build the first 2-round connector for the 6-round attack.

    This is the deterministic base of the adaptive 3-round connector. The next
    step will re-express the returned equation system over the second round's
    input and add non-full linearizations for the third round.
    """
    states = trail_states(core)
    beta1_choice, beta0_choice = choose_beta1_beta0_pair(
        states.alpha2,
        seed=seed,
        beta_attempts=beta_attempts,
        beta1_candidates=beta1_candidates,
        beta0_samples=beta0_samples,
    )
    connector, _assigned, _rows = build_bitwise_prefix_connector(
        transitions_from_pair(beta0_choice.beta0, beta1_choice.alpha),
        transitions_from_pair(beta1_choice.beta, states.alpha2),
        rate=1440,
        padding_bits=1,
        seed=seed,
        row_order=CORE3_CONNECTOR_ROW_ORDER,
        row_retries=row_retries,
    )
    return FirstTwoRoundConnector(
        trail=states,
        beta1_choice=beta1_choice,
        beta0_choice=beta0_choice,
        connector=connector,
    )


def choose_beta1_beta0_pair(
    alpha2: int,
    seed: int,
    beta_attempts: int,
    beta1_candidates: int,
    beta0_samples: int,
    min_active_alpha_sboxes: int = 320,
    beta0_basis: int = 768,
) -> tuple[BetaChoice, Beta0ConcreteChoice]:
    """Pick a beta1/beta0 pair, scoring beta0 value-phase cost.

    The old 5-round code accepted the first beta1 whose alpha1 was fully
    active. For the 6-round connector this choice matters more, so we can test
    several beta1 candidates and keep the one whose sampled beta0 induces fewer
    first-round equations.
    """
    if beta1_candidates <= 1:
        beta1_choice = search_beta_for_alpha(
            alpha2,
            min_active_alpha_sboxes=min_active_alpha_sboxes,
            attempts=beta_attempts,
            seed=seed,
            strict=True,
        )
        beta0_diff = run_beta0_difference_phase(
            beta1_choice.alpha,
            rate=1440,
            padding_bits=1,
            seed=seed,
        )
        beta0_choice = choose_best_concrete_beta0(
            beta0_diff,
            rng=random.Random(seed),
            samples=beta0_samples,
            max_basis=beta0_basis,
        )
        return beta1_choice, beta0_choice

    rng = random.Random(seed)
    best: tuple[tuple[int, int, int, int], BetaChoice, Beta0ConcreteChoice] | None = None
    attempts = 0
    accepted = 0
    while attempts < beta_attempts and accepted < beta1_candidates:
        attempts += 1
        beta1_choice = choose_beta_for_alpha(alpha2, rng=rng, randomize_ties=True)
        if beta1_choice.active_alpha_sboxes < min_active_alpha_sboxes:
            continue
        accepted += 1
        beta0_diff = run_beta0_difference_phase(
            beta1_choice.alpha,
            rate=1440,
            padding_bits=1,
            seed=seed ^ (accepted * 0x9E3779B97F4A7C15),
        )
        beta0_choice = choose_best_concrete_beta0(
            beta0_diff,
            rng=rng,
            samples=beta0_samples,
            max_basis=beta0_basis,
        )
        key = (
            not beta0_choice.compatible,
            beta0_choice.first_round_equations,
            beta0_choice.ddt2_transitions,
            -beta0_choice.ddt8_transitions,
        )
        if best is None or key < best[0]:
            best = (key, beta1_choice, beta0_choice)
    if best is None:
        raise RuntimeError(
            "failed to find beta with "
            f"{min_active_alpha_sboxes} active alpha S-boxes in {beta_attempts} attempts"
        )
    return best[1], best[2]


def verify_first_two_round_sample(
    reproduction: FirstTwoRoundConnector,
    seed: int = 1,
    max_basis: int | None = 96,
) -> bool:
    """Sample one connector solution and directly check the first two rounds."""
    _l_columns, linv_columns = load_or_build_matrices()
    x = sample_solution(reproduction.connector.system, random.Random(seed), max_basis=max_basis)
    message1 = apply_matrix_columns(linv_columns, x)
    message2 = message1 ^ reproduction.beta0_choice.alpha0
    return (rounds_int(message1, 2) ^ rounds_int(message2, 2)) == reproduction.trail.alpha2


def _z_affine_in_x(
    z_mask: int,
    first_connector: IncrementalConnector,
    l_columns: list[int],
) -> tuple[int, int]:
    coeff, rhs = substitute_second_round_equation(
        z_mask,
        0,
        list(first_connector.y_coeffs),
        list(first_connector.y_consts),
        l_columns,
        round_constant_state(0),
    )
    return coeff, rhs


def _model_candidates_for_chi1(
    beta1_alpha2: list,
) -> list:
    by_index = {transition.sbox_index: transition for transition in beta1_alpha2}
    inactive = all_two_dim_linearizable_models()
    candidates = []
    for sbox_index in range(SBOXES_1600):
        transition = by_index.get(sbox_index)
        if transition is None or (transition.delta_in == 0 and transition.delta_out == 0):
            models = inactive
        else:
            models = linearizable_models_for_transition(transition.delta_in, transition.delta_out)
        if not models:
            raise ValueError(f"no chi1 model candidates for S-box {sbox_index}")
        candidates.append(models)
    return candidates


def build_full_linearized_third_round_connector(
    reproduction: FirstTwoRoundConnector,
    seed: int = CORE3_CONNECTOR_SEED,
    forced_model_choices: dict[int, int] | None = None,
) -> ThirdRoundConnector:
    """Add a conservative third-round connector layer.

    This is intentionally stricter than the paper's adaptive non-full
    linearization. It is useful as an executable correctness scaffold: whenever
    it returns a non-empty system, direct execution must reach `alpha3`.
    """
    rng = random.Random(seed)
    l_columns, _linv_columns = load_or_build_matrices()
    system = reproduction.connector.system.copy()
    beta1_alpha2 = transitions_from_pair(reproduction.beta1_choice.beta, reproduction.trail.alpha2)
    candidates = _model_candidates_for_chi1(beta1_alpha2)
    forced_model_choices = forced_model_choices or {}

    third_rows = second_round_g_equations(
        transitions_from_pair(reproduction.trail.beta2, reproduction.trail.alpha3)
    )
    needed_sboxes: set[int] = set()
    prepared_y_masks: list[tuple[int, int, int]] = []
    for zprime_mask, rhs in third_rows:
        yprime_mask = transpose_apply(l_columns, zprime_mask)
        prepared_y_masks.append((zprime_mask, rhs, yprime_mask))
        value = yprime_mask
        while value:
            low = value & -value
            bit = low.bit_length() - 1
            lane = bit // 64
            z = bit % 64
            y = lane // 5
            needed_sboxes.add(y * 64 + z)
            value ^= low

    assigned = {}
    for sbox_index in sorted(needed_sboxes):
        if sbox_index in forced_model_choices:
            order = [candidates[sbox_index][forced_model_choices[sbox_index] % len(candidates[sbox_index])]]
        else:
            order = list(candidates[sbox_index])
            rng.shuffle(order)
        for model in order:
            lifted = lift_model(sbox_index, model)
            equations = []
            for z_mask, rhs in lifted.input_equations:
                coeff, const = _z_affine_in_x(z_mask, reproduction.connector, l_columns)
                equations.append((coeff, rhs ^ const))
            if system.add_equations(equations):
                assigned[sbox_index] = lifted
                break
        else:
            system.inconsistent = True
            return ThirdRoundConnector(
                system=system,
                linearized_sboxes=len(assigned),
                third_round_rows=len(third_rows),
                consistent=False,
            )

    for zprime_mask, rhs, yprime_mask in prepared_y_masks:
        coeff = 0
        const = (zprime_mask & apply_matrix_columns(l_columns, round_constant_state(1))).bit_count() & 1
        value = yprime_mask
        while value:
            low = value & -value
            bit = low.bit_length() - 1
            lane = bit // 64
            local_bit = lane % 5
            z = bit % 64
            y = lane // 5
            sbox_index = y * 64 + z
            lifted = assigned[sbox_index]
            z_expr_mask = lifted.output_masks[local_bit]
            z_expr_const = lifted.output_constants[local_bit]
            z_coeff, z_const = _z_affine_in_x(z_expr_mask, reproduction.connector, l_columns)
            coeff ^= z_coeff
            const ^= z_const ^ z_expr_const
            value ^= low
        if not system.add_equation(coeff, rhs ^ const):
            return ThirdRoundConnector(
                system=system,
                linearized_sboxes=len(assigned),
                third_round_rows=len(third_rows),
                consistent=False,
            )

    return ThirdRoundConnector(
        system=system,
        linearized_sboxes=len(assigned),
        third_round_rows=len(third_rows),
        consistent=not system.inconsistent,
    )


def _state_bit_for_sbox_local(sbox_index: int, local_bit: int) -> int:
    y, z = sbox_index_to_yz(sbox_index)
    return bit_position(local_bit, y, z)


def build_known_value_bitwise_third_round_connector(
    reproduction: FirstTwoRoundConnector,
    chi1_input: int,
) -> ThirdRoundConnector:
    """Add the third-round connector with output-bit non-full linearization.

    This follows the paper's spirit more closely than full S-box
    linearization: for a needed chi output bit
    y_j = z_j + (1 + z_{j+1}) z_{j+2}, fix only one neighboring input bit
    to its known value from the printed collision, making that output bit
    affine while preserving more degrees of freedom.
    """
    l_columns, _linv_columns = load_or_build_matrices()
    system = reproduction.connector.system.copy()
    third_rows = second_round_g_equations(
        transitions_from_pair(reproduction.trail.beta2, reproduction.trail.alpha3)
    )

    z_cache: dict[int, tuple[int, int]] = {}
    output_cache: dict[tuple[int, int], tuple[int, int]] = {}
    linearized_sboxes: set[int] = set()

    def z_expr(state_bit: int) -> tuple[int, int]:
        cached = z_cache.get(state_bit)
        if cached is None:
            cached = _z_affine_in_x(1 << state_bit, reproduction.connector, l_columns)
            z_cache[state_bit] = cached
        return cached

    def force_bit(state_bit: int, value: int, trial_system: GF2System) -> bool:
        coeff, const = z_expr(state_bit)
        return trial_system.add_equation(coeff, value ^ const)

    def output_expr(sbox_index: int, local_bit: int) -> tuple[int, int] | None:
        key = (sbox_index, local_bit)
        cached = output_cache.get(key)
        if cached is not None:
            return cached

        bit_j = _state_bit_for_sbox_local(sbox_index, local_bit)
        bit_a = _state_bit_for_sbox_local(sbox_index, (local_bit + 1) % 5)
        bit_b = _state_bit_for_sbox_local(sbox_index, (local_bit + 2) % 5)
        value_a = (chi1_input >> bit_a) & 1
        value_b = (chi1_input >> bit_b) & 1
        coeff_j, const_j = z_expr(bit_j)
        coeff_a, const_a = z_expr(bit_a)
        coeff_b, const_b = z_expr(bit_b)

        # Candidate 1: fix z_{j+1}=a, then y_j = z_j + (1+a) z_{j+2}.
        if value_a == 0:
            expr_a = (coeff_j ^ coeff_b, const_j ^ const_b)
        else:
            expr_a = (coeff_j, const_j)

        # Candidate 2: fix z_{j+2}=b, then y_j = z_j + b + b z_{j+1}.
        if value_b == 0:
            expr_b = (coeff_j, const_j)
        else:
            expr_b = (coeff_j ^ coeff_a, const_j ^ const_a ^ 1)

        candidates = [(bit_a, value_a, expr_a), (bit_b, value_b, expr_b)]
        best: tuple[int, GF2System, tuple[int, int]] | None = None
        for forced_bit, forced_value, expr in candidates:
            trial = system.copy()
            before = trial.rank
            if not force_bit(forced_bit, forced_value, trial):
                continue
            cost = trial.rank - before
            if best is None or cost < best[0]:
                best = (cost, trial, expr)
        if best is None:
            system.inconsistent = True
            return None
        system.rows = best[1].rows
        system.inconsistent = best[1].inconsistent
        output_cache[key] = best[2]
        linearized_sboxes.add(sbox_index)
        return best[2]

    for zprime_mask, rhs in third_rows:
        yprime_mask = transpose_apply(l_columns, zprime_mask)
        coeff = 0
        const = (zprime_mask & apply_matrix_columns(l_columns, round_constant_state(1))).bit_count() & 1
        value = yprime_mask
        while value:
            low = value & -value
            bit = low.bit_length() - 1
            lane = bit // 64
            local_bit = lane % 5
            z = bit % 64
            y = lane // 5
            sbox_index = y * 64 + z
            expr = output_expr(sbox_index, local_bit)
            if expr is None:
                return ThirdRoundConnector(
                    system=system,
                    linearized_sboxes=len(linearized_sboxes),
                    third_round_rows=len(third_rows),
                    consistent=False,
                )
            coeff ^= expr[0]
            const ^= expr[1]
            value ^= low
        if not system.add_equation(coeff, rhs ^ const):
            return ThirdRoundConnector(
                system=system,
                linearized_sboxes=len(linearized_sboxes),
                third_round_rows=len(third_rows),
                consistent=False,
            )

    return ThirdRoundConnector(
        system=system,
        linearized_sboxes=len(linearized_sboxes),
        third_round_rows=len(third_rows),
        consistent=not system.inconsistent,
    )


def verify_third_round_sample(
    reproduction: FirstTwoRoundConnector,
    third: ThirdRoundConnector,
    seed: int = 1,
    max_basis: int | None = 96,
) -> bool:
    _l_columns, linv_columns = load_or_build_matrices()
    x = sample_solution(third.system, random.Random(seed), max_basis=max_basis)
    message1 = apply_matrix_columns(linv_columns, x)
    message2 = message1 ^ reproduction.beta0_choice.alpha0
    return (rounds_int(message1, 3) ^ rounds_int(message2, 3)) == reproduction.trail.alpha3


def expected_seconds(pair_exponent: float, pairs_per_second: float) -> float:
    return (2.0 ** pair_exponent) / pairs_per_second
