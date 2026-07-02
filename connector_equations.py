"""Equation construction for the Section 4.2 two-round connector."""

from __future__ import annotations

from dataclasses import dataclass

from gf2 import GF2System
from linear_layer import (
    STATE_BITS,
    apply_l_int,
    load_or_build_matrices,
    preimage_bit_equation,
    round_constant_state,
    transpose_apply,
)
from sbox_constraints import (
    AffineSBoxModel,
    all_two_dim_linearizable_models,
    linearizable_models_for_transition,
    value_constraints_for_transition,
)
from state_lift import SBOXES_1600, lift_local_mask, lift_model, state_positions_for_sbox


Equation = tuple[int, int]


@dataclass(frozen=True)
class SBoxTransition:
    sbox_index: int
    delta_in: int
    delta_out: int


@dataclass(frozen=True)
class ConnectorSystem:
    system: GF2System
    y_coeffs: tuple[int, ...]
    y_consts: tuple[int, ...]
    first_round_equations: int
    second_round_equations: int
    fixed_bit_equations: int


def choose_first_round_models(
    beta0_alpha1: list[SBoxTransition],
    choices: list[int] | None = None,
) -> list[AffineSBoxModel]:
    """Choose one linearizable model for every first-round S-box.

    For active S-boxes the model must satisfy the specified differential
    transition. For non-active S-boxes we still restrict values to an arbitrary
    two-dimensional linearizable plane, because the whole first chi layer must
    be linearized.
    """
    by_index = {transition.sbox_index: transition for transition in beta0_alpha1}
    if choices is None:
        choices = [0] * SBOXES_1600
    if len(choices) != SBOXES_1600:
        raise ValueError("choices must contain 320 entries")

    inactive_models = all_two_dim_linearizable_models()
    selected: list[AffineSBoxModel] = []
    for sbox_index in range(SBOXES_1600):
        transition = by_index.get(sbox_index)
        if transition is None or (transition.delta_in == 0 and transition.delta_out == 0):
            models = inactive_models
        else:
            models = linearizable_models_for_transition(
                transition.delta_in,
                transition.delta_out,
            )
        if not models:
            raise ValueError(f"no first-round model for S-box {sbox_index}")
        selected.append(models[choices[sbox_index] % len(models)])
    return selected


def build_y_affine(models: list[AffineSBoxModel]) -> tuple[list[int], list[int], list[Equation]]:
    """Build y=chi_L*x+chi_C plus A*x=t from selected S-box models."""
    y_coeffs = [0] * STATE_BITS
    y_consts = [0] * STATE_BITS
    equations: list[Equation] = []

    for sbox_index, model in enumerate(models):
        lifted = lift_model(sbox_index, model)
        equations.extend(lifted.input_equations)
        positions = state_positions_for_sbox(sbox_index)
        for local_bit, out_bit in enumerate(positions):
            y_coeffs[out_bit] = lifted.output_masks[local_bit]
            y_consts[out_bit] = lifted.output_constants[local_bit]

    return y_coeffs, y_consts, equations


def second_round_g_equations(beta1_alpha2: list[SBoxTransition]) -> list[Equation]:
    equations: list[Equation] = []
    for transition in beta1_alpha2:
        if transition.delta_in == 0 and transition.delta_out == 0:
            continue
        local = value_constraints_for_transition(transition.delta_in, transition.delta_out)
        if not local:
            raise ValueError(f"incompatible second-round transition {transition}")
        for local_mask, rhs in local:
            equations.append((lift_local_mask(transition.sbox_index, local_mask), rhs))
    return equations


def substitute_second_round_equation(
    z_mask: int,
    rhs: int,
    y_coeffs: list[int],
    y_consts: list[int],
    l_columns: list[int],
    rc_after_chi0: int,
) -> Equation:
    """Substitute z=L(y+RC[0]) and y=chi_L*x+chi_C into one G row."""
    y_mask = transpose_apply(l_columns, z_mask)
    rc_const = (z_mask & apply_l_int(rc_after_chi0)).bit_count() & 1

    coeff_x = 0
    const = rc_const
    value = y_mask
    while value:
        low = value & -value
        y_bit = low.bit_length() - 1
        coeff_x ^= y_coeffs[y_bit]
        const ^= y_consts[y_bit]
        value ^= low

    return coeff_x, rhs ^ const


def fixed_initial_bit_equations(rate: int, padding_bits: int, linv_columns: list[int]) -> list[Equation]:
    """Equations for L^-1(x)'s fixed padding/capacity bits."""
    equations: list[Equation] = []
    for bit in range(rate - padding_bits, rate):
        equations.append((preimage_bit_equation(linv_columns, bit), 1))
    for bit in range(rate, STATE_BITS):
        equations.append((preimage_bit_equation(linv_columns, bit), 0))
    return equations


def build_connector_system(
    beta0_alpha1: list[SBoxTransition],
    beta1_alpha2: list[SBoxTransition],
    rate: int,
    padding_bits: int,
    choices: list[int] | None = None,
) -> ConnectorSystem:
    """Build EM for a fixed set of first/second round differential choices."""
    l_columns, linv_columns = load_or_build_matrices()
    models = choose_first_round_models(beta0_alpha1, choices)
    y_coeffs, y_consts, first_equations = build_y_affine(models)
    second_g = second_round_g_equations(beta1_alpha2)
    fixed = fixed_initial_bit_equations(rate, padding_bits, linv_columns)

    system = GF2System(STATE_BITS)
    if not system.add_equations(first_equations):
        system.inconsistent = True
        return ConnectorSystem(
            system=system,
            y_coeffs=tuple(y_coeffs),
            y_consts=tuple(y_consts),
            first_round_equations=len(first_equations),
            second_round_equations=0,
            fixed_bit_equations=len(fixed),
        )
    if not system.add_equations(fixed):
        system.inconsistent = True
        return ConnectorSystem(
            system=system,
            y_coeffs=tuple(y_coeffs),
            y_consts=tuple(y_consts),
            first_round_equations=len(first_equations),
            second_round_equations=0,
            fixed_bit_equations=len(fixed),
        )

    substituted = [
        substitute_second_round_equation(
            z_mask,
            rhs,
            y_coeffs,
            y_consts,
            l_columns,
            round_constant_state(0),
        )
        for z_mask, rhs in second_g
    ]
    if not system.add_equations(substituted):
        system.inconsistent = True

    return ConnectorSystem(
        system=system,
        y_coeffs=tuple(y_coeffs),
        y_consts=tuple(y_consts),
        first_round_equations=len(first_equations),
        second_round_equations=len(substituted),
        fixed_bit_equations=len(fixed),
    )


def demo() -> None:
    # A tiny artificial setup: not a real paper trail, but exercises the full
    # Section 4.2 equation assembly path.
    beta0_alpha1 = [
        SBoxTransition(0, 0x03, 0x02),
        SBoxTransition(1, 0x01, 0x01),
    ]
    beta1_alpha2 = [
        SBoxTransition(0, 0x03, 0x02),
        SBoxTransition(1, 0x01, 0x01),
    ]
    connector = build_connector_system(
        beta0_alpha1=beta0_alpha1,
        beta1_alpha2=beta1_alpha2,
        rate=1440,
        padding_bits=1,
    )
    print("toy connector system")
    print(f"  first-round A equations: {connector.first_round_equations}")
    print(f"  fixed initial equations: {connector.fixed_bit_equations}")
    print(f"  substituted second-round equations: {connector.second_round_equations}")
    print(f"  final rank: {connector.system.rank}")
    print(f"  final dimension: {connector.system.dimension}")
    print(f"  consistent: {not connector.system.inconsistent}")


if __name__ == "__main__":
    demo()
