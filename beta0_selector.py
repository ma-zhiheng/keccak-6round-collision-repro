"""Section 4.4 beta0 difference-phase selection."""

from __future__ import annotations

import random
from dataclasses import dataclass

from gf2 import GF2System
from linear_layer import STATE_BITS, apply_matrix_columns, load_or_build_matrices, preimage_bit_equation
from sbox_linearization import DDT, affine_equations, input_difference_planes
from state_lift import SBOXES_1600, bit_position, lift_local_mask, sbox_index_to_yz
from trail_verify import sbox_value, transition_stats


@dataclass
class DifferencePlaneChoice:
    sbox_index: int
    delta_out: int
    planes: list[frozenset[int]]
    pointer: int = 0

    def reset(self) -> None:
        self.pointer = 0


@dataclass(frozen=True)
class Beta0DifferenceResult:
    alpha1: int
    equations: GF2System
    choices: tuple[DifferencePlaneChoice, ...]
    order: tuple[int, ...]


@dataclass(frozen=True)
class Beta0ConcreteChoice:
    beta0: int
    alpha0: int
    transitions: tuple[tuple[int, int, int], ...]
    compatible: bool
    first_round_equations: int = 0
    ddt2_transitions: int = 0
    ddt4_transitions: int = 0
    ddt8_transitions: int = 0


def fixed_initial_difference_bits(rate: int, padding_bits: int) -> range:
    return range(rate - padding_bits, STATE_BITS)


def active_sboxes_in_state(state: int) -> list[int]:
    return [
        sbox_index for sbox_index in range(SBOXES_1600)
        if sbox_value(state, sbox_index) != 0
    ]


def local_plane_equations(sbox_index: int, plane: frozenset[int]) -> list[tuple[int, int]]:
    return [
        (lift_local_mask(sbox_index, mask), rhs)
        for mask, rhs in affine_equations(plane)
    ]


def equations_fix_sbox_value(sbox_index: int, value: int) -> list[tuple[int, int]]:
    y, z = sbox_index_to_yz(sbox_index)
    return [
        (1 << bit_position(x, y, z), (value >> x) & 1)
        for x in range(5)
    ]


def build_difference_choices(alpha1: int) -> list[DifferencePlaneChoice]:
    choices: list[DifferencePlaneChoice] = []
    for sbox_index in active_sboxes_in_state(alpha1):
        delta_out = sbox_value(alpha1, sbox_index)
        planes = input_difference_planes(delta_out)
        if not planes:
            raise ValueError(f"no input-difference plane for S-box {sbox_index}")
        choices.append(DifferencePlaneChoice(sbox_index, delta_out, planes))
    return choices


def initialize_e_delta(alpha1: int, rate: int, padding_bits: int, linv_columns: list[int]) -> GF2System:
    system = GF2System(STATE_BITS)

    for bit in fixed_initial_difference_bits(rate, padding_bits):
        system.add_equation(preimage_bit_equation(linv_columns, bit), 0)

    for sbox_index in range(SBOXES_1600):
        if sbox_value(alpha1, sbox_index) != 0:
            continue
        for equation in equations_fix_sbox_value(sbox_index, 0):
            system.add_equation(*equation)

    return system


def run_beta0_difference_phase(
    alpha1: int,
    rate: int,
    padding_bits: int,
    threshold: int = 200,
    seed: int = 1,
) -> Beta0DifferenceResult:
    """Build E_delta for beta0 candidates compatible with alpha1."""
    _, linv_columns = load_or_build_matrices()
    rng = random.Random(seed)
    choices = build_difference_choices(alpha1)
    order = list(range(len(choices)))

    for attempt in range(threshold):
        system = initialize_e_delta(alpha1, rate, padding_bits, linv_columns)
        rng.shuffle(order)
        for choice in choices:
            choice.reset()

        success = True
        for order_pos in order:
            choice = choices[order_pos]
            plane_order = list(range(len(choice.planes)))
            rng.shuffle(plane_order)
            for plane_index in plane_order:
                plane = choice.planes[plane_index]
                equations = local_plane_equations(choice.sbox_index, plane)
                if system.add_equations(equations):
                    choice.pointer = plane_index
                    break
            else:
                success = False
                break

        if success and not system.inconsistent:
            return Beta0DifferenceResult(alpha1, system, tuple(choices), tuple(order))

    raise RuntimeError("failed to construct beta0 difference phase")


def choose_concrete_beta0(
    result: Beta0DifferenceResult,
    rng: random.Random | None = None,
    max_basis: int | None = None,
) -> Beta0ConcreteChoice:
    """Pick one concrete beta0 from E_delta's affine space.

    This is a deterministic baseline. The full value phase will later try
    different concrete choices when the connector equations are inconsistent.
    """
    _, linv_columns = load_or_build_matrices()
    beta0 = result.equations.particular_solution()
    if rng is not None:
        basis = result.equations.nullspace_basis()
        if max_basis is not None:
            basis = basis[:max_basis]
        for vector in basis:
            if rng.getrandbits(1):
                beta0 ^= vector
    alpha0 = apply_matrix_columns(linv_columns, beta0)
    return concrete_choice_from_beta0(result, beta0)


def concrete_choice_from_beta0(result: Beta0DifferenceResult, beta0: int) -> Beta0ConcreteChoice:
    _, linv_columns = load_or_build_matrices()
    alpha0 = apply_matrix_columns(linv_columns, beta0)
    transitions = tuple(
        (
            sbox_index,
            sbox_value(beta0, sbox_index),
            sbox_value(result.alpha1, sbox_index),
        )
        for sbox_index in range(SBOXES_1600)
        if sbox_value(beta0, sbox_index) or sbox_value(result.alpha1, sbox_index)
    )
    stats = transition_stats(beta0, result.alpha1)
    first_round_equations = 0
    ddt2_transitions = 0
    ddt4_transitions = 0
    ddt8_transitions = 0
    for _sbox_index, delta_in, delta_out in transitions:
        entry = DDT[delta_in][delta_out]
        if entry == 2:
            first_round_equations += 4
            ddt2_transitions += 1
        elif entry in (4, 8):
            first_round_equations += 3
            if entry == 4:
                ddt4_transitions += 1
            else:
                ddt8_transitions += 1
        elif entry == 32:
            # Inactive S-boxes still get linearized on a 2-dimensional plane.
            first_round_equations += 3
        elif entry == 0:
            first_round_equations += 5
        else:
            raise ValueError(f"unexpected DDT entry {entry}")
    return Beta0ConcreteChoice(
        beta0=beta0,
        alpha0=alpha0,
        transitions=transitions,
        compatible=stats.compatible,
        first_round_equations=first_round_equations,
        ddt2_transitions=ddt2_transitions,
        ddt4_transitions=ddt4_transitions,
        ddt8_transitions=ddt8_transitions,
    )


def choose_best_concrete_beta0(
    result: Beta0DifferenceResult,
    rng: random.Random,
    samples: int = 2048,
    max_basis: int | None = None,
) -> Beta0ConcreteChoice:
    """Sample beta0 candidates and keep the one with the lightest value phase.

    The paper's value phase tries concrete beta0 choices until the connector
    works with enough freedom. This baseline scores candidates by the number of
    first-round linearization equations, which is mainly driven by avoiding
    DDT=2 transitions.
    """
    basis = result.equations.nullspace_basis()
    if max_basis is not None:
        basis = basis[:max_basis]
    base = result.equations.particular_solution()
    best = concrete_choice_from_beta0(result, base)
    best_key = (
        not best.compatible,
        best.first_round_equations,
        best.ddt2_transitions,
        -best.ddt8_transitions,
    )
    for _ in range(samples):
        beta0 = base
        for vector in basis:
            if rng.getrandbits(1):
                beta0 ^= vector
        candidate = concrete_choice_from_beta0(result, beta0)
        key = (
            not candidate.compatible,
            candidate.first_round_equations,
            candidate.ddt2_transitions,
            -candidate.ddt8_transitions,
        )
        if key < best_key:
            best = candidate
            best_key = key
    return best


def demo() -> None:
    # Smoke test with alpha1=0. The only valid beta0 under non-active S-box
    # constraints is beta0=0.
    result = run_beta0_difference_phase(0, rate=1440, padding_bits=1, threshold=1)
    concrete = choose_concrete_beta0(result)
    assert concrete.beta0 == 0
    assert concrete.alpha0 == 0
    assert concrete.compatible
    print("beta0 selector checks passed")


if __name__ == "__main__":
    demo()
