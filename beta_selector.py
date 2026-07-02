"""Heuristic beta choices for connector construction."""

from __future__ import annotations

import random
from dataclasses import dataclass

from linear_layer import load_or_build_matrices, apply_matrix_columns
from sbox_linearization import DDT
from state_lift import SBOXES_1600, bit_position, sbox_index_to_yz
from trail_parser import active_sboxes
from trail_verify import sbox_value, transition_stats


@dataclass(frozen=True)
class BetaChoice:
    beta: int
    alpha: int
    transition_weight: int
    active_alpha_sboxes: int
    compatible: bool


def set_sbox_value(state: int, sbox_index: int, value: int) -> int:
    y, z = sbox_index_to_yz(sbox_index)
    for x in range(5):
        bit = bit_position(x, y, z)
        if (value >> x) & 1:
            state |= 1 << bit
        else:
            state &= ~(1 << bit)
    return state


def compatible_inputs_for_output(delta_out: int) -> list[int]:
    return [delta_in for delta_in in range(32) if DDT[delta_in][delta_out] > 0]


def sbox_weight(delta_in: int, delta_out: int) -> int:
    entry = DDT[delta_in][delta_out]
    if entry == 0:
        raise ValueError("incompatible S-box transition")
    return 5 - (entry.bit_length() - 1)


def choose_beta_for_alpha(
    alpha_out: int,
    rng: random.Random | None = None,
    randomize_ties: bool = True,
) -> BetaChoice:
    """Choose beta input differences compatible with output difference alpha_out.

    This constructs the beta in a S-box-local manner. The caller can then reject
    choices whose alpha=L^-1(beta) does not satisfy extra heuristic conditions.
    """
    rng = rng or random.Random()
    beta = 0
    total_weight = 0
    for sbox_index in range(SBOXES_1600):
        delta_out = sbox_value(alpha_out, sbox_index)
        candidates = compatible_inputs_for_output(delta_out)
        best_weight = min(sbox_weight(delta_in, delta_out) for delta_in in candidates)
        best = [
            delta_in for delta_in in candidates
            if sbox_weight(delta_in, delta_out) == best_weight
        ]
        delta_in = rng.choice(best) if randomize_ties else best[0]
        beta = set_sbox_value(beta, sbox_index, delta_in)
        total_weight += best_weight if delta_out or delta_in else 0

    _, linv_columns = load_or_build_matrices()
    alpha = apply_matrix_columns(linv_columns, beta)
    stats = transition_stats(beta, alpha_out)
    return BetaChoice(
        beta=beta,
        alpha=alpha,
        transition_weight=total_weight,
        active_alpha_sboxes=active_sboxes(alpha),
        compatible=stats.compatible,
    )


def search_beta_for_alpha(
    alpha_out: int,
    min_active_alpha_sboxes: int = 320,
    attempts: int = 1000,
    seed: int = 1,
    strict: bool = False,
) -> BetaChoice:
    rng = random.Random(seed)
    best: BetaChoice | None = None
    for _ in range(attempts):
        choice = choose_beta_for_alpha(alpha_out, rng=rng, randomize_ties=True)
        if best is None:
            best = choice
        elif (
            choice.active_alpha_sboxes > best.active_alpha_sboxes
            or (
                choice.active_alpha_sboxes == best.active_alpha_sboxes
                and choice.transition_weight < best.transition_weight
            )
        ):
            best = choice
        if choice.active_alpha_sboxes >= min_active_alpha_sboxes:
            return choice
    assert best is not None
    if strict:
        raise RuntimeError(
            "failed to find beta with "
            f"{min_active_alpha_sboxes} active alpha S-boxes in {attempts} attempts; "
            f"best was {best.active_alpha_sboxes}"
        )
    return best


def demo() -> None:
    choice = choose_beta_for_alpha(0, randomize_ties=False)
    assert choice.beta == 0
    assert choice.alpha == 0
    assert choice.compatible
    assert choice.transition_weight == 0
    print("beta selector checks passed")


if __name__ == "__main__":
    demo()
