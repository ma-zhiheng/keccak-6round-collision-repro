"""High-level connector assembly from a target alpha2 difference."""

from __future__ import annotations

import random
from dataclasses import dataclass

from beta0_selector import choose_concrete_beta0, run_beta0_difference_phase
from beta_selector import choose_beta_for_alpha
from connector_equations import ConnectorSystem, SBoxTransition, build_connector_system
from trail_verify import sbox_value


@dataclass(frozen=True)
class ConnectorAttempt:
    alpha2: int
    beta1: int
    alpha1: int
    beta0: int
    alpha0: int
    connector: ConnectorSystem
    system_rank: int
    system_dimension: int
    consistent: bool
    beta1_weight: int
    beta0_transitions: int


def transitions_from_pair(input_difference: int, output_difference: int) -> list[SBoxTransition]:
    transitions: list[SBoxTransition] = []
    for sbox_index in range(320):
        delta_in = sbox_value(input_difference, sbox_index)
        delta_out = sbox_value(output_difference, sbox_index)
        if delta_in or delta_out:
            transitions.append(SBoxTransition(sbox_index, delta_in, delta_out))
    return transitions


def build_connector_from_alpha2(
    alpha2: int,
    rate: int,
    padding_bits: int,
    seed: int = 1,
    randomize_model_choices: bool = False,
    sample_beta0: bool = False,
    beta0_max_basis: int | None = 256,
) -> ConnectorAttempt:
    rng = random.Random(seed)
    beta1_choice = choose_beta_for_alpha(alpha2, rng=rng, randomize_ties=True)
    beta0_diff = run_beta0_difference_phase(
        beta1_choice.alpha,
        rate=rate,
        padding_bits=padding_bits,
        seed=seed,
    )
    beta0_choice = choose_concrete_beta0(
        beta0_diff,
        rng=rng if sample_beta0 else None,
        max_basis=beta0_max_basis,
    )

    beta0_alpha1 = transitions_from_pair(beta0_choice.beta0, beta1_choice.alpha)
    beta1_alpha2 = transitions_from_pair(beta1_choice.beta, alpha2)
    choices = None
    if randomize_model_choices:
        choices = [rng.randrange(80) for _ in range(320)]
    connector = build_connector_system(
        beta0_alpha1=beta0_alpha1,
        beta1_alpha2=beta1_alpha2,
        rate=rate,
        padding_bits=padding_bits,
        choices=choices,
    )
    return ConnectorAttempt(
        alpha2=alpha2,
        beta1=beta1_choice.beta,
        alpha1=beta1_choice.alpha,
        beta0=beta0_choice.beta0,
        alpha0=beta0_choice.alpha0,
        connector=connector,
        system_rank=connector.system.rank,
        system_dimension=connector.system.dimension,
        consistent=not connector.system.inconsistent,
        beta1_weight=beta1_choice.transition_weight,
        beta0_transitions=len(beta0_alpha1),
    )


def search_connector_from_alpha2(
    alpha2: int,
    rate: int,
    padding_bits: int,
    attempts: int = 100,
    seed: int = 1,
) -> ConnectorAttempt | None:
    rng = random.Random(seed)
    best: ConnectorAttempt | None = None
    for _ in range(attempts):
        attempt_seed = rng.randrange(1 << 60)
        attempt = build_connector_from_alpha2(
            alpha2,
            rate=rate,
            padding_bits=padding_bits,
            seed=attempt_seed,
            randomize_model_choices=True,
            sample_beta0=True,
        )
        if best is None or attempt.system_rank < best.system_rank:
            best = attempt
        if attempt.consistent:
            return attempt
    return best


def demo() -> None:
    attempt = build_connector_from_alpha2(0, rate=1440, padding_bits=1, seed=1)
    assert attempt.beta1 == 0
    assert attempt.alpha1 == 0
    assert attempt.beta0 == 0
    assert attempt.consistent
    print("connector runner checks passed")
    print(f"  rank={attempt.system_rank}")
    print(f"  dimension={attempt.system_dimension}")


if __name__ == "__main__":
    demo()
