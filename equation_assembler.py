"""Partial state-level equation assembly for the two-round connector."""

from __future__ import annotations

from dataclasses import dataclass

from sbox_constraints import (
    linearizable_models_for_transition,
    value_constraints_for_transition,
)
from state_lift import LiftedSBoxModel, lift_local_mask, lift_model


Equation = tuple[int, int]


@dataclass(frozen=True)
class SBoxTransition:
    sbox_index: int
    delta_in: int
    delta_out: int


@dataclass(frozen=True)
class FirstRoundAssembly:
    equations_a: tuple[Equation, ...]
    lifted_models: tuple[LiftedSBoxModel, ...]


def assemble_second_round_g(transitions: list[SBoxTransition]) -> tuple[Equation, ...]:
    """Assemble local value constraints into state-level G*z=m rows."""
    equations: list[Equation] = []
    for transition in transitions:
        for local_mask, rhs in value_constraints_for_transition(
            transition.delta_in,
            transition.delta_out,
        ):
            equations.append((lift_local_mask(transition.sbox_index, local_mask), rhs))
    return tuple(equations)


def assemble_first_round_a(
    transitions: list[SBoxTransition],
    model_choices: list[int] | None = None,
) -> FirstRoundAssembly:
    """Assemble first-round linearization constraints A*x=t.

    `model_choices` selects one linearizable model per active S-box. This mirrors
    the heuristic choices in the paper's basic linearization procedure.
    """
    if model_choices is None:
        model_choices = [0 for _ in transitions]
    if len(model_choices) != len(transitions):
        raise ValueError("model_choices length mismatch")

    equations: list[Equation] = []
    lifted_models: list[LiftedSBoxModel] = []
    for transition, choice in zip(transitions, model_choices):
        models = linearizable_models_for_transition(transition.delta_in, transition.delta_out)
        if not models:
            raise ValueError(f"incompatible transition {transition}")
        model = models[choice % len(models)]
        lifted = lift_model(transition.sbox_index, model)
        equations.extend(lifted.input_equations)
        lifted_models.append(lifted)

    return FirstRoundAssembly(tuple(equations), tuple(lifted_models))


def demo() -> None:
    transitions = [
        SBoxTransition(0, 0x03, 0x02),
        SBoxTransition(1, 0x01, 0x01),
    ]
    second = assemble_second_round_g(transitions)
    first = assemble_first_round_a(transitions, [0, 2])

    print("toy state-level assembly")
    print(f"  second-round G*z=m equations: {len(second)}")
    print(f"  first-round A*x=t equations: {len(first.equations_a)}")
    print(f"  lifted first-round S-box models: {len(first.lifted_models)}")
    print("  first G row coefficient bit length:", second[0][0].bit_length())


if __name__ == "__main__":
    demo()
