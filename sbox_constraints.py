"""Local linear constraints used by the two-round connector.

The full connector in Section 4.2 is a 1600-bit GF(2) system. This file builds
the 5-bit S-box pieces that later become diagonal blocks in the paper's A and G
matrices.
"""

from __future__ import annotations

from dataclasses import dataclass

from sbox_linearization import (
    DDT,
    WIDTH,
    affine_equations,
    chi5,
    contained_two_dim_linearizable_subspaces,
    is_linearizable_affine_subspace,
    linearizable_affine_subspaces,
    value_solution_set,
)


@dataclass(frozen=True)
class AffineSBoxModel:
    """chi5(x) = affine(x) for x restricted to subset."""

    subset: frozenset[int]
    input_equations: tuple[tuple[int, int], ...]
    output_masks: tuple[int, ...]
    output_constants: tuple[int, ...]

    def apply(self, value: int) -> int:
        out = 0
        for bit, (mask, const) in enumerate(zip(self.output_masks, self.output_constants)):
            out_bit = ((mask & value).bit_count() & 1) ^ const
            out |= out_bit << bit
        return out

    def verify(self) -> None:
        for value in self.subset:
            assert self.apply(value) == chi5(value), (value, self.apply(value), chi5(value))


def affine_bit_expression(values: frozenset[int], outputs: dict[int, int]) -> tuple[int, int]:
    """Find mask,const so outputs[x] = parity(mask & x) xor const on values."""
    for mask in range(1 << WIDTH):
        for const in (0, 1):
            if all((((mask & value).bit_count() & 1) ^ const) == outputs[value] for value in values):
                return mask, const
    raise ValueError("no affine expression on this subset")


def model_for_subset(values: frozenset[int]) -> AffineSBoxModel:
    if not is_linearizable_affine_subspace(values):
        raise ValueError("subset is not linearizable")

    masks: list[int] = []
    constants: list[int] = []
    for bit in range(WIDTH):
        outputs = {value: (chi5(value) >> bit) & 1 for value in values}
        mask, const = affine_bit_expression(values, outputs)
        masks.append(mask)
        constants.append(const)

    model = AffineSBoxModel(
        subset=values,
        input_equations=tuple(affine_equations(values)),
        output_masks=tuple(masks),
        output_constants=tuple(constants),
    )
    model.verify()
    return model


def all_two_dim_linearizable_models() -> list[AffineSBoxModel]:
    return [model_for_subset(plane) for plane in linearizable_affine_subspaces(2)]


def value_constraints_for_transition(delta_in: int, delta_out: int) -> tuple[tuple[int, int], ...]:
    """Equations defining V={x:S(x)+S(x+din)=dout}.

    This is the local block used in the paper's G*z=m constraints for the second
    chi layer. For example, DDT=8 gives a 3-dimensional V, hence only two
    equations are needed.
    """
    if DDT[delta_in][delta_out] == 0:
        return tuple()

    values = value_solution_set(delta_in, delta_out)
    return tuple(affine_equations(values))


def linearizable_models_for_transition(delta_in: int, delta_out: int) -> list[AffineSBoxModel]:
    """Linearizable models compatible with S(x)+S(x+din)=dout.

    This is the local block used for the first chi layer. If DDT is 8, the full
    value set is 3-dimensional and cannot be linearized, so we split it into the
    six 2-dimensional linearizable planes described in the paper.
    """
    if DDT[delta_in][delta_out] == 0:
        return []

    values = value_solution_set(delta_in, delta_out)
    if DDT[delta_in][delta_out] in (2, 4):
        return [model_for_subset(values)]
    if DDT[delta_in][delta_out] == 8:
        return [model_for_subset(plane) for plane in contained_two_dim_linearizable_subspaces(values)]

    raise NotImplementedError(f"unexpected DDT entry {DDT[delta_in][delta_out]}")


def demo_transition(delta_in: int = 0x03, delta_out: int = 0x02) -> str:
    value_equations = value_constraints_for_transition(delta_in, delta_out)
    models = linearizable_models_for_transition(delta_in, delta_out)
    if not value_equations and not models:
        return "no compatible transition"

    lines = [
        f"transition din={delta_in:02X}, dout={delta_out:02X}, DDT={DDT[delta_in][delta_out]}",
        f"second-round value equations: {len(value_equations)}",
        f"linearizable models: {len(models)}",
    ]
    for mask, rhs in value_equations:
        lines.append(f"  V equation: {mask:05b} * x = {rhs}")
    for i, model in enumerate(models[:3]):
        lines.append(f"  model {i}: subset={{{', '.join(f'{v:X}' for v in sorted(model.subset))}}}")
        lines.append("    input equations:")
        for mask, rhs in model.input_equations:
            lines.append(f"      {mask:05b} * x = {rhs}")
        lines.append("    output bits y_j = mask_j*x + const_j:")
        for bit, (mask, const) in enumerate(zip(model.output_masks, model.output_constants)):
            lines.append(f"      y{bit}: {mask:05b}, {const}")
    return "\n".join(lines)
