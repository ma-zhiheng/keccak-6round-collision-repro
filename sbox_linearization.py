"""Keccak S-box linearization tools for Qiao et al. EUROCRYPT 2017.

This module reproduces the observations in Section 4.1 of
"New Collision Attacks on Round-Reduced Keccak" without using the paper's
appendix tables as input.
"""

from __future__ import annotations

from itertools import combinations


WIDTH = 5
SPACE_SIZE = 1 << WIDTH


def chi5(value: int) -> int:
    """The 5-bit Keccak chi S-box on one row."""
    bits = [(value >> x) & 1 for x in range(WIDTH)]
    out = 0
    for x in range(WIDTH):
        bit = bits[x] ^ ((1 ^ bits[(x + 1) % WIDTH]) & bits[(x + 2) % WIDTH])
        out |= bit << x
    return out


def build_ddt() -> list[list[int]]:
    table = [[0 for _ in range(SPACE_SIZE)] for _ in range(SPACE_SIZE)]
    for delta_in in range(SPACE_SIZE):
        for value in range(SPACE_SIZE):
            delta_out = chi5(value) ^ chi5(value ^ delta_in)
            table[delta_in][delta_out] += 1
    return table


DDT = build_ddt()


def rank(vectors: list[int]) -> int:
    basis: dict[int, int] = {}
    for vector in vectors:
        reduced = vector
        while reduced:
            pivot = reduced.bit_length() - 1
            existing = basis.get(pivot)
            if existing is None:
                basis[pivot] = reduced
                break
            reduced ^= existing
    return len(basis)


def linear_subspaces(dim: int) -> list[frozenset[int]]:
    """Enumerate all dim-dimensional linear subspaces of GF(2)^5."""
    result: set[frozenset[int]] = set()
    for vectors in combinations(range(1, SPACE_SIZE), dim):
        if rank(list(vectors)) != dim:
            continue
        subspace = {0}
        for vector in vectors:
            subspace |= {item ^ vector for item in list(subspace)}
        result.add(frozenset(subspace))
    return sorted(result, key=lambda values: sorted(values))


def affine_subspaces(dim: int) -> list[frozenset[int]]:
    """Enumerate all dim-dimensional affine subspaces of GF(2)^5."""
    result: set[frozenset[int]] = set()
    for linear in linear_subspaces(dim):
        for base in range(SPACE_SIZE):
            result.add(frozenset(base ^ item for item in linear))
    return sorted(result, key=lambda values: sorted(values))


def is_affine_subset(values: set[int] | frozenset[int]) -> bool:
    if not values:
        return False
    base = next(iter(values))
    linear = {base ^ value for value in values}
    if 0 not in linear:
        return False
    return all((a ^ b) in linear for a in linear for b in linear)


def is_linearizable_affine_subspace(values: set[int] | frozenset[int]) -> bool:
    """Return whether chi5 restricted to values is affine-linear.

    An affine map sends affine subspaces to affine subspaces and preserves the
    parallelogram relations inside the restricted domain. Since the domain is
    itself an affine subspace, this condition is enough here.
    """
    if not is_affine_subset(values):
        return False
    return is_affine_subset({chi5(value) for value in values})


def linearizable_affine_subspaces(dim: int) -> list[frozenset[int]]:
    return [
        values for values in affine_subspaces(dim)
        if is_linearizable_affine_subspace(values)
    ]


def value_solution_set(delta_in: int, delta_out: int) -> frozenset[int]:
    return frozenset(
        value for value in range(SPACE_SIZE)
        if chi5(value) ^ chi5(value ^ delta_in) == delta_out
    )


def contained_two_dim_linearizable_subspaces(values: set[int] | frozenset[int]) -> list[frozenset[int]]:
    """All 4-point linearizable affine planes contained in values."""
    universe = set(values)
    return [
        plane for plane in linearizable_affine_subspaces(2)
        if set(plane) <= universe
    ]


def possible_input_differences(delta_out: int) -> list[int]:
    return [delta_in for delta_in in range(SPACE_SIZE) if DDT[delta_in][delta_out] > 0]


def input_difference_planes(delta_out: int) -> list[frozenset[int]]:
    """4-point affine planes inside {delta_in: DDT(delta_in, delta_out)>0}.

    These are used in the Section 4.4 difference phase for beta0 selection.
    """
    possible = set(possible_input_differences(delta_out))
    planes: set[frozenset[int]] = set()
    for base in possible:
        for u in range(1, SPACE_SIZE):
            for v in range(u + 1, SPACE_SIZE):
                plane = frozenset({base, base ^ u, base ^ v, base ^ u ^ v})
                if len(plane) == 4 and plane <= possible:
                    planes.add(plane)
    return sorted(
        planes,
        key=lambda plane: sorted((DDT[delta_in][delta_out] for delta_in in plane), reverse=True),
        reverse=True,
    )


def affine_equations(values: set[int] | frozenset[int]) -> list[tuple[int, int]]:
    """Return independent equations mask*x = rhs defining values."""
    if not is_affine_subset(values):
        raise ValueError("values must be an affine subset")
    dimension = (len(values)).bit_length() - 1
    needed = WIDTH - dimension
    equations: list[tuple[int, int]] = []
    equation_basis: dict[int, int] = {}

    for mask in range(1, SPACE_SIZE):
        rhs_values = {((mask & value).bit_count() & 1) for value in values}
        if len(rhs_values) != 1:
            continue
        rhs = next(iter(rhs_values))
        reduced = mask
        while reduced:
            pivot = reduced.bit_length() - 1
            existing = equation_basis.get(pivot)
            if existing is None:
                equation_basis[pivot] = reduced
                equations.append((mask, rhs))
                break
            reduced ^= existing
        if len(equations) == needed:
            return equations

    raise RuntimeError("failed to derive enough affine equations")


def format_subset(values: set[int] | frozenset[int]) -> str:
    return "{" + ", ".join(f"{value:X}" for value in sorted(values)) + "}"
