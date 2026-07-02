"""Sample states from a connector system and verify two-round differences."""

from __future__ import annotations

import random
from dataclasses import dataclass

from connector_runner import ConnectorAttempt, build_connector_from_alpha2
from gf2 import GF2System
from keccak_state import rounds_int
from linear_layer import apply_matrix_columns, load_or_build_matrices


@dataclass(frozen=True)
class ConnectorSample:
    x: int
    message1_state: int
    message2_state: int
    output1: int
    output2: int
    output_difference: int
    verifies: bool


def sample_solution(system: GF2System, rng: random.Random, max_basis: int | None = None) -> int:
    value = system.particular_solution()
    basis = system.nullspace_basis()
    if max_basis is not None:
        basis = basis[:max_basis]
    for vector in basis:
        if rng.getrandbits(1):
            value ^= vector
    return value


def sample_and_verify(
    attempt: ConnectorAttempt,
    rng: random.Random | None = None,
    max_basis: int | None = 64,
) -> ConnectorSample:
    rng = rng or random.Random()
    _, linv_columns = load_or_build_matrices()
    x = sample_solution(attempt.connector.system, rng, max_basis=max_basis)
    message1 = apply_matrix_columns(linv_columns, x)
    message2 = message1 ^ attempt.alpha0
    output1 = rounds_int(message1, 2)
    output2 = rounds_int(message2, 2)
    output_difference = output1 ^ output2
    return ConnectorSample(
        x=x,
        message1_state=message1,
        message2_state=message2,
        output1=output1,
        output2=output2,
        output_difference=output_difference,
        verifies=output_difference == attempt.alpha2,
    )


def demo() -> None:
    attempt = build_connector_from_alpha2(0, rate=1440, padding_bits=1, seed=1)
    sample = sample_and_verify(attempt, random.Random(1), max_basis=32)
    assert sample.verifies
    print("connector sampling checks passed")
    print(f"  output difference: {sample.output_difference:X}")


if __name__ == "__main__":
    demo()
