"""Hand-entered appendix trail-core data.

The PDF stores Tables 7-9 as non-text graphics, so the reliable path is to
transcribe the 5x5 lane matrices into this file and validate them with
`trail_verify.py`.

Each matrix is five rows. Within a row, lanes are separated by `|`. A dash means
hex zero exactly as in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass

from linear_layer import apply_matrix_columns, load_or_build_matrices
from trail_parser import active_sboxes, parse_state_matrix
from trail_verify import transition_stats, verify_l_relation


@dataclass(frozen=True)
class TrailCoreData:
    name: str
    beta2: str
    beta3: str
    beta4: str


# Table 7, Trail core No. 2, used for Keccak[1440,160,5,160].
# This is the first core we want to fully validate and use as a real alpha2
# source. The strings below are intentionally stored in the same visual format
# as the paper to make manual auditing easy.
TRAIL_CORE_2_PARTIAL = TrailCoreData(
    name="table7_core2_keccak_1440_160_5_160",
    beta2="""
    |----------------|----------------|----------------|----------------|----------------|
    |--------1-------|-----------8----|-------------1--|--------1----1--|----------------|
    |--------1-------|----------------|-1--------------|-1--------------|-1--------------|
    |----------------|-----------8----|----------------|----------------|-1--------------|
    |----------------|----------------|-------------1--|-------------1--|----------------|
    """,
    beta3="""
    |----------------|8---------------|----------------|----------------|----------------|
    |----------------|----------------|--------8-------|---------------1|----------------|
    |----------------|----------------|----------------|---------------1|----------------|
    |----------------|---------------1|----------------|----------------|---------------1|
    |----------------|8---------------|--------8-------|----------------|----------------|
    """,
    beta4="""
    |----------------|----------------|----------------|----------------|----------------|
    |----------------|----------------|----------------|----2-----------|--------1-------|
    |---------------1|------2---------|---------2------|----------------|----------------|
    |----------------|----------------|-------------4--|----------------|----------------|
    |----------------|--8-------------|----------------|----------------|---------------2|
    """,
)


def state_from_matrix(matrix: str) -> int:
    return parse_state_matrix(matrix)


def summarize_core(core: TrailCoreData) -> None:
    beta2 = state_from_matrix(core.beta2)
    beta3 = state_from_matrix(core.beta3)
    beta4 = state_from_matrix(core.beta4)
    _, linv = load_or_build_matrices()
    alpha2 = apply_matrix_columns(linv, beta2)
    alpha3 = apply_matrix_columns(linv, beta3)
    alpha4 = apply_matrix_columns(linv, beta4)

    print(core.name)
    print(f"  active beta2/beta3/beta4: {active_sboxes(beta2)}, {active_sboxes(beta3)}, {active_sboxes(beta4)}")
    print(f"  active alpha2/alpha3/alpha4: {active_sboxes(alpha2)}, {active_sboxes(alpha3)}, {active_sboxes(alpha4)}")
    print(f"  L(alpha2)=beta2: {verify_l_relation(alpha2, beta2)}")
    print(f"  L(alpha3)=beta3: {verify_l_relation(alpha3, beta3)}")
    print(f"  L(alpha4)=beta4: {verify_l_relation(alpha4, beta4)}")
    stats23 = transition_stats(beta2, alpha3)
    stats34 = transition_stats(beta3, alpha4)
    print(f"  beta2 -> alpha3 compatible: {stats23.compatible}, weight={stats23.weight}")
    print(f"  beta3 -> alpha4 compatible: {stats34.compatible}, weight={stats34.weight}")


def main() -> None:
    summarize_core(TRAIL_CORE_2_PARTIAL)


if __name__ == "__main__":
    main()
