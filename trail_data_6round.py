"""Trail-core data for Keccak[1440,160,6,160].

The first target is the journal Table 11 / trail core No. 5. The table is an
image in the PDF, so this file keeps the visual 5x5 matrix format and validates
the transcription mechanically before it is used by connector code.
"""

from __future__ import annotations

from dataclasses import dataclass

from linear_layer import apply_matrix_columns, load_or_build_matrices
from trail_parser import active_sboxes, parse_state_matrix
from trail_verify import transition_stats, verify_l_relation


@dataclass(frozen=True)
class SixRoundTrailCoreData:
    name: str
    beta2: str
    beta3: str
    beta4: str
    beta5: str | None = None


TRAIL_CORE_5_KECCAK_1440_160_6_160 = SixRoundTrailCoreData(
    name="table11_core5_keccak_1440_160_6_160",
    beta2="""
    |----------------|-----8----------|-----8------4---|----------------|------------4---|
    |-----------2---8|-----------2----|---------------8|-----------2---8|-----------2----|
    |----------------|-----8----------|----------------|----------------|-----8------4---|
    |-----------2----|----------------|-----8----------|-----------2---8|----------------|
    |----------------|----------------|----------------|----------------|----------------|
    """,
    beta3="""
    |----------------|----------------|----------------|---------1------|----------------|
    |----------------|----------------|----------------|---------1------|----------------|
    |----------------|----------------|----------------|----------------|----------------|
    |-----2----------|--2-------------|--2-------------|-4--------------|----------------|
    |-----2----------|-4--------------|--2-------------|-4--------------|----------------|
    """,
    beta4="""
    |----------------|----------------|----------------|------------8---|----------------|
    |--1-------------|----------------|----------------|----------------|---4------------|
    |----------------|----------------|----------------|----------------|-8--------------|
    |----------------|----------------|----------------|--------------1-|---4------------|
    |----------------|------------8---|----------------|-----------4----|----------------|
    """,
    beta5="""
    |-8--------------|-------1--------|48-1---1--------|---------2-2----|-------12--4---C|
    |-----8---1------|------48-1---84-|4---------------|-------2--------|---4-------12--4|
    |--2--------1----|-8-------24--8--|----------2-----|84-------48-1---|------------2---|
    |----24--8--18---|--------81------|4---------------|-------48-1---12|--1----------4--|
    |---8-------24-8-|---8-8----------|-24--8-418------|------1---------|--4--------2----|
    """,
)


def state_from_matrix(matrix: str) -> int:
    return parse_state_matrix(matrix)


def summarize_core(core: SixRoundTrailCoreData) -> bool:
    beta2 = state_from_matrix(core.beta2)
    beta3 = state_from_matrix(core.beta3)
    beta4 = state_from_matrix(core.beta4)
    beta5 = state_from_matrix(core.beta5) if core.beta5 is not None else None
    _, linv = load_or_build_matrices()
    alpha2 = apply_matrix_columns(linv, beta2)
    alpha3 = apply_matrix_columns(linv, beta3)
    alpha4 = apply_matrix_columns(linv, beta4)
    alpha5 = apply_matrix_columns(linv, beta5) if beta5 is not None else None

    stats23 = transition_stats(beta2, alpha3)
    stats34 = transition_stats(beta3, alpha4)
    stats45 = transition_stats(beta4, alpha5) if alpha5 is not None else None

    checks = {
        "L(alpha2)=beta2": verify_l_relation(alpha2, beta2),
        "L(alpha3)=beta3": verify_l_relation(alpha3, beta3),
        "L(alpha4)=beta4": verify_l_relation(alpha4, beta4),
        "beta2->alpha3 compatible": stats23.compatible,
        "beta3->alpha4 compatible": stats34.compatible,
        "beta2->alpha3 weight=25": stats23.weight == 25,
        "beta3->alpha4 weight=18": stats34.weight == 18,
    }
    if beta5 is not None and stats45 is not None:
        beta5_checks = {
            "L(alpha5)=beta5": verify_l_relation(alpha5, beta5),
            "beta4->alpha5 compatible": stats45.compatible,
            "beta4->alpha5 weight=16": stats45.weight == 16,
        }
    else:
        beta5_checks = {}

    print(core.name)
    if beta5 is None or alpha5 is None:
        print(f"  active beta2/beta3/beta4: {active_sboxes(beta2)}, {active_sboxes(beta3)}, {active_sboxes(beta4)}")
        print(f"  active alpha2/alpha3/alpha4: {active_sboxes(alpha2)}, {active_sboxes(alpha3)}, {active_sboxes(alpha4)}")
    else:
        print(f"  active beta2/beta3/beta4/beta5: {active_sboxes(beta2)}, {active_sboxes(beta3)}, {active_sboxes(beta4)}, {active_sboxes(beta5)}")
        print(f"  active alpha2/alpha3/alpha4/alpha5: {active_sboxes(alpha2)}, {active_sboxes(alpha3)}, {active_sboxes(alpha4)}, {active_sboxes(alpha5)}")
    print(f"  beta2 -> alpha3: compatible={stats23.compatible}, weight={stats23.weight}")
    print(f"  beta3 -> alpha4: compatible={stats34.compatible}, weight={stats34.weight}")
    if stats45 is not None:
        print(f"  beta4 -> alpha5: compatible={stats45.compatible}, weight={stats45.weight}")
    for name, ok in checks.items():
        print(f"  {name}: {ok}")
    for name, ok in beta5_checks.items():
        print(f"  provisional {name}: {ok}")
    if beta5_checks and not all(beta5_checks.values()):
        print("  beta5 transcription is provisional and is not used yet")
    return all(checks.values())


def main() -> None:
    ok = summarize_core(TRAIL_CORE_5_KECCAK_1440_160_6_160)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
