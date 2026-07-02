"""Export the paper-derived 3-round connector for C++/CUDA search."""

from __future__ import annotations

from pathlib import Path

from beta0_selector import Beta0ConcreteChoice
from beta_selector import BetaChoice
from connector_runner import transitions_from_pair
from core3_connector import (
    FirstTwoRoundConnector,
    build_known_value_bitwise_third_round_connector,
    trail_states,
)
from derive_paper_first_two_connector import forced_models_from_message
from incremental_connector import build_bitwise_prefix_connector
from keccak_state import round_int, rounds_int
from linear_layer import apply_l_int, apply_matrix_columns, load_or_build_matrices
from paper_collisions import TABLE18_KECCAK_1440_160_6_160
from trail_data_6round import TRAIL_CORE_5_KECCAK_1440_160_6_160
from trail_parser import parse_state_matrix
from trail_verify import transition_stats


OUT = Path(__file__).with_name("core3_search_data.hpp")


def words25(value: int) -> list[int]:
    return [(value >> (64 * index)) & ((1 << 64) - 1) for index in range(25)]


def array_literal(words: list[int]) -> str:
    return "{" + ", ".join(f"0x{word:016x}ULL" for word in words) + "}"


def build_paper_reproduction() -> tuple[FirstTwoRoundConnector, object]:
    m1 = parse_state_matrix(TABLE18_KECCAK_1440_160_6_160.m1)
    m2 = parse_state_matrix(TABLE18_KECCAK_1440_160_6_160.m2)
    alpha0 = m1 ^ m2
    beta0 = apply_l_int(alpha0)
    alpha1 = round_int(m1, 0) ^ round_int(m2, 0)
    beta1 = apply_l_int(alpha1)
    alpha2 = rounds_int(m1, 2) ^ rounds_int(m2, 2)
    states = trail_states(TRAIL_CORE_5_KECCAK_1440_160_6_160)
    forced = forced_models_from_message(beta0, alpha1, apply_l_int(m1))
    connector, _assigned, _rows = build_bitwise_prefix_connector(
        transitions_from_pair(beta0, alpha1),
        transitions_from_pair(beta1, alpha2),
        rate=1440,
        padding_bits=1,
        seed=1,
        row_order="large",
        row_retries=1,
        forced_model_choices=forced,
    )
    beta1_stats = transition_stats(beta1, alpha2)
    beta0_stats = transition_stats(beta0, alpha1)
    reproduction = FirstTwoRoundConnector(
        trail=states,
        beta1_choice=BetaChoice(
            beta=beta1,
            alpha=alpha1,
            transition_weight=beta1_stats.weight,
            active_alpha_sboxes=320,
            compatible=beta1_stats.compatible,
        ),
        beta0_choice=Beta0ConcreteChoice(
            beta0=beta0,
            alpha0=alpha0,
            transitions=tuple((t.sbox_index, t.delta_in, t.delta_out) for t in transitions_from_pair(beta0, alpha1)),
            compatible=beta0_stats.compatible,
        ),
        connector=connector,
    )
    third = build_known_value_bitwise_third_round_connector(
        reproduction,
        chi1_input=apply_l_int(round_int(m1, 0)),
    )
    return reproduction, third


def main() -> None:
    _l_columns, linv_columns = load_or_build_matrices()
    reproduction, third = build_paper_reproduction()
    if not third.consistent:
        raise RuntimeError("paper-derived third connector is inconsistent")
    particular_x = third.system.particular_solution()
    basis_x = third.system.nullspace_basis()
    particular_m = apply_matrix_columns(linv_columns, particular_x)
    basis_m = [apply_matrix_columns(linv_columns, vector) for vector in basis_x]

    lines = [
        "#pragma once",
        "",
        "#include <array>",
        "#include <cstdint>",
        "",
        "namespace core3_data {",
        "static constexpr std::uint64_t CONNECTOR_SEED = 1ULL;",
        f"static constexpr std::size_t BASIS_SIZE = {len(basis_m)};",
        f"static constexpr std::array<std::uint64_t, 25> PARTICULAR_MESSAGE = {array_literal(words25(particular_m))};",
        f"static constexpr std::array<std::uint64_t, 25> ALPHA0 = {array_literal(words25(reproduction.beta0_choice.alpha0))};",
        f"static constexpr std::array<std::uint64_t, 25> ALPHA2 = {array_literal(words25(reproduction.trail.alpha2))};",
        f"static constexpr std::array<std::uint64_t, 25> ALPHA3 = {array_literal(words25(reproduction.trail.alpha3))};",
        f"static constexpr std::array<std::uint64_t, 25> ALPHA4 = {array_literal(words25(reproduction.trail.alpha4))};",
        "static constexpr std::array<std::array<std::uint64_t, 25>, BASIS_SIZE> BASIS_MESSAGE = {{",
    ]
    for vector in basis_m:
        lines.append(f"    {array_literal(words25(vector))},")
    lines.extend(["}};", "}  // namespace core3_data", ""])
    OUT.write_text("\n".join(lines), encoding="ascii")
    print(f"wrote {OUT}")
    print(f"basis size: {len(basis_m)}")
    print(f"connector rank/dimension: {third.system.rank}/{third.system.dimension}")


if __name__ == "__main__":
    main()
