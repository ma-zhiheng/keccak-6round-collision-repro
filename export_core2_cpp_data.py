"""Export the reproduced Table 7 core No. 2 connector for C++ search."""

from __future__ import annotations

from pathlib import Path

from core2_connector import (
    CORE2_CONNECTOR_BETA0_SAMPLES,
    CORE2_CONNECTOR_ROW_ORDER,
    CORE2_CONNECTOR_ROW_RETRIES,
    CORE2_CONNECTOR_SEED,
    build_reproduced_core2_connector,
)
from linear_layer import apply_matrix_columns, load_or_build_matrices
from trail_data import TRAIL_CORE_2_PARTIAL, state_from_matrix


OUT = Path(__file__).with_name("core2_search_data.hpp")


def words25(value: int) -> list[int]:
    return [(value >> (64 * index)) & ((1 << 64) - 1) for index in range(25)]


def array_literal(words: list[int]) -> str:
    return "{" + ", ".join(f"0x{word:016x}ULL" for word in words) + "}"


def main() -> None:
    _l_columns, linv_columns = load_or_build_matrices()
    reproduction = build_reproduced_core2_connector()
    system = reproduction.connector.system
    particular_x = system.particular_solution()
    basis_x = system.nullspace_basis()

    particular_m = apply_matrix_columns(linv_columns, particular_x)
    basis_m = [apply_matrix_columns(linv_columns, vector) for vector in basis_x]

    beta3 = state_from_matrix(TRAIL_CORE_2_PARTIAL.beta3)
    beta4 = state_from_matrix(TRAIL_CORE_2_PARTIAL.beta4)
    alpha3 = apply_matrix_columns(linv_columns, beta3)
    alpha4 = apply_matrix_columns(linv_columns, beta4)

    lines = [
        "#pragma once",
        "",
        "#include <array>",
        "#include <cstdint>",
        "",
        "namespace core2_data {",
        f"static constexpr std::uint64_t CONNECTOR_SEED = {CORE2_CONNECTOR_SEED}ULL;",
        f"static constexpr int ROW_RETRIES = {CORE2_CONNECTOR_ROW_RETRIES};",
        f"static constexpr int BETA0_SAMPLES = {CORE2_CONNECTOR_BETA0_SAMPLES};",
        f'static constexpr const char* ROW_ORDER = "{CORE2_CONNECTOR_ROW_ORDER}";',
        f"static constexpr std::size_t BASIS_SIZE = {len(basis_m)};",
        f"static constexpr std::array<std::uint64_t, 25> PARTICULAR_MESSAGE = {array_literal(words25(particular_m))};",
        f"static constexpr std::array<std::uint64_t, 25> ALPHA0 = {array_literal(words25(reproduction.beta0_choice.alpha0))};",
        f"static constexpr std::array<std::uint64_t, 25> ALPHA2 = {array_literal(words25(reproduction.alpha2))};",
        f"static constexpr std::array<std::uint64_t, 25> ALPHA3 = {array_literal(words25(alpha3))};",
        f"static constexpr std::array<std::uint64_t, 25> ALPHA4 = {array_literal(words25(alpha4))};",
        "static constexpr std::array<std::array<std::uint64_t, 25>, BASIS_SIZE> BASIS_MESSAGE = {{",
    ]
    for vector in basis_m:
        lines.append(f"    {array_literal(words25(vector))},")
    lines.extend([
        "}};",
        "}  // namespace core2_data",
        "",
    ])
    OUT.write_text("\n".join(lines), encoding="ascii")
    print(f"wrote {OUT}")
    print(f"basis size: {len(basis_m)}")
    print(f"connector dimension: {reproduction.connector.system.dimension}")


if __name__ == "__main__":
    main()
