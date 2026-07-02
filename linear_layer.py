"""Keccak-f[1600] linear layer helpers for connector equations."""

from __future__ import annotations

import pickle
from pathlib import Path


MASK64 = (1 << 64) - 1
STATE_BITS = 1600
CACHE_FILE = Path(__file__).with_name("_linear_layer_cache.pkl")

ROUND_CONSTANTS = [
    0x0000000000000001, 0x0000000000008082,
    0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088,
    0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B,
    0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080,
    0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080,
    0x0000000080000001, 0x8000000080008008,
]

ROTATION = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
]


def index(x: int, y: int) -> int:
    return x + 5 * y


def rotl64(value: int, amount: int) -> int:
    amount %= 64
    return ((value << amount) | (value >> (64 - amount))) & MASK64


def int_to_lanes(value: int) -> list[int]:
    return [(value >> (64 * lane)) & MASK64 for lane in range(25)]


def lanes_to_int(state: list[int]) -> int:
    value = 0
    for lane, word in enumerate(state):
        value |= (word & MASK64) << (64 * lane)
    return value


def theta(state: list[int]) -> None:
    columns = [0] * 5
    for x in range(5):
        for y in range(5):
            columns[x] ^= state[index(x, y)]
    delta = [columns[(x - 1) % 5] ^ rotl64(columns[(x + 1) % 5], 1) for x in range(5)]
    for x in range(5):
        for y in range(5):
            state[index(x, y)] ^= delta[x]


def rho_pi(state: list[int]) -> None:
    moved = [0] * 25
    for x in range(5):
        for y in range(5):
            new_x = y
            new_y = (2 * x + 3 * y) % 5
            moved[index(new_x, new_y)] = rotl64(state[index(x, y)], ROTATION[x][y])
    state[:] = moved


def apply_l_lanes(state: list[int]) -> list[int]:
    out = list(state)
    theta(out)
    rho_pi(out)
    return out


def apply_l_int(value: int) -> int:
    return lanes_to_int(apply_l_lanes(int_to_lanes(value)))


def apply_matrix_columns(columns: list[int], value: int) -> int:
    out = 0
    while value:
        low = value & -value
        bit = low.bit_length() - 1
        out ^= columns[bit]
        value ^= low
    return out


def build_l_columns() -> list[int]:
    return [apply_l_int(1 << bit) for bit in range(STATE_BITS)]


def invert_matrix_columns(columns: list[int]) -> list[int]:
    rows = [0] * STATE_BITS
    for col_index, col_value in enumerate(columns):
        value = col_value
        while value:
            low = value & -value
            row_index = low.bit_length() - 1
            rows[row_index] |= 1 << col_index
            value ^= low

    inverse_rows = [1 << i for i in range(STATE_BITS)]
    for pivot in range(STATE_BITS):
        pivot_row = None
        for row in range(pivot, STATE_BITS):
            if (rows[row] >> pivot) & 1:
                pivot_row = row
                break
        if pivot_row is None:
            raise ValueError("matrix is singular")
        if pivot_row != pivot:
            rows[pivot], rows[pivot_row] = rows[pivot_row], rows[pivot]
            inverse_rows[pivot], inverse_rows[pivot_row] = inverse_rows[pivot_row], inverse_rows[pivot]

        for row in range(STATE_BITS):
            if row != pivot and ((rows[row] >> pivot) & 1):
                rows[row] ^= rows[pivot]
                inverse_rows[row] ^= inverse_rows[pivot]

    inverse_columns = [0] * STATE_BITS
    for row_index, row_value in enumerate(inverse_rows):
        value = row_value
        while value:
            low = value & -value
            col_index = low.bit_length() - 1
            inverse_columns[col_index] |= 1 << row_index
            value ^= low
    return inverse_columns


def load_or_build_matrices() -> tuple[list[int], list[int]]:
    if CACHE_FILE.exists():
        with CACHE_FILE.open("rb") as handle:
            return pickle.load(handle)
    l_columns = build_l_columns()
    linv_columns = invert_matrix_columns(l_columns)
    with CACHE_FILE.open("wb") as handle:
        pickle.dump((l_columns, linv_columns), handle)
    return l_columns, linv_columns


def transpose_apply(columns: list[int], output_mask: int) -> int:
    """Return input mask c such that c*x == output_mask*M*x."""
    coeff = 0
    for input_bit, column in enumerate(columns):
        if (output_mask & column).bit_count() & 1:
            coeff |= 1 << input_bit
    return coeff


def round_constant_state(round_number: int) -> int:
    return ROUND_CONSTANTS[round_number]


def preimage_bit_equation(linv_columns: list[int], bit: int) -> int:
    coeff = 0
    for y_bit, column in enumerate(linv_columns):
        if (column >> bit) & 1:
            coeff |= 1 << y_bit
    return coeff
