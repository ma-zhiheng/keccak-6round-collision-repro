"""Parsing helpers for Keccak trail matrices in Qiao et al. appendices."""

from __future__ import annotations

import re
from dataclasses import dataclass

from state_lift import LANE_SIZE, STATE_BITS_1600, bit_position, lane_index


LANE_RE = re.compile(r"[0-9A-Fa-f-]+")


@dataclass(frozen=True)
class TrailCore:
    name: str
    beta_states: tuple[int, ...]


def parse_lane_hex(text: str, lane_bits: int = 64) -> int:
    """Parse a paper lane: hexadecimal little-endian with '-' replacing zero."""
    cleaned = text.strip().replace("-", "0")
    if not cleaned:
        return 0
    value = int(cleaned, 16)
    if value >= (1 << lane_bits):
        raise ValueError(f"lane exceeds {lane_bits} bits: {text}")
    return value


def state_from_lanes(lanes: list[int], lane_bits: int = 64) -> int:
    if len(lanes) != 25:
        raise ValueError(f"expected 25 lanes, got {len(lanes)}")
    state = 0
    for y in range(5):
        for x in range(5):
            lane = lanes[lane_index(x, y)]
            for z in range(lane_bits):
                if (lane >> z) & 1:
                    state |= 1 << bit_position(x, y, z)
    return state


def lanes_from_state(state: int, lane_bits: int = 64) -> list[int]:
    lanes = [0] * 25
    for y in range(5):
        for x in range(5):
            lane = 0
            for z in range(lane_bits):
                if (state >> bit_position(x, y, z)) & 1:
                    lane |= 1 << z
            lanes[lane_index(x, y)] = lane
    return lanes


def parse_state_matrix(text: str, lane_bits: int = 64) -> int:
    """Parse a 5x5 matrix of lanes separated by vertical bars.

    The parser accepts either full rows like:
        |----|0001|----|----|----|
    or whitespace-separated lane tokens. Exactly 25 lane tokens are required.
    """
    tokens = LANE_RE.findall(text)
    tokens = [token for token in tokens if token]
    if len(tokens) != 25:
        raise ValueError(f"expected 25 lane tokens, got {len(tokens)}")
    lanes = [parse_lane_hex(token, lane_bits) for token in tokens]
    return state_from_lanes(lanes, lane_bits)


def format_state_matrix(state: int, lane_bits: int = 64, zero: str = "-") -> str:
    lanes = lanes_from_state(state, lane_bits)
    width = lane_bits // 4
    rows = []
    for y in range(5):
        cells = []
        for x in range(5):
            lane = lanes[lane_index(x, y)]
            text = f"{lane:0{width}X}"
            if zero == "-":
                text = text.replace("0", "-")
            cells.append(text)
        rows.append("|" + "|".join(cells) + "|")
    return "\n".join(rows)


def active_sboxes(state: int, lane_bits: int = 64) -> int:
    count = 0
    for y in range(5):
        for z in range(lane_bits):
            value = 0
            for x in range(5):
                value |= ((state >> bit_position(x, y, z)) & 1) << x
            if value:
                count += 1
    return count


def self_test() -> None:
    text = """
    |----------------|----------------|---------------1|-------4--------|----------------|
    |----------------|----------------|----------------|----------------|----------------|
    |----------------|----------------|----------------|----------------|----------------|
    |----------------|----------------|----------------|-------4--------|----8-----------|
    |----------------|----------------|---------------1|----------------|----8-----------|
    """
    state = parse_state_matrix(text)
    assert active_sboxes(state) == 6
    rendered = format_state_matrix(state)
    assert parse_state_matrix(rendered) == state
    assert state.bit_length() <= STATE_BITS_1600


if __name__ == "__main__":
    self_test()
    print("trail parser checks passed")
