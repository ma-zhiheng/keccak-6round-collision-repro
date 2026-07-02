"""Experimental OCR for appendix trail tables.

The appendix tables are graphics rather than text. This helper detects digit
components in a cropped table image and maps them to 5x5 lanes. It is meant as
an auditing aid, not as a trusted parser.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


DIGIT_TEMPLATES = {
    "1": np.array([
        "00100",
        "01100",
        "00100",
        "00100",
        "00100",
        "00100",
        "01110",
    ]),
    "2": np.array([
        "01110",
        "10001",
        "00001",
        "00010",
        "00100",
        "01000",
        "11111",
    ]),
    "4": np.array([
        "00010",
        "00110",
        "01010",
        "10010",
        "11111",
        "00010",
        "00010",
    ]),
    "8": np.array([
        "01110",
        "10001",
        "10001",
        "01110",
        "10001",
        "10001",
        "01110",
    ]),
}


def normalize_component(binary: np.ndarray) -> np.ndarray:
    resized = cv2.resize(binary.astype(np.uint8), (5, 7), interpolation=cv2.INTER_AREA)
    return (resized > 0).astype(np.uint8)


def classify_digit(binary: np.ndarray) -> str:
    norm = normalize_component(binary)
    best_digit = "?"
    best_score = 10**9
    for digit, template_strings in DIGIT_TEMPLATES.items():
        template = np.array([[int(ch) for ch in row] for row in template_strings], dtype=np.uint8)
        score = int(np.sum(norm != template))
        if score < best_score:
            best_score = score
            best_digit = digit
    return best_digit


def detect_digit_components(image_path: Path) -> list[tuple[int, int, int, int, str]]:
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(image_path)
    _, threshold = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(threshold, 8)
    result: list[tuple[int, int, int, int, str]] = []
    for label in range(1, nlabels):
        x, y, w, h, area = stats[label]
        if h < 8 or h > 40 or w < 2 or w > 25 or area < 10:
            continue
        component = threshold[y:y + h, x:x + w] > 0
        digit = classify_digit(component)
        result.append((x, y, w, h, digit))
    return sorted(result, key=lambda item: (item[1], item[0]))


def main() -> None:
    path = Path("results/crops/core2_full.png")
    components = detect_digit_components(path)
    print(f"components: {len(components)}")
    for x, y, w, h, digit in components:
        print(f"{digit} x={x:4d} y={y:4d} w={w:2d} h={h:2d}")


def components_to_matrix(
    components: list[tuple[int, int, int, int, str]],
    row_centers: list[int],
    x_left: int = 82,
    lane_width: int = 189,
    char_width: float = 11.8,
) -> str:
    lanes = [["-" * 16 for _ in range(5)] for _ in range(5)]
    mutable = [[list(lane) for lane in row] for row in lanes]
    for x, y, w, h, digit in components:
        if w <= 3:
            continue
        center_y = y + h / 2
        row = min(range(5), key=lambda i: abs(center_y - row_centers[i]))
        if abs(center_y - row_centers[row]) > 10:
            continue
        center_x = x + w / 2
        lane = int((center_x - x_left) // lane_width)
        if not 0 <= lane < 5:
            continue
        char = int(round((center_x - (x_left + lane * lane_width)) / char_width))
        char = max(0, min(15, char))
        mutable[row][lane][char] = digit
    rows = ["|" + "|".join("".join(lane) for lane in row) + "|" for row in mutable]
    return "\n".join(rows)


def dump_core2_guess() -> None:
    components = detect_digit_components(Path("results/crops/core2_full.png"))
    groups = {
        "beta2": [66, 91, 115, 139, 163],
        "beta3": [188, 212, 236, 260, 284],
        "beta4": [310, 334, 358, 382, 406],
    }
    for name, centers in groups.items():
        print(name)
        print(components_to_matrix(components, centers))
        print()


if __name__ == "__main__":
    dump_core2_guess()
