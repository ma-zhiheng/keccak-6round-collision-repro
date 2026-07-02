"""Pure-Python digit detector for appendix trail-table images.

This is a fallback for environments without cv2/numpy. It uses ImageMagick's
`convert` to obtain grayscale pixels, then runs a tiny connected-component pass.
The output is only an auditing aid; trail data must still pass DDT validation.
"""

from __future__ import annotations

import argparse
import subprocess
from collections import deque
from pathlib import Path


TEMPLATES = {
    "1": [
        "00100",
        "01100",
        "00100",
        "00100",
        "00100",
        "00100",
        "01110",
    ],
    "2": [
        "01110",
        "10001",
        "00001",
        "00010",
        "00100",
        "01000",
        "11111",
    ],
    "4": [
        "00010",
        "00110",
        "01010",
        "10010",
        "11111",
        "00010",
        "00010",
    ],
    "8": [
        "01110",
        "10001",
        "10001",
        "01110",
        "10001",
        "10001",
        "01110",
    ],
    "C": [
        "01111",
        "10000",
        "10000",
        "10000",
        "10000",
        "10000",
        "01111",
    ],
}


def image_to_pgm(path: Path) -> tuple[int, int, bytes]:
    data = subprocess.check_output(["convert", str(path), "-colorspace", "Gray", "pgm:-"])
    tokens: list[bytes] = []
    index = 0
    while len(tokens) < 4:
        while data[index:index + 1].isspace():
            index += 1
        if data[index:index + 1] == b"#":
            while data[index:index + 1] != b"\n":
                index += 1
            continue
        start = index
        while index < len(data) and not data[index:index + 1].isspace():
            index += 1
        tokens.append(data[start:index])
    magic, width_b, height_b, max_b = tokens
    if magic != b"P5" or int(max_b) != 255:
        raise ValueError("expected binary PGM with max value 255")
    while data[index:index + 1].isspace():
        index += 1
    width = int(width_b)
    height = int(height_b)
    pixels = data[index:index + width * height]
    if len(pixels) != width * height:
        raise ValueError("truncated PGM data")
    return width, height, pixels


def component_boxes(width: int, height: int, pixels: bytes, threshold: int = 160) -> list[tuple[int, int, int, int, int]]:
    seen = bytearray(width * height)
    boxes: list[tuple[int, int, int, int, int]] = []
    for start in range(width * height):
        if seen[start] or pixels[start] >= threshold:
            continue
        seen[start] = 1
        q: deque[int] = deque([start])
        min_x = max_x = start % width
        min_y = max_y = start // width
        area = 0
        while q:
            pos = q.popleft()
            area += 1
            x = pos % width
            y = pos // width
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue
                nxt = ny * width + nx
                if seen[nxt] or pixels[nxt] >= threshold:
                    continue
                seen[nxt] = 1
                q.append(nxt)
        boxes.append((min_x, min_y, max_x - min_x + 1, max_y - min_y + 1, area))
    return boxes


def normalize(width: int, height: int, bitmap: set[tuple[int, int]]) -> list[str]:
    rows: list[str] = []
    for oy in range(7):
        chars: list[str] = []
        y0 = oy * height / 7
        y1 = (oy + 1) * height / 7
        for ox in range(5):
            x0 = ox * width / 5
            x1 = (ox + 1) * width / 5
            black = 0
            total = 0
            for y in range(int(y0), max(int(y0) + 1, int(y1 + 0.999))):
                for x in range(int(x0), max(int(x0) + 1, int(x1 + 0.999))):
                    if 0 <= x < width and 0 <= y < height:
                        total += 1
                        black += (x, y) in bitmap
            chars.append("1" if black * 2 >= max(total, 1) else "0")
        rows.append("".join(chars))
    return rows


def classify(width: int, height: int, pixels: bytes, box: tuple[int, int, int, int, int], threshold: int = 160) -> str:
    x, y, w, h, _area = box
    bitmap = {
        (xx, yy)
        for yy in range(h)
        for xx in range(w)
        if pixels[(y + yy) * width + (x + xx)] < threshold
    }
    norm = normalize(w, h, bitmap)
    best = "?"
    best_score = 10**9
    for digit, template in TEMPLATES.items():
        score = sum(a != b for row_a, row_b in zip(norm, template) for a, b in zip(row_a, row_b))
        if score < best_score:
            best = digit
            best_score = score
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    args = parser.parse_args()
    width, height, pixels = image_to_pgm(Path(args.image))
    boxes = component_boxes(width, height, pixels)
    filtered = [
        box for box in boxes
        if 8 <= box[2] <= 35 and 15 <= box[3] <= 45 and 30 <= box[4] <= 500
    ]
    for box in sorted(filtered, key=lambda item: (item[1], item[0])):
        digit = classify(width, height, pixels, box)
        x, y, w, h, area = box
        print(f"{digit} x={x:4d} y={y:4d} w={w:2d} h={h:2d} area={area:3d}")


if __name__ == "__main__":
    main()
