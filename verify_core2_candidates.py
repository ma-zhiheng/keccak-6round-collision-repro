"""Verify CUDA-found candidates for the Table 7 core No. 2 experiment."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from keccak_state import round_int, rounds_int, squeeze_digest
from linear_layer import apply_matrix_columns, load_or_build_matrices
from trail_data import TRAIL_CORE_2_PARTIAL, state_from_matrix


@dataclass(frozen=True)
class Candidate:
    label: str
    message1: int
    message2: int


def parse_candidates(path: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    label = ""
    message1: int | None = None
    message2: int | None = None

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("FOUND "):
            label = line
            message1 = None
            message2 = None
            continue
        if line.startswith("M1 "):
            message1 = int(line.split(maxsplit=1)[1], 16)
            continue
        if line.startswith("M2 "):
            message2 = int(line.split(maxsplit=1)[1], 16)
            if not label:
                raise ValueError(f"M2 before candidate label in {path}")
            if message1 is None:
                raise ValueError(f"M2 before M1 in {path}")
            candidates.append(Candidate(label, message1, message2))

    return candidates


def candidate_id(label: str) -> str:
    match = re.search(r"sample=(\d+)", label)
    return match.group(1) if match else label


def digest_hex(message: int, digest_bits: int) -> str:
    width = (digest_bits + 3) // 4
    return f"{squeeze_digest(rounds_int(message, 5), digest_bits):0{width}x}"


def verify_candidate(
    candidate: Candidate,
    alpha0: int,
    alpha2: int,
    alpha3: int,
    alpha4: int,
    digest_bits: int,
) -> tuple[bool, dict[str, bool | str]]:
    state2_1 = rounds_int(candidate.message1, 2)
    state2_2 = rounds_int(candidate.message2, 2)
    state3_1 = round_int(state2_1, 2)
    state3_2 = round_int(state2_2, 2)
    state4_1 = round_int(state3_1, 3)
    state4_2 = round_int(state3_2, 3)
    state5_1 = round_int(state4_1, 4)
    state5_2 = round_int(state4_2, 4)

    diff5 = state5_1 ^ state5_2
    digest1 = squeeze_digest(state5_1, digest_bits)
    digest2 = squeeze_digest(state5_2, digest_bits)

    checks: dict[str, bool | str] = {
        "input_diff": (candidate.message1 ^ candidate.message2) == alpha0,
        "alpha2": (state2_1 ^ state2_2) == alpha2,
        "alpha3": (state3_1 ^ state3_2) == alpha3,
        "alpha4": (state4_1 ^ state4_2) == alpha4,
        "digest_zero": squeeze_digest(diff5, digest_bits) == 0,
        "digest_equal": digest1 == digest2,
        "digest": f"{digest1:0{(digest_bits + 3) // 4}x}",
    }
    ok = all(value for key, value in checks.items() if key != "digest")
    return ok, checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default="results/core2_cuda_candidates.txt",
        help="candidate file produced by core2_trail_search_cuda",
    )
    parser.add_argument("--digest-bits", type=int, default=160)
    args = parser.parse_args()

    path = Path(args.path)
    candidates = parse_candidates(path)

    _, linv = load_or_build_matrices()
    beta2 = state_from_matrix(TRAIL_CORE_2_PARTIAL.beta2)
    beta3 = state_from_matrix(TRAIL_CORE_2_PARTIAL.beta3)
    beta4 = state_from_matrix(TRAIL_CORE_2_PARTIAL.beta4)
    alpha2 = apply_matrix_columns(linv, beta2)
    alpha3 = apply_matrix_columns(linv, beta3)
    alpha4 = apply_matrix_columns(linv, beta4)

    if not candidates:
        raise SystemExit(f"no candidates found in {path}")

    alpha0 = candidates[0].message1 ^ candidates[0].message2
    all_ok = True
    print(f"verifying {len(candidates)} candidates from {path}")
    for index, candidate in enumerate(candidates, start=1):
        ok, checks = verify_candidate(
            candidate,
            alpha0,
            alpha2,
            alpha3,
            alpha4,
            args.digest_bits,
        )
        all_ok &= ok
        print(f"candidate {index} sample={candidate_id(candidate.label)} ok={ok}")
        print(f"  input_diff: {checks['input_diff']}")
        print(f"  alpha2/alpha3/alpha4: {checks['alpha2']}/{checks['alpha3']}/{checks['alpha4']}")
        print(f"  digest_zero/equal: {checks['digest_zero']}/{checks['digest_equal']}")
        print(f"  digest: {checks['digest']}")

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
