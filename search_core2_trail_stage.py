"""Sample the reproduced connector and test the remaining Table 7 trail."""

from __future__ import annotations

import argparse
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from core2_connector import CORE2_CONNECTOR_SEED, build_reproduced_core2_connector
from keccak_state import round_int, rounds_int, squeeze_digest
from linear_layer import apply_l_int, apply_matrix_columns, load_or_build_matrices
from trail_data import TRAIL_CORE_2_PARTIAL, state_from_matrix


WORKER_PARTICULAR = 0
WORKER_BASIS: list[int] = []
WORKER_LINV: list[int] = []
WORKER_ALPHA0 = 0
WORKER_ALPHA2 = 0
WORKER_ALPHA3 = 0
WORKER_ALPHA4 = 0
WORKER_DIGEST_BITS = 160


@dataclass(frozen=True)
class TrailStageResult:
    follows_alpha2: bool
    follows_alpha3: bool
    follows_alpha4: bool
    digest_zero: bool
    output_difference: int


@dataclass(frozen=True)
class WorkerResult:
    samples: int
    hits_alpha2: int
    hits_alpha3: int
    hits_alpha4: int
    hits_digest: int
    alpha3_pair: tuple[int, int] | None = None
    hit_pair: tuple[int, int] | None = None


def random_solution(particular: int, basis: list[int], rng: random.Random) -> int:
    value = particular
    for vector in basis:
        if rng.getrandbits(1):
            value ^= vector
    return value


def init_worker(
    particular: int,
    basis: list[int],
    linv: list[int],
    alpha0: int,
    alpha2: int,
    alpha3: int,
    alpha4: int,
    digest_bits: int,
) -> None:
    global WORKER_PARTICULAR, WORKER_BASIS, WORKER_LINV, WORKER_ALPHA0
    global WORKER_ALPHA2, WORKER_ALPHA3, WORKER_ALPHA4, WORKER_DIGEST_BITS
    WORKER_PARTICULAR = particular
    WORKER_BASIS = basis
    WORKER_LINV = linv
    WORKER_ALPHA0 = alpha0
    WORKER_ALPHA2 = alpha2
    WORKER_ALPHA3 = alpha3
    WORKER_ALPHA4 = alpha4
    WORKER_DIGEST_BITS = digest_bits


def evaluate_pair(message1: int, message2: int, alpha2: int, alpha3: int, alpha4: int, digest_bits: int) -> TrailStageResult:
    state2_1 = rounds_int(message1, 2)
    state2_2 = rounds_int(message2, 2)
    diff2 = state2_1 ^ state2_2
    follows_alpha2 = diff2 == alpha2

    state3_1 = round_int(state2_1, 2)
    state3_2 = round_int(state2_2, 2)
    diff3 = state3_1 ^ state3_2
    follows_alpha3 = diff3 == alpha3

    state4_1 = round_int(state3_1, 3)
    state4_2 = round_int(state3_2, 3)
    diff4 = state4_1 ^ state4_2
    follows_alpha4 = diff4 == alpha4

    state5_1 = round_int(state4_1, 4)
    state5_2 = round_int(state4_2, 4)
    output_difference = state5_1 ^ state5_2
    digest_zero = squeeze_digest(output_difference, digest_bits) == 0

    return TrailStageResult(
        follows_alpha2=follows_alpha2,
        follows_alpha3=follows_alpha3,
        follows_alpha4=follows_alpha4,
        digest_zero=digest_zero,
        output_difference=output_difference,
    )


def run_worker(seed: int, samples: int) -> WorkerResult:
    rng = random.Random(seed)
    hits_alpha2 = 0
    hits_alpha3 = 0
    hits_alpha4 = 0
    hits_digest = 0
    alpha3_pair = None
    for _ in range(samples):
        x = random_solution(WORKER_PARTICULAR, WORKER_BASIS, rng)
        message1 = apply_matrix_columns(WORKER_LINV, x)
        message2 = message1 ^ WORKER_ALPHA0
        result = evaluate_pair(
            message1,
            message2,
            WORKER_ALPHA2,
            WORKER_ALPHA3,
            WORKER_ALPHA4,
            WORKER_DIGEST_BITS,
        )
        hits_alpha2 += int(result.follows_alpha2)
        hits_alpha3 += int(result.follows_alpha3)
        hits_alpha4 += int(result.follows_alpha4)
        hits_digest += int(result.digest_zero)
        if result.follows_alpha3 and alpha3_pair is None:
            alpha3_pair = (message1, message2)
        if result.follows_alpha4 and result.digest_zero:
            return WorkerResult(
                samples=hits_alpha2,
                hits_alpha2=hits_alpha2,
                hits_alpha3=hits_alpha3,
                hits_alpha4=hits_alpha4,
                hits_digest=hits_digest,
                alpha3_pair=alpha3_pair,
                hit_pair=(message1, message2),
            )
    return WorkerResult(
        samples=samples,
        hits_alpha2=hits_alpha2,
        hits_alpha3=hits_alpha3,
        hits_alpha4=hits_alpha4,
        hits_digest=hits_digest,
        alpha3_pair=alpha3_pair,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=CORE2_CONNECTOR_SEED)
    parser.add_argument("--digest-bits", type=int, default=160)
    parser.add_argument("--max-basis", type=int, default=0, help="0 means use the whole connector basis")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=10000)
    args = parser.parse_args()

    _, linv = load_or_build_matrices()
    reproduction = build_reproduced_core2_connector()
    connector = reproduction.connector
    alpha2 = reproduction.alpha2
    beta3 = state_from_matrix(TRAIL_CORE_2_PARTIAL.beta3)
    beta4 = state_from_matrix(TRAIL_CORE_2_PARTIAL.beta4)
    alpha3 = apply_matrix_columns(linv, beta3)
    alpha4 = apply_matrix_columns(linv, beta4)

    if apply_l_int(alpha2) != state_from_matrix(TRAIL_CORE_2_PARTIAL.beta2):
        raise RuntimeError("alpha2 does not match beta2")
    if apply_l_int(alpha3) != beta3:
        raise RuntimeError("alpha3 does not match beta3")
    if apply_l_int(alpha4) != beta4:
        raise RuntimeError("alpha4 does not match beta4")

    particular = connector.system.particular_solution()
    basis = connector.system.nullspace_basis()
    if args.max_basis:
        basis = basis[: args.max_basis]

    rng = random.Random(args.seed)
    hits_alpha2 = 0
    hits_alpha3 = 0
    hits_alpha4 = 0
    hits_digest = 0
    first_alpha3_pair: tuple[int, int] | None = None
    best_message_pair: tuple[int, int] | None = None
    start = time.time()

    if args.workers <= 1:
        for index in range(1, args.samples + 1):
            x = random_solution(particular, basis, rng)
            message1 = apply_matrix_columns(linv, x)
            message2 = message1 ^ reproduction.beta0_choice.alpha0
            result = evaluate_pair(message1, message2, alpha2, alpha3, alpha4, args.digest_bits)

            hits_alpha2 += int(result.follows_alpha2)
            hits_alpha3 += int(result.follows_alpha3)
            hits_alpha4 += int(result.follows_alpha4)
            hits_digest += int(result.digest_zero)
            if result.follows_alpha3 and first_alpha3_pair is None:
                first_alpha3_pair = (message1, message2)
            if result.follows_alpha4 and result.digest_zero:
                best_message_pair = (message1, message2)
                print("  FOUND full trail/digest candidate", flush=True)
                break
            if index % max(1, args.samples // 10) == 0:
                print(
                    f"  sampled={index}, elapsed={time.time()-start:.1f}s, "
                    f"alpha3_hits={hits_alpha3}, alpha4_hits={hits_alpha4}, digest_hits={hits_digest}",
                    flush=True,
                )
    else:
        workers = min(args.workers, os.cpu_count() or 1)
        chunk_count = (args.samples + args.chunk_size - 1) // args.chunk_size
        jobs = []
        for chunk_index in range(chunk_count):
            chunk_samples = min(args.chunk_size, args.samples - chunk_index * args.chunk_size)
            jobs.append((rng.randrange(1 << 60), chunk_samples))
        index = 0
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_worker,
            initargs=(
                particular,
                basis,
                linv,
                reproduction.beta0_choice.alpha0,
                alpha2,
                alpha3,
                alpha4,
                args.digest_bits,
            ),
        ) as pool:
            futures = [pool.submit(run_worker, seed, samples) for seed, samples in jobs]
            for future in as_completed(futures):
                result = future.result()
                index += result.samples
                hits_alpha2 += result.hits_alpha2
                hits_alpha3 += result.hits_alpha3
                hits_alpha4 += result.hits_alpha4
                hits_digest += result.hits_digest
                if result.alpha3_pair is not None and first_alpha3_pair is None:
                    first_alpha3_pair = result.alpha3_pair
                if result.hit_pair is not None:
                    best_message_pair = result.hit_pair
                    print("  FOUND full trail/digest candidate", flush=True)
                    pool.shutdown(cancel_futures=True)
                    break
                print(
                    f"  sampled={index}, elapsed={time.time()-start:.1f}s, "
                    f"alpha3_hits={hits_alpha3}, alpha4_hits={hits_alpha4}, digest_hits={hits_digest}",
                    flush=True,
                )

    print("Table 7 core No. 2 post-connector sampling")
    print(f"  samples: {index}")
    print(f"  connector dimension: {connector.system.dimension}")
    print(f"  basis used: {len(basis)}")
    print(f"  alpha2 hits: {hits_alpha2}")
    print(f"  alpha3 hits: {hits_alpha3}")
    print(f"  alpha4 hits: {hits_alpha4}")
    print(f"  digest-zero hits ({args.digest_bits} bits): {hits_digest}")
    if first_alpha3_pair is not None:
        message1, message2 = first_alpha3_pair
        print("  first alpha3 hit:")
        print(f"    M1: {message1:0400x}")
        print(f"    M2: {message2:0400x}")
    if best_message_pair is not None:
        message1, message2 = best_message_pair
        print(f"  M1: {message1:0400x}")
        print(f"  M2: {message2:0400x}")


if __name__ == "__main__":
    main()
