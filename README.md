# Keccak 5/6-Round Collision Reproduction

This repository contains a reproduction-oriented implementation for the
Keccak collision attacks developed in:

- Ling Song, Guohong Liao, Jian Guo,
  **"Non-full Sbox Linearization: Applications to Collision Attacks on
  Round-Reduced Keccak"**, CRYPTO 2017.
- Jian Guo, Guohong Liao, Guozhen Liu, Meicheng Liu, Kexin Qiao, Ling Song,
  **"Practical Collision Attacks against Round-Reduced SHA-3"**,
  Journal of Cryptology 2020.

The code is forked from a completed reproduction of Qiao et al.'s 2017
two-round connector attack. It extends that baseline toward the CRYPTO 2017 /
J. Cryptol 6-round Keccak challenge attack using non-full S-box linearization.

## Current Status

Implemented and checked locally:

- Keccak-f[1600] round implementation and state/digest helpers.
- Keccak 5-bit chi S-box, DDT, affine subspace enumeration, and the
  linearization observations used by the connector papers.
- The completed 5-round baseline for `Keccak[1440,160,5,160]`:
  - Table 7 trail core No. 2 validation;
  - two-round connector reproduction;
  - C++/CUDA post-connector search tooling;
  - saved collision candidates in `results/core2_cuda_candidates.txt`;
  - independent Python verification of those candidates.
- The 6-round `Keccak[1440,160,6,160]` staged reproduction:
  - the printed Table 18 collision from the journal paper verifies under the
    local Keccak implementation;
  - Table 11 trail core No. 5 is transcribed and partially validated;
  - a paper-derived first two-round connector is reconstructed from the printed
    collision path;
  - a conservative third-round connector and a bitwise non-full third-round
    connector are constructed;
  - direct Python samples verify `R^3(M1) xor R^3(M2) = alpha3`;
  - C++/CUDA search data for the 3-round connector can be exported.

Not completed yet:

- A long CUDA search for a newly found 6-round collision has not been run from
  this checkout.
- The `beta5` transcription in `trail_data_6round.py` is marked provisional and
  is not used by the connector code.
- The current 6-round route uses the paper's printed collision to derive a
  successful connector path. It is not yet a full independent re-search of all
  connector choices from scratch.

## Quick Checks

Run the baseline Python checks:

```bash
python run_all.py
```

Expected highlights:

```text
2-dimensional linearizable affine subspaces: 80
3-dimensional linearizable affine subspaces: 0
Table 7 core No. 2 connector reproduced
added G rows: 195/195
dimension: 180
verifies R^2(M1)+R^2(M2)=alpha2: True
```

Verify saved 5-round candidates:

```bash
python verify_core2_candidates.py results/core2_cuda_candidates.txt
```

Expected result: all candidates print `ok=True`.

Verify the printed 6-round collision from the paper:

```bash
python verify_paper_6round_collision.py
```

Expected highlights:

```text
collision: True
matches printed digest: True
```

Check the 6-round trail transcription:

```bash
python trail_data_6round.py
```

Expected highlights:

```text
table11_core5_keccak_1440_160_6_160
beta2->alpha3 compatible: True
beta3->alpha4 compatible: True
beta2->alpha3 weight=25: True
beta3->alpha4 weight=18: True
beta5 transcription is provisional and is not used yet
```

## Reproducing the 6-Round Connector Path

Construct the paper-derived first two-round connector and the third-round
non-full connector:

```bash
python derive_paper_first_two_connector.py --build-third
```

Expected highlights:

```text
paper-derived first 2-round connector
  added G rows: 292/292
  rank/dimension: 1420/180
  sample verifies R^2 target: True

paper-derived bitwise third-round connector
  third-round rows: 25
  rank/dimension: 1544/56
  sample verifies R^3 target: True
```

Export native search data:

```bash
python export_core3_cpp_data.py
```

Expected highlights:

```text
basis size: 56
connector rank/dimension: 1544/56
```

The script `reproduce_core3_connector.py` is a staged overview entry point. It
prints the expected long-search scale and validates the available trail data:

```bash
python reproduce_core3_connector.py
```

## Native Search

Build the CPU and CUDA samplers on a Linux server:

```bash
make core3-search
make core3-search-cuda
```

Small CPU calibration:

```bash
./core3_trail_search \
  --samples 10000000 \
  --threads 32 \
  --report 2000000
```

CUDA calibration on all GPUs:

```bash
./core3_trail_search_cuda \
  --samples 1000000000 \
  --devices all \
  --max-alpha3-print 0 \
  --threads-per-block 256 \
  --blocks-per-sm 4
```

Long searches should be run in resumable chunks and append candidates to a
result file, for example:

```bash
./core3_trail_search_cuda \
  --samples 100000000000 \
  --devices all \
  --seed 917306210421 \
  --max-alpha3-print 0 \
  --threads-per-block 256 \
  --blocks-per-sm 4 \
  --candidate-file results/core3_cuda_candidates.txt
```

Planning numbers from the staged script:

```text
expected 2^47.81 at 1e9/s: about 68.5 hours
paper actual 2^49.07 at 1e9/s: about 164.2 hours
```

These are probability estimates, not guarantees.

## Project Layout

Core modules:

- `keccak_state.py`: Keccak-f[1600] execution and digest helpers.
- `sbox_linearization.py`: chi S-box, DDT, and affine linearization checks.
- `sbox_constraints.py`: local S-box transition equations and linear models.
- `gf2.py`: GF(2) linear equation system.
- `linear_layer.py`: Keccak linear layer and inverse matrix helpers.
- `state_lift.py`: maps local S-box equations to 1600-bit state coordinates.
- `connector_equations.py`: full-linearized two-round connector equations.
- `incremental_connector.py`: incremental connector construction primitives.
- `core2_connector.py`: completed 5-round baseline connector.
- `core3_connector.py`: paper-derived and bitwise non-full 6-round connector
  helpers.

Trail and verification data:

- `trail_data.py`: Table 7 core No. 2 for the completed 5-round baseline.
- `trail_data_6round.py`: Table 11 core No. 5 for the 6-round target.
- `paper_collisions.py`: printed collision data helpers.
- `verify_paper_6round_collision.py`: verifies the printed 6-round collision.
- `verify_core2_candidates.py`: verifies saved 5-round candidates.

Main reproduction scripts:

- `run_all.py`: baseline Python self-checks.
- `reproduce_core2_connector.py`: deterministic 5-round baseline connector.
- `derive_paper_first_two_connector.py`: derives the 6-round first two-round
  connector and builds the third-round non-full connector.
- `reproduce_core3_connector.py`: staged 6-round overview and trail check.
- `export_core3_cpp_data.py`: exports the 3-round connector for native search.

Native search:

- `core2_trail_search.cpp`, `core2_trail_search_cuda.cu`: completed 5-round
  post-connector samplers.
- `core3_trail_search.cpp`, `core3_trail_search_cuda.cu`: 6-round
  post-connector samplers.
- `Makefile`: build targets.

Exploratory scripts:

- `search_core3_beta_pairs.py`
- `search_core3_first_two_parallel.py`
- `search_core3_first_two_repair.py`
- `search_core3_forced_models.py`
- `search_core3_row_reorder.py`
- `search_core3_with_beta_pair.py`
- `repair_core3_with_beta_pair.py`

These are kept to document the route taken before switching to the
paper-derived connector path.

## Notes

- This repository is for reduced-round cryptanalysis research and reproduction.
  It does not affect the security of full 24-round SHA-3.
- Generated matrix caches, compiled binaries, exported native headers, and
  long-search logs are ignored by `.gitignore`.
- `README_6ROUND_PLAN.md` contains a more detailed development log for the
  6-round work.
