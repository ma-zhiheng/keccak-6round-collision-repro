# Keccak[1440,160,6,160] Reproduction Plan

This work directory is forked from the completed 5-round reproduction so the
6-round implementation can evolve without disturbing the known-good baseline.

## Target

Reproduce the practical collision attack on the Keccak challenge instance:

- permutation width: 1600
- rate/capacity: `r=1440`, `c=160`
- reduced rounds: `nr=6`
- digest bits: `d=160`

The plan follows Song, Liao, Guo, CRYPTO 2017 Section 7, with the journal
version's Table 11 trail data used for the sharper probability accounting.

## What Should Be Known Before Long Search

The final CUDA search should only be a probability experiment. Before running
it for many hours, the deterministic parts must already pass:

1. The local Keccak implementation verifies the printed 6-round collision.
2. The transcribed trail core verifies `L(alpha_i)=beta_i`.
3. The S-box DDT weights of the trail transitions match the paper.
4. A 2-round connector returns a message space that always reaches `alpha2`.
5. The adaptive 3-round connector returns subspaces that always reach `alpha3`.
6. A small CUDA calibration sees alpha/digest hit rates compatible with the
   expected differential weights.

Only after those checks pass should a long search be treated as "time and
probability", not as debugging.

## Paper Timing Baseline

For the 6-round target, the CRYPTO 2017 paper reports:

- first 2-round connector: 4.5 core hours, DF = 174
- adaptive 3-round connector output: each subspace has roughly `2^32..2^35`
  messages
- first collision: 112 hours on three NVIDIA GTX970 GPUs
- actual work: about `2^49.07` message-pair evaluations

The journal version reports the same target with a last-three-round probability
of about `2^-47.81` when multiple trails are counted. That is a better planning
number than the older single-trail-style estimate.

## Implementation Milestones

1. `verify_paper_6round_collision.py`
   Verify the printed Table 8 / Table 18 collision under local conventions.
   Status: implemented and passing.

2. `trail_data_6round.py`
   Hold the transcribed Table 11 trail core No. 5 and validate active S-box
   counts and transition weights.
   Status: `beta2`, `beta3`, and `beta4` are validated. `beta5` is currently
   marked provisional and is not used by connector code yet.

3. `core3_connector.py`
   Build the first 2-round connector, re-express its equations over the second
   round input, then enumerate adaptive non-full linearizations for round 2.
   Status: first 2-round connector builder and conservative third-round
   correctness scaffold are implemented.

4. `reproduce_core3_connector.py`
   Deterministically reproduce one or more 3-round connector subspaces and
   verify `R^3(M1)+R^3(M2)=alpha3` by direct Keccak execution.
   Status: staged entry point implemented. It waits for a successful first
   2-round connector seed.

5. `search_core3_first_two_parallel.py`
   Search independent seeds for the first 2-round connector.
   Status: implemented. Quick searches have not found a full connector yet.

6. `search_core3_beta_pairs.py`
   Pre-search and persist promising `beta1/beta0` pairs before spending time
   on row/model connector choices.
   Status: implemented. The best saved value-phase score currently has
   `first_eq=1034`, `ddt2=74`, `ddt8=61`; the best connector prefix currently
   comes from pair index 2, with `first_eq=1043`, `ddt2=83`, `ddt8=69`.

7. `search_core3_with_beta_pair.py`
   Reuse saved beta pairs and search many row/model seeds cheaply.
   Status: implemented. The best observed fixed-pair greedy prefix is `233/292`
   G rows for pair index 2 with row seed `1014163185563850369`.

8. `repair_core3_with_beta_pair.py`
   Rebuild from the best fixed-pair prefix, back up by a configurable window,
   and run model-level backtracking from that point.
   Status: implemented. Short repair runs did not improve over the best greedy
   prefix, so row-order steering became the more promising path.

9. `search_core3_row_reorder.py`
   Continue from the best fixed-pair connector attempt by applying chained
   targeted row reorders near the failing frontier.
   Status: implemented, but superseded by the paper-derived connector path.
   Random/model search reached `283/292` G rows and then became inefficient.

10. `derive_paper_first_two_connector.py`
   Use the printed 6-round collision as a deterministic successful path:
   derive `alpha0`, `beta0`, `alpha1`, `beta1`, force first-round S-box
   linearization planes containing the paper message values, and verify the
   first 2-round connector.
   Status: implemented and passing. First 2-round connector adds `292/292`
   rows with rank/dimension `1420/180`.

11. `build_known_value_bitwise_third_round_connector`
   Non-full third-round linearization: for each needed chi output bit, fix one
   neighboring input bit from the paper collision value, instead of full S-box
   plane linearization.
   Status: implemented and passing. The conservative full-linearized third
   connector has dimension `1`; the bitwise non-full connector has
   rank/dimension `1544/56` and direct samples verify `R^3=alpha3`.

12. `export_core3_cpp_data.py`
   Export multiple 3-round connector subspaces for C++/CUDA sampling.
   Status: implemented for the paper-derived bitwise third connector.

13. `core3_trail_search.cpp` / `core3_trail_search_cuda.cu`
   Search the remaining rounds from verified 3-round connector samples. The
   first mode should be calibration-only; the long mode appends candidates.
   Status: implemented. CPU calibration over `10^7` samples gives `alpha4=38`,
   matching the `2^-18` expectation. CUDA calibration over `10^9` samples on
   all GPUs gives `alpha3=10^9`, `alpha4=3893`, `digest=0`, rate about
   `668M/s` with `--threads-per-block 256 --blocks-per-sm 4`.

## Practical Run Strategy

Start small:

```bash
python verify_paper_6round_collision.py
python trail_data_6round.py
python derive_paper_first_two_connector.py --build-third
python export_core3_cpp_data.py
make core3-search core3-search-cuda
./core3_trail_search --samples 10000000 --threads 32 --report 2000000
./core3_trail_search_cuda --samples 1000000000 --devices all --max-alpha3-print 0 --threads-per-block 256 --blocks-per-sm 4
```

Then calibrate:

```bash
make core3-search-cuda
./core3_trail_search_cuda --samples 1000000000 --devices all --calibrate
```

Only after calibration matches expectations, run long searches in resumable
chunks and append candidates to `results/core3_cuda_candidates.txt`.
