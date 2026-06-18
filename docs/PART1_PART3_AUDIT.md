# Part 1–3 Audit

## Scope

This audit covers only the currently implemented work:

- `Part 1` — Scalar baseline
- `Part 2` — RVV vector reduction
- `Part 3` — SIMD-like RVV parallelization

`Part 4` and `Part 5` are not implemented and are intentionally excluded from this audit.

## Reference Check

Authoritative specification:

- [reference/CA_Final_Project.pdf](/home/york/ca_final_project/reference/CA_Final_Project.pdf)

Relevant assignment requirements confirmed from the PDF:

- Part 1 must be a non-toy, computationally intensive C/C++ workload.
- At least one nested-loop portion must contain a reduction form
  `f(ai[1], bi[1]) + ... + f(ai[n], bi[n])`.
- Part 2 must use actual RVV vector reduction.
- Part 3 must use SIMD-like across-iteration RVV mapping, strided access, and no vector reduction.

## Repository Inventory

Implemented source files:

- [common/ofdm_params.h](/home/york/ca_final_project/common/ofdm_params.h)
- [common/ofdm_data.h](/home/york/ca_final_project/common/ofdm_data.h)
- [common/ofdm_io.h](/home/york/ca_final_project/common/ofdm_io.h)
- [common/ofdm_model.h](/home/york/ca_final_project/common/ofdm_model.h)
- [common/ofdm_verify.h](/home/york/ca_final_project/common/ofdm_verify.h)
- [data_gen/generate_ofdm_data.cpp](/home/york/ca_final_project/data_gen/generate_ofdm_data.cpp)
- [part1/main.cpp](/home/york/ca_final_project/part1/main.cpp)
- [part2/main.cpp](/home/york/ca_final_project/part2/main.cpp)
- [part3/main.cpp](/home/york/ca_final_project/part3/main.cpp)

Build entry points:

- [part1/Makefile](/home/york/ca_final_project/part1/Makefile)
- [part2/Makefile](/home/york/ca_final_project/part2/Makefile)
- [part3/Makefile](/home/york/ca_final_project/part3/Makefile)

Documentation updated to match code:

- [part1/Part1_MMSE.md](/home/york/ca_final_project/part1/Part1_MMSE.md)
- [part2/Part2_RVV_Reduction.md](/home/york/ca_final_project/part2/Part2_RVV_Reduction.md)
- [part3/Part3_SIMD_Like_RVV.md](/home/york/ca_final_project/part3/Part3_SIMD_Like_RVV.md)
- [Overall_Results_Analysis.md](/home/york/ca_final_project/Overall_Results_Analysis.md)

## Algorithm Mapping

Project workload:

- Stage 1: pilot-based LS channel estimation
- Stage 2: one-tap LMMSE equalization

Reduction form used for Part 1 / Part 2 / Part 3 Stage 1:

```text
Hhat[k] = sum_p Ypilot[k,p] * pilot_w[p]
```

Mapping to assignment notation:

- outer iteration `i` -> subcarrier `k`
- inner index -> pilot index `p`
- `a[p] = Ypilot[k,p]`
- `b[p] = pilot_w[p]`
- `f(a,b) = a * b`

This satisfies the required nested-loop reduction form while still being a full OFDM receiver pipeline rather than a toy dot product demo.

## Compliance Table

| Requirement | Part 1 | Part 2 | Part 3 | Evidence |
| --- | --- | --- | --- | --- |
| Non-toy C/C++ workload | Yes | Yes | Yes | Two-stage OFDM receiver pipeline |
| Nested-loop reduction form | Yes | Yes | Yes | `estimate_channel_*()` loops |
| Scalar baseline only | Yes | N/A | N/A | Part 1 binary has no RVV instructions |
| Actual RVV reduction | N/A | Yes | No by design | `vfredusum.vs` in Part 2 binary |
| SIMD-like lane-per-output mapping | N/A | N/A | Yes | Part 3 Stage 1 uses `vlse32.v` across `k` |
| No vector reduction in Part 3 | N/A | N/A | Yes | No `vfred*` or `vred*` in Part 3 binary |
| gem5 runnable | Yes | Yes | Yes | All three `make` runs completed |
| Output correctness maintained | Yes | Yes | Yes | `Verification = PASS` in all three runs |

## Build Matrix

Environment used:

- Docker image: `weisheng505/gem5-rvv-image:v1`
- gem5: `/root/gem5/build/RISCV/gem5.opt`

Commands executed:

| Component | Clean | Build / Run | Result |
| --- | --- | --- | --- |
| Part 1 host | `make clean` | `make host` | PASS |
| Part 1 gem5 | `make clean` | `make` | PASS |
| Part 2 gem5 | `make clean` | `make` | PASS |
| Part 3 gem5 | `make clean` | `make` | PASS |

## Numerical Validation

| Metric | Part 1 | Part 2 | Part 3 |
| --- | --- | --- | --- |
| `H_MSE` | `0.00001250` | `0.00001250` | `0.00001250` |
| `MSE_RX_BEFORE_EQ` | `0.14093372` | `0.14093372` | `0.14093372` |
| `MSE_LMMSE` | `0.00681053` | `0.00681052` | `0.00681053` |
| `Verification` | `PASS` | `PASS` | `PASS` |

The tiny checksum and `MSE_LMMSE` differences are consistent with floating-point ordering differences in RVV execution.

## RVV Disassembly Evidence

Part 1 scalar baseline:

- `riscv64-linux-gnu-objdump -d main` shows no `vsetvli`, `vle32.v`, `vlse32.v`, `vfredusum.vs`, or `vfdiv.vv`.

Part 2:

- Stage 1 reduction evidence:
  - `vsetvli`
  - `vle32.v`
  - `vfmul.vv`
  - `vfredusum.vs`
- Stage 2 RVV equalizer evidence:
  - `vfmacc.vv`
  - `vfdiv.vv`
  - `vse32.v`

Part 3:

- Stage 1 SIMD-like evidence:
  - `vsetvli`
  - `vlse32.v`
  - `vle32.v`
  - `vse32.v`
- Stage 2 RVV equalizer evidence:
  - `vfmacc.vv`
  - `vfdiv.vv`
  - `vse32.v`
- Reduction absence check:
  - no `vfredusum.vs`
  - no `vredsum.vs`

## gem5 Statistics

| Metric | Part 1 | Part 2 | Part 3 |
| --- | --- | --- | --- |
| `simSeconds` | `0.069028` | `0.054827` | `0.072715` |
| `simInsts` | `17,066,142` | `11,395,259` | `11,208,742` |
| `numCycles` | `138,056,924` | `109,653,580` | `145,430,124` |
| `CPI` | `8.089506` | `9.622709` | `12.974667` |
| `IPC` | `0.123617` | `0.103921` | `0.077073` |
| `D-cache miss rate` | `0.114745` | `0.170969` | `0.229839` |
| `I-cache miss rate` | `0.000046` | `0.000071` | `0.000051` |

## Confirmed Fixes

| Classification | Fix |
| --- | --- |
| `RVV_COMPLIANCE` | Replaced the overly strict `__riscv_vector` guard with `OFDM_HAS_RVV`, which matches the course toolchain macros |
| `BLOCKER` | Fixed inline asm `vl` output constraints to early-clobber form so pointer registers are not accidentally overwritten |
| `DOCUMENTATION` | Updated Markdown files to match actual binaries, current measurements, and current project scope |
| `ROBUSTNESS` | Added consistent `Verification = PASS/FAIL` output and initialized Part 1 `g_sink` like Parts 2–3 |

## Remaining Notes

- The project currently uses fixed compile-time dimensions: `512` subcarriers, `256` pilots, `512` data symbols.
- Part 3 chooses its active lane count through `vsetvli`; on the current gem5 configuration this means up to 8 FP32 lanes for `e32,m1`.
- `Part 4` and `Part 5` remain future work only.

## Final Verdict

| Part | Verdict | Reason |
| --- | --- | --- |
| Part 1 | Compliant | Scalar nested-loop reduction baseline under gem5 |
| Part 2 | Compliant | Uses actual RVV reduction and preserves correctness |
| Part 3 | Compliant with documented limitations | Uses strided SIMD-like RVV and no reduction; performance is worse than Part 2 because of memory behavior, not because of a compliance failure |
