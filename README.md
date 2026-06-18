# CA Final Project

This repository contains a Computer Architecture final project built around:

```text
LS Channel Estimation + LMMSE/MMSE Equalizer Acceleration
```

The workload is a pilot-based OFDM receiver with two computation stages:

1. LS channel estimation
2. One-tap LMMSE equalization

## Status

Currently implemented:

- `Part 1` — Scalar baseline
- `Part 2` — RVV vector reduction
- `Part 3` — SIMD-like RVV parallelization
- `Part 4` — CUDA SIMT single-pattern implementation

Prepared but not yet implemented:

- `Part 5` — Multi-pattern GPU parallelism

## Repository Structure

```text
common/      Shared parameters, data, I/O, model, verification
data_gen/    Host-side OFDM input generator
data/        Generated binary input
part1/       Scalar baseline
part2/       RVV reduction implementation
part3/       SIMD-like RVV implementation
part4/       CUDA SIMT implementation and experiment scripts
part5/       CUDA work directory placeholder
reference/   Course references and PDFs
docs/        Audit / supplementary notes
report.md    Report draft content
```

## Mathematical Model

Pilot model:

```text
X_pilot[p,k] = 1 + j0
Y_pilot[p,k] = H[k] + N_pilot[p,k]
```

LS channel estimation:

```text
Hhat[k] = sum_p Y_pilot[p,k] * w[p]
w[p] = 1 / P
```

Data model:

```text
Y_data[s,k] = H[k] * X_data[s,k] + N_data[s,k]
```

LMMSE equalizer:

```text
Xmmse[s,k] = Ydata[s,k] * conj(Hhat[k])
             / (|Hhat[k]|^2 + NOISE_VAR_OVER_SYMBOL_POWER + EPSILON)
```

## Build and Run

### Part 1

Host correctness check:

```bash
cd part1
make clean
make host
```

gem5 run:

```bash
cd part1
make clean
make
```

### Part 2

```bash
cd part2
make clean
make
```

### Part 3

```bash
cd part3
make clean
make
```

Each `make` run:

- regenerates `data/ofdm_input.bin` on the host
- cross-compiles the target program
- runs gem5 with `TimingSimpleCPU`
- prints key stats from `m5out/stats.txt`

## gem5 Results

| Metric | Part 1 | Part 2 | Part 3 |
| --- | --- | --- | --- |
| `simSeconds` | `0.069028` | `0.054827` | `0.072715` |
| `simInsts` | `17,066,142` | `11,395,259` | `11,208,742` |
| `numCycles` | `138,056,924` | `109,653,580` | `145,430,124` |
| `CPI` | `8.089506` | `9.622709` | `12.974667` |
| `IPC` | `0.123617` | `0.103921` | `0.077073` |
| `D-cache miss rate` | `0.114745` | `0.170969` | `0.229839` |
| `I-cache miss rate` | `0.000046` | `0.000071` | `0.000051` |

Interpretation:

- `Part 2` has the lowest `simSeconds` and `numCycles` in the current gem5 runs.
- `Part 3` keeps the required across-`k` SIMD-like mapping, but the strided `vlse32.v` path raises the miss rate and cycle count.

## Verification

All implemented parts currently pass:

```text
Verification = PASS
```

The shared validation checks:

- `H_MSE`
- `MSE_RX_BEFORE_EQ`
- `MSE_LMMSE`
- `checksum`

## Environment Notes

### RISC-V / gem5

- Uses `riscv64-linux-gnu-g++`
- Runs under gem5 in Docker
- `part2/` and `part3/` require RVV-enabled compilation

### CUDA

The CUDA environment is already usable for `Part 4`, and the same setup will be reused for `Part 5`:

- GPU: `NVIDIA GeForce RTX 3060`
- Compute Capability: `8.6`
- CUDA Docker runtime: available
- `nvcc 12.4`: available
- `-arch=sm_86`: verified
- CUDA kernel launch: verified
- CUDA event timing: verified
- PTX generation: verified
- PTXAS resource report: verified
- `ncu` / Nsight Compute: verified
- GPU performance counters: verified

Note:

- The host driver reports `CUDA Version 13.0`
- The project container uses `CUDA Toolkit 12.4.1`
- This combination is working correctly in the current environment

No extra CUDA setup is needed before continuing with `Part 4` or starting `Part 5`.
