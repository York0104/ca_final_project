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
part5/       Multi-pattern CUDA implementation and experiment scripts
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

All commands below assume you start from the repository root:

```bash
cd /home/york/ca_final_project
```

### Environment Requirements

For `Part 1` to `Part 3` you need:

- Docker with the gem5 / RVV image from the course material, or an equivalent environment that provides:
  - `g++`
  - `riscv64-linux-gnu-g++`
  - gem5 at `/root/gem5/build/RISCV/gem5.opt`

For `Part 4` and `Part 5` you need:

- an NVIDIA GPU with a CUDA-capable driver
- Docker with the CUDA image from the course material, or an equivalent environment that provides:
  - `nvcc`
  - `ncu`
  - GPU access inside the container

The checked-in Markdown files summarize the current saved runs in this repository, but rebuilding and rerunning still requires the environments above.

### Part 1

Host correctness check:

```bash
make -C part1 clean
make -C part1 host
```

gem5 run:

```bash
make -C part1 clean
make -C part1
```

### Part 2

```bash
make -C part2 clean
make -C part2
```

### Part 3

```bash
make -C part3 clean
make -C part3
```

Each `make` run:

- regenerates `data/ofdm_input.bin` on the host
- cross-compiles the target program
- runs gem5 with `TimingSimpleCPU`
- prints key stats from `m5out/stats.txt`

Current gem5 output directories:

- `part1/m5out/`
- `part2/m5out/`
- `part3/m5out/`

### Part 4

```bash
make -C part4 clean
make -C part4
make -C part4 run
```

Extra commands:

```bash
make -C part4 ptx
make -C part4 profile
make -C part4 sweep
```

Current Part 4 result directory:

- `part4/results/`

### Part 5

```bash
make -C part5 clean
make -C part5
make -C part5 run
```

Optional CPU baseline:

```bash
make -C part5 run_cpu
```

Extra commands:

```bash
make -C part5 ptx
make -C part5 profile
make -C part5 sweep
```

Current Part 5 result directory:

- `part5/results/`

## Cleaning Old Results

Remove build outputs and regenerateable logs:

```bash
make -C part1 clean
make -C part2 clean
make -C part3 clean
make -C part4 clean
make -C part5 clean
```

The checked-in `m5out/` and `results/` folders contain the current saved measured outputs. They are report artifacts, not source code.

## gem5 Results

| Metric | Part 1 | Part 2 | Part 3 |
| --- | --- | --- | --- |
| `simSeconds` | `0.069028` | `0.054755` | `0.072706` |
| `simInsts` | `17,066,251` | `11,395,368` | `11,208,851` |
| `numCycles` | `138,055,892` | `109,510,852` | `145,412,866` |
| `CPI` | `8.089394` | `9.610092` | `12.973001` |
| `IPC` | `0.123619` | `0.104057` | `0.077083` |
| `D-cache miss rate` | `0.114745` | `0.170969` | `0.229838` |
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

For `Part 1` to `Part 3`, these values are printed by the executable itself and recorded in the checked-in Markdown summaries.

For `Part 4` and `Part 5`, the checked-in `results/*_run.txt` logs and the 2026-06-19 rerun copies under `rerun_outputs/` both include the same functional checks together with GPU timing output.

## Environment Notes

### RISC-V / gem5

- Uses `riscv64-linux-gnu-g++`
- Runs under gem5 in Docker
- `part2/` and `part3/` require RVV-enabled compilation

### CUDA

The CUDA environment is already being used for `Part 4` and `Part 5`:

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

## Result Files and Figures

Primary report-facing artifacts in the repository:

- `Overall_Results_Analysis.md`
- `part1/Part1_MMSE.md`
- `part2/Part2_RVV_Reduction.md`
- `part3/Part3_SIMD_Like_RVV.md`
- `part4/Part4_CUDA_SIMT.md`
- `part5/Part5_Multi_Pattern_GPU.md`
- `report.md`

Saved raw outputs:

- gem5 stats: `part1/m5out/stats.txt`, `part2/m5out/stats.txt`, `part3/m5out/stats.txt`
- CUDA run logs: `part4/results/*_run.txt`, `part5/results/*_run.txt`
- PTXAS / PTX / NCU: `part4/results/`, `part5/results/`

Generated figures:

- `figures/fig_part123_numcycles.svg`
- `figures/fig_part123_dcache_miss.svg`
- `figures/fig_part4_tpb_sweep.svg`
- `figures/fig_part5_pattern_scaling.svg`

Figure regeneration script:

```bash
python3 tools/generate_figures.py
```
