# CA Final Project Part 5

## Scope

Part 5 extends the Part 4 CUDA pipeline from one OFDM input pattern to multiple independent patterns.

The mathematical workload stays the same:

- Stage 1: LS channel estimation
- Stage 2: one-tap LMMSE equalization

The new part is the pattern dimension and the GPU mapping around it.

## Problem Statement

The assignment notes that the previous parts process only one input pattern. Part 5 should exploit another level of parallelism:

- parallelism across independent patterns

For this project, the concrete question is:

> If Part 4 already parallelizes within one OFDM frame, what changes when we also let the GPU process many OFDM frames at the same time?

## Requirement Mapping

From [reference/CA_Final_Project.pdf](/home/york/ca_final_project/reference/CA_Final_Project.pdf), the key Part 5 requirements are:

- use the GPU to process multiple input patterns
- exploit parallelism across independent patterns
- a 2D CUDA grid is appropriate
- `blockIdx.y` can be used as the pattern index
- compile with `nvcc`
- inspect PTXAS / PTX
- profile with `ncu --set basic`

## Implementation Mapping

### Input

Part 5 uses a dedicated multi-pattern binary input file:

```text
../data/ofdm_input_multi.bin
```

The generator is:

- [generate_ofdm_multi.cpp](/home/york/ca_final_project/part5/generate_ofdm_multi.cpp)

### Stage 1

`ls_channel_estimation_multi_kernel()`

- `blockIdx.x -> subcarrier k`
- `blockIdx.y -> pattern index`
- `threadIdx.x -> pilot contributions`

Stage 1 still uses a block-cooperative shared-memory reduction, but the grid now covers both:

- subcarrier dimension
- pattern dimension

### Stage 2

`lmmse_equalization_multi_kernel()`

- `blockIdx.x * blockDim.x + threadIdx.x -> flattened output index inside one pattern`
- `blockIdx.y -> pattern index`

Each thread still computes one output task. The difference from Part 4 is that many patterns are active together.

## Evidence in Code

- CUDA source: [main.cu](/home/york/ca_final_project/part5/main.cu)
- CPU baseline: [main_cpu.cpp](/home/york/ca_final_project/part5/main_cpu.cpp)
- Makefile: [Makefile](/home/york/ca_final_project/part5/Makefile)
- Sweep script: [run_experiments.sh](/home/york/ca_final_project/part5/run_experiments.sh)

The core evidence for multi-pattern mapping is:

- `dim3 grid_ls(NUM_SUBCARRIERS, num_patterns)`
- `dim3 grid_eq(ceil(TOTAL_DATA / TPB_EQ), num_patterns)`
- kernel code that reads `blockIdx.y` as the pattern index

## Verification

Part 5 keeps the same verification direction as Parts 1–4:

- `H_MSE < 0.01`
- `MSE_LMMSE < MSE_RX_BEFORE_EQ`
- `checksum != 0`

The CPU baseline provides an additional scalar reference path for the same multi-pattern input.

## Current Measured Result

The current default run uses:

- `NUM_PATTERNS = 16`
- `TPB_LS = 256`
- `TPB_EQ = 256`

Logs:

- [results/default_gpu_run.txt](/home/york/ca_final_project/part5/results/default_gpu_run.txt)
- [results/default_cpu_run.txt](/home/york/ca_final_project/part5/results/default_cpu_run.txt)

本節數字對齊 2026-06-19 的 fresh rerun。

Measured output:

| Metric | Value |
| --- | --- |
| `H_MSE` | `0.00001289` |
| `MSE_RX_BEFORE_EQ` | `0.14082038` |
| `MSE_LMMSE` | `0.00682142` |
| `LS_KERNEL_MS` | `0.128993` |
| `LMMSE_KERNEL_MS` | `0.308572` |
| `PIPELINE_KERNEL_MS` | `0.434780` |
| `Verification` | `PASS` |

CPU baseline on the same input:

| Metric | Value |
| --- | --- |
| `CPU_PIPELINE_MS` | `12.268151` |
| `H_MSE` | `0.00001289` |
| `MSE_LMMSE` | `0.00682142` |
| `Verification` | `PASS` |

This confirms that the multi-pattern CUDA path preserves the same OFDM computation and correctness checks.

## Pattern Sweep

The current sweep summary is stored in:

- [results/Part5_Experiment_Summary.md](/home/york/ca_final_project/part5/results/Part5_Experiment_Summary.md)

下列表格對齊目前 repo 內保存的最新 rerun sweep。

Summary table:

| Patterns | Verification | LS ms | LMMSE ms | Pipeline ms |
| --- | --- | --- | --- | --- |
| `1` | `PASS` | `0.032481` | `0.037750` | `0.069048` |
| `4` | `PASS` | `0.039578` | `0.117176` | `0.141583` |
| `8` | `PASS` | `0.059996` | `0.177853` | `0.214241` |
| `16` | `PASS` | `0.086354` | `0.347571` | `0.404536` |
| `32` | `PASS` | `0.223770` | `0.745861` | `0.926991` |

Interpretation:

- Total pipeline time increases as more patterns are processed, which is expected because total work also increases.
- The more useful view is time per pattern. In the current sweep, pipeline time per pattern drops from about `0.069048 ms` at `1` pattern to about `0.028968 ms` at `32` patterns.
- This matches the Part 5 goal: adding patterns gives the GPU more independent blocks and warps to schedule, so the work is amortized better than a one-pattern launch.

## Nsight Compute Evidence

The current Nsight Compute capture is:

- [results/ncu_default.txt](/home/york/ca_final_project/part5/results/ncu_default.txt)

For the default `16`-pattern run:

### Stage 1

- grid size: `8192`
- block size: `256`
- registers per thread: `16`
- dynamic shared memory per block: about `2.05 KB`
- achieved occupancy: about `89.53%`
- memory throughput: about `55.97%`
- compute throughput: about `55.97%`

### Stage 2

- grid size: `16384`
- block size: `256`
- registers per thread: `21`
- dynamic shared memory per block: `0`
- achieved occupancy: about `82.77%`
- memory throughput: about `91.74%`
- compute throughput: about `28.55%`

Interpretation:

- Stage 1 still looks like a shared-memory reduction kernel, now replicated across many patterns.
- Stage 2 remains the more memory-heavy kernel.
- The larger 2D launch gives the GPU many more blocks and waves per SM than the single-pattern version, which is the main architectural point of Part 5.

## PTXAS and PTX Evidence

The current build log is:

- [results/build.txt](/home/york/ca_final_project/part5/results/build.txt)

The current PTX file is:

- [main.ptx](/home/york/ca_final_project/part5/main.ptx)

From the current build:

- Stage 1 uses `15` registers per thread
- Stage 2 uses `21` registers per thread
- both kernels show `0` spill stores and `0` spill loads

The PTX file can be used to inspect:

- global load/store instructions in Stage 2
- shared-memory usage in Stage 1
- the 2D-grid kernel structure that extends the Part 4 design to multiple patterns

## Expected Performance Question

Part 5 is not mainly about changing the math inside one kernel. It is about changing how much independent work is available to the GPU.

The main performance questions are:

- why Part 5 should be more GPU-friendly than Part 4
- whether more patterns always improve performance
- when the gain starts to flatten
- how pattern count affects occupancy, active warps, waves per SM, and latency hiding

## Run Commands

Build and run:

```bash
cd part5
make
make run
```

CPU baseline:

```bash
cd part5
make run_cpu
```

PTX:

```bash
cd part5
make ptx
```

Nsight Compute:

```bash
cd part5
make profile
```

Pattern sweep:

```bash
cd part5
make sweep
```

## Current Status

Implemented:

- multi-pattern input generator
- CPU baseline
- CUDA Part 5 main program
- 2D grid mapping with `blockIdx.y`
- shared-memory Stage 1 reduction
- element-wise Stage 2 equalization
- sweep script for pattern-count experiments

Next step after implementation:

- extend the pattern sweep if needed
- compare Part 4 and Part 5 more directly in the final report
- decide how many pattern counts to include in the final submission figures
