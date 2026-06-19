# Part 5

## Scope

Part 5 is the multi-pattern GPU extension of the Part 4 pipeline:

- Stage 1: LS channel estimation
- Stage 2: one-tap LMMSE equalization

The key difference from Part 4 is that Part 5 processes multiple independent OFDM input patterns in one GPU launch sequence.

Default logs are kept in:

- [results/default_gpu_run.txt](/home/york/ca_final_project/part5/results/default_gpu_run.txt)
- [results/default_cpu_run.txt](/home/york/ca_final_project/part5/results/default_cpu_run.txt)

## Problem and Goal

According to [reference/CA_Final_Project.pdf](/home/york/ca_final_project/reference/CA_Final_Project.pdf), the previous parts only process one input pattern. Part 5 should move to:

- multiple independent patterns
- GPU parallelism across patterns
- 2D CUDA grid mapping

In this repository, that means:

- keep the same OFDM math as Part 4
- keep the same Stage 1 shared-memory reduction
- keep the same Stage 2 thread-per-output mapping
- add a pattern dimension and map it to `blockIdx.y`

## File Layout

- source: [main.cu](/home/york/ca_final_project/part5/main.cu)
- CPU baseline: [main_cpu.cpp](/home/york/ca_final_project/part5/main_cpu.cpp)
- multi-pattern generator: [generate_ofdm_multi.cpp](/home/york/ca_final_project/part5/generate_ofdm_multi.cpp)
- local format helpers: [ofdm_multi_common.h](/home/york/ca_final_project/part5/ofdm_multi_common.h)
- build script: [Makefile](/home/york/ca_final_project/part5/Makefile)
- experiment script: [run_experiments.sh](/home/york/ca_final_project/part5/run_experiments.sh)

## Data Model

The generator writes a dedicated multi-pattern input file:

```text
../data/ofdm_input_multi.bin
```

Each pattern uses the same OFDM dimensions as Parts 1–4:

- `NUM_SUBCARRIERS = 512`
- `NUM_PILOTS = 256`
- `NUM_DATA_SYMBOLS = 512`

Patterns share the same pilot weights and deep-fade channel shape, while QPSK symbols and AWGN samples vary deterministically with the pattern index.

## CUDA Mapping

### Stage 1

`ls_channel_estimation_multi_kernel()`

- `blockIdx.x -> subcarrier k`
- `blockIdx.y -> pattern index`
- `threadIdx.x -> pilot contributions`
- `__shared__` is used for block-level reduction

### Stage 2

`lmmse_equalization_multi_kernel()`

- `blockIdx.x * blockDim.x + threadIdx.x -> flattened data index inside one pattern`
- `blockIdx.y -> pattern index`
- one thread computes one `Xmmse[s,k]`

This is the 2D grid mapping requested by the assignment:

- x-dimension: output/thread index
- y-dimension: pattern index

## Build

```bash
cd part5
make
```

## Run

```bash
cd part5
make run
```

Optional CPU baseline:

```bash
cd part5
make run_cpu
```

## PTX and Nsight Compute

```bash
cd part5
make ptx
make profile
```

Current analysis artifacts:

- [main.ptx](/home/york/ca_final_project/part5/main.ptx)
- [results/build.txt](/home/york/ca_final_project/part5/results/build.txt)
- [results/ncu_default.txt](/home/york/ca_final_project/part5/results/ncu_default.txt)
- [results/default_gpu_run.txt](/home/york/ca_final_project/part5/results/default_gpu_run.txt)
- [results/default_cpu_run.txt](/home/york/ca_final_project/part5/results/default_cpu_run.txt)

## Pattern Sweep

```bash
cd part5
make sweep
```

This sweep currently tests:

- `1`
- `4`
- `8`
- `16`
- `32`

patterns and writes logs to:

```text
part5/results/
```

The current sweep summary is:

- [results/Part5_Experiment_Summary.md](/home/york/ca_final_project/part5/results/Part5_Experiment_Summary.md)

One Nsight Compute capture is also stored at:

- [results/ncu_default.txt](/home/york/ca_final_project/part5/results/ncu_default.txt)
