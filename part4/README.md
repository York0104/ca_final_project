# Part 4

## Scope

Part 4 is the CUDA SIMT version of the current project workload:

- Stage 1: LS channel estimation
- Stage 2: one-tap LMMSE equalization

This part processes a single OFDM input pattern from:

```text
../data/ofdm_input.bin
```

It does not implement the multi-pattern batching used later in Part 5.

This directory is meant to be self-contained for the single-pattern CUDA submission:

- source: [main.cu](/home/york/ca_final_project/part4/main.cu)
- build script: [Makefile](/home/york/ca_final_project/part4/Makefile)
- experiment script: [run_experiments.sh](/home/york/ca_final_project/part4/run_experiments.sh)
- report note: [Part4_CUDA_SIMT.md](/home/york/ca_final_project/part4/Part4_CUDA_SIMT.md)
- measured outputs: [results/](/home/york/ca_final_project/part4/results)

## Kernel Mapping

### Kernel 1

`ls_channel_estimation_shared_kernel()`

- one block -> one subcarrier `k`
- one thread -> one pilot contribution or multiple pilot contributions with thread striding
- uses `__shared__` memory for block-level reduction

### Kernel 2

`lmmse_equalization_kernel()`

- one thread -> one output `Xmmse[s,k]`
- flattened index:
  - `idx = blockIdx.x * blockDim.x + threadIdx.x`

## Default Configuration

- `TPB_LS = 256`
- `TPB_EQ = 256`
- `WARMUP_ITERS = 10`
- `TIMED_ITERS = 100`
- `-arch=sm_86`

The executable also accepts optional runtime arguments:

```text
./main [input_file] [warmup_iters] [timed_iters] [ls_mode]
```

Example:

```bash
./main ../data/ofdm_input.bin 1 1
./main ../data/ofdm_input.bin 1 1 shared
./main ../data/ofdm_input.bin 1 1 serial
```

This is mainly for `ncu`, so profiling does not replay the long timing loop.

The optional `ls_mode` argument supports:

- `shared` -> block-cooperative shared-memory reduction
- `serial` -> one-thread-per-subcarrier serial LS baseline

## Build

```bash
cd part4
make
```

The default target compiles with:

- `nvcc`
- `-O2`
- `-std=c++17`
- `-arch=sm_86`
- `-lineinfo`
- `-Xptxas -v`

## Run

```bash
cd part4
make run
```

The runtime output includes:

- functional metrics: `H_MSE`, `MSE_RX_BEFORE_EQ`, `MSE_LMMSE`, `checksum`, `Verification`
- configuration: `TPB_LS`, `TPB_EQ`, `LS_KERNEL_MODE`, warmup/timed iterations
- timing: `LS_KERNEL_MS`, `LMMSE_KERNEL_MS`, `PIPELINE_KERNEL_MS`

`PIPELINE_KERNEL_MS` is the average GPU kernel-only time for running Stage 1 and Stage 2 back-to-back inside the timing loop. It does not include input loading, host-side verification, H2D/D2H copies, allocation, or setup.

## PTX

```bash
cd part4
make ptx
```

## Nsight Compute

```bash
cd part4
make profile
```

This uses a short command-line configuration:

```bash
./main ../data/ofdm_input.bin 1 1
```

The goal is to reduce replay cost during profiling. The `ncu` numbers should be interpreted as profiling data, not as the main timing source for the sweep table.

## TPB / Shared-Memory Experiments

Run the current experiment sweep:

```bash
cd part4
make sweep
```

This currently generates:

- LS shared-memory TPB sweep: `64 / 128 / 256`
- LMMSE TPB sweep: `128 / 256 / 512`
- shared vs serial LS comparison

Outputs are written to:

```text
part4/results/
```

The key files are:

- [Part4_Experiment_Summary.md](/home/york/ca_final_project/part4/results/Part4_Experiment_Summary.md)
- `*_build.txt`
- `*_run.txt`
- `main_shared_256_256.ptx`
- `ncu_shared_256_256.txt`

## Notes

- Stage 1 is the main `__shared__` use case because threads inside one block must cooperatively reduce pilot partial sums into one `Hhat[k]`.
- Stage 2 is primarily element-wise, so the current version keeps the straightforward global-memory path instead of adding shared-memory traffic and synchronization with little reuse.
- Correctness is checked with the same metrics as Parts 1–3:
  - `H_MSE`
  - `MSE_RX_BEFORE_EQ`
  - `MSE_LMMSE`
- `checksum`
- The shared-vs-serial comparison should be read as a comparison between:
  - block-cooperative shared-memory reduction
  - one-thread-per-subcarrier serial Stage 1
  It is not evidence that shared memory alone explains the full timing gap.
- `ncu` output is useful for throughput, waves, register, and memory observations, but the sweep table should still be read from the `cudaEvent` kernel timing in `results/*_run.txt`.
