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

## Run

```bash
cd part4
make run
```

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

## Notes

- Stage 1 is the main `__shared__` use case because threads inside one block must cooperatively reduce pilot partial sums into one `Hhat[k]`.
- Stage 2 is primarily element-wise, so the current version keeps the straightforward global-memory path instead of adding shared-memory traffic and synchronization with little reuse.
- Correctness is checked with the same metrics as Parts 1–3:
  - `H_MSE`
  - `MSE_RX_BEFORE_EQ`
  - `MSE_LMMSE`
  - `checksum`
