# CA Final Project Part 4

## Scope

Part 4 covers the CUDA single-pattern version of the same OFDM pipeline:

```text
CUDA SIMT LS Channel Estimation + CUDA LMMSE Equalization
```

Current scope:

- `single OFDM input pattern`
- 讀取同一份 `../data/ofdm_input.bin`
- 不使用 multi-pattern batching
- 不使用 2D grid

Part 5 is implemented separately in [`part5/`](/home/york/ca_final_project/part5) and is not part of this single-pattern CUDA file.

At this stage, Part 4 already includes source code, build scripts, PTX/PTXAS materials, Nsight Compute output, TPB sweep logs, and a shared-vs-serial Stage 1 comparison.

## Requirement Mapping

According to [reference/CA_Final_Project.pdf](/home/york/ca_final_project/reference/CA_Final_Project.pdf), Part 4 requires:

- 使用 CUDA SIMT execution model
- each thread computes one output task
- 使用 `threadIdx.x / blockIdx.x / blockDim.x`
- 使用 `__shared__`
- 使用 `-arch=sm_xx`
- 產生 PTX
- 觀察 PTXAS 與 `ncu --set basic`

The repository already contains evidence for each item above:

| Requirement | Evidence in repo |
| --- | --- |
| CUDA SIMT implementation | [main.cu](/home/york/ca_final_project/part4/main.cu) |
| each thread computes one output task | `lmmse_equalization_kernel()` |
| `threadIdx.x / blockIdx.x / blockDim.x` mapping | Stage 1 and Stage 2 kernels in `main.cu` |
| `__shared__` usage | `ls_channel_estimation_shared_kernel()` |
| `-arch=sm_xx` build | [Makefile](/home/york/ca_final_project/part4/Makefile) with `-arch=sm_86` |
| PTX generation | [results/main_shared_256_256.ptx](/home/york/ca_final_project/part4/results/main_shared_256_256.ptx) |
| PTXAS output | `results/*_build.txt` |
| `ncu --set basic` profiling | [results/ncu_shared_256_256.txt](/home/york/ca_final_project/part4/results/ncu_shared_256_256.txt) |
| TPB analysis | [results/Part4_Experiment_Summary.md](/home/york/ca_final_project/part4/results/Part4_Experiment_Summary.md) |

## Kernel Design

### Stage 1

`ls_channel_estimation_shared_kernel()`

- one block -> one subcarrier `k`
- one thread -> one pilot partial contribution
- block 內使用 `__shared__` 做 tree reduction

Stage 1 computes:

```text
Hhat[k] = sum_p Ypilot[k,p] * pilot_w[p]
```

### Stage 1 Baseline

`ls_channel_estimation_serial_kernel()`

- one thread -> one subcarrier `k`
- thread 內用 scalar loop 跑完所有 `256` 個 pilots

This is the baseline used for the shared-vs-serial Stage 1 comparison.

### Stage 2

`lmmse_equalization_kernel()`

- one thread -> one output `Xmmse[s,k]`
- flattened mapping:

```text
idx = blockIdx.x * blockDim.x + threadIdx.x
k   = idx % NUM_SUBCARRIERS
```

This stage is primarily an element-wise global-memory workload, so the current implementation does not force shared memory into it.

## Build and Analysis

主要檔案：

- [part4/main.cu](/home/york/ca_final_project/part4/main.cu)
- [part4/Makefile](/home/york/ca_final_project/part4/Makefile)
- [part4/README.md](/home/york/ca_final_project/part4/README.md)

編譯：

```bash
cd part4
make
```

產生 PTX：

```bash
cd part4
make ptx
```

Nsight Compute：

```bash
cd part4
make profile
```

## Correctness

Part 4 uses the same verification policy as Parts 1–3:

- `H_MSE < 0.01`
- `MSE_LMMSE < MSE_RX_BEFORE_EQ`
- `checksum != 0`

Both LS modes already pass:

```text
Verification = PASS
```

## PTXAS Evidence

以 `TPB_LS=256`、`TPB_EQ=256` 為例：

- shared LS kernel:
  - `14 registers`
  - `0 spill stores`
  - `0 spill loads`
- serial LS kernel:
  - `40 registers`
  - `0 spill stores`
  - `0 spill loads`
- LMMSE kernel:
  - `21 registers`
  - `0 spill stores`
  - `0 spill loads`

In this build, the shared Stage 1 kernel uses fewer registers than the serial baseline.

## PTX Evidence

已保存：

- [part4/results/main_shared_256_256.ptx](/home/york/ca_final_project/part4/results/main_shared_256_256.ptx)

- LS shared kernel 可見：
  - `extern .shared`
  - `st.shared`
  - `ld.shared`
  - `bar.sync`
- LMMSE kernel 可見：
  - `ld.global`
  - `st.global`
  - FP multiply / add / divide

Interpretation:

- Stage 1 PTX shows shared-memory traffic and synchronization.
- Stage 2 PTX is dominated by global-memory access and floating-point arithmetic.

## Nsight Compute Evidence

已保存：

- [part4/results/ncu_shared_256_256.txt](/home/york/ca_final_project/part4/results/ncu_shared_256_256.txt)

This profile was collected with `TPB_LS=256`, `TPB_EQ=256`, and `shared` LS mode:

### LS shared kernel

- `Memory Throughput` 約 `35.55%`
- `DRAM Throughput` 約 `29.55%`
- `Compute (SM) Throughput` 約 `35.55%`
- `Registers Per Thread` = `16`
- `Dynamic Shared Memory Per Block` 約 `2.05 KB`
- `Waves Per SM` = `2.84`
- `Achieved Occupancy` 約 `87.37%`
- `Achieved Active Warps Per SM` 約 `41.94`

### LMMSE kernel

- `Memory Throughput` 約 `75.26%`
- `DRAM Throughput` 約 `75.26%`
- `Compute (SM) Throughput` 約 `23.55%`
- `Registers Per Thread` = `21`
- `Dynamic Shared Memory Per Block` = `0`
- `Waves Per SM` = `5.69`
- `Achieved Occupancy` 約 `104.46%`（NCU 報表值；不拿這個值當主要結論）
- `Achieved Active Warps Per SM` 約 `50.14`

Interpretation:

- Stage 1 looks like a block-cooperative shared-memory reduction.
- Stage 2 is closer to a memory-bound kernel.
- The `Achieved Occupancy > 100%` line is treated as a profiling anomaly, not as evidence that physical occupancy exceeded the hardware limit.

## TPB Sweep Results

Current summary:

- [part4/results/Part4_Experiment_Summary.md](/home/york/ca_final_project/part4/results/Part4_Experiment_Summary.md)

下列表格對齊 2026-06-19 的 fresh rerun sweep。

### LS shared kernel sweep

| Case | TPB_LS | TPB_EQ | LS Mode | LS ms | Pipeline ms |
| --- | --- | --- | --- | --- | --- |
| `ls_shared_64` | `64` | `256` | `shared` | `0.017032` | `0.040740` |
| `ls_shared_128` | `128` | `256` | `shared` | `0.015666` | `0.039000` |
| `ls_shared_256` | `256` | `256` | `shared` | `0.017372` | `0.043345` |

In this sweep, `TPB_LS=128` gives the lowest measured pipeline kernel-only time.

### LMMSE kernel sweep

| Case | TPB_LS | TPB_EQ | LS Mode | LMMSE ms | Pipeline ms |
| --- | --- | --- | --- | --- | --- |
| `eq_shared_128` | `256` | `128` | `shared` | `0.021142` | `0.062669` |
| `eq_shared_256` | `256` | `256` | `shared` | `0.024146` | `0.044678` |
| `eq_shared_512` | `256` | `512` | `shared` | `0.021271` | `0.034696` |

In this sweep, `TPB_EQ=512` gives the lowest measured pipeline kernel-only time.

## Shared vs Serial

| Case | LS Mode | LS ms | Pipeline ms | Verification |
| --- | --- | --- | --- | --- |
| `ls_shared_256` | `shared` | `0.017372` | `0.043345` | `PASS` |
| `ls_serial_256` | `serial` | `0.135022` | `0.118535` | `PASS` |

## Shared vs Serial Interpretation

- The shared version is a block-cooperative reduction with `__shared__`.
- The serial version is a one-thread-per-subcarrier baseline.
- The timing gap measures the combined effect of the parallel reduction mapping and shared-memory cooperation. It should not be reduced to "shared memory alone."

## Submission Status

Completed in the current repository:

- single-pattern CUDA SIMT pipeline
- shared-memory LS reduction kernel
- element-wise LMMSE kernel
- CUDA correctness verification
- cudaEvent timing
- TPB sweep
- shared vs serial LS comparison
- PTXAS / PTX / NCU evidence

Items still left for later work:

- broader TPB exploration
- a more polished NCU write-up
