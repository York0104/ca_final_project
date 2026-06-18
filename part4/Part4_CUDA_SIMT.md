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

Part 5 is still reserved for multi-pattern GPU parallelism.

## Requirement Mapping

根據 [reference/CA_Final_Project.pdf](/home/york/ca_final_project/reference/CA_Final_Project.pdf)，Part 4 需要：

- 使用 CUDA SIMT execution model
- each thread computes one output task
- 使用 `threadIdx.x / blockIdx.x / blockDim.x`
- 使用 `__shared__`
- 使用 `-arch=sm_xx`
- 產生 PTX
- 觀察 PTXAS 與 `ncu --set basic`

The current source and result files provide evidence for each item above.

## Kernel 設計

### Stage 1

`ls_channel_estimation_shared_kernel()`

- one block -> one subcarrier `k`
- one thread -> one pilot partial contribution
- block 內使用 `__shared__` 做 tree reduction

數學形式：

```text
Hhat[k] = sum_p Ypilot[k,p] * pilot_w[p]
```

### Stage 1 Baseline

`ls_channel_estimation_serial_kernel()`

- one thread -> one subcarrier `k`
- thread 內用 scalar loop 跑完所有 `256` 個 pilots

這是 Part 4 用來比較 `shared-memory reduction` 是否有效的 baseline。

### Stage 2

`lmmse_equalization_kernel()`

- one thread -> one output `Xmmse[s,k]`
- flattened mapping:

```text
idx = blockIdx.x * blockDim.x + threadIdx.x
k   = idx % NUM_SUBCARRIERS
```

這段主要是 element-wise global-memory workload，不強制使用 shared memory。

## 編譯與分析

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

目前 Part 4 與 Parts 1–3 使用相同的驗證標準：

- `H_MSE < 0.01`
- `MSE_LMMSE < MSE_RX_BEFORE_EQ`
- `checksum != 0`

目前 shared 與 serial 兩種 LS 模式都已驗證：

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

The shared kernel uses fewer registers than the serial Stage 1 baseline in this build.

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

- Stage 1 PTX shows shared-memory traffic and barriers.
- Stage 2 PTX is dominated by global-memory access and floating-point arithmetic.

## Nsight Compute Evidence

已保存：

- [part4/results/ncu_shared_256_256.txt](/home/york/ca_final_project/part4/results/ncu_shared_256_256.txt)

The file below was collected with `TPB_LS=256`, `TPB_EQ=256`, and `shared` LS mode:

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

完整表已保存：

- [part4/results/Part4_Experiment_Summary.md](/home/york/ca_final_project/part4/results/Part4_Experiment_Summary.md)

### LS shared kernel sweep

| Case | TPB_LS | TPB_EQ | LS Mode | LS ms | Pipeline ms |
| --- | --- | --- | --- | --- | --- |
| `ls_shared_64` | `64` | `256` | `shared` | `0.017875` | `0.043228` |
| `ls_shared_128` | `128` | `256` | `shared` | `0.017576` | `0.038370` |
| `ls_shared_256` | `256` | `256` | `shared` | `0.018781` | `0.052552` |

In this sweep, `TPB_LS=128` gives the lowest pipeline kernel time.

### LMMSE kernel sweep

| Case | TPB_LS | TPB_EQ | LS Mode | LMMSE ms | Pipeline ms |
| --- | --- | --- | --- | --- | --- |
| `eq_shared_128` | `256` | `128` | `shared` | `0.023600` | `0.055852` |
| `eq_shared_256` | `256` | `256` | `shared` | `0.037985` | `0.069458` |
| `eq_shared_512` | `256` | `512` | `shared` | `0.021407` | `0.043108` |

In this sweep, `TPB_EQ=512` gives the lowest pipeline kernel time.

## Shared vs Serial

| Case | LS Mode | LS ms | Pipeline ms | Verification |
| --- | --- | --- | --- | --- |
| `ls_shared_256` | `shared` | `0.018781` | `0.052552` | `PASS` |
| `ls_serial_256` | `serial` | `0.136742` | `0.133509` | `PASS` |

## Shared vs Serial Interpretation

- The shared version is a block-cooperative reduction with `__shared__`.
- The serial version is a one-thread-per-subcarrier baseline.
- The timing gap measures the combined effect of the parallel reduction mapping and shared-memory cooperation. It should not be reduced to "shared memory alone."

## Current State

Completed in the current repository:

- single-pattern CUDA SIMT pipeline
- shared-memory LS reduction kernel
- element-wise LMMSE kernel
- CUDA correctness verification
- cudaEvent timing
- TPB sweep
- shared vs serial LS comparison
- PTXAS / PTX / NCU 初步分析材料

Still left for later work:

- 更完整的 TPB search
- 更完整的 NCU 報告整理
- Part 5 multi-pattern 2D grid mapping
