# CA Final Project Part 4

## 範圍

Part 4 是相同 OFDM pipeline 的 CUDA 單一 pattern 版本：

```text
CUDA SIMT LS Channel Estimation + CUDA LMMSE Equalization
```

目前範圍：

- `single OFDM input pattern`
- 讀取同一份 `../data/ofdm_input.bin`
- 不使用 multi-pattern batching
- 不使用 2D grid

Part 5 另行實作於 [`part5/`](../part5/)，不屬於這份單一 pattern CUDA 文件的範圍。

目前 repository 中的 Part 4 已包含 source code、build scripts、PTX/PTXAS 材料、Nsight Compute 輸出、TPB sweep logs，以及 shared-vs-serial Stage 1 comparison。

## 題目要求對應

依照 [reference/CA_Final_Project.pdf](../reference/CA_Final_Project.pdf)，Part 4 需要：

- 使用 CUDA SIMT execution model
- each thread computes one output task
- 使用 `threadIdx.x / blockIdx.x / blockDim.x`
- 使用 `__shared__`
- 使用 `-arch=sm_xx`
- 產生 PTX
- 觀察 PTXAS 與 `ncu --set basic`

目前 repository 中對應的證據如下：

| Requirement | Repository 中的證據 |
| --- | --- |
| CUDA SIMT implementation | [main.cu](main.cu) |
| each thread computes one output task | `lmmse_equalization_kernel()` |
| `threadIdx.x / blockIdx.x / blockDim.x` mapping | `main.cu` 中的 Stage 1 與 Stage 2 kernels |
| `__shared__` usage | `ls_channel_estimation_shared_kernel()` |
| `-arch=sm_xx` build | [Makefile](Makefile) 內的 `-arch=sm_86` |
| PTX generation | [results/main_shared_256_256.ptx](results/main_shared_256_256.ptx) |
| PTXAS output | `results/*_build.txt` |
| `ncu --set basic` profiling | [results/ncu_shared_256_256.txt](results/ncu_shared_256_256.txt) |
| TPB 分析 | [results/Part4_Experiment_Summary.md](results/Part4_Experiment_Summary.md) |

## Kernel 設計

### Stage 1

`ls_channel_estimation_shared_kernel()`

- one block -> one subcarrier `k`
- one thread -> one pilot partial contribution
- block 內使用 `__shared__` 做 tree reduction

Stage 1 計算：

```text
Hhat[k] = sum_p Ypilot[k,p] * pilot_w[p]
```

### Stage 1 Baseline

`ls_channel_estimation_serial_kernel()`

- one thread -> one subcarrier `k`
- thread 內用 scalar loop 跑完全部 `256` 個 pilots

這就是 shared-vs-serial Stage 1 comparison 的 baseline。

### Stage 2

`lmmse_equalization_kernel()`

- one thread -> one output `Xmmse[s,k]`
- flattened mapping：

```text
idx = blockIdx.x * blockDim.x + threadIdx.x
k   = idx % NUM_SUBCARRIERS
```

這一段主要是 element-wise global-memory workload，因此目前實作沒有硬把 shared memory 加進來。

## 建置與分析

主要檔案：

- [part4/main.cu](main.cu)
- [part4/Makefile](Makefile)
- [part4/README.md](README.md)

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

Part 4 使用與 Parts 1–3 相同的 verification policy：

- `H_MSE < 0.01`
- `MSE_LMMSE < MSE_RX_BEFORE_EQ`
- `checksum != 0`

兩種 LS mode 目前都能通過：

```text
Verification = PASS
```

## PTXAS 證據

以 `TPB_LS=256`、`TPB_EQ=256` 為例：

- shared LS kernel：
  - `14 registers`
  - `0 spill stores`
  - `0 spill loads`
- serial LS kernel：
  - `40 registers`
  - `0 spill stores`
  - `0 spill loads`
- LMMSE kernel：
  - `21 registers`
  - `0 spill stores`
  - `0 spill loads`

在目前這個 build 中，shared Stage 1 kernel 的 register 使用量低於 serial baseline。

## PTX 證據

已保存：

- [part4/results/main_shared_256_256.ptx](results/main_shared_256_256.ptx)

LS shared kernel 可見：

- `extern .shared`
- `st.shared`
- `ld.shared`
- `bar.sync`

LMMSE kernel 可見：

- `ld.global`
- `st.global`
- FP multiply / add / divide

解讀：

- Stage 1 的 PTX 顯示 shared-memory traffic 與 synchronization
- Stage 2 的 PTX 主要由 global-memory access 與 floating-point arithmetic 組成

## Nsight Compute 證據

已保存：

- [part4/results/ncu_shared_256_256.txt](results/ncu_shared_256_256.txt)

這份 profile 使用 `TPB_LS=256`、`TPB_EQ=256` 與 `shared` LS mode 量測。

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
- `Achieved Occupancy` 約 `104.46%`（NCU 報表值；不拿來當主要結論）
- `Achieved Active Warps Per SM` 約 `50.14`

解讀：

- Stage 1 看起來像典型的 block-cooperative shared-memory reduction
- Stage 2 更接近 memory-bound kernel
- `Achieved Occupancy > 100%` 這一行應視為 profiling anomaly，而不是硬體 occupancy 超出上限的證據

## TPB Sweep 結果

目前摘要：

- [part4/results/Part4_Experiment_Summary.md](results/Part4_Experiment_Summary.md)

下列表格對齊目前 repo 內保存的最新 rerun sweep。

### LS shared kernel sweep

| Case | TPB_LS | TPB_EQ | LS Mode | LS ms | Pipeline ms |
| --- | --- | --- | --- | --- | --- |
| `ls_shared_64` | `64` | `256` | `shared` | `0.017032` | `0.040740` |
| `ls_shared_128` | `128` | `256` | `shared` | `0.015666` | `0.039000` |
| `ls_shared_256` | `256` | `256` | `shared` | `0.017372` | `0.043345` |

在這組 sweep 中，`TPB_LS=128` 有最低的 pipeline kernel-only time。

### LMMSE kernel sweep

| Case | TPB_LS | TPB_EQ | LS Mode | LMMSE ms | Pipeline ms |
| --- | --- | --- | --- | --- | --- |
| `eq_shared_128` | `256` | `128` | `shared` | `0.021142` | `0.062669` |
| `eq_shared_256` | `256` | `256` | `shared` | `0.024146` | `0.044678` |
| `eq_shared_512` | `256` | `512` | `shared` | `0.021271` | `0.034696` |

在這組 sweep 中，`TPB_EQ=512` 有最低的 pipeline kernel-only time。

## Shared vs Serial

| Case | LS Mode | LS ms | Pipeline ms | Verification |
| --- | --- | --- | --- | --- |
| `ls_shared_256` | `shared` | `0.017372` | `0.043345` | `PASS` |
| `ls_serial_256` | `serial` | `0.135022` | `0.118535` | `PASS` |

## Shared vs Serial 解讀

- `shared` 版本是 block-cooperative reduction，並使用 `__shared__`
- `serial` 版本是 one-thread-per-subcarrier baseline
- 兩者的 timing gap 應解讀為 parallel reduction mapping 與 shared-memory cooperation 的整體效果，而不是單純說「全部都來自 shared memory」

## 目前完成狀態

目前 repository 中已完成：

- single-pattern CUDA SIMT pipeline
- shared-memory LS reduction kernel
- element-wise LMMSE kernel
- CUDA correctness verification
- cudaEvent timing
- TPB sweep
- shared vs serial LS comparison
- PTXAS / PTX / NCU evidence

後續若還要延伸，可再做：

- 更廣的 TPB sweep
- 更完整的 NCU 報告整理
