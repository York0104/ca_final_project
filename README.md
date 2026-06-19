# CA Final Project

本 repo 的CA期末專題設計：

```text
LS Channel Estimation + LMMSE/MMSE Equalizer Acceleration
```

整體 workload 為 pilot-based OFDM receiver，含兩個主要 computation stages：

1. LS channel estimation
2. one-tap LMMSE equalization

## 目前完成狀態

- `Part 1`：Scalar baseline
- `Part 2`：RVV vector reduction
- `Part 3`：SIMD-like RVV parallelization
- `Part 4`：CUDA SIMT single-pattern implementation
- `Part 5`：Multi-pattern GPU parallelism

## Repository 結構

```text
common/      共用參數、資料、I/O、model與驗證
data_gen/    Host 端 OFDM 輸入產生器
data/        產生出的 binary input
part1/       Scalar baseline
part2/       RVV reduction implementation
part3/       SIMD-like RVV implementation
part4/       CUDA SIMT implementation 與Shell script
part5/       Multi-pattern CUDA implementation 與Shell script
reference/   參考資料
figures/     實驗圖表
```

## 數學模型

### Pilot model

```text
X_pilot[p,k] = 1 + j0
Y_pilot[p,k] = H[k] + N_pilot[p,k]
```

### LS channel estimation

```text
Hhat[k] = sum_p Y_pilot[p,k] * w[p]
w[p] = 1 / P
```

### Data model

```text
Y_data[s,k] = H[k] * X_data[s,k] + N_data[s,k]
```

### LMMSE equalizer

```text
Xmmse[s,k] = Ydata[s,k] * conj(Hhat[k])
             / (|Hhat[k]|^2 + NOISE_VAR_OVER_SYMBOL_POWER + EPSILON)
```

## 建置與執行

從 repository root 開始：

```bash
cd [your repo]
```

### 環境需求

`Part 1` 到 `Part 3` 需要：

- Docker 搭配課程使用的 gem5 / RVV image

`Part 4` 與 `Part 5` 需要：

- 支援 CUDA 的 NVIDIA GPU
- Docker 搭配課程使用的 CUDA image



### Part 1

Host correctness check：

```bash
make -C part1 clean
make -C part1 host
```

gem5 run：

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

每次 `make` 會：

- 在 host 端重新產生 `data/ofdm_input.bin`
- cross-compile 目標程式
- 在 gem5 `TimingSimpleCPU` 設定下執行
- 印出 `m5out/stats.txt` 中的重要統計

gem5 輸出目錄：

- `part1/m5out/`
- `part2/m5out/`
- `part3/m5out/`

### Part 4

```bash
make -C part4 clean
make -C part4
make -C part4 run
```

其他：

```bash
make -C part4 ptx
make -C part4 profile
make -C part4 sweep
```

Part 4 結果目錄：

- `part4/results/`

### Part 5

```bash
make -C part5 clean
make -C part5
make -C part5 run
```

CPU baseline：

```bash
make -C part5 run_cpu
```

其他常用指令：

```bash
make -C part5 ptx
make -C part5 profile
make -C part5 sweep
```

Part 5 結果目錄：

- `part5/results/`


## gem5 結果

| Metric | Part 1 | Part 2 | Part 3 |
| --- | --- | --- | --- |
| `simSeconds` | `0.069028` | `0.054755` | `0.072706` |
| `simInsts` | `17,066,251` | `11,395,368` | `11,208,851` |
| `numCycles` | `138,055,892` | `109,510,852` | `145,412,866` |
| `CPI` | `8.089394` | `9.610092` | `12.973001` |
| `IPC` | `0.123619` | `0.104057` | `0.077083` |
| `D-cache miss rate` | `0.114745` | `0.170969` | `0.229838` |
| `I-cache miss rate` | `0.000046` | `0.000071` | `0.000051` |

註：

- `Part 2` 在目前 gem5 結果中有最低的 `simSeconds` 與 `numCycles`
- `Part 3` 維持題目要求的 across-`k` SIMD-like mapping，但 `vlse32.v` 的 strided access 讓 miss rate 與 cycle count 上升

## CUDA 結果

`Part 4` 與 `Part 5` 使用的是 GPU kernel-only timing，來源為 `cudaEvent`。
這些數字不包含 input loading、allocation、H2D/D2H copy 與 host-side verification，
因此不直接和上面的 gem5 `simSeconds` 比較。

### Part 4

預設 case 為 `TPB_LS=256`、`TPB_EQ=256`、`LS_KERNEL_MODE=shared`：

| Metric | 數值 |
| --- | --- |
| `Verification` | `PASS` |
| `H_MSE` | `0.00001250` |
| `MSE_LMMSE` | `0.00681053` |
| `LS_KERNEL_MS` | `0.023186` |
| `LMMSE_KERNEL_MS` | `0.024146` |
| `PIPELINE_KERNEL_MS` | `0.044678` |

Part 4 的 sweep 結果顯示：

- LS shared reduction 的最佳 pipeline time 出現在 `TPB_LS=128`，為 `0.039000 ms`
- LMMSE sweep 中最低 pipeline time 出現在 `TPB_EQ=512`，為 `0.034696 ms`
- shared LS 與 one-thread-per-subcarrier serial LS 相比，`TPB_LS=256` case 的 pipeline time 從 `0.118535 ms` 降到 `0.043345 ms`

### Part 5

預設 multi-pattern case 為 `NUM_PATTERNS=16`、`TPB_LS=256`、`TPB_EQ=256`：

| Metric | GPU | CPU baseline |
| --- | --- | --- |
| `Verification` | `PASS` | `PASS` |
| `H_MSE` | `0.00001289` | `0.00001289` |
| `MSE_LMMSE` | `0.00682142` | `0.00682142` |
| `PIPELINE_MS` | `0.434780` | `12.268151` |

以這組 `16 patterns` 的預設量測來看，GPU kernel-only pipeline time 約為 CPU baseline 的 `28.22x` 加速。

Pattern sweep：

| `NUM_PATTERNS` | `PIPELINE_KERNEL_MS` | `ms / pattern` |
| --- | --- | --- |
| `1` | `0.069048` | `0.069048` |
| `4` | `0.141583` | `0.035396` |
| `8` | `0.214241` | `0.026780` |
| `16` | `0.404536` | `0.025284` |
| `32` | `0.926991` | `0.028968` |

重點：當 pattern 數增加時，GPU 可以同時暴露更多獨立 blocks 與 warps，
固定 launch / scheduling overhead 也能被更多工作量攤提。

此改善並非線性無限延伸：

- 在 `32 patterns` 時，總時間仍繼續上升
- per-pattern time 也開始回升
- 代表資源利用與 memory traffic 的限制開始變明顯


## 環境說明

### RISC-V / gem5

- 使用 `riscv64-linux-gnu-g++`
- 在 Docker 內的 gem5 環境執行
- `part2/` 與 `part3/` 需要 RVV 編譯設定

### CUDA

`Part 4` 與 `Part 5` 使用：

- GPU：`NVIDIA GeForce RTX 3060`
- Compute Capability：`8.6`
- CUDA Docker runtime：可用
- `nvcc 12.4`：可用
- `-arch=sm_86`：已驗證
- CUDA kernel launch：已驗證
- CUDA event timing：已驗證
- PTX generation：已驗證
- PTXAS resource report：已驗證
- `ncu` / Nsight Compute：已驗證
- GPU performance counters：已驗證
