# CA Final Project Part 5


Part 5 是將 Part 4 的 CUDA pipeline 從單一 OFDM input pattern 延伸到多個彼此獨立的 patterns。

數學維持不變：
- Stage 1：LS channel estimation
- Stage 2：one-tap LMMSE equalization

新增主要在於 pattern dimension 與 GPU mapping。

## Q

前面幾個 parts 都只處理單一 input pattern，而 Part 5 要再利用一層新的平行性：跨多個獨立 patterns 的平行性


根據 [reference/CA_Final_Project.pdf](../reference/CA_Final_Project.pdf)，Part 5 ：

- 使用 GPU 處理多個 input patterns
- 利用 independent patterns 之間的平行性
- 2D CUDA grid 是合理的 mapping 方式
- 可使用 `blockIdx.y` 當作 pattern index
- 使用 `nvcc` 編譯
- 檢查 PTXAS / PTX
- 使用 `ncu --set basic` 做 profiling

## 設計

### Input

Part 5 使用獨立的 multi-pattern binary input：

```text
../data/ofdm_input_multi.bin
```

generator：[generate_ofdm_multi.cpp](./generate_ofdm_multi.cpp)

### Stage 1

`ls_channel_estimation_multi_kernel()`

- `blockIdx.x -> subcarrier k`
- `blockIdx.y -> pattern index`
- `threadIdx.x -> pilot contributions`

Stage 1 仍是 block-cooperative shared-memory reduction，只是 grid 現在同時覆蓋：

- subcarrier dimension
- pattern dimension

### Stage 2

`lmmse_equalization_multi_kernel()`

- `blockIdx.x * blockDim.x + threadIdx.x -> 單一 pattern 內的 flattened output index`
- `blockIdx.y -> pattern index`

每個 thread 仍然計算一個輸出 task。和 Part 4 的主要差別在於現在會同時有更多 patterns 活躍在 GPU 上。

## 證據

- CUDA source：[main.cu](./main.cu)
- CPU baseline：[main_cpu.cpp](./main_cpu.cpp)
- Makefile：[Makefile](Makefile)
- sweep script：[run_experiments.sh](./run_experiments.sh)

multi-pattern mapping 證據：

- `dim3 grid_ls(NUM_SUBCARRIERS, num_patterns)`
- `dim3 grid_eq(ceil(TOTAL_DATA / TPB_EQ), num_patterns)`
- kernel code 直接把 `blockIdx.y` 當作 pattern index



## 量測結果

預設 run 使用：

- `NUM_PATTERNS = 16`
- `TPB_LS = 256`
- `TPB_EQ = 256`

對應 logs：

- [results/default_gpu_run.txt](./results/default_gpu_run.txt)
- [results/default_cpu_run.txt](./results/default_cpu_run.txt)



輸出：

| Metric | Value |
| --- | --- |
| `H_MSE` | `0.00001289` |
| `MSE_RX_BEFORE_EQ` | `0.14082038` |
| `MSE_LMMSE` | `0.00682142` |
| `LS_KERNEL_MS` | `0.128993` |
| `LMMSE_KERNEL_MS` | `0.308572` |
| `PIPELINE_KERNEL_MS` | `0.434780` |
| `Verification` | `PASS` |

同一份 input 的 CPU baseline：

| Metric | Value |
| --- | --- |
| `CPU_PIPELINE_MS` | `12.268151` |
| `H_MSE` | `0.00001289` |
| `MSE_LMMSE` | `0.00682142` |
| `Verification` | `PASS` |

*  multi-pattern CUDA path 在維持相同 OFDM 運算與 correctness checks 的前提下，可以穩定處理 workload。

## Pattern Sweep

目前的 sweep 摘要保存在：[results/Part5_Experiment_Summary.md](./results/Part5_Experiment_Summary.md)



| Patterns | Verification | LS ms | LMMSE ms | Pipeline ms |
| --- | --- | --- | --- | --- |
| `1` | `PASS` | `0.032481` | `0.037750` | `0.069048` |
| `4` | `PASS` | `0.039578` | `0.117176` | `0.141583` |
| `8` | `PASS` | `0.059996` | `0.177853` | `0.214241` |
| `16` | `PASS` | `0.086354` | `0.347571` | `0.404536` |
| `32` | `PASS` | `0.223770` | `0.745861` | `0.926991` |

註：

- total pipeline time 隨著 pattern 數量增加而上升，因為總工作量也上升
- pipeline time per pattern
    - 在目前 sweep 中，per-pattern time 從 `1` pattern 時的約 `0.069048 ms`，下降到 `32` patterns 時的約 `0.028968 ms`

符合 Part 5 目標：當 pattern 數量增加，GPU 可同時排程的 blocks 與 warps 更多，整體 launch / resource overhead 會被更有效地攤提

## Nsight Compute 證據

目前的 Nsight Compute capture：[./results/ncu_default.txt](./results/ncu_default.txt)

預設 `16`-pattern run：

### Stage 1

- grid size：`8192`
- block size：`256`
- registers per thread：`16`
- dynamic shared memory per block：約 `2.05 KB`
- achieved occupancy：約 `89.53%`
- memory throughput：約 `55.97%`
- compute throughput：約 `55.97%`

### Stage 2

- grid size：`16384`
- block size：`256`
- registers per thread：`21`
- dynamic shared memory per block：`0`
- achieved occupancy：約 `82.77%`
- memory throughput：約 `91.74%`
- compute throughput：約 `28.55%`

註：

- Stage 1 仍呈現 shared-memory reduction kernel 的特徵，只是現在複製到更多 patterns 上
- Stage 2 仍偏 memory-heavy 的 kernel
- 2D launch 讓 GPU 同時看到更多 blocks 與 waves per SM

## PTXAS 與 PTX 證據

目前的 build log：[results/build.txt](./results/build.txt)

目前的 PTX：[main.ptx](./main.ptx)

觀察：

- Stage 1 使用 `15` registers per thread
- Stage 2 使用 `21` registers per thread
- 兩個都是 `0 spill stores`、`0 spill loads`



## 執行指令

建置與執行：

```bash
cd part5
make
make run
```

CPU baseline：

```bash
cd part5
make run_cpu
```

PTX：

```bash
cd part5
make ptx
```

Nsight Compute：

```bash
cd part5
make profile
```

Pattern sweep：

```bash
cd part5
make sweep
```
