# Part 5

## 範圍

Part 5 是 Part 4 CUDA pipeline 的 multi-pattern GPU 延伸版本：

- Stage 1：LS channel estimation
- Stage 2：one-tap LMMSE equalization

和 Part 4 的主要差異在於：Part 5 會在一次 GPU launch sequence 中，同時處理多個彼此獨立的 OFDM input patterns。

預設輸出紀錄保存在：

- [results/default_gpu_run.txt](results/default_gpu_run.txt)
- [results/default_cpu_run.txt](results/default_cpu_run.txt)

## 問題與目標

依照 [reference/CA_Final_Project.pdf](../reference/CA_Final_Project.pdf)，前面幾個 parts 都只處理單一 input pattern。Part 5 的重點是進一步利用：

- 多個獨立 patterns 之間的平行性
- GPU 上跨 patterns 的平行執行
- 2D CUDA grid mapping

在本 repository 中，這表示：

- 維持與 Part 4 相同的 OFDM 數學模型
- 維持相同的 Stage 1 shared-memory reduction
- 維持相同的 Stage 2 thread-per-output mapping
- 新增 pattern dimension，並映射到 `blockIdx.y`

## 檔案結構

- source：[main.cu](main.cu)
- CPU baseline：[main_cpu.cpp](main_cpu.cpp)
- multi-pattern generator：[generate_ofdm_multi.cpp](generate_ofdm_multi.cpp)
- local format helpers：[ofdm_multi_common.h](ofdm_multi_common.h)
- build script：[Makefile](Makefile)
- experiment script：[run_experiments.sh](run_experiments.sh)

## 資料模型

generator 會輸出專用的 multi-pattern input：

```text
../data/ofdm_input_multi.bin
```

每個 pattern 都沿用與 Parts 1–4 相同的 OFDM 維度：

- `NUM_SUBCARRIERS = 512`
- `NUM_PILOTS = 256`
- `NUM_DATA_SYMBOLS = 512`

所有 patterns 共用相同的 pilot weights 與 deep-fade channel shape，而 QPSK symbols 與 AWGN samples 則依 pattern index 以 deterministic 方式變化。

## CUDA Mapping

### Stage 1

`ls_channel_estimation_multi_kernel()`

- `blockIdx.x -> subcarrier k`
- `blockIdx.y -> pattern index`
- `threadIdx.x -> pilot contributions`
- 使用 `__shared__` 進行 block-level reduction

### Stage 2

`lmmse_equalization_multi_kernel()`

- `blockIdx.x * blockDim.x + threadIdx.x -> 單一 pattern 內的 flattened data index`
- `blockIdx.y -> pattern index`
- one thread 計算一個 `Xmmse[s,k]`

這就是作業要求的 2D CUDA grid mapping：

- x-dimension：output / thread index
- y-dimension：pattern index

## 建置

```bash
cd part5
make
```

## 執行

```bash
cd part5
make run
```

可選的 CPU baseline：

```bash
cd part5
make run_cpu
```

## PTX 與 Nsight Compute

```bash
cd part5
make ptx
make profile
```

目前保存的分析 artifact：

- [main.ptx](main.ptx)
- [results/build.txt](results/build.txt)
- [results/ncu_default.txt](results/ncu_default.txt)
- [results/default_gpu_run.txt](results/default_gpu_run.txt)
- [results/default_cpu_run.txt](results/default_cpu_run.txt)

## Pattern Sweep

```bash
cd part5
make sweep
```

目前 sweep 測試：

- `1`
- `4`
- `8`
- `16`
- `32`

個 patterns，並將結果寫入：

```text
part5/results/
```

目前 sweep 摘要：

- [results/Part5_Experiment_Summary.md](results/Part5_Experiment_Summary.md)

另有一份 Nsight Compute capture：

- [results/ncu_default.txt](results/ncu_default.txt)
