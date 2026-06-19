# Part 4

## 範圍

Part 4 是 CUDA SIMT 單一 pattern ：

- Stage 1：LS channel estimation
- Stage 2：one-tap LMMSE equalization

單一 pattern CUDA 需要的主要檔案：

- source：[main.cu](/home/york/ca_final_project/part4/main.cu)
- build script：[Makefile](/home/york/ca_final_project/part4/Makefile)
- experiment script：[run_experiments.sh](/home/york/ca_final_project/part4/run_experiments.sh)
- 說明文件：[Part4_CUDA_SIMT.md](/home/york/ca_final_project/part4/Part4_CUDA_SIMT.md)
- 結果輸出：[results/](/home/york/ca_final_project/part4/results)

## Kernel Mapping

### Kernel 1

`ls_channel_estimation_shared_kernel()`

- one block -> 一個 subcarrier `k`
- one thread -> 一筆或多筆 pilot contribution
- 使用 `__shared__` memory 做 block-level reduction

### Kernel 2

`lmmse_equalization_kernel()`

- one thread -> 一個輸出 `Xmmse[s,k]`
- flattened index：
  - `idx = blockIdx.x * blockDim.x + threadIdx.x`

## 設定

- `TPB_LS = 256`
- `TPB_EQ = 256`
- `WARMUP_ITERS = 10`
- `TIMED_ITERS = 100`
- `-arch=sm_86`

執行檔也支援選用的 runtime arguments：

```text
./main [input_file] [warmup_iters] [timed_iters] [ls_mode]
```

例如：

```bash
./main ../data/ofdm_input.bin 1 1
./main ../data/ofdm_input.bin 1 1 shared
./main ../data/ofdm_input.bin 1 1 serial
```

`ls_mode` ：

- `shared` -> block-cooperative shared-memory reduction
- `serial` -> one-thread-per-subcarrier 的 scalar LS baseline

## 建置

```bash
cd part4
make
```

預設會使用：

- `nvcc`
- `-O2`
- `-std=c++17`
- `-arch=sm_86`
- `-lineinfo`
- `-Xptxas -v`

## 執行

```bash
cd part4
make run
```

程式輸出包含：

- functional metrics：`H_MSE`、`MSE_RX_BEFORE_EQ`、`MSE_LMMSE`、`checksum`、`Verification`
- configuration：`TPB_LS`、`TPB_EQ`、`LS_KERNEL_MODE`、warmup / timed iterations
- timing：`LS_KERNEL_MS`、`LMMSE_KERNEL_MS`、`PIPELINE_KERNEL_MS`

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

這會使用較短的 command-line 設定：

```bash
./main ../data/ofdm_input.bin 1 1
```

目的是降低 profiling replay 成本。因此 `ncu` 輸出應視為 profiling 資料，不是 sweep 表格的主要 timing 來源。

## TPB / Shared-Memory 實驗

執行目前的 sweep：

```bash
cd part4
make sweep
```

目前 sweep 會產生：

- LS shared-memory TPB sweep：`64 / 128 / 256`
- LMMSE TPB sweep：`128 / 256 / 512`
- shared vs serial LS comparison

輸出會寫入：

```text
part4/results/
```

主要檔案包括：

- [Part4_Experiment_Summary.md](./results/Part4_Experiment_Summary.md)
- `*_build.txt`
- `*_run.txt`
- `main_shared_256_256.ptx`
- `ncu_shared_256_256.txt`
