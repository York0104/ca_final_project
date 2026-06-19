# Part 4 實驗摘要

由 `run_experiments.sh` 產生。

此處的 timing 為 `main.cu` 內使用 `cudaEvent` 量測的 GPU kernel-only timing，
不包含 input loading、allocation、H2D/D2H copy 與 host-side verification。

## LS Shared Kernel 的 TPB Sweep

| Case | TPB_LS | TPB_EQ | LS Mode | Verification | LS ms | LMMSE ms | Pipeline ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ls_shared_64 | 64 | 256 | shared | PASS | 0.017032 | 0.019973 | 0.040740 |
| ls_shared_128 | 128 | 256 | shared | PASS | 0.015666 | 0.020561 | 0.039000 |
| ls_shared_256 | 256 | 256 | shared | PASS | 0.017372 | 0.020174 | 0.043345 |

## LMMSE Kernel 的 TPB Sweep

| Case | TPB_LS | TPB_EQ | LS Mode | Verification | LS ms | LMMSE ms | Pipeline ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| eq_shared_128 | 256 | 128 | shared | PASS | 0.024464 | 0.021142 | 0.062669 |
| eq_shared_256 | 256 | 256 | shared | PASS | 0.023186 | 0.024146 | 0.044678 |
| eq_shared_512 | 256 | 512 | shared | PASS | 0.035142 | 0.021271 | 0.034696 |

## Shared 與 Serial LS 比較

| Case | TPB_LS | TPB_EQ | LS Mode | Verification | LS ms | LMMSE ms | Pipeline ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ls_shared_256 | 256 | 256 | shared | PASS | 0.017372 | 0.020174 | 0.043345 |
| ls_serial_256 | 256 | 256 | serial | PASS | 0.135022 | 0.019535 | 0.118535 |

## 註

- `TPB_LS` 並非越大越快
  - 在目前 LS sweep 中，`TPB_LS=128` 有最低的 pipeline kernel-only time
- LMMSE sweep 呈現非單調結果
  - 在目前量測中，`TPB_EQ=512` 最佳
