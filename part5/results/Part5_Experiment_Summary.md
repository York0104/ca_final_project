# Part 5 實驗摘要

由 `run_experiments.sh` 產生。

此處的 timing 為 `main.cu` 內使用 `cudaEvent` 量測的 GPU kernel-only timing，
不包含 input loading、allocation、H2D/D2H copy 與 host-side verification。

## Pattern Sweep

| Patterns | Verification | LS ms | LMMSE ms | Pipeline ms |
| --- | --- | --- | --- | --- |
| 1 | PASS | 0.032481 | 0.037750 | 0.069048 |
| 4 | PASS | 0.039578 | 0.117176 | 0.141583 |
| 8 | PASS | 0.059996 | 0.177853 | 0.214241 |
| 16 | PASS | 0.086354 | 0.347571 | 0.404536 |
| 32 | PASS | 0.223770 | 0.745861 | 0.926991 |

## 註

- Part 5 把 Part 4 從單一 OFDM pattern 延伸到多個彼此獨立的 patterns
- 主要新增的 mapping 是 `blockIdx.y -> pattern index`
    - 在單一 pattern workload 無法充分餵飽 GPU 時
    - 比 Part 4 暴露出更多獨立的 blocks 與 warps
- 隨著 pattern count 增加，total pipeline time 上升
    - 因為總工作量上升
- pipeline time per pattern隨著 pattern 數量增加，GPU launch overhead 與資源利用的overhead amortization會更明顯
