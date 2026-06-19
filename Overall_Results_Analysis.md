# CA Final Project 整體結果分析

## 範圍

主要設計：

```text
LS Channel Estimation + LMMSE/MMSE Equalizer Acceleration
```

目前納入比較的三個 parts：

| Part | Computation Stage 1 | Computation Stage 2 |
| --- | --- | --- |
| `Part 1` | Scalar LS channel estimation | Scalar LMMSE equalization |
| `Part 2` | RVV reduction LS channel estimation | RVV LMMSE equalization |
| `Part 3` | SIMD-like RVV LS channel estimation | RVV LMMSE equalization |

Part 1～3 使用 gem5 simulated statistics，Part 4～5 使用 `cudaEvent` 量測 GPU kernel-only timing。
這兩種量測不同，因此分成 gem5 與 CUDA 兩段整理。


---

## 共同數學模型

### Pilot stage

```text
X_pilot[p,k] = 1 + j0
Y_pilot[p,k] = H[k] + N_pilot[p,k]
```

### LS / averaging channel estimation

```text
Hhat[k] = sum_p Y_pilot[p,k] * w[p]
w[p] = 1 / P
```

### Data stage

```text
Y_data[s,k] = H[k] * X_data[s,k] + N_data[s,k]
```

### LMMSE / MMSE one-tap equalizer

```text
Xmmse[s,k] = Ydata[s,k] * conj(Hhat[k])
             / (|Hhat[k]|^2 + NOISE_VAR_OVER_SYMBOL_POWER + EPSILON)
```

其中：

```text
NOISE_VAR = complex noise power sigma_n^2
NOISE_VAR_OVER_SYMBOL_POWER = sigma_n^2 / sigma_x^2
```

---

## 驗證方式

Part 1～3 共用四個驗證量：

| Metric | 用途 |
| --- | --- |
| `H_MSE` | 驗證 channel estimation |
| `MSE_RX_BEFORE_EQ` | 未等化前 baseline |
| `MSE_LMMSE` | 驗證 equalization 效果 |
| `checksum` | 防止 compiler 移除 `Hhat / Xmmse` |

Verification 條件：

```text
MSE_LMMSE < MSE_RX_BEFORE_EQ
H_MSE < 0.01
checksum != 0
```

---

## Part 1 結果

### Host correctness

| Metric | 數值 |
| --- | --- |
| `H_MSE` | `0.00001250` |
| `MSE_RX_BEFORE_EQ` | `0.14093372` |
| `MSE_LMMSE` | `0.00681053` |
| `Xmmse[0]` | `1.01242745 + j1.08173215` |
| `checksum` | `584.37127686` |
| `Verification` | `PASS` |

### gem5 baseline

| Metric | Part 1 Scalar |
| --- | --- |
| `simSeconds` | `0.069028` |
| `simInsts` | `17,066,251` |
| `numCycles` | `138,055,892` |
| `CPI` | `8.089394` |
| `IPC` | `0.123619` |
| `D-cache miss rate` | `0.114745` |
| `I-cache miss rate` | `0.000046` |

這組數據是 Part 2 / Part 3 的 gem5 baseline。

---

## Part 2 結果

### Correctness

| Metric | 數值 |
| --- | --- |
| `H_MSE` | `0.00001250` |
| `MSE_RX_BEFORE_EQ` | `0.14093372` |
| `MSE_LMMSE` | `0.00681052` |
| `Verification` | `PASS` |

### gem5 stats

| Metric | Part 2 RVV |
| --- | --- |
| `simSeconds` | `0.054755` |
| `simInsts` | `11,395,368` |
| `numCycles` | `109,510,852` |
| `CPI` | `9.610092` |
| `IPC` | `0.104057` |
| `D-cache miss rate` | `0.170969` |
| `I-cache miss rate` | `0.000071` |

## Part 3 結果

### Correctness

| Metric | 數值 |
| --- | --- |
| `H_MSE` | `0.00001250` |
| `MSE_RX_BEFORE_EQ` | `0.14093372` |
| `MSE_LMMSE` | `0.00681053` |
| `Verification` | `PASS` |

### gem5 stats

| Metric | Part 3 SIMD-like RVV |
| --- | --- |
| `simSeconds` | `0.072706` |
| `simInsts` | `11,208,851` |
| `numCycles` | `145,412,866` |
| `CPI` | `12.973001` |
| `IPC` | `0.077083` |
| `D-cache miss rate` | `0.229838` |
| `I-cache miss rate` | `0.000051` |

---

## Part 4 結果

### Correctness

預設 case：`TPB_LS=256`、`TPB_EQ=256`、`LS_KERNEL_MODE=shared`

| Metric | 數值 |
| --- | --- |
| `H_MSE` | `0.00001250` |
| `MSE_RX_BEFORE_EQ` | `0.14093372` |
| `MSE_LMMSE` | `0.00681053` |
| `checksum` | `584.37121582` |
| `Verification` | `PASS` |

### GPU kernel-only timing

| Metric | Part 4 CUDA SIMT |
| --- | --- |
| `LS_KERNEL_MS` | `0.023186` |
| `LMMSE_KERNEL_MS` | `0.024146` |
| `PIPELINE_KERNEL_MS` | `0.044678` |

### Sweep 摘要

LS shared kernel 的 TPB sweep：

| Case | `TPB_LS` | `PIPELINE_KERNEL_MS` |
| --- | --- | --- |
| `ls_shared_64` | `64` | `0.040740` |
| `ls_shared_128` | `128` | `0.039000` |
| `ls_shared_256` | `256` | `0.043345` |

LMMSE kernel 的 TPB sweep：

| Case | `TPB_EQ` | `PIPELINE_KERNEL_MS` |
| --- | --- | --- |
| `eq_shared_128` | `128` | `0.062669` |
| `eq_shared_256` | `256` | `0.044678` |
| `eq_shared_512` | `512` | `0.034696` |

Shared vs serial LS：

| Case | `LS mode` | `PIPELINE_KERNEL_MS` |
| --- | --- | --- |
| `ls_shared_256` | `shared` | `0.043345` |
| `ls_serial_256` | `serial` | `0.118535` |

解讀：

- Part 4 對應 `reference/CA_Final_Project.pdf` 的重點是 CUDA SIMT 與 `__shared__` 的使用。
- 設計把 Stage 1 映射成 one-block-per-subcarrier，block 內 threads 共同完成 pilot reduction，因此 shared memory 的角色是 block-cooperative reduction。
- `TPB_LS` 與 `TPB_EQ` 都呈現非單調結果，在目前 workload 下，較大的 block 不一定最好。
- shared LS 比 serial LS 快
  - 「block-level parallel reduction + shared memory + 更多可同時執行的 threads」的整體效果

---

## Part 5 結果

### Correctness

預設 case：`NUM_PATTERNS=16`、`TPB_LS=256`、`TPB_EQ=256`

| Metric | GPU | CPU baseline |
| --- | --- | --- |
| `H_MSE` | `0.00001289` | `0.00001289` |
| `MSE_RX_BEFORE_EQ` | `0.14082038` | `0.14082038` |
| `MSE_LMMSE` | `0.00682142` | `0.00682142` |
| `checksum` | `9294.77246094` | `9294.76074219` |
| `Verification` | `PASS` | `PASS` |

### GPU kernel-only timing 與 CPU baseline

| Metric | GPU | CPU baseline |
| --- | --- | --- |
| `LS_KERNEL_MS` | `0.128993` | `N/A` |
| `LMMSE_KERNEL_MS` | `0.308572` | `N/A` |
| `PIPELINE_MS` | `0.434780` | `12.268151` |

以 `16 patterns` 的預設 case 來看，GPU kernel-only pipeline time 約為 CPU baseline 的 `28.22x` 加速。

### Pattern sweep

| `NUM_PATTERNS` | `PIPELINE_KERNEL_MS` | `ms / pattern` |
| --- | --- | --- |
| `1` | `0.069048` | `0.069048` |
| `4` | `0.141583` | `0.035396` |
| `8` | `0.214241` | `0.026780` |
| `16` | `0.404536` | `0.025284` |
| `32` | `0.926991` | `0.028968` |

解讀：

- `reference/CA_Final_Project.pdf` 對 Part 5 的要求是利用 GPU 同時處理多個彼此獨立的 input patterns，並以 `blockIdx.y` 等 2D grid mapping 暴露這種 pattern-level parallelism。
- 目前結果符合這個方向：pattern 數增加時，總 kernel time 會上升，但每個 pattern 的平均時間先下降，表示 GPU launch / scheduling overhead 與固定成本被更多工作量攤提。
- 效果非無限延伸。到 `32 patterns` 時，`ms / pattern` 從 `0.025284` 回升到 `0.028968`，顯示 occupancy、memory traffic 與整體資源利用的改善開始趨緩。
- Part 5 的重點不是單一 pattern latency 更小，而是 batch workload 下的 throughput 更適合 GPU。

---

## 摘要

- Part 1～Part 5 都通過各自的 correctness checks
- Part 1～3 的 `H_MSE` 與 `MSE_LMMSE` 維持接近，Part 4 與 Part 5 也與對應 baseline 對齊
- 在目前 gem5 結果中，Part 2 有最低的 `simSeconds` 與 `numCycles`


## 正式比較表

| Metric | Part 1 Scalar | Part 2 RVV | Part 3 SIMD-like RVV |
| --- | --- | --- | --- |
| `simSeconds` | `0.069028` | `0.054755` | `0.072706` |
| `simInsts` | `17,066,251` | `11,395,368` | `11,208,851` |
| `numCycles` | `138,055,892` | `109,510,852` | `145,412,866` |
| `CPI` | `8.089394` | `9.610092` | `12.973001` |
| `IPC` | `0.123619` | `0.104057` | `0.077083` |
| `D-cache miss rate` | `0.114745` | `0.170969` | `0.229838` |
| `I-cache miss rate` | `0.000046` | `0.000071` | `0.000051` |

## 解讀

- Part 2 相較 Part 1：
  - `simSeconds` 約下降 `20.68%`
  - `numCycles` 約下降 `20.68%`
  - `simInsts` 約下降 `33.23%`

- Part 2 的 CPI 與 miss rate 雖然上升，但 instruction count 的下降幅度更大，因此 total cycles 仍然下降。

- Part 3 相較 Part 1：
  - `simSeconds` 約增加 `5.33%`
  - `numCycles` 約增加 `5.33%`
  - `D-cache miss rate` 明顯升高到 `0.229838`

- Part 3 維持題目要求的 across-`k` mapping，但 `vlse32.v` 的 access pattern 破壞了 spatial locality，讓 CPI 與 D-cache miss rate 都上升。

- 這不代表 Part 3 寫錯或不合規。題目本身明確要求 Part 3 採用 across-`k` SIMD-like mapping、禁止 vector reduction，且需要 strided memory access。在目前的 layout `pilot_index(k, p) = k * NUM_PILOTS + p` 下，Stage 1 自然要承擔 `vlse32.v` 在跨 `k` 時的 `NUM_PILOTS * sizeof(float) = 1024` byte stride。

- 可支撐這個結論的 reference 包括：
  - `reference/CA_Final_Project.pdf`：Part 3 明確寫出 `Do NOT use Vector Reduction Operations`，也指出 strided memory access 是必要的，並以 `vlse32.v` / `vsse32.v` 為例。同一節也要求說明為什麼 cycle reduction 與 instruction reduction 不一定相同，以及 Part 3 和 Part 2 的 tradeoff。
  - `reference/riscv-v-spec-1.0.pdf`：`vle32.v` 是 unit-stride vector load，而 `vlse32.v` 是以 byte stride 為基礎的 strided vector load。這說明了即使 Part 2 與 Part 3 都減少 scalar loop overhead，兩者仍可能有不同的 memory behavior。

- 目前 Part 3 的實作還會在每次 pilot iteration 中透過記憶體更新 partial channel estimates。這讓 mapping 忠於 Part 3 的目標，但也帶來額外 memory traffic；相較之下，Part 2 可沿著 contiguous pilot dimension `p` 做直接的 vector reduction。

- 因此，目前 Part 3 的結果屬於合理的架構現象：instruction count 雖然下降，但 cache behavior 惡化到足以讓 total cycles 不降反升。這正是 Part 3 題目希望觀察的 tradeoff。

- 反組譯也和這個 mapping 一致：
  - Part 2 binary 可見 `vfredusum.vs`
  - Part 3 binary 可見 `vlse32.v`
  - Part 3 binary 不含 vector reduction instruction

- 在目前 Part 1～3 的 gem5 比較中，Part 2 是最快的版本。
