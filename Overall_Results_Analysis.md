# CA Final Project Overall Results Analysis

## 正式主題

本專題目前的正式主題為：

```text
LS Channel Estimation + LMMSE/MMSE Equalizer Acceleration
```

正式三個 Part 定義如下：

| Part | Computation Stage 1 | Computation Stage 2 |
| --- | --- | --- |
| `Part 1` | Scalar LS channel estimation | Scalar LMMSE equalization |
| `Part 2` | RVV reduction LS channel estimation | RVV LMMSE equalization |
| `Part 3` | SIMD-like RVV LS channel estimation | RVV LMMSE equalization |

這代表目前的正式 workload 是一個簡化但完整的 OFDM receiver pipeline，而不是單一 dot product benchmark。

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
             / (|Hhat[k]|^2 + NOISE_VAR + EPSILON)
```

---

## 正式驗證指標

目前正式主流程只保留四個驗證量：

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

| Metric | Value |
| --- | --- |
| `H_MSE` | `0.00001250` |
| `MSE_RX_BEFORE_EQ` | `0.14093372` |
| `MSE_LMMSE` | `0.00694250` |
| `Xmmse[0]` | `0.97359151 + j1.04023767` |
| `checksum` | `584.51818848` |
| `Verification` | `PASS` |

### gem5 baseline

| Metric | Part 1 Scalar |
| --- | --- |
| `simSeconds` | `0.074765` |
| `simInsts` | `17,063,351` |
| `numCycles` | `149,530,538` |
| `CPI` | `8.763241` |
| `IPC` | `0.114113` |
| `D-cache miss rate` | `0.114764` |
| `I-cache miss rate` | `0.000035` |

Part 1 已可作為正式 baseline。

---

## Part 2 結果

### Correctness

| Metric | Value |
| --- | --- |
| `H_MSE` | `0.00001250` |
| `MSE_RX_BEFORE_EQ` | `0.14093372` |
| `MSE_LMMSE` | `0.00694250` |
| `Verification` | `PASS` |

### gem5 stats

| Metric | Part 2 RVV |
| --- | --- |
| `simSeconds` | `0.067093` |
| `simInsts` | `16,442,183` |
| `numCycles` | `134,185,122` |
| `CPI` | `8.161012` |
| `IPC` | `0.122534` |
| `D-cache miss rate` | `0.120423` |
| `I-cache miss rate` | `0.000048` |

## Part 3 結果

### Correctness

| Metric | Value |
| --- | --- |
| `H_MSE` | `0.00001250` |
| `MSE_RX_BEFORE_EQ` | `0.14093372` |
| `MSE_LMMSE` | `0.00694250` |
| `Verification` | `PASS` |

### gem5 stats

| Metric | Part 3 SIMD-like RVV |
| --- | --- |
| `simSeconds` | `0.074766` |
| `simInsts` | `17,063,365` |
| `numCycles` | `149,531,160` |
| `CPI` | `8.763270` |
| `IPC` | `0.114113` |
| `D-cache miss rate` | `0.114764` |
| `I-cache miss rate` | `0.000035` |

---

## 目前結論

截至目前為止，可以先確立：

1. 專案正式主題已定義清楚。
2. 程式結構已對齊成 `common + data_gen + part1 + part2 + part3`。
3. Part 1 / Part 2 / Part 3 都已在新版正式設計下完成執行。
4. 三個 Part 的 correctness 均通過，`H_MSE` 與 `MSE_LMMSE` 基本一致。
5. 目前最佳 performance 來自 Part 2。

## 正式比較表

| Metric | Part 1 Scalar | Part 2 RVV | Part 3 SIMD-like RVV |
| --- | --- | --- | --- |
| `simSeconds` | `0.074765` | `0.067093` | `0.074766` |
| `simInsts` | `17,063,351` | `16,442,183` | `17,063,365` |
| `numCycles` | `149,530,538` | `134,185,122` | `149,531,160` |
| `CPI` | `8.763241` | `8.161012` | `8.763270` |
| `IPC` | `0.114113` | `0.122534` | `0.114113` |
| `D-cache miss rate` | `0.114764` | `0.120423` | `0.114764` |
| `I-cache miss rate` | `0.000035` | `0.000048` | `0.000035` |

## 初步分析

- Part 2 相較 Part 1 有明顯改善：
  - `simSeconds` 約下降 `10.26%`
  - `numCycles` 約下降 `10.26%`
  - `simInsts` 約下降 `3.64%`
  - `IPC` 上升

- Part 3 目前與 Part 1 幾乎完全相同，表示這版 SIMD-like RVV channel estimation 尚未帶來有效的整體效能優勢。

- 因此目前可以先得出：

```text
Part 2 is the best-performing implementation under the current design.
```
