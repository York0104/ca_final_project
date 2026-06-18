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
             / (|Hhat[k]|^2 + NOISE_VAR_OVER_SYMBOL_POWER + EPSILON)
```

其中：

```text
NOISE_VAR = complex noise power sigma_n^2
NOISE_VAR_OVER_SYMBOL_POWER = sigma_n^2 / sigma_x^2
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
| `MSE_LMMSE` | `0.00681053` |
| `Xmmse[0]` | `1.01242745 + j1.08173215` |
| `checksum` | `584.37127686` |
| `Verification` | `PASS` |

### gem5 baseline

| Metric | Part 1 Scalar |
| --- | --- |
| `simSeconds` | `0.074767` |
| `simInsts` | `17,065,752` |
| `numCycles` | `149,534,050` |
| `CPI` | `8.762213` |
| `IPC` | `0.114126` |
| `D-cache miss rate` | `0.114747` |
| `I-cache miss rate` | `0.000034` |

Part 1 已可作為正式 baseline。

---

## Part 2 結果

### Correctness

| Metric | Value |
| --- | --- |
| `H_MSE` | `0.00001250` |
| `MSE_RX_BEFORE_EQ` | `0.14093372` |
| `MSE_LMMSE` | `0.00681052` |
| `Verification` | `PASS` |

### gem5 stats

| Metric | Part 2 RVV |
| --- | --- |
| `simSeconds` | `0.067096` |
| `simInsts` | `16,444,742` |
| `numCycles` | `134,192,468` |
| `CPI` | `8.160189` |
| `IPC` | `0.122546` |
| `D-cache miss rate` | `0.120403` |
| `I-cache miss rate` | `0.000048` |

## Part 3 結果

### Correctness

| Metric | Value |
| --- | --- |
| `H_MSE` | `0.00001250` |
| `MSE_RX_BEFORE_EQ` | `0.14093372` |
| `MSE_LMMSE` | `0.00681053` |
| `Verification` | `PASS` |

### gem5 stats

| Metric | Part 3 SIMD-like RVV |
| --- | --- |
| `simSeconds` | `0.069027` |
| `simInsts` | `17,065,386` |
| `numCycles` | `138,053,396` |
| `CPI` | `8.089658` |
| `IPC` | `0.123615` |
| `D-cache miss rate` | `0.114750` |
| `I-cache miss rate` | `0.000046` |

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
| `simSeconds` | `0.074767` | `0.067096` | `0.069027` |
| `simInsts` | `17,065,752` | `16,444,742` | `17,065,386` |
| `numCycles` | `149,534,050` | `134,192,468` | `138,053,396` |
| `CPI` | `8.762213` | `8.160189` | `8.089658` |
| `IPC` | `0.114126` | `0.122546` | `0.123615` |
| `D-cache miss rate` | `0.114747` | `0.120403` | `0.114750` |
| `I-cache miss rate` | `0.000034` | `0.000048` | `0.000046` |

## 初步分析

- Part 2 相較 Part 1 有明顯改善：
  - `simSeconds` 約下降 `10.26%`
  - `numCycles` 約下降 `10.26%`
  - `simInsts` 約下降 `3.64%`
  - `IPC` 上升

- Part 3 相較 Part 1 也有改善：
  - `simSeconds` 約下降 `7.68%`
  - `numCycles` 約下降 `7.68%`
  - `simInsts` 幾乎持平
  - `IPC` 明顯上升

- Part 3 仍略慢於 Part 2。這表示 SIMD-like RVV 確實有效，但 Stage 1 的 `vlse32.v` strided memory access 與較差的 spatial locality，會抵消一部分理論上的平行化收益。

- 因此目前可以先得出：

```text
Part 2 is the best-performing implementation under the current design.
```
