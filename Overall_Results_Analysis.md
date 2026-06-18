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

目前實作進度只到 `Part 1 ~ Part 3`。`Part 4` 與 `Part 5` 尚未實作，不納入本結果分析。

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
| `simSeconds` | `0.069028` |
| `simInsts` | `17,066,142` |
| `numCycles` | `138,056,924` |
| `CPI` | `8.089506` |
| `IPC` | `0.123617` |
| `D-cache miss rate` | `0.114745` |
| `I-cache miss rate` | `0.000046` |

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
| `simSeconds` | `0.054827` |
| `simInsts` | `11,395,259` |
| `numCycles` | `109,653,580` |
| `CPI` | `9.622709` |
| `IPC` | `0.103921` |
| `D-cache miss rate` | `0.170969` |
| `I-cache miss rate` | `0.000071` |

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
| `simSeconds` | `0.072715` |
| `simInsts` | `11,208,742` |
| `numCycles` | `145,430,124` |
| `CPI` | `12.974667` |
| `IPC` | `0.077073` |
| `D-cache miss rate` | `0.229839` |
| `I-cache miss rate` | `0.000051` |

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
| `simSeconds` | `0.069028` | `0.054827` | `0.072715` |
| `simInsts` | `17,066,142` | `11,395,259` | `11,208,742` |
| `numCycles` | `138,056,924` | `109,653,580` | `145,430,124` |
| `CPI` | `8.089506` | `9.622709` | `12.974667` |
| `IPC` | `0.123617` | `0.103921` | `0.077073` |
| `D-cache miss rate` | `0.114745` | `0.170969` | `0.229839` |
| `I-cache miss rate` | `0.000046` | `0.000071` | `0.000051` |

## 初步分析

- Part 2 相較 Part 1 有明顯改善：
  - `simSeconds` 約下降 `20.57%`
  - `numCycles` 約下降 `20.57%`
  - `simInsts` 約下降 `33.23%`

- Part 2 的 CPI 與 cache miss rate 雖然變差，但總指令數下降更多，因此整體仍最快。

- Part 3 相較 Part 1 並沒有帶來速度優勢：
  - `simSeconds` 約增加 `5.34%`
  - `numCycles` 約增加 `5.34%`
  - `D-cache miss rate` 明顯升高到 `0.229839`

- Part 3 慢於 Part 2 的主要原因，是 Stage 1 使用 `vlse32.v` 做 strided memory access。這種 across-`k` 的存取模式 spatial locality 較差，會把 CPI 與 cache miss rate 明顯推高。

- 反組譯檢查也支持上述結果：
  - Part 2 binary 可見 `vfredusum.vs`
  - Part 3 binary 可見 `vlse32.v`
  - Part 3 binary 不含 vector reduction instruction

- 因此目前可以先得出：

```text
Part 2 is the best-performing implementation under the current design.
```
