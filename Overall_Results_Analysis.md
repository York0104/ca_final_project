# CA Final Project Overall Results Analysis

## Scope

本專題目前聚焦在：

```text
LS Channel Estimation + LMMSE/MMSE Equalizer Acceleration
```

目前納入比較的三個 parts：

| Part | Computation Stage 1 | Computation Stage 2 |
| --- | --- | --- |
| `Part 1` | Scalar LS channel estimation | Scalar LMMSE equalization |
| `Part 2` | RVV reduction LS channel estimation | RVV LMMSE equalization |
| `Part 3` | SIMD-like RVV LS channel estimation | RVV LMMSE equalization |

這份表整理的是同一個 OFDM receiver workload 在 Part 1～3 的 gem5 結果，不包含 Part 4 與 Part 5 的 CUDA kernel timing。

Part 4 和 Part 5 都已實作，但量測方式是 GPU kernel-only timing，和 gem5 simulated statistics 不同，所以分別放在各自的 CUDA 文件。

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

## Verification

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

這組數據用作 Part 2 / Part 3 的 gem5 baseline。

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

## Summary

- Part 1, Part 2, and Part 3 all pass the shared correctness checks.
- `H_MSE` and `MSE_LMMSE` stay close across the three implementations.
- In the current gem5 runs, Part 2 has the lowest `simSeconds` and `numCycles`.

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

## Interpretation

- Part 2 relative to Part 1:
  - `simSeconds` 約下降 `20.57%`
  - `numCycles` 約下降 `20.57%`
  - `simInsts` 約下降 `33.23%`

- CPI and miss rate increase in Part 2, but the instruction count drops more than enough to reduce total cycles.

- Part 3 relative to Part 1:
  - `simSeconds` 約增加 `5.34%`
  - `numCycles` 約增加 `5.34%`
  - `D-cache miss rate` 明顯升高到 `0.229839`

- Part 3 keeps the required across-`k` mapping, but the `vlse32.v` access pattern hurts spatial locality and pushes up both CPI and D-cache miss rate.

- The disassembly check matches the code-level mapping:
  - Part 2 binary 可見 `vfredusum.vs`
  - Part 3 binary 可見 `vlse32.v`
  - Part 3 binary 不含 vector reduction instruction

- Within the current Part 1～3 gem5 comparison, Part 2 is the fastest version.
