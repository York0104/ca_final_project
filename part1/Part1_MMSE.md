# CA Final Project Part 1

## Scope

Part 1 uses the same OFDM workload as the later RVV and CUDA parts:

```text
LS Channel Estimation + LMMSE/MMSE Equalizer Acceleration
```

```text
Scalar LS Channel Estimation + Scalar LMMSE Equalization
```

---

## Workload

本專題不是單純的 dot product benchmark，而是一個簡化但完整的 OFDM receiver computation pipeline。

Part 1 包含兩個主要 computation stages：

1. `LS Channel Estimation`
2. `LMMSE Equalization`

- Stage 1 carries the nested weighted summation required by the assignment.
- Stage 2 carries the element-wise complex arithmetic used by the equalizer.

---

## 數學模型

### Pilot model

固定 pilot 設定：

```text
X_pilot[p,k] = 1 + j0
```

received pilot：

```text
Y_pilot[p,k] = H[k] + N_pilot[p,k]
```

### LS / averaging channel estimation

對每個 subcarrier `k`：

```text
Hhat[k] = sum_{p=0}^{P-1} Y_pilot[p,k] * w[p]
```

其中：

```text
w[p] = 1 / P
```

拆成 real / imag：

```text
Hhat_r[k] = sum_p Ypilot_r[p,k] * w[p]
Hhat_i[k] = sum_p Ypilot_i[p,k] * w[p]
```

這就是 Part 1 的主要 reduction loop。

### Data model

```text
Y_data[s,k] = H[k] * X_data[s,k] + N_data[s,k]
```

### LMMSE / MMSE one-tap equalizer

```text
Xmmse[s,k] = Ydata[s,k] * conj(Hhat[k])
             / (|Hhat[k]|^2 + NOISE_VAR_OVER_SYMBOL_POWER + EPSILON)
```

若令：

```text
D[k] = Hhat_r[k]^2 + Hhat_i[k]^2 + NOISE_VAR_OVER_SYMBOL_POWER + EPSILON
```

則：

```text
Xmmse_r[s,k] = (Ydata_r[s,k] * Hhat_r[k] + Ydata_i[s,k] * Hhat_i[k]) / D[k]
Xmmse_i[s,k] = (Ydata_i[s,k] * Hhat_r[k] - Ydata_r[s,k] * Hhat_i[k]) / D[k]
```

其中：

```text
NOISE_VAR = complex noise power sigma_n^2
NOISE_VAR_OVER_SYMBOL_POWER = sigma_n^2 / sigma_x^2
```

由於本專題的 QPSK symbol 採用 `±1 ± j1`，所以：

```text
SYMBOL_POWER = 2
```

因此正式 LMMSE denominator 採用較標準的：

```text
|Hhat[k]|^2 + sigma_n^2 / sigma_x^2 + EPSILON
```

---

## 程式結構

目前專案已重構成共用架構：

```text
common/
data_gen/
part1/
part2/
part3/
```

### C++ data generator

使用：

```text
data_gen/generate_ofdm_data.cpp
```

由 C++ 直接產生共同輸入檔：

```text
data/ofdm_input.bin
```

所有 Part 都讀同一份 binary input，因此比較更公平。

### Part 1 main.cpp

Part 1 的 [main.cpp](/home/york/ca_final_project/part1/main.cpp) 只保留本 Part 特有的兩個 computation stages：

1. `estimate_channel_ls_average_scalar()`
2. `equalize_lmmse_scalar()`

共用的資料、I/O 與驗證則放在 `common/`。

---

## Verification

The runtime prints four shared verification values:

```text
H_MSE
MSE_RX_BEFORE_EQ
MSE_LMMSE
checksum
```

用途如下：

| Metric | 用途 |
| --- | --- |
| `H_MSE` | 驗證 LS channel estimation 是否正確 |
| `MSE_RX_BEFORE_EQ` | 作為未等化前 baseline |
| `MSE_LMMSE` | 驗證 LMMSE equalizer 是否有效 |
| `checksum` | 防止 compiler 移除 `Hhat / Xmmse` 計算 |

Verification 條件為：

```text
MSE_LMMSE < MSE_RX_BEFORE_EQ
H_MSE < 0.01
checksum != 0
```

---

## Part 1 結果

### Host correctness

本次 host 執行結果：

| Metric | Value |
| --- | --- |
| `H_MSE` | `0.00001250` |
| `MSE_RX_BEFORE_EQ` | `0.14093372` |
| `MSE_LMMSE` | `0.00681053` |
| `Xmmse[0]` | `1.01242745 + j1.08173215` |
| `checksum` | `584.37127686` |
| `Verification` | `PASS` |

Interpretation:

- `H_MSE` is small.
- `MSE_LMMSE` is well below `MSE_RX_BEFORE_EQ`.
- `checksum` is non-zero, so the outputs are consumed after the kernels.

### gem5 baseline

本次 gem5 TimingSimpleCPU + caches 結果：

| Metric | Part 1 Scalar |
| --- | --- |
| `simSeconds` | `0.069028` |
| `simInsts` | `17,066,142` |
| `numCycles` | `138,056,924` |
| `CPI` | `8.089506` |
| `IPC` | `0.123617` |
| `D-cache miss rate` | `0.114745` |
| `I-cache miss rate` | `0.000046` |

---

## Summary

Part 1 is the scalar reference used for the later RVV comparisons.
