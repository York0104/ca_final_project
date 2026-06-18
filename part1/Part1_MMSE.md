# CA Final Project Part 1

## 主題

Part 1 的正式主題為：

```text
LS Channel Estimation + LMMSE/MMSE Equalizer Acceleration
```

Part 1 對應的實作是：

```text
Scalar LS Channel Estimation
+ Scalar LMMSE Equalization
```

---

## 演算法定位

本專題不是單純的 dot product benchmark，而是一個簡化但完整的 OFDM receiver computation pipeline。

Part 1 包含兩個主要 computation stages：

1. `LS Channel Estimation`
2. `LMMSE Equalization`

其中：

- Stage 1 提供題目要求的 nested-loop summation / reduction form
- Stage 2 提供 element-wise complex arithmetic workload

因此整體 workload 同時具備：

- DSP-related numerical computing 特性
- channel estimation reduction kernel
- equalizer data-parallel arithmetic kernel

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

## 驗證指標

正式主流程保留四個驗證量：

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

這表示：

- LS channel estimation 正確
- LMMSE equalizer 明顯改善資料恢復誤差
- 結果有被實際使用，未被 compiler 移除

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

## 結論

Part 1 已完成正式 scalar baseline：

```text
Scalar LS Channel Estimation
+ Scalar LMMSE Equalization
```

這組結果可作為後續比較基準：

- Part 2：`RVV reduction LS + RVV LMMSE`
- Part 3：`SIMD-like RVV LS + RVV LMMSE`

目前依照新版正式結果來看：

- Part 2 相較 Part 1 有明顯加速
- Part 3 目前與 Part 1 幾乎相同

因此 Part 1 的角色已可明確定義為：

```text
formal scalar baseline for all later RVV comparisons
```
