# CA Final Project Part 2



Part 2 keeps the same OFDM workload and changes the mapping:

```text
LS Channel Estimation + LMMSE/MMSE Equalizer Acceleration
```

```text
RVV Reduction LS Channel Estimation + RVV LMMSE Equalization
```

---

## 與 Part 1 的差異

Part 1：

```text
Scalar LS Channel Estimation + Scalar LMMSE Equalization
```

Part 2：

```text
RVV Reduction LS Channel Estimation + RVV LMMSE Equalization
```

* Stage 1 uses RVV reduction.
* Stage 2 uses RVV element-wise vectorization.

---

## Computation Stage 1

### RVV Reduction LS Channel Estimation

數學與 Part 1 相同：

```text
Hhat[k] = sum_p Y_pilot[p,k] * w[p]
```

但 Part 2 使用 RVV reduction 完成：

- `vsetvli`
- `vle32.v`
- `vfmul.vv`
- `vfredusum.vs`
- `vfmv.f.s`

### CA mapping

```text
vector lanes -> different pilot observations p for the same subcarrier k
reduction    -> sum over p
```

---

## Computation Stage 2

### RVV Element-wise LMMSE Equalization

Part 2 的第二段主要計算為：

```text
Xmmse[s,k] = Ydata[s,k] * conj(Hhat[k]) / (|Hhat[k]|^2 + NOISE_VAR_OVER_SYMBOL_POWER + EPSILON)
```

這段沿著 subcarrier `k` 做 unit-stride RVV element-wise vectorization。

其中：

```text
NOISE_VAR = complex noise power sigma_n^2
NOISE_VAR_OVER_SYMBOL_POWER = sigma_n^2 / sigma_x^2
```

### CA mapping

```text
vector lanes -> different subcarriers k for the same data symbol s
```

Part 2 contains two different mappings:

- reduction parallelism
- element-wise data parallelism

---

## 程式結構

[part2/main.cpp](main.cpp) 只保留本 Part 特有的兩段 computation stages：

1. `estimate_channel_ls_average_rvv_reduction()`
2. `equalize_lmmse_rvv()`

共用部分則放在：

- [common/ofdm_params.h](../common/ofdm_params.h)
- [common/ofdm_data.h](../common/ofdm_data.h)
- [common/ofdm_io.h](../common/ofdm_io.h)
- [common/ofdm_verify.h](../common/ofdm_verify.h)

所有 Part 都讀取同一份：

```text
../data/ofdm_input.bin
```

---

## Verification

Part 2 與 Part 1 使用相同的正式驗證量：

| Metric | 用途 |
| --- | --- |
| `H_MSE` | 驗證 channel estimation |
| `MSE_RX_BEFORE_EQ` | 作為未等化前 baseline |
| `MSE_LMMSE` | 驗證 LMMSE equalization |
| `checksum` | 防止 compiler 移除 `Hhat / Xmmse` |

Verification 條件：

```text
MSE_LMMSE < MSE_RX_BEFORE_EQ
H_MSE < 0.01
checksum != 0
```

---

## 結果紀錄

### Evidence

本次 Part 2 輸出：

| Metric | Value |
| --- | --- |
| `H_MSE` | `0.00001250` |
| `MSE_RX_BEFORE_EQ` | `0.14093372` |
| `MSE_LMMSE` | `0.00681052` |
| `Xmmse[0]` | `1.01242721 + j1.08173215` |
| `checksum` | `584.37054443` |
| `Verification` | `PASS` |



### gem5 stats

| Metric | Part 2 RVV |
| --- | --- |
| `simSeconds` | `0.054755` |
| `simTicks` | `54,755,426,000` |
| `hostSeconds` | `21.49` |
| `simInsts` | `11,395,368` |
| `simOps` | `11,395,405` |
| `numCycles` | `109,510,852` |
| `CPI` | `9.610092` |
| `IPC` | `0.104057` |
| `D-cache misses` | `560,245` |
| `D-cache miss rate` | `0.170969` |
| `I-cache misses` | `928` |
| `I-cache miss rate` | `0.000071` |

### gem5 Comparison Against Part 1

| Metric | Part 1 Scalar | Part 2 RVV | Change |
| --- | --- | --- | --- |
| `simSeconds` | `0.069028` | `0.054755` | `-20.68%` |
| `simInsts` | `17,066,251` | `11,395,368` | `-33.23%` |
| `numCycles` | `138,055,892` | `109,510,852` | `-20.68%` |
| `CPI` | `8.089394` | `9.610092` | `+18.80%` |
| `IPC` | `0.123619` | `0.104057` | `-15.82%` |
| `D-cache miss rate` | `0.114745` | `0.170969` | `+48.95%` |
| `I-cache miss rate` | `0.000046` | `0.000071` | `+54.35%` |
