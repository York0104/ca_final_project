# CA Final Project Part 2

## 主題

Part 2 的正式主題為：

```text
LS Channel Estimation + LMMSE/MMSE Equalizer Acceleration
```

Part 2 對應的實作是：

```text
RVV Reduction LS Channel Estimation
+ RVV LMMSE Equalization
```

---

## 與 Part 1 的差異

Part 1：

```text
Scalar LS Channel Estimation
+ Scalar LMMSE Equalization
```

Part 2：

```text
RVV Reduction LS Channel Estimation
+ RVV LMMSE Equalization
```

因此 Part 2 不只是把 channel estimation 改成 RVV reduction，也同時把 equalization 改成 RVV element-wise vectorization。

---

## Computation Stage 1

### RVV Reduction LS Channel Estimation

數學形式仍然與 Part 1 相同：

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

這正好對應題目對 Part 2 的要求：必須使用 vector reduction operation。

---

## Computation Stage 2

### RVV Element-wise LMMSE Equalization

Part 2 的第二段主要計算為：

```text
Xmmse[s,k] = Ydata[s,k] * conj(Hhat[k])
             / (|Hhat[k]|^2 + NOISE_VAR_OVER_SYMBOL_POWER + EPSILON)
```

這段不需要 reduction，而是沿著 subcarrier `k` 做 unit-stride RVV element-wise vectorization。

其中：

```text
NOISE_VAR = complex noise power sigma_n^2
NOISE_VAR_OVER_SYMBOL_POWER = sigma_n^2 / sigma_x^2
```

因此 Part 2 的正式 LMMSE equalizer 使用較標準的 normalized denominator。

### CA mapping

```text
vector lanes -> different subcarriers k for the same data symbol s
```

因此 Part 2 同時展示兩類平行化：

- reduction parallelism
- element-wise data parallelism

---

## 程式結構

目前 [part2/main.cpp](/home/york/ca_final_project/part2/main.cpp) 只保留本 Part 特有的兩段 computation stages：

1. `estimate_channel_ls_average_rvv_reduction()`
2. `equalize_lmmse_rvv()`

共用部分則放在：

- [common/ofdm_params.h](/home/york/ca_final_project/common/ofdm_params.h)
- [common/ofdm_data.h](/home/york/ca_final_project/common/ofdm_data.h)
- [common/ofdm_io.h](/home/york/ca_final_project/common/ofdm_io.h)
- [common/ofdm_verify.h](/home/york/ca_final_project/common/ofdm_verify.h)

所有 Part 都讀取同一份：

```text
../data/ofdm_input.bin
```

---

## 驗證指標

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

### Correctness

本次 Part 2 輸出：

| Metric | Value |
| --- | --- |
| `H_MSE` | `0.00001250` |
| `MSE_RX_BEFORE_EQ` | `0.14093372` |
| `MSE_LMMSE` | `0.00681052` |
| `Xmmse[0]` | `1.01242721 + j1.08173215` |
| `checksum` | `584.37054443` |
| `Verification` | `PASS` |

這表示：

- RVV reduction LS channel estimation 與 scalar 版本維持相同正確性
- RVV LMMSE equalization 也維持正確
- checksum 僅有極小的 floating-point rounding 差異
- `objdump` 可見 `vfredusum.vs` 與 `vfdiv.vv / vse32.v`，表示兩個 stages 都確實使用 RVV 指令

### gem5 stats

| Metric | Part 2 RVV |
| --- | --- |
| `simSeconds` | `0.054827` |
| `simTicks` | `54,826,790,000` |
| `hostSeconds` | `22.91` |
| `simInsts` | `11,395,259` |
| `simOps` | `11,395,296` |
| `numCycles` | `109,653,580` |
| `CPI` | `9.622709` |
| `IPC` | `0.103921` |
| `D-cache misses` | `560,245` |
| `D-cache miss rate` | `0.170969` |
| `I-cache misses` | `928` |
| `I-cache miss rate` | `0.000071` |

### 與 Part 1 比較

| Metric | Part 1 Scalar | Part 2 RVV | Change |
| --- | --- | --- | --- |
| `simSeconds` | `0.069028` | `0.054827` | `-20.57%` |
| `simInsts` | `17,066,142` | `11,395,259` | `-33.23%` |
| `numCycles` | `138,056,924` | `109,653,580` | `-20.57%` |
| `CPI` | `8.089506` | `9.622709` | `+18.95%` |
| `IPC` | `0.123617` | `0.103921` | `-15.93%` |
| `D-cache miss rate` | `0.114745` | `0.170969` | `+48.95%` |
| `I-cache miss rate` | `0.000046` | `0.000071` | `+54.35%` |

初步解讀：

- Part 2 在正式 workload 下優於 Part 1
- Stage 1 的 RVV reduction 與 Stage 2 的 RVV LMMSE equalization 一起帶來整體加速
- 雖然 CPI 與 cache miss rate 上升，但 instruction count 大幅下降，因此總 cycles 與 simulated time 仍明顯改善

---

## 結論

Part 2 的正式定位是：

```text
RVV Reduction LS Channel Estimation
+ RVV LMMSE Equalization
```

其中：

- Stage 1 用來滿足 Part 2 的 vector reduction requirement
- Stage 2 用來展示 MMSE/LMMSE equalizer 的 RVV acceleration

這比「只加速 channel estimation 的 weighted sum」更完整，也更貼近本專題真正的目標。
