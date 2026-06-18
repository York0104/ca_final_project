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
| `checksum` | `584.37023926` |
| `Verification` | `PASS` |
| `__riscv_vector` | `enabled` |
| `Stage1_RVV_inline_asm` | `yes` |
| `Stage2_RVV_inline_asm` | `yes` |

這表示：

- RVV reduction LS channel estimation 與 scalar 版本維持相同正確性
- RVV LMMSE equalization 也維持正確
- checksum 僅有極小的 floating-point rounding 差異
- Part 2 的兩個 computation stages 都確實進入 RVV inline assembly path

### gem5 stats

| Metric | Part 2 RVV |
| --- | --- |
| `simSeconds` | `0.067096` |
| `simTicks` | `67,096,234,000` |
| `hostSeconds` | `20.99` |
| `simInsts` | `16,444,742` |
| `simOps` | `16,444,779` |
| `numCycles` | `134,192,468` |
| `CPI` | `8.160189` |
| `IPC` | `0.122546` |
| `D-cache misses` | `560,240` |
| `D-cache miss rate` | `0.120403` |
| `I-cache misses` | `926` |
| `I-cache miss rate` | `0.000048` |

### 與 Part 1 比較

| Metric | Part 1 Scalar | Part 2 RVV | Change |
| --- | --- | --- | --- |
| `simSeconds` | `0.074767` | `0.067096` | `-10.26%` |
| `simInsts` | `17,065,752` | `16,444,742` | `-3.64%` |
| `numCycles` | `149,534,050` | `134,192,468` | `-10.26%` |
| `CPI` | `8.762213` | `8.160189` | `-6.87%` |
| `IPC` | `0.114126` | `0.122546` | `+7.38%` |
| `D-cache miss rate` | `0.114747` | `0.120403` | `+4.93%` |
| `I-cache miss rate` | `0.000034` | `0.000048` | `+41.18%` |

初步解讀：

- Part 2 在正式 workload 下優於 Part 1
- Stage 1 的 RVV reduction 與 Stage 2 的 RVV LMMSE equalization 一起帶來整體加速
- 雖然 cache miss rate 略高，但 instruction count、cycles 與 simulated time 仍下降

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
