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
             / (|Hhat[k]|^2 + NOISE_VAR + EPSILON)
```

這段不需要 reduction，而是沿著 subcarrier `k` 做 unit-stride RVV element-wise vectorization。

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

由於本次專案架構已大幅重構成：

```text
common/ + data_gen/ + part1/part2/part3/
```

且正式主題也已改成：

```text
LS Channel Estimation + LMMSE/MMSE Equalizer Acceleration
```

因此舊版 Part 2 結果已不再代表目前正式設計。

目前這份文件僅保留新的設計與實作說明；新的 Part 2 統計數據應以重跑後結果為準，再更新至本文件與總分析表。

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
