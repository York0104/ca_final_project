# CA Final Project Part 3

## 主題

Part 3 的正式主題為：

```text
LS Channel Estimation + LMMSE/MMSE Equalizer Acceleration
```

Part 3 對應的實作是：

```text
SIMD-like RVV LS Channel Estimation
+ RVV LMMSE Equalization
```

---

## 與 Part 2 的差異

Part 2：

```text
RVV reduction LS channel estimation
+ RVV LMMSE equalization
```

Part 3：

```text
SIMD-like RVV LS channel estimation
+ RVV LMMSE equalization
```

因此 Part 2 與 Part 3 的主要差別在於 Stage 1 的 parallelization strategy。

---

## Computation Stage 1

### SIMD-like RVV LS Channel Estimation

數學形式仍然是：

```text
Hhat[k] = sum_p Y_pilot[p,k] * w[p]
```

但 Part 3 的實作重點是：

- 不使用 vector reduction
- 每個 vector lane 對應不同的輸出 subcarrier `k`
- 對固定的 pilot index `p`，同時更新多個 `Hhat[k]`
- 由於 layout 採用 `pilot_index(k, p) = k * NUM_PILOTS + p`
  ，所以 across-`k` 的存取需要 strided load

### CA mapping

```text
vector lanes -> different output subcarriers k
no reduction
strided access -> vlse32.v
```

這正好符合 Part 3 題目要求：

- SIMD-like RVV
- 不使用 reduction
- 需要 strided memory access

---

## Computation Stage 2

### RVV Element-wise LMMSE Equalization

Part 3 的第二段與 Part 2 相同：

```text
Xmmse[s,k] = Ydata[s,k] * conj(Hhat[k])
             / (|Hhat[k]|^2 + NOISE_VAR + EPSILON)
```

這段沿著 subcarrier `k` 做 unit-stride RVV vectorization。

---

## 程式結構

[part3/main.cpp](/home/york/ca_final_project/part3/main.cpp) 只保留本 Part 的兩段主要 computation stages：

1. `estimate_channel_ls_average_rvv_simd_like()`
2. `equalize_lmmse_rvv()`

並與 Part 1 / Part 2 共用：

- [common/ofdm_params.h](/home/york/ca_final_project/common/ofdm_params.h)
- [common/ofdm_data.h](/home/york/ca_final_project/common/ofdm_data.h)
- [common/ofdm_io.h](/home/york/ca_final_project/common/ofdm_io.h)
- [common/ofdm_verify.h](/home/york/ca_final_project/common/ofdm_verify.h)

---

## 驗證指標

Part 3 與 Part 1 / Part 2 相同，只保留：

- `H_MSE`
- `MSE_RX_BEFORE_EQ`
- `MSE_LMMSE`
- `checksum`

這樣可以把主題聚焦在：

```text
LS channel estimation + LMMSE equalization acceleration
```

而不被額外的 ZF comparison 分散。

---

## 結果紀錄

由於本次架構與正式主題均已更新，Part 3 的正式 gem5 結果應以新版程式重跑後為準，再補進本文件與總分析表。

---

## 結論

Part 3 的正式定位是：

```text
SIMD-like RVV LS Channel Estimation
+ RVV LMMSE Equalization
```

其中 Stage 1 展示的是「不使用 reduction 的 across-output SIMD-like accumulation」，Stage 2 則延續 Part 2 的 RVV LMMSE equalization。
