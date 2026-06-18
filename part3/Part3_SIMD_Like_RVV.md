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
- 每一組實際處理的 lane 數量由 `vsetvli` 在執行時決定；在目前 gem5 的 `VLEN=256`、`SEW=32` 設定下，最多同時處理 8 個 FP32 lane
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
             / (|Hhat[k]|^2 + NOISE_VAR_OVER_SYMBOL_POWER + EPSILON)
```

這段沿著 subcarrier `k` 做 unit-stride RVV vectorization。

其中：

```text
NOISE_VAR = complex noise power sigma_n^2
NOISE_VAR_OVER_SYMBOL_POWER = sigma_n^2 / sigma_x^2
```

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

### Correctness

本次 Part 3 輸出：

| Metric | Value |
| --- | --- |
| `H_MSE` | `0.00001250` |
| `MSE_RX_BEFORE_EQ` | `0.14093372` |
| `MSE_LMMSE` | `0.00681053` |
| `Xmmse[0]` | `1.01242745 + j1.08173215` |
| `checksum` | `584.37115479` |
| `Verification` | `PASS` |

Part 3 與 Part 1 / Part 2 的 correctness 幾乎一致。`objdump` 可見 `vlse32.v`，且未出現 `vfredusum.vs` 或其他 vector reduction instruction。

### gem5 stats

| Metric | Part 3 SIMD-like RVV |
| --- | --- |
| `simSeconds` | `0.072715` |
| `simTicks` | `72,715,062,000` |
| `hostSeconds` | `26.86` |
| `simInsts` | `11,208,742` |
| `simOps` | `11,438,155` |
| `numCycles` | `145,430,124` |
| `CPI` | `12.974667` |
| `IPC` | `0.077073` |
| `D-cache misses` | `790,640` |
| `D-cache miss rate` | `0.229839` |
| `I-cache misses` | `907` |
| `I-cache miss rate` | `0.000051` |

### 與 Part 1 / Part 2 比較

| Metric | Part 1 Scalar | Part 2 RVV | Part 3 SIMD-like RVV |
| --- | --- | --- | --- |
| `simSeconds` | `0.069028` | `0.054827` | `0.072715` |
| `simInsts` | `17,066,142` | `11,395,259` | `11,208,742` |
| `numCycles` | `138,056,924` | `109,653,580` | `145,430,124` |
| `CPI` | `8.089506` | `9.622709` | `12.974667` |
| `IPC` | `0.123617` | `0.103921` | `0.077073` |
| `D-cache miss rate` | `0.114745` | `0.170969` | `0.229839` |
| `I-cache miss rate` | `0.000046` | `0.000071` | `0.000051` |

初步解讀：

- Part 3 在目前設計下慢於 Part 2，也略慢於 Part 1
- 相較 Part 1，Part 3 的 `simSeconds` 約增加 `5.34%`
- 相較 Part 2，Part 3 的 `simSeconds` 約增加 `32.63%`
- 更合理的原因是 Stage 1 使用 `vlse32.v` 進行 strided memory access，跨 `k` 的 stride 為 `NUM_PILOTS * sizeof(float)`，spatial locality 較差
- 較差的 spatial locality 會大幅提高 D-cache miss rate，並把 CPI 推高到 `12.97`
- 因此即使 Part 3 的 instruction count 低於 Part 1，最終 cycles 與 simulated time 仍沒有優勢
- 相較之下，Part 2 的 RVV reduction + RVV LMMSE equalization 目前最有效

---

## 結論

Part 3 的正式定位是：

```text
SIMD-like RVV LS Channel Estimation
+ RVV LMMSE Equalization
```

其中 Stage 1 展示的是「不使用 reduction 的 across-output SIMD-like accumulation」，Stage 2 則延續 Part 2 的 RVV LMMSE equalization。
