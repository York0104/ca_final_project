# CA Final Project Part 3

## Scope

Part 3 keeps the same OFDM workload and changes the Stage 1 mapping:

```text
LS Channel Estimation + LMMSE/MMSE Equalizer Acceleration
```

```text
SIMD-like RVV LS Channel Estimation + RVV LMMSE Equalization
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

The main difference from Part 2 is the Stage 1 mapping.

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

This stage uses SIMD-like RVV lanes, keeps one partial sum per lane, and relies on strided access instead of vector reduction.

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

## Verification

Part 3 與 Part 1 / Part 2 相同，只保留：

- `H_MSE`
- `MSE_RX_BEFORE_EQ`
- `MSE_LMMSE`
- `checksum`

The printed checks are the same ones used in Parts 1 and 2.

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

`objdump` shows `vlse32.v`, and it does not show `vfredusum.vs` or other vector reduction instructions.

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

## Interpretation

- `simInsts` drops relative to Part 1, but `simSeconds` and `numCycles` increase.
- The Stage 1 `vlse32.v` pattern uses a `NUM_PILOTS * sizeof(float)` byte stride across `k`, so the cache behavior is worse than in Part 2.
- The miss-rate increase is large enough to wipe out the benefit from the lower instruction count.
