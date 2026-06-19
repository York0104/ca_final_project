# CA Final Project Part 3


Part 3 keeps the same OFDM and changes the Stage 1 mapping:

```text
LS Channel Estimation + LMMSE/MMSE Equalizer Acceleration
```

```text
SIMD-like RVV LS Channel Estimation + RVV LMMSE Equalization
```

---

## 與 Part 2 差異

The main difference from Part 2 is the Stage 1 mapping.

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



---

## Computation Stage 1

### SIMD-like RVV LS Channel Estimation

數學形式：

```text
Hhat[k] = sum_p Y_pilot[p,k] * w[p]
```

 Part 3 差異：

- 不使用 vector reduction
- 每個 vector lane 對應不同的輸出 subcarrier `k`
- 對固定的 pilot index `p`，同時更新多個 `Hhat[k]`
- 每一組實際處理的 lane 數量由 `vsetvli` 在執行時決定
  - 在目前 gem5 的 `VLEN=256`、`SEW=32` 設定下，最多同時處理 8 個 FP32 lane


### CA mapping

```text
vector lanes -> different output subcarriers k no reduction
strided access -> vlse32.v
```
---

## Computation Stage 2

### RVV Element-wise LMMSE Equalization

與 Part 2 相同：

```text
Xmmse[s,k] = Ydata[s,k] * conj(Hhat[k]) / (|Hhat[k]|^2 + NOISE_VAR_OVER_SYMBOL_POWER + EPSILON)
```

這段沿著 subcarrier `k` 做 unit-stride RVV vectorization。

其中：

```text
NOISE_VAR = complex noise power sigma_n^2
NOISE_VAR_OVER_SYMBOL_POWER = sigma_n^2 / sigma_x^2
```

---

## 程式結構

[part3/main.cpp](main.cpp) 只保留本 Part 的兩段主要 computation stages：

1. `estimate_channel_ls_average_rvv_simd_like()`
2. `equalize_lmmse_rvv()`

並與 Part 1 / Part 2 共用：

- [common/ofdm_params.h](../common/ofdm_params.h)
- [common/ofdm_data.h](../common/ofdm_data.h)
- [common/ofdm_io.h](../common/ofdm_io.h)
- [common/ofdm_verify.h](../common/ofdm_verify.h)



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
| `simSeconds` | `0.072706` |
| `simTicks` | `72,706,433,000` |
| `hostSeconds` | `23.37` |
| `simInsts` | `11,208,851` |
| `simOps` | `11,438,264` |
| `numCycles` | `145,412,866` |
| `CPI` | `12.973001` |
| `IPC` | `0.077083` |
| `D-cache misses` | `790,639` |
| `D-cache miss rate` | `0.229838` |
| `I-cache misses` | `907` |
| `I-cache miss rate` | `0.000051` |

### 與 Part 1 / Part 2 比較

| Metric | Part 1 Scalar | Part 2 RVV | Part 3 SIMD-like RVV |
| --- | --- | --- | --- |
| `simSeconds` | `0.069028` | `0.054755` | `0.072706` |
| `simInsts` | `17,066,251` | `11,395,368` | `11,208,851` |
| `numCycles` | `138,055,892` | `109,510,852` | `145,412,866` |
| `CPI` | `8.089394` | `9.610092` | `12.973001` |
| `IPC` | `0.123619` | `0.104057` | `0.077083` |
| `D-cache miss rate` | `0.114745` | `0.170969` | `0.229838` |
| `I-cache miss rate` | `0.000046` | `0.000071` | `0.000051` |
