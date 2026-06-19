# CA Final Project Overall Results Analysis

## Scope

本專題目前聚焦在：

```text
LS Channel Estimation + LMMSE/MMSE Equalizer Acceleration
```

目前納入比較的三個 parts：

| Part | Computation Stage 1 | Computation Stage 2 |
| --- | --- | --- |
| `Part 1` | Scalar LS channel estimation | Scalar LMMSE equalization |
| `Part 2` | RVV reduction LS channel estimation | RVV LMMSE equalization |
| `Part 3` | SIMD-like RVV LS channel estimation | RVV LMMSE equalization |

這份表整理的是同一個 OFDM receiver workload 在 Part 1～3 的 gem5 結果，不包含 Part 4 與 Part 5 的 CUDA kernel timing。

Part 4 和 Part 5 都已實作，但量測方式是 GPU kernel-only timing，和 gem5 simulated statistics 不同，所以分別放在各自的 CUDA 文件。
本頁的 gem5 表格對齊目前 repo 內保存的最新 rerun 結果。

---

## 共同數學模型

### Pilot stage

```text
X_pilot[p,k] = 1 + j0
Y_pilot[p,k] = H[k] + N_pilot[p,k]
```

### LS / averaging channel estimation

```text
Hhat[k] = sum_p Y_pilot[p,k] * w[p]
w[p] = 1 / P
```

### Data stage

```text
Y_data[s,k] = H[k] * X_data[s,k] + N_data[s,k]
```

### LMMSE / MMSE one-tap equalizer

```text
Xmmse[s,k] = Ydata[s,k] * conj(Hhat[k])
             / (|Hhat[k]|^2 + NOISE_VAR_OVER_SYMBOL_POWER + EPSILON)
```

其中：

```text
NOISE_VAR = complex noise power sigma_n^2
NOISE_VAR_OVER_SYMBOL_POWER = sigma_n^2 / sigma_x^2
```

---

## Verification

Part 1～3 共用四個驗證量：

| Metric | 用途 |
| --- | --- |
| `H_MSE` | 驗證 channel estimation |
| `MSE_RX_BEFORE_EQ` | 未等化前 baseline |
| `MSE_LMMSE` | 驗證 equalization 效果 |
| `checksum` | 防止 compiler 移除 `Hhat / Xmmse` |

Verification 條件：

```text
MSE_LMMSE < MSE_RX_BEFORE_EQ
H_MSE < 0.01
checksum != 0
```

---

## Part 1 結果

### Host correctness

| Metric | Value |
| --- | --- |
| `H_MSE` | `0.00001250` |
| `MSE_RX_BEFORE_EQ` | `0.14093372` |
| `MSE_LMMSE` | `0.00681053` |
| `Xmmse[0]` | `1.01242745 + j1.08173215` |
| `checksum` | `584.37127686` |
| `Verification` | `PASS` |

### gem5 baseline

| Metric | Part 1 Scalar |
| --- | --- |
| `simSeconds` | `0.069028` |
| `simInsts` | `17,066,251` |
| `numCycles` | `138,055,892` |
| `CPI` | `8.089394` |
| `IPC` | `0.123619` |
| `D-cache miss rate` | `0.114745` |
| `I-cache miss rate` | `0.000046` |

這組數據用作 Part 2 / Part 3 的 gem5 baseline。

---

## Part 2 結果

### Correctness

| Metric | Value |
| --- | --- |
| `H_MSE` | `0.00001250` |
| `MSE_RX_BEFORE_EQ` | `0.14093372` |
| `MSE_LMMSE` | `0.00681052` |
| `Verification` | `PASS` |

### gem5 stats

| Metric | Part 2 RVV |
| --- | --- |
| `simSeconds` | `0.054755` |
| `simInsts` | `11,395,368` |
| `numCycles` | `109,510,852` |
| `CPI` | `9.610092` |
| `IPC` | `0.104057` |
| `D-cache miss rate` | `0.170969` |
| `I-cache miss rate` | `0.000071` |

## Part 3 結果

### Correctness

| Metric | Value |
| --- | --- |
| `H_MSE` | `0.00001250` |
| `MSE_RX_BEFORE_EQ` | `0.14093372` |
| `MSE_LMMSE` | `0.00681053` |
| `Verification` | `PASS` |

### gem5 stats

| Metric | Part 3 SIMD-like RVV |
| --- | --- |
| `simSeconds` | `0.072706` |
| `simInsts` | `11,208,851` |
| `numCycles` | `145,412,866` |
| `CPI` | `12.973001` |
| `IPC` | `0.077083` |
| `D-cache miss rate` | `0.229838` |
| `I-cache miss rate` | `0.000051` |

---

## Summary

- Part 1, Part 2, and Part 3 all pass the shared correctness checks.
- `H_MSE` and `MSE_LMMSE` stay close across the three implementations.
- In the current gem5 runs, Part 2 has the lowest `simSeconds` and `numCycles`.

## 正式比較表

| Metric | Part 1 Scalar | Part 2 RVV | Part 3 SIMD-like RVV |
| --- | --- | --- | --- |
| `simSeconds` | `0.069028` | `0.054755` | `0.072706` |
| `simInsts` | `17,066,251` | `11,395,368` | `11,208,851` |
| `numCycles` | `138,055,892` | `109,510,852` | `145,412,866` |
| `CPI` | `8.089394` | `9.610092` | `12.973001` |
| `IPC` | `0.123619` | `0.104057` | `0.077083` |
| `D-cache miss rate` | `0.114745` | `0.170969` | `0.229838` |
| `I-cache miss rate` | `0.000046` | `0.000071` | `0.000051` |

## Interpretation

- Part 2 relative to Part 1:
  - `simSeconds` 約下降 `20.68%`
  - `numCycles` 約下降 `20.68%`
  - `simInsts` 約下降 `33.23%`

- CPI and miss rate increase in Part 2, but the instruction count drops more than enough to reduce total cycles.

- Part 3 relative to Part 1:
  - `simSeconds` 約增加 `5.33%`
  - `numCycles` 約增加 `5.33%`
  - `D-cache miss rate` 明顯升高到 `0.229838`

- Part 3 keeps the required across-`k` mapping, but the `vlse32.v` access pattern hurts spatial locality and pushes up both CPI and D-cache miss rate.

- This does not indicate a Part 3 correctness or compliance problem. The assignment explicitly asks Part 3 to use across-`k` SIMD-like mapping, forbids vector reduction, and notes that strided memory access is required. Under the current layout `pilot_index(k, p) = k * NUM_PILOTS + p`, Stage 1 therefore pays the cost of `vlse32.v` with a `NUM_PILOTS * sizeof(float) = 1024` byte stride across `k`.

- The supporting references are:
  - `reference/CA_Final_Project.pdf`: Part 3 explicitly says `Do NOT use Vector Reduction Operations`, states that strided memory access will be required, and gives `vlse32.v` / `vsse32.v` as examples. The same section also asks us to think about why cycle reduction and instruction reduction are not identical, and what tradeoffs Part 3 has relative to Part 2.
  - `reference/riscv-v-spec-1.0.pdf`: `vle32.v` is a unit-stride vector load, while `vlse32.v` is a strided vector load with a byte stride. This explains why Part 2 and Part 3 can have different memory behavior even when both reduce scalar loop overhead.

- The current Part 3 implementation also updates partial channel estimates through memory on every pilot iteration. That keeps the mapping faithful to the Part 3 objective, but it increases memory traffic compared with Part 2, where each subcarrier reduction can be handled with contiguous loads and a direct vector reduction.

- Therefore, the current Part 3 result is a reasonable architectural outcome for this workload: instruction count goes down, but cache behavior gets worse enough that total cycles do not improve. This is exactly the kind of tradeoff the Part 3 prompt asks us to analyze.

- The disassembly check matches the code-level mapping:
  - Part 2 binary 可見 `vfredusum.vs`
  - Part 3 binary 可見 `vlse32.v`
  - Part 3 binary 不含 vector reduction instruction

- Within the current Part 1～3 gem5 comparison, Part 2 is the fastest version.
