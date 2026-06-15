# CA Final Project Part 2：RVV Vector Reduction

## 1. Part 2 目標

Part 2 延續 Part 1 的 **Pilot-based OFDM MMSE Channel Equalizer**，但把 channel estimation 的 scalar reduction 改成 **RISC-V Vector (RVV) floating-point vector reduction**。

本部分的核心目標是：

```text
將 Part 1 的 weighted summation
H_hat[k] = sum_p Y_pilot[k][p] * w[p]

改寫成：
vector load -> vector multiply -> vector reduction -> scalar result
```

因此，Part 2 的優化重點只放在 **channel estimation**，而 **MMSE equalization** 仍維持 scalar 實作，方便和 Part 1 做公平比較。

---

## 2. 與 Part 1 的設計差異

### 2.1 演算法本身沒有改變

Part 1 與 Part 2 使用完全相同的通訊模型、資料集與驗證方式：

- `NUM_SUBCARRIERS = 512`
- `NUM_PILOTS = 256`
- `NUM_DATA_SYMBOLS = 512`
- `pilot_w[p] = 1 / NUM_PILOTS`
- `Ypilot[k][p]` 採用 contiguous layout

也就是說，兩者的差別不在數學模型，而在 **channel estimation kernel 的實作方式**。

### 2.2 Part 1：scalar weighted reduction

Part 1 的 channel estimation 形式為：

```cpp
for (int p = 0; p < NUM_PILOTS; ++p)
{
    int idx = pilot_index(k, p);
    float w = pilot_w[p];
    acc_r += Ypilot_r[idx] * w;
    acc_i += Ypilot_i[idx] * w;
}
```

這是標準的 scalar dot-product / reduction。

### 2.3 Part 2：RVV dot-product reduction helper

Part 2 將上面的 scalar inner loop 改成 RVV helper：

```cpp
static inline float rvv_dot_product_reduction_f32(const float *a,
                                                  const float *b,
                                                  int n)
```

它的工作流程是：

1. `vsetvli` 根據剩餘元素數量設定 `vl`
2. `vle32.v` 載入 `a` 與 `b`
3. `vfmul.vv` 做 element-wise multiply
4. `vfredusum.vs` 將 vector 內元素加總成 scalar
5. 由 C++ 外層累加 partial sum，直到 `n = 256` 全部處理完

對每個 subcarrier `k`：

```cpp
const float *yr = &Ypilot_r[pilot_index(k, 0)];
const float *yi = &Ypilot_i[pilot_index(k, 0)];

Hhat_r[k] = rvv_dot_product_reduction_f32(yr, pilot_w, NUM_PILOTS);
Hhat_i[k] = rvv_dot_product_reduction_f32(yi, pilot_w, NUM_PILOTS);
```

因此 Part 2 完整對應題目要求的形式：

```text
a_k[p] = Ypilot[k][p]
b_k[p] = pilot_w[p]
f(a_k[p], b_k[p]) = a_k[p] * b_k[p]
Hhat[k] = sum_p f(a_k[p], b_k[p])
```

### 2.4 Makefile 也和 Part 1 不同

Part 2 使用 RVV inline assembly，因此不能用 x86 host `g++` 直接執行。

目前 `Makefile` 已改成只保留 gem5 執行流程，並使用：

```make
CFLAGS := -static -O2 -std=c++11 -march=rv64gcv -mabi=lp64d
```

這代表 Part 2 必須：

```text
用 riscv64-linux-gnu-g++ 編譯
用 gem5 執行
```

---

## 3. 正確性結果

Part 2 gem5 執行輸出：

| Metric | Value |
| --- | --- |
| `H_MSE` | `0.00001260` |
| `MSE_RX` | `0.33055928` |
| `MSE_MMSE` | `0.01459648` |
| `Verification` | `PASS` |

這表示：

- RVV reduction 沒有破壞 channel estimation 正確性
- MMSE equalization 後的 MSE 仍明顯低於未 equalize 的 `MSE_RX`
- Part 2 與 Part 1 的數值結果幾乎一致，只存在極小的浮點累加順序差異

---

## 4. Part 2 gem5 Stats

本次 Part 2 結果如下：


| Metric | Part 1 Scalar | Part 2 RVV | Change |
| --- | --- | --- | --- |
| `simSeconds` | `0.138000` | `0.130226` | `-5.63%` |
| `simInsts` | `34,616,971` | `33,996,109` | `-1.79%` |
| `numCycles` | `276,000,316` | `260,452,526` | `-5.63%` |
| `CPI` | `7.972973` | `7.661241` | `-3.91%` |
| `IPC` | `0.125424` | `0.130527` | `+4.07%` |
| `D-cache miss rate` | `0.120870` | `0.124492` | `+3.00%` |
| `I-cache miss rate` | `0.000014` | `0.000016` | `+14.29%` |

---

## 6. Part 2 相較於 Part 1 的解讀

### 6.1 整體效能有提升

Part 2 最重要的結果是：

- `simSeconds` 從 `0.138000` 降到 `0.130226`
- `numCycles` 從 `276,000,316` 降到 `260,452,526`
- `simInsts` 也略微下降

這表示 RVV vector reduction 確實成功降低了 channel estimation 的執行成本。



### 6.3 CPI 下降、IPC 上升，代表 vector reduction 有發揮效果

從：

- `CPI: 7.972973 -> 7.661241`
- `IPC: 0.125424 -> 0.130527`

可以看出在相同 TimingSimpleCPU + cache 模型下，Part 2 每條 instruction 的平均成本略降，整體 instruction throughput 略升。

這與預期一致，因為一部分原本需要 scalar loop 完成的 multiply-and-sum，已被 RVV reduction 指令取代。

### 6.4 Cache miss rate 沒有明顯改善

Part 2 的 D-cache miss rate 從 `0.120870` 小幅上升到 `0.124492`，I-cache miss rate 也略升。



### 6.5 正確性維持一致

Part 2 的 `H_MSE` 與 `MSE_MMSE` 幾乎與 Part 1 相同：

- Part 1: `MSE_MMSE = 0.01459649`
- Part 2: `MSE_MMSE = 0.01459648`


---

## 7. 結論

Part 2 成功將 Part 1 的 scalar channel estimation 改寫為 **RVV Vector Reduction**，且結果符合預期：

- 數學模型不變
- 正確性維持 `Verification: PASS`
- `simSeconds`、`numCycles`、`simInsts` 均優於 Part 1
- `CPI` 下降、`IPC` 上升

因此可以得出結論：

```text
Part 2 的 RVV vector reduction 已成功加速 channel estimation kernel，
並成為後續 Part 3 SIMD-like RVV parallelization 的良好基礎。
```
