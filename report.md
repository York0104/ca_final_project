> **文件用途：** 本文件為 Computer Architecture Final Project 報告／簡報草稿，目前整理共同 workload、數學模型、實驗環境、驗證方法與 Part 1 scalar baseline，可直接貼到 Notion，再轉為投影片或正式 `Report.pdf`。
>
>
> **目前 repository 狀態：** GitHub repository 已完成 Part 1～Part 5；本文件整理共同 workload、Part 1～Part 5 的實作證據、量測結果與整體比較，可作為正式報告主稿。
> 

## 1. 研究動機與專題目標

!Conceptual OFDM Receiver Background

Conceptual OFDM Receiver Background

現代處理器會以不同 execution models 利用 parallelism。Scalar processor 一次處理一個資料元素；vector processor 則以多個 vector lanes，在一條 instruction 中處理多筆資料；GPU 進一步使用大量 threads 的 SIMT execution。課程期末專題要求比較 Scalar、RVV Vector Reduction、SIMD-like RVV、CUDA SIMT 與 multi-pattern GPU parallelism 等不同實作方式。

期末專題選擇以 pilot-based OFDM receiver 作為加速主題，其主要由兩個互補的 DSP kernels 組成

- 第一段： repeated-pilot Least-Squares（LS）Channel Estimation
    - 具有 nested weighted summation 
    → 適合 vector reduction
- 第二段： LMMSE/MMSE-form Equalization
    - 在不同 data symbols 與 subcarriers 上產生彼此獨立的 outputs
    → 適合研究 element-wise data-level parallelism。

本階段的核心重點為：

> 在完全相同的 OFDM receiver algorithm、OFDM parameters、輸入資料下，Scalar execution、RVV reduction mapping 與 SIMD-like RVV mapping，會如何影響 instruction count、cycle count、simulated execution time 與 memory-access behavior？
> 

## 2. OFDM Receiver

目前實作的是簡化但完整的 frequency-domain OFDM receiver ：

!image.png

系統使用 512 個 subcarriers、每個 subcarrier 256 筆 repeated pilot observations，以及 512 個 OFDM data symbols。

- 因此，LS stage 需要處理 512 × 256 = 131,072 筆 pilot observations
- equalization stage 則需要輸出 512 × 512 = 262,144 筆 complex equalized symbols

| Parameter | Value | 註 |
| --- | --- | --- |
| `NUM_SUBCARRIERS` | 512 | Frequency-domain subcarriers，index：`k` |
| `NUM_PILOTS` | 256 | 每個 subcarrier 用於估測 channel coefficient 的 repeated pilots |
| `NUM_DATA_SYMBOLS` | 512 | OFDM data symbols，index：`s` |
| `TOTAL_PILOT` | 131,072 | # complex pilot observations  |
| `TOTAL_DATA` | 262,144 | # complex equalization outputs  |
| Modulation | QPSK，`±1 ± j1` | Transmitted data symbols |
| `SYMBOL_POWER` | 2 | Average symbol power |
| `NOISE_STD` | 0.0404 |  real dimension 的 AWGN standard deviation |

資料由固定 seed 的 generator 產生，並輸出成共同輸入檔 `data/ofdm_input.bin`。Part 1、Part 2、Part 3 均讀取同一份 channel、pilot observations、transmitted QPSK data、received data 與 noise realization。避免後續效能受到不同 test vectors 的影響。

- Reference channel 包含 controlled deep fades：
    - 每 64 個 subcarriers 中，前 3 個 subcarriers 的 channel magnitude 設為 `0.20`，其餘為 `1.00`
- memory layout 為：
    
    ```
    pilot_index(k, p) = k × NUM_PILOTS + p
    data_index(s, k)  = s × NUM_SUBCARRIERS + k
    ```
    
    - 固定 subcarrier `k` 時，pilot index `p` 方向的資料是 contiguous
    - 固定 pilot `p`、跨不同 subcarriers `k` 時，資料間距為
    
    ```
    NUM_PILOTS × sizeof(float) = 256 × 4 bytes = 1024 bytes
    ```
    

→ Part 2 可利用 unit-stride access；Part 3 則必須處理 strided memory access

## 3. 數學模型

- Pilot symbols 固定為：$X_{\mathrm{pilot}}[p,k] = 1 + j0$
- Received pilot model 為：$Y_{\mathrm{pilot}}[p,k] = H[k] + N_{\mathrm{pilot}}[p,k]$
- LS Channel Estimation：
    
                                      $\hat{H}[k] =
    \sum_{p=0}^{P-1}
    Y_{\mathrm{pilot}}[p,k] \cdot w[p],
    \qquad
    w[p] = \frac{1}{P}$
    
    ∵ pilot amplitude = 1
    

- Real 與 imaginary components ：
    - $\hat{H}_r[k] =
    \sum_p Y_{\mathrm{pilot},r}[p,k] \cdot w[p]$
    - $\hat{H}_i[k] =
    \sum_p Y_{\mathrm{pilot},i}[p,k] \cdot w[p]$

對應 weighted summation ⇒  reduction kernel，符合作業對 Part 1 / Part 2 所要求的形式：
                                                $f(a_1, b_1) + f(a_2, b_2) + ... + f(a_n, b_n)$

- `f(a,b) = a × b`
    - `a[p] = Ypilot[k,p]`
    - `b[p] = pilot_w[p]`

Stage 1  RVV mapping：

```
Part 2：vector lanes → 同一個 k 的不同 pilot observations p
        需要 vector reduction 將多個乘積加總為 Hhat[k]

Part 3：vector lanes → 不同 output subcarriers k
        固定 p，同時更新多個 Hhat[k]
        不需要 vector reduction，但需要 strided memory access
```

- OFDM data model 為：$Y_{\mathrm{data}}[s,k] =
H[k]X_{\mathrm{data}}[s,k] + N_{\mathrm{data}}[s,k]$
- LMMSE/MMSE-form equalizer：$\hat{X}_{\mathrm{mmse}}[s,k] =
\frac{
Y_{\mathrm{data}}[s,k] \cdot \hat{H}^{*}[k]
}{
\left|\hat{H}[k]\right|^2 +
\sigma_n^2 / \sigma_x^2 +
\epsilon
}$
- 其 real-valued implementation ：
    - $D[k] =
    \hat{H}_r[k]^2 +
    \hat{H}_i[k]^2 +
    \mathrm{NOISE\_VAR\_OVER\_SYMBOL\_POWER} +
    \epsilon$
        - `EPSILON` ：避免 denominator 在 deep fade 時過小而造成 numerical instability
    - $\hat{X}_{r}[s,k] =
    \frac{
    Y_r[s,k]\hat{H}_r[k] +
    Y_i[s,k]\hat{H}_i[k]
    }{
    D[k]
    }$
    - $\hat{X}_{i}[s,k] =
    \frac{
    Y_i[s,k]\hat{H}_r[k] -
    Y_r[s,k]\hat{H}_i[k]
    }{
    D[k]
    }$

 element-wise data-parallel kernel ⇒ 以 RVV 的 unit-stride vector operations 進行加速。

## 4. 實驗環境、工具與驗證方法

本專題使用 Docker 建立可重現的 Linux development environment，使用 `riscv64-linux-gnu-g++` 進行 RISC-V cross-compilation，並以 gem5 模擬 scalar RISC-V 與 RVV execution。課程資料說明 gem5 可用於 architecture research，能量測 timing、instruction details 與 memory-system behavior，並支援 scalar RISC-V 與 RISC-V Vector Extension workloads。

Part 1 的 scalar program 編譯方式為：

```bash
riscv64-linux-gnu-g++ -static -O2 -std=c++11 main.cpp -o main
```

gem5 設定使用 `TimingSimpleCPU`、`256 MB` main memory、啟用 caches、`32 kB` 8-way L1 instruction cache、`32 kB` 8-way L1 data cache，以及 `32-byte` cache line：

```bash
gem5.opt se.py -c ./main \
  --cpu-type=TimingSimpleCPU \
  --mem-size=256MB \
  --caches \
  --l1i_size=32kB --l1i_assoc=8 \
  --l1d_size=32kB --l1d_assoc=8 \
  --cacheline=32
```

Host-side generator 在 gem5 simulation 之前執行，負責建立共同的 binary input；其執行時間不納入 gem5 measured workload。現有 `simSeconds`、`simInsts` 與 `numCycles` 因此表示：

```
End-to-end simulated program statistics after input generation
```

也就是包含 binary input loading、兩個 computation stages、correctness checks、checksum 與結果輸出，但不包含 host-side input generation。這樣的比較對 Part 1～Part 3 是公平的；不過正式報告中應稱為 **end-to-end simulated program statistics**，而不是 pure kernel-only time。

主要 performance metrics 為：

| Metric | 解讀 |
| --- | --- |
| `simSeconds` | gem5 回報的 simulated execution time |
| `simInsts` | simulated instruction count |
| `numCycles` | total simulated CPU cycles |
| `CPI` | cycles per instruction |
| `IPC` | instructions per cycle |
| L1 miss rate | gem5 觀察到的 memory-access behavior |

所有 Part 共用下列 correctness metrics：

| Metric | 用途 |
| --- | --- |
| `H_MSE` | 比較 `Hhat` 與已知 reference channel `Htrue` 的 Mean Squared Error |
| `MSE_RX_BEFORE_EQ` | received symbols 在 equalization 前相對 transmitted symbols 的誤差 |
| `MSE_LMMSE` | equalization 後 `Xmmse` 相對 transmitted QPSK data 的誤差 |
| `checksum` | 確保 `Hhat` 與 `Xmmse` 的運算結果被後續使用，避免 compiler 移除 |

Verification condition 為：

```
MSE_LMMSE < MSE_RX_BEFORE_EQ
H_MSE < 0.01
checksum ≠ 0
```

## 5. Part 1 — Scalar Baseline

Part 1 是本專題的 formal scalar baseline。它執行與後續 RVV implementations 完全相同的 OFDM LS Channel Estimation 與 one-tap LMMSE equalization algorithm，但所有計算都由 scalar C++ loops 完成。

Stage 1 的 outer loop 掃過 subcarriers `k`，inner loop 掃過 pilots `p`。對固定的 `k`，程式累加 weighted real 與 imaginary pilot observations，得到一個 complex channel estimate `Hhat[k]`：

```cpp
for (int k = 0; k < NUM_SUBCARRIERS; ++k) {
    float acc_r = 0.0f;
    float acc_i = 0.0f;

    for (int p = 0; p < NUM_PILOTS; ++p) {
        int idx = pilot_index(k, p);
        float w = pilot_w[p];
        acc_r += Ypilot_r[idx] * w;
        acc_i += Ypilot_i[idx] * w;
    }

    Hhat_r[k] = acc_r;
    Hhat_i[k] = acc_i;
}
```

這個 nested loop 是後續兩種 RVV strategy 的共同 scalar reference：

```
Part 2：
inner pilot dimension p → RVV vector lanes
vfredusum.vs → 將多個 lane products reduction 成 Hhat[k]

Part 3：
different output subcarriers k → RVV vector lanes
固定 p，平行更新多個 Hhat[k]
不使用 reduction，改用 vlse32.v 處理 strided load
```

Stage 2 對每個 data symbol `s` 與 subcarrier `k` 計算一筆獨立的 complex equalized output，因此具有明確的 data-level parallelism：

```cpp
float denom = hr * hr + hi * hi
            + NOISE_VAR_OVER_SYMBOL_POWER + EPSILON;

Xmmse_r[idx] = (yr * hr + yi * hi) / denom;
Xmmse_i[idx] = (yi * hr - yr * hi) / denom;
```

Part 1 host correctness 結果如下：

| Metric | Result |
| --- | --- |
| `H_MSE` | 0.00001250 |
| `MSE_RX_BEFORE_EQ` | 0.14093372 |
| `MSE_LMMSE` | 0.00681053 |
| `Xmmse[0]` | `1.01242745 + j1.08173215` |
| `checksum` | 584.37127686 |
| Verification | PASS |

`MSE_LMMSE` 顯著低於 equalization 前的 `MSE_RX_BEFORE_EQ`，而 `H_MSE` 也遠低於 threshold，因此 Part 1 提供了有效的 functional reference，可用於檢查後續 RVV implementations 是否仍維持相同演算法行為。

Part 1 gem5 baseline 結果如下：

| Metric | Scalar Part 1 |
| --- | --- |
| `simSeconds` | 0.069028 |
| `simInsts` | 17,066,251 |
| `numCycles` | 138,055,892 |
| `CPI` | 8.089394 |
| `IPC` | 0.123619 |
| `D-cache miss rate` | 0.114745 |
| `I-cache miss rate` | 0.000046 |

Part 1 不應被描述成「尚未完成的版本」或「只用來陪襯的未最佳化 code」。它的正式角色是後續所有 RVV experiments 的 scalar reference point：它固定了 algorithmic behavior、workload size、memory layout、input data 與 correctness methodology，使 Part 2、Part 3 的差異可以被合理地歸因於 execution mapping 與 memory-access pattern。

## 6. Part 2 — RVV Reduction + RVV LMMSE

Part 2 保持相同的 OFDM workload，改變的是兩段 computation stage 的 execution mapping：

- Stage 1：RVV reduction LS channel estimation
- Stage 2：RVV element-wise LMMSE equalization

Stage 1 仍然計算：

```text
Hhat[k] = sum_p Y_pilot[p,k] * w[p]
```

但 Part 2 將同一個 subcarrier `k` 的不同 pilot observations `p` 映射到 RVV lanes，並在 lane 內完成乘法後，再使用真正的 vector reduction instruction 將結果加總成一個 `Hhat[k]`。

對應的 CA mapping 為：

```text
vector lanes -> different pilot observations p for the same subcarrier k
reduction    -> sum over p
```

目前程式與反組譯可對應到的關鍵 RVV 指令包括：

- `vsetvli`
- `vle32.v`
- `vfmul.vv`
- `vfredusum.vs`
- `vfmv.f.s`

Stage 2 則沿著 subcarrier `k` 做 unit-stride RVV element-wise vectorization，維持與 Part 1 相同的數學模型：

```text
Xmmse[s,k] = Ydata[s,k] * conj(Hhat[k])
             / (|Hhat[k]|^2 + NOISE_VAR_OVER_SYMBOL_POWER + EPSILON)
```

Part 2 correctness 結果如下：

| Metric | Result |
| --- | --- |
| `H_MSE` | 0.00001250 |
| `MSE_RX_BEFORE_EQ` | 0.14093372 |
| `MSE_LMMSE` | 0.00681052 |
| `Xmmse[0]` | `1.01242721 + j1.08173215` |
| `checksum` | 584.37054443 |
| Verification | PASS |

Part 2 gem5 結果如下：

| Metric | Part 2 RVV |
| --- | --- |
| `simSeconds` | 0.054755 |
| `simInsts` | 11,395,368 |
| `numCycles` | 109,510,852 |
| `CPI` | 9.610092 |
| `IPC` | 0.104057 |
| `D-cache miss rate` | 0.170969 |
| `I-cache miss rate` | 0.000071 |

相較 Part 1，Part 2 的 `simInsts`、`numCycles` 與 `simSeconds` 都下降。雖然 CPI 與 miss rate 上升，但 instruction-count reduction 大於這個代價，因此整體 gem5 end-to-end simulated program statistics 仍然改善。

## 7. Part 3 — SIMD-like RVV + RVV LMMSE

Part 3 的數學工作仍然與 Part 1 相同，但 Stage 1 改成 across-`k` SIMD-like mapping。這也是它和 Part 2 最主要的差異。

Stage 1 的設計重點是：

- 不使用 vector reduction
- 每個 vector lane 對應不同的輸出 subcarrier `k`
- 對固定的 pilot index `p`，同時更新多個 `Hhat[k]`
- 因為 `pilot_index(k, p) = k * NUM_PILOTS + p`，across-`k` 載入必須用 strided access

對應的 CA mapping 為：

```text
vector lanes   -> different output subcarriers k
no reduction
strided access -> vlse32.v
```

因此 Part 3 的重點證據不是 reduction，而是：

- binary 可見 `vlse32.v`
- binary 不含 `vfredusum.vs` 或其他 vector reduction instruction

Stage 2 仍然沿著 subcarrier `k` 做 RVV element-wise LMMSE equalization，因此 Stage 2 與 Part 2 相同。

Part 3 correctness 結果如下：

| Metric | Result |
| --- | --- |
| `H_MSE` | 0.00001250 |
| `MSE_RX_BEFORE_EQ` | 0.14093372 |
| `MSE_LMMSE` | 0.00681053 |
| `Xmmse[0]` | `1.01242745 + j1.08173215` |
| `checksum` | 584.37115479 |
| Verification | PASS |

Part 3 gem5 結果如下：

| Metric | Part 3 SIMD-like RVV |
| --- | --- |
| `simSeconds` | 0.072706 |
| `simInsts` | 11,208,851 |
| `numCycles` | 145,412,866 |
| `CPI` | 12.973001 |
| `IPC` | 0.077083 |
| `D-cache miss rate` | 0.229838 |
| `I-cache miss rate` | 0.000051 |

Part 3 的 `simInsts` 低於 Part 1，但 `numCycles` 與 `simSeconds` 反而更高。這表示 instruction reduction 並沒有直接轉換成 cycle reduction。主要原因是 Stage 1 的 `vlse32.v` strided access 破壞 spatial locality，使 D-cache miss rate 升高到 `0.229838`，進而拉高 CPI。

這個結果不代表 Part 3 寫錯。依照題目在 `CA_Final_Project.pdf` 的要求，Part 3 本來就必須採用 across-`k` SIMD-like mapping、不可使用 reduction，且需要 strided memory access。對目前的資料布局

```text
pilot_index(k, p) = k * NUM_PILOTS + p
```

來說，固定 `p`、同時處理多個 `k` 時，Stage 1 的 byte stride 會是：

```text
NUM_PILOTS * sizeof(float) = 256 * 4 = 1024 bytes
```

這使得 Part 3 天生比 Part 2 更不利於 cache locality。Part 2 的 reduction path 可以沿著 contiguous pilot dimension `p` 使用 unit-stride load 與 `vfredusum.vs`，而 Part 3 為了符合 no-reduction 題意，必須承擔 strided load 的代價。

這個判斷可以直接對應兩份 reference：

- `reference/CA_Final_Project.pdf`
  - Part 3 明確要求 `Do NOT use Vector Reduction Operations`
  - 明確指出 `Strided memory access will be required`
  - 並以 `vlse32.v`、`vsse32.v` 作為例子
  - 同時要求比較 Part 1 / Part 2 / Part 3 的 simulation details，並思考為什麼 cycle reduction 與 instruction reduction 不完全相同
- `reference/riscv-v-spec-1.0.pdf`
  - `vle32.v` 是 unit-stride vector load
  - `vlse32.v` 是 strided vector load，stride 以 byte 為單位

因此可以形成完整的證據鏈：題目要求 Part 3 使用 SIMD-like across-`k` mapping 與 strided access；RVV spec 說明 `vlse32.v` 的語意本來就和 contiguous `vle32.v` 不同；而目前 gem5 stats 又顯示 Part 3 的 `D-cache miss rate = 0.229838`、`CPI = 12.973001`，所以即使 `simInsts` 下降，`numCycles` 與 `simSeconds` 仍可能上升。

另外，當前 Part 3 的 Stage 1 會在每個 pilot iteration 中反覆讀寫 partial channel estimates `Hhat[k]`。這種寫法忠實保留了 Part 3 的 SIMD-like accumulation 形式，但也增加了 memory traffic。因此目前觀察到的結果是：instruction count 雖然下降，cache miss rate 與 CPI 卻明顯上升，最後總 cycles 沒有比 Part 1 更好。這正是 Part 3 題目希望比較的 architectural tradeoff，而不是實作錯誤。

## 8. Part 1～Part 3 gem5 正式比較

| Metric | Part 1 Scalar | Part 2 RVV | Part 3 SIMD-like RVV |
| --- | --- | --- | --- |
| `simSeconds` | 0.069028 | 0.054755 | 0.072706 |
| `simInsts` | 17,066,251 | 11,395,368 | 11,208,851 |
| `numCycles` | 138,055,892 | 109,510,852 | 145,412,866 |
| `CPI` | 8.089394 | 9.610092 | 12.973001 |
| `IPC` | 0.123619 | 0.104057 | 0.077083 |
| `D-cache miss rate` | 0.114745 | 0.170969 | 0.229838 |
| `I-cache miss rate` | 0.000046 | 0.000071 | 0.000051 |

從目前 gem5 結果來看：

- Part 2 是 Part 1～Part 3 中最快的版本
- Part 2 相較 Part 1：
  - `simSeconds` 約下降 `20.68%`
  - `numCycles` 約下降 `20.68%`
  - `simInsts` 約下降 `33.23%`
- Part 3 雖然也降低了 instruction count，但 strided memory access 讓 cache behavior 惡化，最後 `simSeconds` 反而比 Part 1 高約 `5.33%`

這組比較剛好對應作業想看的重點：不同 parallel mapping 不只影響 instruction count，也會改變 memory behavior 與 CPI，因此不能只用「指令變少」來推論「一定更快」。

## 9. Part 4 — CUDA SIMT Single-Pattern Pipeline

Part 4 將相同的 OFDM pipeline 移到 NVIDIA GPU 上，處理單一 OFDM input pattern。這一部分量測的不是 gem5 simulated statistics，而是 GPU kernel-only timing。

Part 4 的兩個 CUDA kernels 為：

- `ls_channel_estimation_shared_kernel()`
- `lmmse_equalization_kernel()`

Stage 1 的 mapping 為：

- one block -> one subcarrier `k`
- one thread -> one pilot partial contribution
- block 內使用 `__shared__` 做 tree reduction

Stage 2 的 mapping 為：

- one thread -> one output `Xmmse[s,k]`
- `idx = blockIdx.x * blockDim.x + threadIdx.x`

也就是說，Part 4 同時滿足：

- CUDA SIMT execution model
- thread/block mapping evidence
- `__shared__` usage
- PTX / PTXAS / Nsight Compute analysis

Part 4 在目前預設 `TPB_LS=256`、`TPB_EQ=256`、shared Stage 1 模式下，correctness 維持 PASS，且與 Parts 1～3 使用同一組 verification policy。

Part 4 的 shared-vs-serial Stage 1 比較如下：

| Case | LS Mode | LS ms | Pipeline ms | Verification |
| --- | --- | --- | --- | --- |
| `ls_shared_256` | `shared` | 0.017372 | 0.043345 | PASS |
| `ls_serial_256` | `serial` | 0.135022 | 0.118535 | PASS |

這裡的 timing gap 應解讀為：

- block-cooperative parallel reduction + shared memory
- 相對 one-thread-per-subcarrier serial baseline 的整體效果

而不是把所有差距都歸因為 shared memory 本身。

Part 4 的 TPB sweep 摘要如下：

### LS shared kernel sweep

| Case | TPB_LS | TPB_EQ | LS Mode | LS ms | Pipeline ms |
| --- | --- | --- | --- | --- | --- |
| `ls_shared_64` | 64 | 256 | shared | 0.017032 | 0.040740 |
| `ls_shared_128` | 128 | 256 | shared | 0.015666 | 0.039000 |
| `ls_shared_256` | 256 | 256 | shared | 0.017372 | 0.043345 |

### LMMSE kernel sweep

| Case | TPB_LS | TPB_EQ | LS Mode | LMMSE ms | Pipeline ms |
| --- | --- | --- | --- | --- | --- |
| `eq_shared_128` | 256 | 128 | shared | 0.021142 | 0.062669 |
| `eq_shared_256` | 256 | 256 | shared | 0.024146 | 0.044678 |
| `eq_shared_512` | 256 | 512 | shared | 0.021271 | 0.034696 |

這些結果說明：

- `TPB_LS` 與 `TPB_EQ` 都不是越大越快
- 單看 LS kernel 與看整體 pipeline，不一定會得到相同的最佳設定
- 在目前量測中，`TPB_LS=128` 的 pipeline kernel-only time 最低，`TPB_EQ=512` 的 pipeline kernel-only time 也最低

Nsight Compute 與 PTXAS 證據則顯示：

- Stage 1 有 shared-memory traffic 與 synchronization
- Stage 2 更接近 memory-heavy element-wise kernel
- Stage 2 的 `Achieved Occupancy > 100%` 應視為 profiling anomaly，不拿來當主要結論

## 10. Part 5 — Multi-Pattern GPU Parallelism

Part 5 將 Part 4 的 single-pattern CUDA pipeline 擴展到多個彼此獨立的 OFDM input patterns。數學模型不變，新增的是 pattern dimension 與 2D grid mapping。

Part 5 的主要映射為：

- Stage 1：
  - `blockIdx.x -> subcarrier k`
  - `blockIdx.y -> pattern index`
  - `threadIdx.x -> pilot contributions`
- Stage 2：
  - `blockIdx.x * blockDim.x + threadIdx.x -> flattened output index inside one pattern`
  - `blockIdx.y -> pattern index`

因此 Part 5 的關鍵不是改變單一 frame 內的運算，而是讓 GPU 同時看到更多獨立 patterns，以增加可同時排程的 blocks 與 warps。

目前預設 `NUM_PATTERNS=16`、`TPB_LS=256`、`TPB_EQ=256` 的 GPU 執行結果如下：

| Metric | Value |
| --- | --- |
| `H_MSE` | 0.00001289 |
| `MSE_RX_BEFORE_EQ` | 0.14082038 |
| `MSE_LMMSE` | 0.00682142 |
| `LS_KERNEL_MS` | 0.128993 |
| `LMMSE_KERNEL_MS` | 0.308572 |
| `PIPELINE_KERNEL_MS` | 0.434780 |
| Verification | PASS |

同一份 multi-pattern input 的 CPU baseline 結果如下：

| Metric | Value |
| --- | --- |
| `CPU_PIPELINE_MS` | 12.268151 |
| `H_MSE` | 0.00001289 |
| `MSE_LMMSE` | 0.00682142 |
| `Verification` | PASS |

這說明 Part 5 的 GPU 路徑在維持相同演算法與 correctness checks 的前提下，已經能穩定處理 multi-pattern workload。

Part 5 pattern sweep 摘要如下：

| Patterns | Verification | LS ms | LMMSE ms | Pipeline ms |
| --- | --- | --- | --- | --- |
| 1 | PASS | 0.032481 | 0.037750 | 0.069048 |
| 4 | PASS | 0.039578 | 0.117176 | 0.141583 |
| 8 | PASS | 0.059996 | 0.177853 | 0.214241 |
| 16 | PASS | 0.086354 | 0.347571 | 0.404536 |
| 32 | PASS | 0.223770 | 0.745861 | 0.926991 |

這組結果不應只看總時間，而要看 pattern-level amortization。隨著 patterns 增加，GPU 看到更多獨立 blocks 與 warps，pipeline time per pattern 下降，這正是 Part 5 想展示的架構重點。

## 11. 結論

目前 repository 的主體工作已經形成一條清楚的比較線：

- Part 1：scalar nested-loop baseline
- Part 2：RVV reduction + RVV element-wise equalization
- Part 3：SIMD-like RVV + strided access + no reduction
- Part 4：single-pattern CUDA SIMT pipeline
- Part 5：multi-pattern GPU parallelism

從 Part 1～Part 3 的 gem5 結果來看，Part 2 在目前設定下提供了最佳的 end-to-end simulated program performance；Part 3 雖然保留了 SIMD-like mapping 與 no-reduction requirement，但 strided access 帶來的 memory penalty 抵銷了 instruction-count reduction 的收益。

從 Part 4 與 Part 5 的 GPU 結果來看，shared-memory block-cooperative reduction 與大量 thread-level parallelism 能有效支撐 OFDM pipeline，而 multi-pattern mapping 進一步提高了 GPU 可用的獨立工作量，使 Part 5 更能展現 GPU 在 throughput-oriented workload 上的優勢。

## 資料來源與報告引用依據

1. **課程資料：`CA_Final_Project.pdf`**
    - Project overview：Part 1 Scalar Baseline、Part 2 RVV Vector Reduction、Part 3 SIMD-like RVV Parallelization、Part 4 CUDA SIMT、Part 5 Multi-pattern GPU Parallelism。
    - Part 1 requirement：演算法需具 computational intensity，且至少一段包含 nested-loop summation。
    - Part 2 requirement：必須使用 RVV vectors、vector arithmetic 與 Vector Reduction Operations。
    - Part 3 requirement：使用 across-k SIMD-like mapping、不得使用 reduction，並需使用 strided memory access。
    - gem5 workflow、metrics 與報告提醒：gem5 simulated time 與實際 CPU/GPU runtime 不可混為一談。
2. **RISC-V International：`riscv-v-spec-1.0.pdf`**
    - RVV programmer model：vector registers `v0`～`v31`、`vl`、`vtype`、`vsetvli`。
    - Vector memory instructions：unit-stride load/store 與 strided load/store。
    - Vector floating-point instructions：`vfmul.vv`、`vfmacc.vv`、`vfsub.vv`、`vfdiv.vv`。
    - Vector Reduction Operations：`vfredusum.vs`。
    - Strip-mining：以 `vsetvli` 和 `remaining` 支援 implementation-dependent vector length。
3. **目前實作 repository：`York0104/ca_final_project`**
    - `Overall_Results_Analysis.md`
    - `part1/Part1_MMSE.md`
    - `part1/main.cpp`
    - `part1/Makefile`
    - `part2/main.cpp`
    - `part3/main.cpp`
    - `common/ofdm_params.h`
    - `common/ofdm_model.h`
    - `data_gen/generate_ofdm_data.cpp`

> **更新注意事項：** `Overall_Results_Analysis.md` 與 `part3/Part3_SIMD_Like_RVV.md` 已記錄最新 Part 3 結果；若有舊文件仍寫「Part 3 與 Part 1 幾乎相同」，正式報告與簡報應以最新 overall results 為準。

https://ww2.mathworks.cn/help/lte/ug/channel-estimation.html
