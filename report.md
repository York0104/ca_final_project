> **文件用途：** 本文件整理專題前半部共用內容，包含研究動機、OFDM workload、數學模型、實驗環境、驗證方法與 Part 1 scalar baseline，可作為簡報或正式報告的母稿。
> 
> 
> **目前完成範圍：** Repository 已完成 Part 1～Part 4。本文主體仍以共同 workload 與 Part 1 baseline 為主；Part 2、Part 3、Part 4 的細節與數據請搭配各自文件與總分析閱讀。
> 

## 1. 研究動機與專題目標

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

## 2. OFDM Receiver Workload 與資料配置

目前實作的是簡化但完整的 frequency-domain OFDM receiver ：

```
Pilot observations ──> LS Channel Estimation ──> Hhat[k]
                                                   │
Received OFDM data ───────────────────────────────┼──> One-tap LMMSE Equalization ──> Xmmse[s,k]
```

系統使用 512 個 subcarriers、每個 subcarrier 256 筆 repeated pilot observations，以及 512 個 OFDM data symbols。

- 因此，LS stage 需要處理 512 × 256 = 131,072 筆 pilot observations
- qualization stage 則需要輸出 512 × 512 = 262,144 筆 complex equalized symbols

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

另外，CUDA 開發環境已完成驗證。主機端可使用 `NVIDIA GeForce RTX 3060`，Compute Capability 為 `8.6`；Docker GPU runtime 可正常使用，container 內已確認 `nvcc 12.4`、`-arch=sm_86` 編譯、CUDA kernel launch、CUDA event timing、PTX generation、PTXAS resource report、`Nsight Compute (ncu)` 與 GPU performance counters 皆可正常運作。主機驅動顯示的 `CUDA Version` 為 `13.0`，而作業 container 內工具鏈為 `CUDA Toolkit 12.4.1`；目前這組合已透過 `cuda_smoke` 與 `ncu` 驗證可用。Part 4 可直接使用這套環境，Part 5 目錄也已預先建立，但程式尚未實作。

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

`MSE_LMMSE` 明顯低於 equalization 前的 `MSE_RX_BEFORE_EQ`，而 `H_MSE` 也遠低於 threshold。這組結果提供了後續 RVV 與 CUDA 版本的功能性對照基準。

Part 1 gem5 baseline 結果如下：

| Metric | Scalar Part 1 |
| --- | --- |
| `simSeconds` | 0.074767 |
| `simInsts` | 17,065,752 |
| `numCycles` | 149,534,050 |
| `CPI` | 8.762213 |
| `IPC` | 0.114126 |
| `D-cache miss rate` | 0.114747 |
| `I-cache miss rate` | 0.000034 |

Part 1 不是只用來陪襯的未最佳化版本。它固定了 algorithmic behavior、workload size、memory layout、input data 與 correctness methodology，因此 Part 2、Part 3 的差異可以回到 execution mapping 與 memory-access pattern 來解讀。

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

> **更新注意事項：** 具體數字請以 `Overall_Results_Analysis.md`、`part2/Part2_RVV_Reduction.md`、`part3/Part3_SIMD_Like_RVV.md` 與 `part4/Part4_CUDA_SIMT.md` 為準；若舊草稿仍保留較早的描述，應以目前 repository 內的最新量測結果覆蓋。
>
