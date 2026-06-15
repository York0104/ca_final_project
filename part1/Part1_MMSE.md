# CA Final Project Part 1：Scalar Baseline + gem5 執行紀錄

## 1. Part 1 總覽

本次 Part 1 為：Pilot-based OFDM MMSE Channel Equalizer

此版本不是完整 OFDM PHY，而是聚焦於 OFDM receiver 中的 **frequency-domain channel equalization workload**。

主要目的為建立 scalar baseline，後續可延伸到：

- Part 2：RVV Vector Reduction
- Part 3：SIMD-like RVV Parallelization
- Part 4：CUDA SIMT Implementation
- Part 5：Multi-pattern GPU Parallelism

目前已完成：

```
1. Docker / gem5 環境確認
2. Part 1 scalar C++ baseline 實作
3. Host g++ 驗證通過
4. RISC-V binary 編譯成功
5. gem5 TimingSimpleCPU + caches 執行成功
6. m5out/stats.txt performance data 擷取完成
```

---

## 2. 環境與工具

本次使用：

- Windows + WSL2 Ubuntu
- Docker Desktop
- Docker image：`weisheng505/gem5-rvv-image:v1`
- RISC-V compiler：`riscv64-linux-gnu-g++`
- gem5：`/root/gem5/build/RISCV/gem5.opt`
- gem5 config：`/root/gem5/configs/deprecated/example/se.py`
- CPU model：`TimingSimpleCPU`
- Cache setting：
    - L1 I-cache：32KB，8-way
    - L1 D-cache：32KB，8-way
    - cache line：32 bytes
- Memory size：256MB

---

## 3. Docker 環境確認

### 檢查 Docker 版本

```bash
docker --version
```

本次結果：

```
Docker version 29.5.3, build d1c06ef
```

### 測試 Docker

```bash
docker run hello-world
```

成功看到：

```
Hello from Docker!
This message shows that your installation appears to be working correctly.
```

代表 Docker 安裝與 WSL2 integration 正常。

---

## 4. 建立 Part 1 專案資料夾

```bash
mkdir -p ~/ca_final_project/part1
cd ~/ca_final_project/part1
```

本次 Part 1 資料夾主要包含：

```
main.cpp
Makefile
```

確認檔案：

```bash
ls -al
```

---

## 5. 啟動 gem5 Docker Container

第一次建立 container：

```bash
docker run -it --name ca-fp-gem5 \
-v ~/ca_final_project:/workspace \
-w /workspace \
weisheng505/gem5-rvv-image:v1
```

若 container 已存在，進入 container：

```bash
docker exec -it ca-fp-gem5 /bin/bash
```

進入 Part 1：

```bash
cd /workspace/part1
```

---

# 6. 演算法設計：Pilot-based OFDM MMSE Channel Equalizer

## 6.1 設計定位

本次 Part 1 使用 **Pilot-based OFDM MMSE Channel Equalizer** 作為 scalar baseline。

此 workload 模擬 OFDM receiver 中的 channel equalization computation kernel。

為了讓程式聚焦於 CA final project 所需的 scalar / vector / SIMT 比較，本版本採用 **fixed pilot**，不實作完整 OFDM PHY，例如 IFFT / FFT、CP insertion / removal、synchronization 或 channel coding。

---

## 6.2 系統參數

程式參數：

```cpp
static constexpr int NUM_SUBCARRIERS  = 512;
static constexpr int NUM_PILOTS       = 256;
static constexpr int NUM_DATA_SYMBOLS = 512;

static constexpr int TOTAL_PILOT = NUM_SUBCARRIERS * NUM_PILOTS;
static constexpr int TOTAL_DATA  = NUM_SUBCARRIERS * NUM_DATA_SYMBOLS;

static constexpr float NOISE_VAR = 0.0025f;
static constexpr float EPSILON   = 1.0e-6f;
```

對應數量：

| Parameter | Value | Meaning |
| --- | --- | --- |
| `NUM_SUBCARRIERS` | 512 | OFDM subcarriers 數量 |
| `NUM_PILOTS` | 256 | 每個 subcarrier 的 pilot observations 數量 |
| `NUM_DATA_SYMBOLS` | 512 | data OFDM symbols 數量 |
| `TOTAL_PILOT` | 131,072 | pilot samples 總數 |
| `TOTAL_DATA` | 262,144 | data samples 總數 |
| `NOISE_VAR` | 0.0025 | MMSE regularization / noise variance |

---

## 6.3 通訊數學模型

### Fixed pilot 設定

本版本使用固定 pilot：

```
X_pilot[p][k] = 1 + j0
```

因此 received pilot observation 為：

```
Y_pilot[k][p] = H[k] + N_pilot[k][p]
```

其中：

| Symbol | Meaning |
| --- | --- |
| `k` | subcarrier index |
| `p` | pilot observation index |
| `H[k]` | true channel frequency response |
| `N_pilot[k][p]` | pilot noise |

---

### Data symbol channel model

每個 data symbol 經過 channel 後：

```
Y_data[s][k] = H[k] X_data[s][k] + N_data[s][k]
```

其中：

| Symbol | Meaning |
| --- | --- |
| `s` | OFDM data symbol index |
| `X_data[s][k]` | transmitted QPSK-like symbol |
| `Y_data[s][k]` | received data symbol |
| `N_data[s][k]` | data noise |

---

## 6.4 Channel Estimation

因為 pilot 固定為 `1 + j0`，所以 channel estimation 可寫成對 received pilot observations 的 weighted reduction，其中每個權重皆為 `1 / NUM_PILOTS`：

```
H_hat[k] = sum_{p=0}^{NUM_PILOTS-1} Y_pilot[k][p] * w[p]
```

以 real / imaginary 分開實作：

```
H_hat_r[k] = sum_p Ypilot_r[k][p] * w[p]
H_hat_i[k] = sum_p Ypilot_i[k][p] * w[p]
```

這是本程式最重要的 nested-loop reduction kernel。

---

## 6.5 對應題目要求的 summation form

題目要求的形式可寫成：

```
f(a_i[1], b_i[1]) + f(a_i[2], b_i[2]) + ... + f(a_i[n], b_i[n])
```

本程式可對應為：

```
i-th iteration = 第 k 個 subcarrier 的 channel estimation
n = NUM_PILOTS
a_k[p] = Y_pilot[k][p]
b_k[p] = w[p]
f(a_k[p], b_k[p]) = a_k[p] * b_k[p]
```

因此：

```
H_hat[k] = sum_p f(Y_pilot[k][p], w[p])
```

由於 complex value 以 real / imaginary 分別儲存，所以程式中對 `Ypilot_r` 與 `Ypilot_i` 各自做 reduction。

---

## 6.6 MMSE Equalization

MMSE equalizer 使用：

```
X_hat[s][k] = Y_data[s][k] * conj(H_hat[k]) / (|H_hat[k]|^2 + NOISE_VAR)
```

令：

```
Y_data[s][k] = yr + j yi
H_hat[k]     = hr + j hi
```

則：

```
denom = hr^2 + hi^2 + NOISE_VAR + EPSILON
```

實作成：

```
X_hat_r = (yr * hr + yi * hi) / denom
X_hat_i = (yi * hr - yr * hi) / denom
```

此部分是主要 equalization workload，具備大量 independent subcarrier-level / symbol-level parallelism。

---

## 6.7 MSE Verification

### Equalization 前 MSE

```
MSE_RX = (1 / TOTAL_DATA) * sum_idx |Y_data[idx] - X_data[idx]|^2
```

### MMSE Equalization 後 MSE

```
MSE_MMSE = (1 / TOTAL_DATA) * sum_idx |X_hat[idx] - X_data[idx]|^2
```

### Channel Estimation MSE

```
H_MSE = (1 / NUM_SUBCARRIERS) * sum_k |H_hat[k] - H_true[k]|^2
```

驗證條件：

```
MSE_MMSE < MSE_RX
H_MSE < 0.01
checksum != 0
```

本次結果符合條件，因此輸出：

```
Verification: PASS
```

---

# 7. 程式架構

主要 function：

| Function | Purpose |
| --- | --- |
| `init_channel()` | 建立 deterministic frequency-selective channel `Htrue[k]` |
| `init_pilots_and_data()` | 產生 fixed pilot、QPSK-like data，並通過 channel 產生 received samples |
| `estimate_channel_scalar()` | 對 pilot observations 做 nested-loop reduction，估計 `Hhat[k]` |
| `equalize_mmse_scalar()` | 對 received data 做 MMSE equalization |
| `compute_channel_mse()` | 計算 channel estimation error |
| `compute_received_mse()` | 計算 equalization 前 MSE |
| `compute_mmse_mse()` | 計算 equalization 後 MSE |
| `checksum_results()` | 消耗輸出結果，避免 compiler 移除 computation |

---

## 7.1 核心 Channel Estimation Code

```cpp
static void estimate_channel_scalar()
{
    for (int k = 0; k < NUM_SUBCARRIERS; ++k)
    {
        float acc_r = 0.0f;
        float acc_i = 0.0f;

        for (int p = 0; p < NUM_PILOTS; ++p)
        {
            int idx = pilot_index(k, p);
            float w = pilot_w[p];
            acc_r += Ypilot_r[idx] * w;
            acc_i += Ypilot_i[idx] * w;
        }

        Hhat_r[k] = acc_r;
        Hhat_i[k] = acc_i;
    }
}
```

---

## 7.2 核心 MMSE Equalization Code

```cpp
static void equalize_mmse_scalar()
{
    for (int s = 0; s < NUM_DATA_SYMBOLS; ++s)
    {
        for (int k = 0; k < NUM_SUBCARRIERS; ++k)
        {
            int idx = data_index(s, k);

            float yr = Ydata_r[idx];
            float yi = Ydata_i[idx];

            float hr = Hhat_r[k];
            float hi = Hhat_i[k];

            float denom = hr * hr + hi * hi + NOISE_VAR + EPSILON;

            Xhat_r[idx] = (yr * hr + yi * hi) / denom;
            Xhat_i[idx] = (yi * hr - yr * hi) / denom;
        }
    }
}
```

---

# 8. Makefile

本次使用助教風格 Makefile，會直接編譯 RISC-V binary 並以 gem5 執行。

```makefile
GEM5_DIR := /root/gem5
GEM5 := $(GEM5_DIR)/build/RISCV/gem5.opt
CONFIG := $(GEM5_DIR)/configs/deprecated/example/se.py

CC := riscv64-linux-gnu-g++
CFLAGS := -static -O2 -std=c++11

SRC := main.cpp
BIN := main

all: run

$(BIN): $(SRC)
	$(CC) $(CFLAGS) $< -o $@

run: $(BIN)
	$(GEM5) $(CONFIG) -c $(CURDIR)/$(BIN) --cpu-type=TimingSimpleCPU --mem-size=256MB --caches --l1i_size=32kB --l1i_assoc=8 --l1d_size=32kB --l1d_assoc=8 --cacheline=32
	grep -E "simSeconds|simTicks|numCycles|simInsts|simOps|hostSeconds|overallMisses|overallMissRate|system.cpu.cpi|system.cpu.ipc" m5out/stats.txt

host:
	g++ -O2 -std=c++11 $(SRC) -o main_host
	./main_host

clean:
	rm -f $(BIN) main_host
	rm -rf m5out

.PHONY: all run host clean
```

注意：Makefile command line 前面必須是 Tab。

---

# 9. Host 端驗證

先用 x86 host `g++` 驗證程式正確性：

```bash
cd /workspace/part1
make clean
make host
```

本次輸出：

```
Part 1 - Scalar Baseline: Pilot-based OFDM MMSE Channel Equalizer
NUM_SUBCARRIERS  = 512
NUM_PILOTS       = 256
NUM_DATA_SYMBOLS = 512
TOTAL_DATA       = 262144
NOISE_VAR        = 0.00250000
H_MSE            = 0.00001260
MSE_RX           = 0.33055928
MSE_MMSE         = 0.01459649
Xhat[0]          = 0.69086909 + j1.21442187
checksum         = 342.94046021
guard            = 345.01266479
Verification: PASS
```

結論：

```
Host g++ verification passed.
MSE was reduced from 0.33055928 to 0.01459649 after MMSE equalization.
```

---

# 10. RISC-V + gem5 執行

執行：

```bash
make clean
make
```

實際 RISC-V 編譯指令：

```bash
riscv64-linux-gnu-g++ -static -O2 -std=c++11 main.cpp -o main
```

實際 gem5 執行指令：

```bash
/root/gem5/build/RISCV/gem5.opt \
/root/gem5/configs/deprecated/example/se.py \
-c /workspace/part1/main \
--cpu-type=TimingSimpleCPU \
--mem-size=256MB \
--caches \
--l1i_size=32kB \
--l1i_assoc=8 \
--l1d_size=32kB \
--l1d_assoc=8 \
--cacheline=32
```

gem5 正確完成：

```
Verification: PASS
Exiting @ tick 138000158000 because exiting with last active thread context
```

---

# 11. m5out 產生檔案

gem5 執行後產生：

```
m5out/
├── citations.bib
├── config.dot
├── config.dot.pdf
├── config.dot.svg
├── config.ini
├── config.json
├── fs/
└── stats.txt
```

重要檔案：

```
m5out/stats.txt
```

---

# 12. 擷取 Performance Stats

指令：

```bash
grep -E "simSeconds|simTicks|numCycles|simInsts|simOps|hostSeconds|overallMisses|overallMissRate|system.cpu.cpi|system.cpu.ipc" m5out/stats.txt
```

本次結果：

| Metric | Value | Meaning |
| --- | --- | --- |
| `simSeconds` | `0.138000` | gem5 simulated seconds |
| `simTicks` | `138000158000` | simulated ticks |
| `hostSeconds` | `51.58` | host elapsed time |
| `simInsts` | `34,616,971` | simulated instructions |
| `simOps` | `34,616,991` | simulated ops including micro-ops |
| `system.cpu.numCycles` | `276,000,316` | CPU cycles |
| `system.cpu.cpi` | `7.972973` | cycles per instruction |
| `system.cpu.ipc` | `0.125424` | instructions per cycle |
| `D-cache overallMisses` | `952,726` | D-cache misses |
| `D-cache overallMissRate` | `0.120870` | D-cache miss rate |
| `I-cache overallMisses` | `747` | I-cache misses |
| `I-cache overallMissRate` | `0.000014` | I-cache miss rate |

---

# 13. Baseline Summary

| Version | simSeconds | simInsts | numCycles | CPI | IPC | D-cache Miss Rate |
| --- | --- | --- | --- | --- | --- | --- |
| Part 1 Scalar MMSE Equalizer | 0.138000 | 34,616,971 | 276,000,316 | 7.972973 | 0.125424 | 0.120870 |

---

# 14. Performance Interpretation



---

# 15. 備份 stats

儲存 summary：

```bash
grep -E "simSeconds|simTicks|numCycles|simInsts|simOps|hostSeconds|overallMisses|overallMissRate|system.cpu.cpi|system.cpu.ipc" m5out/stats.txt > part1_mmse_stats_summary.txt
```

備份完整 stats：

```bash
cp m5out/stats.txt part1_mmse_full_stats.txt
```

---

# 16. 本次正確流程總結

```bash
docker --version
docker run hello-world

mkdir -p ~/ca_final_project/part1
cd ~/ca_final_project/part1

docker run -it --name ca-fp-gem5 \
-v ~/ca_final_project:/workspace \
-w /workspace \
weisheng505/gem5-rvv-image:v1

docker exec -it ca-fp-gem5 /bin/bash

cd /workspace/part1

make clean
make host

make clean
make

grep -E "simSeconds|simTicks|numCycles|simInsts|simOps|hostSeconds|overallMisses|overallMissRate|system.cpu.cpi|system.cpu.ipc" m5out/stats.txt
```

---

# 17. Part 1 結論

本次 Part 1 已完成 scalar baseline。

最終演算法為：

```
Pilot-based OFDM MMSE Channel Equalizer
```

主要計算包含：

```
1. Fixed pilot-based channel estimation
2. MMSE equalization
3. MSE verification
```

Host 與 gem5 均顯示：

```
Verification: PASS
```

正確性結果：

| Metric | Value |
| --- | --- |
| `H_MSE` | 0.00001260 |
| `MSE_RX` | 0.33055928 |
| `MSE_MMSE` | 0.01459649 |

gem5 baseline performance：

| Metric | Value |
| --- | --- |
| `simInsts` | 34,616,971 |
| `numCycles` | 276,000,316 |
| `CPI` | 7.972973 |
| `IPC` | 0.125424 |
| `D-cache miss rate` | 0.120870 |

Part 1 可作為後續 RVV Vector Reduction、SIMD-like RVV Parallelization 與 CUDA SIMT implementation 的 baseline。
