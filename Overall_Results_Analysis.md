# CA Final Project Overall Results Analysis

## 1. Purpose

This file records the measured results collected so far, including:

- Part 1 full program scalar baseline
- Part 2 full program RVV vector reduction
- Part 1 kernel-only scalar channel estimation
- Part 2 kernel-only RVV reduction channel estimation

The purpose of this summary is to keep all key numbers in one place before writing the final report.

---

## 2. Full Program Results

### 2.1 Part 1 Full Program

| Metric | Value |
| --- | --- |
| `H_MSE` | `0.00001260` |
| `MSE_RX` | `0.33055928` |
| `MSE_MMSE` | `0.01459649` |
| `simSeconds` | `0.138000` |
| `simInsts` | `34,616,971` |
| `numCycles` | `276,000,316` |
| `CPI` | `7.972973` |
| `IPC` | `0.125424` |
| `D-cache miss rate` | `0.120870` |
| `I-cache miss rate` | `0.000014` |

### 2.2 Part 2 Full Program

| Metric | Value |
| --- | --- |
| `H_MSE` | `0.00001260` |
| `MSE_RX` | `0.33055928` |
| `MSE_MMSE` | `0.01459648` |
| `simSeconds` | `0.130226` |
| `simInsts` | `33,996,109` |
| `numCycles` | `260,452,526` |
| `CPI` | `7.661241` |
| `IPC` | `0.130527` |
| `D-cache miss rate` | `0.124492` |
| `I-cache miss rate` | `0.000016` |

### 2.3 Full Program Comparison

| Metric | Part 1 Full | Part 2 Full |
| --- | --- | --- |
| `H_MSE` | `0.00001260` | `0.00001260` |
| `MSE_RX` | `0.33055928` | `0.33055928` |
| `MSE_MMSE` | `0.01459649` | `0.01459648` |
| `simSeconds` | `0.138000` | `0.130226` |
| `simInsts` | `34,616,971` | `33,996,109` |
| `numCycles` | `276,000,316` | `260,452,526` |
| `CPI` | `7.972973` | `7.661241` |
| `IPC` | `0.125424` | `0.130527` |
| `D-cache miss rate` | `0.120870` | `0.124492` |
| `I-cache miss rate` | `0.000014` | `0.000016` |

Initial observation:

- Part 2 full program is faster than Part 1 full program.
- Correctness is maintained.
- The overall speedup is visible but modest, because only the channel estimation kernel was vectorized while MMSE equalization remained scalar.

---

## 3. Kernel-only Results

### 3.1 Part 1 Kernel-only

| Metric | Value |
| --- | --- |
| `H_MSE` | `0.00001260` |
| `checksum_Hhat` | `394.26150513` |
| `simSeconds` | `0.493423` |
| `simInsts` | `122,690,231` |
| `numCycles` | `986,846,688` |
| `CPI` | `8.043400` |
| `IPC` | `0.124326` |
| `D-cache miss rate` | `0.088941` |
| `I-cache miss rate` | `0.000004` |

### 3.2 Part 2 Kernel-only

| Metric | Value |
| --- | --- |
| `H_MSE` | `0.00001260` |
| `checksum_Hhat` | `394.26150513` |
| `simSeconds` | `0.527467` |
| `simInsts` | `162,062,900` |
| `numCycles` | `1,054,933,786` |
| `CPI` | `6.509409` |
| `IPC` | `0.153624` |
| `D-cache miss rate` | `0.066973` |
| `I-cache miss rate` | `0.000003` |

### 3.3 Kernel-only Comparison

| Metric | Part 1 Kernel-only | Part 2 Kernel-only |
| --- | --- | --- |
| `H_MSE` | `0.00001260` | `0.00001260` |
| `checksum_Hhat` | `394.26150513` | `394.26150513` |
| `simSeconds` | `0.493423` | `0.527467` |
| `simInsts` | `122,690,231` | `162,062,900` |
| `numCycles` | `986,846,688` | `1,054,933,786` |
| `CPI` | `8.043400` | `6.509409` |
| `IPC` | `0.124326` | `0.153624` |
| `D-cache miss rate` | `0.088941` | `0.066973` |
| `I-cache miss rate` | `0.000004` | `0.000003` |

Initial observation:

- Part 2 kernel-only preserves correctness exactly.
- Part 2 kernel-only shows lower `CPI`, higher `IPC`, and lower cache miss rates.
- However, total `simInsts`, `numCycles`, and `simSeconds` are higher than Part 1 kernel-only.
- This suggests the current RVV reduction implementation improves per-instruction efficiency, but still incurs higher total execution cost in gem5 for this kernel-only setup.

---

## 4. Cross-Section Observation

Current results show an interesting contrast:

- In the full program, Part 2 is faster than Part 1.
- In the kernel-only experiment, Part 2 is currently slower than Part 1.

This means the kernel-only data should be interpreted carefully. Possible reasons include:

- RVV reduction setup overhead inside the helper
- repeated `vsetvli` and reduction control overhead
- gem5 modeling effects for this specific inline assembly pattern
- `KERNEL_REPEAT = 100` amplifying fixed RVV overheads

This is not necessarily a contradiction, but it is an important discussion point for the final report.

---

## 5. Suggested Report Positioning

Recommended report structure:

- Use **Full Program Performance** as the main required result.
- Use **Kernel-only Performance** as a supplementary analysis.
- If full-program speedup is limited, explain that scalar equalization, verification, checksum, and other non-kernel work dilute the overall RVV benefit.
- If kernel-only RVV is slower than expected, discuss RVV setup/reduction overhead and gem5 behavior honestly instead of hiding the result.

---

## 6. Next Step

Recommended next actions:

1. Add Part 1 kernel-only and Part 2 kernel-only dedicated markdown reports.
2. Compute percentage change tables for all comparisons.
3. Investigate whether the Part 2 kernel-only RVV helper can be optimized further.
