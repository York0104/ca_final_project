#include <cstddef>

#include "../common/ofdm_params.h"
#include "../common/ofdm_data.h"
#include "../common/ofdm_io.h"
#include "../common/ofdm_verify.h"

// ============================================================
// Computation Stage 1 - SIMD-like RVV LS Channel Estimation
//
// CA mapping:
//   Vector lanes cover different output subcarriers k.
//   No vector reduction is used.
//   Strided loads are required because Ypilot[k][p] is stored with
//   stride NUM_PILOTS across k.
// ============================================================
static void estimate_channel_ls_average_rvv_simd_like(){
#if OFDM_HAS_RVV
    const ptrdiff_t pilot_stride_bytes = static_cast<ptrdiff_t>(NUM_PILOTS * sizeof(float));

    for (int p = 0; p < NUM_PILOTS; ++p)
    {
        float w = pilot_w[p];
        int k = 0;

        while (k < NUM_SUBCARRIERS)
        {
            int remaining = NUM_SUBCARRIERS - k;
            int vl = 0;

            const float *yr_ptr = &Ypilot_r[pilot_index(k, p)];
            const float *yi_ptr = &Ypilot_i[pilot_index(k, p)];
            float *hr_ptr = &Hhat_r[k];
            float *hi_ptr = &Hhat_i[k];

            asm volatile(
                "vsetvli %[vl], %[remaining], e32, m1, ta, ma\n\t"
                "vlse32.v v1, (%[yr]), %[stride]\n\t"
                "vlse32.v v2, (%[yi]), %[stride]\n\t"
                "vle32.v v3, (%[hr])\n\t"
                "vle32.v v4, (%[hi])\n\t"
                "vfmv.v.f v5, %[w]\n\t"
                "vfmul.vv v6, v1, v5\n\t"
                "vfmul.vv v7, v2, v5\n\t"
                "vfadd.vv v3, v3, v6\n\t"
                "vfadd.vv v4, v4, v7\n\t"
                "vse32.v v3, (%[hr])\n\t"
                "vse32.v v4, (%[hi])\n\t"
                : [vl] "=&r"(vl)
                : [remaining] "r"(remaining),
                  [yr] "r"(yr_ptr),
                  [yi] "r"(yi_ptr),
                  [hr] "r"(hr_ptr),
                  [hi] "r"(hi_ptr),
                  [stride] "r"(pilot_stride_bytes),
                  [w] "f"(w)
                : "memory"
            );

            k += vl;
        }
    }
#else
    for (int k = 0; k < NUM_SUBCARRIERS; ++k)
    {
        float acc_r = 0.0f;
        float acc_i = 0.0f;

        for (int p = 0; p < NUM_PILOTS; ++p)
        {
            int idx = pilot_index(k, p);
            acc_r += Ypilot_r[idx] * pilot_w[p];
            acc_i += Ypilot_i[idx] * pilot_w[p];
        }

        Hhat_r[k] = acc_r;
        Hhat_i[k] = acc_i;
    }
#endif
}

// ============================================================
// Computation Stage 2 - RVV Element-wise LMMSE Equalization
//
// CA mapping:
//   Same element-wise RVV equalizer as Part 2.
// ============================================================
static void equalize_lmmse_rvv(){
#if OFDM_HAS_RVV
    const float noise_bias = NOISE_VAR_OVER_SYMBOL_POWER + EPSILON;

    for (int s = 0; s < NUM_DATA_SYMBOLS; ++s)
    {
        int k = 0;

        while (k < NUM_SUBCARRIERS)
        {
            int remaining = NUM_SUBCARRIERS - k;
            int vl = 0;

            const float *yr_ptr = &Ydata_r[data_index(s, k)];
            const float *yi_ptr = &Ydata_i[data_index(s, k)];
            const float *hr_ptr = &Hhat_r[k];
            const float *hi_ptr = &Hhat_i[k];
            float *xr_ptr = &Xmmse_r[data_index(s, k)];
            float *xi_ptr = &Xmmse_i[data_index(s, k)];

            asm volatile(
                "vsetvli %[vl], %[remaining], e32, m1, ta, ma\n\t"
                "vle32.v v1, (%[yr])\n\t"
                "vle32.v v2, (%[yi])\n\t"
                "vle32.v v3, (%[hr])\n\t"
                "vle32.v v4, (%[hi])\n\t"
                "vfmul.vv v5, v1, v3\n\t"
                "vfmacc.vv v5, v2, v4\n\t"
                "vfmul.vv v6, v2, v3\n\t"
                "vfmul.vv v7, v1, v4\n\t"
                "vfsub.vv v6, v6, v7\n\t"
                "vfmul.vv v8, v3, v3\n\t"
                "vfmacc.vv v8, v4, v4\n\t"
                "vfmv.v.f v9, %[bias]\n\t"
                "vfadd.vv v8, v8, v9\n\t"
                "vfdiv.vv v5, v5, v8\n\t"
                "vfdiv.vv v6, v6, v8\n\t"
                "vse32.v v5, (%[xr])\n\t"
                "vse32.v v6, (%[xi])\n\t"
                : [vl] "=&r"(vl)
                : [remaining] "r"(remaining),
                  [yr] "r"(yr_ptr),
                  [yi] "r"(yi_ptr),
                  [hr] "r"(hr_ptr),
                  [hi] "r"(hi_ptr),
                  [xr] "r"(xr_ptr),
                  [xi] "r"(xi_ptr),
                  [bias] "f"(noise_bias)
                : "memory"
            );

            k += vl;
        }
    }
#else
    for (int s = 0; s < NUM_DATA_SYMBOLS; ++s)
    {
        for (int k = 0; k < NUM_SUBCARRIERS; ++k)
        {
            int idx = data_index(s, k);
            float yr = Ydata_r[idx];
            float yi = Ydata_i[idx];
            float hr = Hhat_r[k];
            float hi = Hhat_i[k];

            float denom = hr * hr + hi * hi + NOISE_VAR_OVER_SYMBOL_POWER + EPSILON;

            Xmmse_r[idx] = (yr * hr + yi * hi) / denom;
            Xmmse_i[idx] = (yi * hr - yr * hi) / denom;
        }
    }
#endif
}

int main(){
    load_input_binary("../data/ofdm_input.bin");

    // Computation Stage 1: SIMD-like RVV LS channel estimation
    estimate_channel_ls_average_rvv_simd_like();

    // Computation Stage 2: RVV element-wise LMMSE equalization
    equalize_lmmse_rvv();

    float h_mse = compute_channel_mse();
    float mse_rx_before_eq = compute_rx_mse_before_equalization();
    float mse_lmmse = compute_lmmse_mse();
    float checksum = checksum_lmmse_results();

    g_sink = h_mse + mse_rx_before_eq + mse_lmmse + checksum +
             Xmmse_r[0] + Xmmse_i[TOTAL_DATA - 1];

    print_lmmse_results("Part 3 - SIMD-like RVV LS Channel Estimation + RVV LMMSE Equalization",
                        h_mse,
                        mse_rx_before_eq,
                        mse_lmmse,
                        checksum);

    return 0;
}
