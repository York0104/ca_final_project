#include <cstdio>

#include "../common/ofdm_params.h"
#include "../common/ofdm_data.h"
#include "../common/ofdm_io.h"
#include "../common/ofdm_verify.h"

// Stage 1: RVV reduction for LS channel estimation.
// Lanes map to different pilot observations for one subcarrier k, and
// vfredusum.vs reduces the vector products to one Hhat[k].
static inline float rvv_dot_product_reduction_f32(const float *a, const float *b, int n){
    // a = yr = Ypilot_r[k][0...255]
    // b = pilot_w[0...255]
    // n = NUM_PILOTS = 256
    float total = 0.0f;
    int remaining = n;

    while (remaining > 0){
        int vl = 0; // vsetvli 回填「實際處理多少」
        float partial = 0.0f;

        asm volatile(
            "vsetvli %[vl], %[remaining], e32, m1, ta, ma\n\t"
            "vle32.v v1, (%[a])\n\t"
            "vle32.v v2, (%[b])\n\t"
            "vfmul.vv v3, v1, v2\n\t"
            "vmv.v.i v0, 0\n\t"
            "vfredusum.vs v4, v3, v0\n\t"
            "vfmv.f.s ft0, v4\n\t"
            "fsw ft0, (%[partial])\n\t"
            : [vl] "=&r"(vl)
            : [a] "r"(a),
              [b] "r"(b),
              [remaining] "r"(remaining),
              [partial] "r"(&partial)
            : "ft0", "memory"
        ); // result = sum(a[i] * b[i]) for i in [0, vl-1]

        total += partial;
        a += vl;
        b += vl;
        remaining -= vl;
    }

    return total;
}

static void estimate_channel_ls_average_rvv_reduction(){
    for (int k = 0; k < NUM_SUBCARRIERS; ++k){
        const float *yr = &Ypilot_r[pilot_index(k, 0)];
        const float *yi = &Ypilot_i[pilot_index(k, 0)];

        Hhat_r[k] = rvv_dot_product_reduction_f32(yr, pilot_w, NUM_PILOTS);
        Hhat_i[k] = rvv_dot_product_reduction_f32(yi, pilot_w, NUM_PILOTS);
    }
}

// Stage 2: RVV element-wise LMMSE equalization.
// Lanes map to adjacent subcarriers k for one data symbol s.
static void equalize_lmmse_rvv(){
#if OFDM_HAS_RVV
    const float noise_bias = NOISE_VAR_OVER_SYMBOL_POWER + EPSILON;

    for (int s = 0; s < NUM_DATA_SYMBOLS; ++s){
        int k = 0;

        while (k < NUM_SUBCARRIERS){
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
    for (int s = 0; s < NUM_DATA_SYMBOLS; ++s){
        for (int k = 0; k < NUM_SUBCARRIERS; ++k){
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

    estimate_channel_ls_average_rvv_reduction();
    equalize_lmmse_rvv();

    float h_mse = compute_channel_mse();
    float mse_rx_before_eq = compute_rx_mse_before_equalization();
    float mse_lmmse = compute_lmmse_mse();
    float checksum = checksum_lmmse_results();

    g_sink = h_mse + mse_rx_before_eq + mse_lmmse + checksum +
             Xmmse_r[0] + Xmmse_i[TOTAL_DATA - 1];

    print_lmmse_results("Part 2 - RVV Reduction LS Channel Estimation + RVV LMMSE Equalization",
                        h_mse,
                        mse_rx_before_eq,
                        mse_lmmse,
                        checksum);

    return 0;
}
