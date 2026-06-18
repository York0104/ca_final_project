#include "../common/ofdm_params.h"
#include "../common/ofdm_data.h"
#include "../common/ofdm_io.h"
#include "../common/ofdm_verify.h"

// Stage 1: scalar LS channel estimation.
// The inner loop is the weighted summation used as the scalar reference
// for the RVV and CUDA versions.
static void estimate_channel_ls_average_scalar(){
    for (int k = 0; k < NUM_SUBCARRIERS; ++k){
        float acc_r = 0.0f;
        float acc_i = 0.0f;

        for (int p = 0; p < NUM_PILOTS; ++p){
            int idx = pilot_index(k, p);
            float w = pilot_w[p];
            acc_r += Ypilot_r[idx] * w;
            acc_i += Ypilot_i[idx] * w;
        }

        Hhat_r[k] = acc_r;
        Hhat_i[k] = acc_i;
    }
}

// Stage 2: scalar one-tap LMMSE equalization.
// Each output (s, k) is independent, so later parts map this loop to
// vector lanes or CUDA threads without changing the arithmetic.
static void equalize_lmmse_scalar(){
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
}

int main(){
    load_input_binary("../data/ofdm_input.bin");

    estimate_channel_ls_average_scalar();
    equalize_lmmse_scalar();

    float h_mse = compute_channel_mse();
    float mse_rx_before_eq = compute_rx_mse_before_equalization();
    float mse_lmmse = compute_lmmse_mse();

    float checksum = checksum_lmmse_results();
    g_sink = h_mse + mse_rx_before_eq + mse_lmmse + checksum +
             Xmmse_r[0] + Xmmse_i[TOTAL_DATA - 1];

    print_lmmse_results("Part 1 - Scalar LS Channel Estimation + Scalar LMMSE Equalization",
                        h_mse,
                        mse_rx_before_eq,
                        mse_lmmse,
                        checksum);

    return 0;
}
