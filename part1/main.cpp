#include "../common/ofdm_params.h"
#include "../common/ofdm_data.h"
#include "../common/ofdm_io.h"
#include "../common/ofdm_verify.h"

// Stage 1: scalar LS channel estimation.
static void estimate_channel_ls_average_scalar(){
    for (int k = 0; k < NUM_SUBCARRIERS; ++k){
        float acc_r = 0.0f;
        float acc_i = 0.0f;

        // pilot average
        for (int p = 0; p < NUM_PILOTS; ++p){
            int idx = pilot_index(k, p);

            // weighted accumulation
            float w = pilot_w[p];
            acc_r += Ypilot_r[idx] * w;
            acc_i += Ypilot_i[idx] * w;
        }

        // complex channel estimate
        Hhat_r[k] = acc_r;
        Hhat_i[k] = acc_i;
    }
}

// Stage 2: scalar LMMSE equalization.
static void equalize_lmmse_scalar(){
    for (int s = 0; s < NUM_DATA_SYMBOLS; ++s){
        for (int k = 0; k < NUM_SUBCARRIERS; ++k){
            // flattened array index
            int idx = data_index(s, k);

            // received data
            float yr = Ydata_r[idx];
            float yi = Ydata_i[idx];

            // estimated channel
            float hr = Hhat_r[k];
            float hi = Hhat_i[k];

            // LMMSE denominator
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

    // 等化後比等化前好 & channel estimation 沒有明顯錯誤 & output 不為 null
    float checksum = checksum_lmmse_results();
    g_sink = h_mse + mse_rx_before_eq + mse_lmmse + checksum + Xmmse_r[0] + Xmmse_i[TOTAL_DATA - 1];

    print_lmmse_results("Part 1 - Scalar LS Channel Estimation + Scalar LMMSE Equalization",
                        h_mse,
                        mse_rx_before_eq,
                        mse_lmmse,
                        checksum);

    return 0;
}
