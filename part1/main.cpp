#include "../common/ofdm_params.h"
#include "../common/ofdm_data.h"
#include "../common/ofdm_io.h"
#include "../common/ofdm_verify.h"

// ============================================================
// Computation Stage 1 - Scalar LS Channel Estimation
//
// Purpose:
//   Estimate the frequency-domain channel Hhat[k] from repeated
//   pilot observations.
//
// Mathematical form:
//   Hhat[k] = Σ_p Ypilot[k][p] * pilot_w[p]
//
// CA mapping:
//   This is the required nested-loop reduction form:
//   sum_p f(a[p], b[p]), where f(a,b) = a*b.
// ============================================================
static void estimate_channel_ls_average_scalar(){
    for (int k = 0; k < NUM_SUBCARRIERS; ++k){
        float acc_r = 0.0f;
        float acc_i = 0.0f;

        for (int p = 0; p < NUM_PILOTS; ++p){
            int idx = pilot_index(k, p);
            float w = pilot_w[p];
            // Hhat_r[k] = Σ_p Ypilot_r[k][p] · pilot_w[p]
            acc_r += Ypilot_r[idx] * w;
            // Hhat_i[k] = Σ_p Ypilot_i[k][p] · pilot_w[p]
            acc_i += Ypilot_i[idx] * w;
        }

        // estimated channel
        Hhat_r[k] = acc_r;
        Hhat_i[k] = acc_i;
    }
}

// ============================================================
// Computation Stage 2 - Scalar LMMSE Equalization
//
// Purpose:
//   Recover transmitted data symbols using the estimated channel.
//
// Mathematical form:
//   Xmmse[s][k] = Ydata[s][k] * conj(Hhat[k])
//                 / (|Hhat[k]|^2 + NOISE_VAR_OVER_SYMBOL_POWER + EPSILON)
//
// CA mapping:
//   This is an element-wise complex arithmetic workload.
//   Each (s, k) output is independent and parallelizable.
// ============================================================
static void equalize_lmmse_scalar(){ // equalize for each OFDM data symbol
    for (int s = 0; s < NUM_DATA_SYMBOLS; ++s){
        for (int k = 0; k < NUM_SUBCARRIERS; ++k){
            int idx = data_index(s, k);

            // received data symbol
            float yr = Ydata_r[idx];
            float yi = Ydata_i[idx];
            float hr = Hhat_r[k];
            float hi = Hhat_i[k];

            // |Hhat[k]|^2 + noise_term + epsilon
            float denom = hr * hr + hi * hi + NOISE_VAR_OVER_SYMBOL_POWER + EPSILON;

            // estimate ( Ydata · conj(Hhat) ) / denom
            Xmmse_r[idx] = (yr * hr + yi * hi) / denom;
            Xmmse_i[idx] = (yi * hr - yr * hi) / denom;
        }
    }
}

int main(){
    load_input_binary("../data/ofdm_input.bin");

    // Computation Stage 1: scalar LS channel estimation
    estimate_channel_ls_average_scalar();

    // Computation Stage 2: scalar LMMSE equalization
    equalize_lmmse_scalar();

    // check
    // H_MSE = average_k |Hhat[k] - Htrue[k]|^2 for  Stage 1
    float h_mse = compute_channel_mse();
    // MSE_RX_BEFORE_EQ = average |Ydata - Xdata|^2
    float mse_rx_before_eq = compute_rx_mse_before_equalization();
    // MSE_LMMSE = average |Xmmse - Xdata|^2 for stage 2
    float mse_lmmse = compute_lmmse_mse();

    // Force the results  be used
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
