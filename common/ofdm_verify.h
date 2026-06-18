#ifndef OFDM_VERIFY_H
#define OFDM_VERIFY_H

#include <cstdio>

#include "ofdm_params.h"
#include "ofdm_data.h"

using namespace std;

// Hhat vs Htrue：Channel estimation 是否正確
static inline float compute_channel_mse(){
    float acc = 0.0f;

    for (int k = 0; k < NUM_SUBCARRIERS; ++k){
        float er = Hhat_r[k] - Htrue_r[k];
        float ei = Hhat_i[k] - Htrue_i[k];
        acc += er * er + ei * ei;
    }

    return acc / static_cast<float>(NUM_SUBCARRIERS);
}

// Ydata vs Xdata：等化前原始 received signal 的誤差  -> baseline
static inline float compute_rx_mse_before_equalization()
{
    float acc = 0.0f;

    for (int idx = 0; idx < TOTAL_DATA; ++idx){
        float er = Ydata_r[idx] - Xdata_r[idx];
        float ei = Ydata_i[idx] - Xdata_i[idx];
        acc += er * er + ei * ei;
    }

    return acc / static_cast<float>(TOTAL_DATA);
}

// Xmmse vs Xdata：等化後是否更接近原始 QPSK data
static inline float compute_lmmse_mse(){
    float acc = 0.0f;

    for (int idx = 0; idx < TOTAL_DATA; ++idx){
        float er = Xmmse_r[idx] - Xdata_r[idx];
        float ei = Xmmse_i[idx] - Xdata_i[idx];
        acc += er * er + ei * ei;
    }

    return acc / static_cast<float>(TOTAL_DATA);
}

// 等化後比等化前好 & channel estimation 沒有明顯錯誤 & output 不為 null
static inline float checksum_lmmse_results(){
    float sum = 0.0f;

    for (int k = 0; k < NUM_SUBCARRIERS; ++k){
        sum += Hhat_r[k] + Hhat_i[k];
    }

    for (int idx = 0; idx < TOTAL_DATA; ++idx){
        sum += 0.5f * Xmmse_r[idx] + 0.5f * Xmmse_i[idx];
    }

    return sum;
}

static inline void print_lmmse_results(const char *title, float h_mse, float mse_rx_before_eq, float mse_lmmse, float checksum){
    int verification_pass =
        (mse_lmmse < mse_rx_before_eq) &&
        (h_mse < 0.01f) &&
        (abs_float(checksum) > 1.0e-6f);

    printf("%s\n", title);
    printf("NUM_SUBCARRIERS              = %d\n", NUM_SUBCARRIERS);
    printf("NUM_PILOTS                   = %d\n", NUM_PILOTS);
    printf("NUM_DATA_SYMBOLS             = %d\n", NUM_DATA_SYMBOLS);
    printf("TOTAL_DATA                   = %d\n", TOTAL_DATA);
    printf("NOISE_STD                    = %.8f\n", static_cast<double>(NOISE_STD));
    printf("NOISE_VAR                    = %.8f\n", static_cast<double>(NOISE_VAR));
    printf("NOISE_VAR_OVER_SYMBOL_POWER  = %.8f\n",
           static_cast<double>(NOISE_VAR_OVER_SYMBOL_POWER));

    printf("H_MSE                        = %.8f\n", static_cast<double>(h_mse));
    printf("MSE_RX_BEFORE_EQ             = %.8f\n", static_cast<double>(mse_rx_before_eq));
    printf("MSE_LMMSE                    = %.8f\n", static_cast<double>(mse_lmmse));

    printf("Xmmse[0]                     = %.8f + j%.8f\n",
           static_cast<double>(Xmmse_r[0]),
           static_cast<double>(Xmmse_i[0]));

    printf("checksum                     = %.8f\n", static_cast<double>(checksum));
    printf("guard                        = %.8f\n", static_cast<double>(g_sink));
    printf("Verification                 = %s\n", verification_pass ? "PASS" : "FAIL");
}

#endif
