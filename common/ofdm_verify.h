#ifndef OFDM_VERIFY_H
#define OFDM_VERIFY_H

#include <cstdio>

#include "ofdm_params.h"
#include "ofdm_data.h"

static inline float compute_channel_mse()
{
    float acc = 0.0f;

    for (int k = 0; k < NUM_SUBCARRIERS; ++k)
    {
        float er = Hhat_r[k] - Htrue_r[k];
        float ei = Hhat_i[k] - Htrue_i[k];
        acc += er * er + ei * ei;
    }

    return acc / static_cast<float>(NUM_SUBCARRIERS);
}

static inline float compute_rx_mse_before_equalization()
{
    float acc = 0.0f;

    for (int idx = 0; idx < TOTAL_DATA; ++idx)
    {
        float er = Ydata_r[idx] - Xdata_r[idx];
        float ei = Ydata_i[idx] - Xdata_i[idx];
        acc += er * er + ei * ei;
    }

    return acc / static_cast<float>(TOTAL_DATA);
}

static inline float compute_lmmse_mse()
{
    float acc = 0.0f;

    for (int idx = 0; idx < TOTAL_DATA; ++idx)
    {
        float er = Xmmse_r[idx] - Xdata_r[idx];
        float ei = Xmmse_i[idx] - Xdata_i[idx];
        acc += er * er + ei * ei;
    }

    return acc / static_cast<float>(TOTAL_DATA);
}

static inline float checksum_lmmse_results()
{
    float sum = 0.0f;

    for (int k = 0; k < NUM_SUBCARRIERS; ++k)
    {
        sum += Hhat_r[k] + Hhat_i[k];
    }

    for (int idx = 0; idx < TOTAL_DATA; ++idx)
    {
        sum += 0.5f * Xmmse_r[idx] + 0.5f * Xmmse_i[idx];
    }

    return sum;
}

static inline void print_lmmse_results(const char *title,
                                       float h_mse,
                                       float mse_rx_before_eq,
                                       float mse_lmmse,
                                       float checksum)
{
    std::printf("%s\n", title);
    std::printf("NUM_SUBCARRIERS              = %d\n", NUM_SUBCARRIERS);
    std::printf("NUM_PILOTS                   = %d\n", NUM_PILOTS);
    std::printf("NUM_DATA_SYMBOLS             = %d\n", NUM_DATA_SYMBOLS);
    std::printf("TOTAL_DATA                   = %d\n", TOTAL_DATA);
    std::printf("NOISE_STD                    = %.8f\n", static_cast<double>(NOISE_STD));
    std::printf("NOISE_VAR                    = %.8f\n", static_cast<double>(NOISE_VAR));

    std::printf("H_MSE                        = %.8f\n", static_cast<double>(h_mse));
    std::printf("MSE_RX_BEFORE_EQ             = %.8f\n", static_cast<double>(mse_rx_before_eq));
    std::printf("MSE_LMMSE                    = %.8f\n", static_cast<double>(mse_lmmse));

    std::printf("Xmmse[0]                     = %.8f + j%.8f\n",
                static_cast<double>(Xmmse_r[0]),
                static_cast<double>(Xmmse_i[0]));

    std::printf("checksum                     = %.8f\n", static_cast<double>(checksum));
    std::printf("guard                        = %.8f\n", static_cast<double>(g_sink));

    if ((mse_lmmse < mse_rx_before_eq) &&
        (h_mse < 0.01f) &&
        (abs_float(checksum) > 1.0e-6f))
    {
        std::printf("Verification: PASS\n");
    }
    else
    {
        std::printf("Verification: FAIL\n");
    }
}

#endif
