#ifndef OFDM_DATA_H
#define OFDM_DATA_H

#include "ofdm_params.h"

alignas(64) static float Ypilot_r[TOTAL_PILOT];
alignas(64) static float Ypilot_i[TOTAL_PILOT];
alignas(64) static float pilot_w[NUM_PILOTS];

alignas(64) static float Xdata_r[TOTAL_DATA];
alignas(64) static float Xdata_i[TOTAL_DATA];
alignas(64) static float Ydata_r[TOTAL_DATA];
alignas(64) static float Ydata_i[TOTAL_DATA];

alignas(64) static float Htrue_r[NUM_SUBCARRIERS];
alignas(64) static float Htrue_i[NUM_SUBCARRIERS];
alignas(64) static float Hhat_r[NUM_SUBCARRIERS];
alignas(64) static float Hhat_i[NUM_SUBCARRIERS];

alignas(64) static float Xmmse_r[TOTAL_DATA];
alignas(64) static float Xmmse_i[TOTAL_DATA];

static volatile float g_sink = 0.0f;

#endif
