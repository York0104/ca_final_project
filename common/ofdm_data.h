#ifndef OFDM_DATA_H
#define OFDM_DATA_H

#include "ofdm_params.h"

// array 起始位址 % 64 == 0 for 64-byte alignment

// Pilot-related arrays
alignas(64) static float Ypilot_r[TOTAL_PILOT];
alignas(64) static float Ypilot_i[TOTAL_PILOT];
alignas(64) static float pilot_w[NUM_PILOTS];

// Transmitted-data arrays
alignas(64) static float Xdata_r[TOTAL_DATA];
alignas(64) static float Xdata_i[TOTAL_DATA];

// Received-data arrays
alignas(64) static float Ydata_r[TOTAL_DATA];
alignas(64) static float Ydata_i[TOTAL_DATA];

// True channel & estimated channel
alignas(64) static float Htrue_r[NUM_SUBCARRIERS];
alignas(64) static float Htrue_i[NUM_SUBCARRIERS];
alignas(64) static float Hhat_r[NUM_SUBCARRIERS];
alignas(64) static float Hhat_i[NUM_SUBCARRIERS];

// LMMSE output arrays
alignas(64) static float Xmmse_r[TOTAL_DATA];
alignas(64) static float Xmmse_i[TOTAL_DATA];

// // Prevents dead code elimination by forcing memory writes
static volatile float g_sink = 0.0f;

#endif
