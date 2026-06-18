#ifndef OFDM_MODEL_H
#define OFDM_MODEL_H

// Pilot-based OFDM LS Channel Estimation and LMMSE Equalization
// Accelerated kernels ：channel estimation and LMMSE equalization.

// QPSK generator
static inline void make_qpsk_symbol(int a, int b, float *out_r, float *out_i){
    // X∈{−1−j,1−j,−1+j,1+j}
    int pattern = (a * 17 + b * 31 + 7) % 4;
    *out_r = (pattern & 1) ? 1.0f : -1.0f;
    *out_i = (pattern & 2) ? 1.0f : -1.0f;
}

static inline void complex_mul(float ar, float ai, float br, float bi,  float *cr, float *ci){
    *cr = ar * br - ai * bi;
    *ci = ar * bi + ai * br;
}

static inline void init_reference_channel(float *htrue_r, float *htrue_i){
    for (int k = 0; k < NUM_SUBCARRIERS; ++k)
    {   // First 3 enter deep fade
        bool is_deep_fade = ((k % 64) < 3);

        float mag = is_deep_fade ? 0.20f : 1.00f;

        // 0.98^2+0.20^2=1.0004≈1
        float shape_r = 0.98f;
        float shape_i = 0.20f;

        htrue_r[k] = mag * shape_r;
        htrue_i[k] = mag * shape_i;
    }
}

#endif
