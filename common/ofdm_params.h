#ifndef OFDM_PARAMS_H
#define OFDM_PARAMS_H

#include <cstdint>

static constexpr int NUM_SUBCARRIERS = 512;
static constexpr int NUM_PILOTS = 256;
static constexpr int NUM_DATA_SYMBOLS = 512;

static constexpr int TOTAL_PILOT = NUM_SUBCARRIERS * NUM_PILOTS;
static constexpr int TOTAL_DATA = NUM_SUBCARRIERS * NUM_DATA_SYMBOLS;

static constexpr float NOISE_STD = 0.0404f;
static constexpr float SYMBOL_POWER = 2.0f;
static constexpr float NOISE_VAR_PER_REAL = NOISE_STD * NOISE_STD;
static constexpr float COMPLEX_NOISE_POWER = 2.0f * NOISE_VAR_PER_REAL;
static constexpr float NOISE_VAR = COMPLEX_NOISE_POWER;
static constexpr float NOISE_VAR_OVER_SYMBOL_POWER = COMPLEX_NOISE_POWER / SYMBOL_POWER;
static constexpr float EPSILON = 1.0e-6f;

static constexpr uint32_t OFDM_INPUT_MAGIC = 0x4F46444Du;

#if defined(__CUDACC__)
#define OFDM_HD __host__ __device__
#else
#define OFDM_HD
#endif

#if defined(__riscv) && (defined(__riscv_vector) || defined(__riscv_v))
#define OFDM_HAS_RVV 1
#else
#define OFDM_HAS_RVV 0
#endif

static OFDM_HD inline int pilot_index(int k, int p){
    return k * NUM_PILOTS + p;
}

static OFDM_HD inline int data_index(int s, int k){
    return s * NUM_SUBCARRIERS + k;
}

static OFDM_HD inline float abs_float(float v){
    return (v < 0.0f) ? -v : v;
}

#undef OFDM_HD

#endif
