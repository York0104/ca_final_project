#ifndef OFDM_MULTI_COMMON_H
#define OFDM_MULTI_COMMON_H

#include <cstdint>

#include "../common/ofdm_params.h"

static constexpr uint32_t OFDM_MULTI_INPUT_MAGIC = 0x4F464D35u;
static constexpr int DEFAULT_NUM_PATTERNS = 16;

#if defined(__CUDACC__)
#define OFDM_MULTI_HD __host__ __device__
#else
#define OFDM_MULTI_HD
#endif

struct OfdmMultiInputHeader{
    uint32_t magic;
    int32_t num_patterns;
    int32_t num_subcarriers;
    int32_t num_pilots;
    int32_t num_data_symbols;
};

static OFDM_MULTI_HD inline int multi_channel_index(int pattern, int k){
    return pattern * NUM_SUBCARRIERS + k;
}

static OFDM_MULTI_HD inline int multi_pilot_index(int pattern, int k, int p){
    return pattern * TOTAL_PILOT + pilot_index(k, p);
}

static OFDM_MULTI_HD inline int multi_data_index(int pattern, int s, int k){
    return pattern * TOTAL_DATA + data_index(s, k);
}

#undef OFDM_MULTI_HD

#endif
