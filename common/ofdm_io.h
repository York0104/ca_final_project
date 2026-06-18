#ifndef OFDM_IO_H
#define OFDM_IO_H

#include <cstdio>
#include <cstdlib>

#include "ofdm_params.h"
#include "ofdm_data.h"

using namespace std;

struct OfdmInputHeader{
    uint32_t magic;
    int32_t num_subcarriers;
    int32_t num_pilots;
    int32_t num_data_symbols;
};

static inline void read_exact(FILE *fp, void *ptr, size_t bytes, const char *name){
    size_t n = fread(ptr, 1, bytes, fp);

    if (n != bytes){
        printf("Read failed: %s, expected %lu bytes, got %lu bytes\n",
               name,
               static_cast<unsigned long>(bytes),
               static_cast<unsigned long>(n));
        exit(1);
    }
}

static inline void reset_outputs(){
    for (int k = 0; k < NUM_SUBCARRIERS; ++k){
        Hhat_r[k] = 0.0f;
        Hhat_i[k] = 0.0f;
    }

    for (int idx = 0; idx < TOTAL_DATA; ++idx){
        Xmmse_r[idx] = 0.0f;
        Xmmse_i[idx] = 0.0f;
    }
}

static inline void load_input_binary(const char *filename){
    FILE *fp = fopen(filename, "rb");

    if (!fp){
        printf("Cannot open input file: %s\n", filename);
        exit(1);
    }

    OfdmInputHeader header = {};

    read_exact(fp, &header, sizeof(header), "header");

    if (header.magic != OFDM_INPUT_MAGIC || header.num_subcarriers != NUM_SUBCARRIERS ||header.num_pilots != NUM_PILOTS ||header.num_data_symbols != NUM_DATA_SYMBOLS){
        printf("Input header mismatch\n");
        printf("magic = 0x%08x\n", header.magic);
        printf("n_sc = %d, n_pilot = %d, n_data = %d\n",
               header.num_subcarriers,
               header.num_pilots,
               header.num_data_symbols);
        exit(1);
    }

    read_exact(fp, Htrue_r, sizeof(float) * NUM_SUBCARRIERS, "Htrue_r");
    read_exact(fp, Htrue_i, sizeof(float) * NUM_SUBCARRIERS, "Htrue_i");
    read_exact(fp, pilot_w, sizeof(float) * NUM_PILOTS, "pilot_w");

    read_exact(fp, Ypilot_r, sizeof(float) * TOTAL_PILOT, "Ypilot_r");
    read_exact(fp, Ypilot_i, sizeof(float) * TOTAL_PILOT, "Ypilot_i");

    read_exact(fp, Xdata_r, sizeof(float) * TOTAL_DATA, "Xdata_r");
    read_exact(fp, Xdata_i, sizeof(float) * TOTAL_DATA, "Xdata_i");
    read_exact(fp, Ydata_r, sizeof(float) * TOTAL_DATA, "Ydata_r");
    read_exact(fp, Ydata_i, sizeof(float) * TOTAL_DATA, "Ydata_i");

    fclose(fp);
    reset_outputs();
}

#endif
