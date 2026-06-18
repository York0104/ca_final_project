#include <cstdio>
#include <cstdlib>
#include <random>

#include "../common/ofdm_params.h"
#include "../common/ofdm_model.h"
#include "ofdm_multi_common.h"

using namespace std;

static inline void write_exact(FILE *fp, const void *ptr, size_t bytes, const char *name){
    size_t n = fwrite(ptr, 1, bytes, fp);

    if (n != bytes){
        printf("Write failed: %s, expected %lu bytes, got %lu bytes\n",
               name,
               static_cast<unsigned long>(bytes),
               static_cast<unsigned long>(n));
        exit(1);
    }
}

int main(int argc, char **argv){
    const char *filename = (argc >= 2) ? argv[1] : "../data/ofdm_input_multi.bin";
    int num_patterns = (argc >= 3) ? atoi(argv[2]) : DEFAULT_NUM_PATTERNS;

    if (num_patterns <= 0){
        printf("Invalid num_patterns: %d\n", num_patterns);
        return 1;
    }

    size_t total_channel_all = static_cast<size_t>(num_patterns) * NUM_SUBCARRIERS;
    size_t total_pilot_all = static_cast<size_t>(num_patterns) * TOTAL_PILOT;
    size_t total_data_all = static_cast<size_t>(num_patterns) * TOTAL_DATA;

    float *pilot_w = new float[NUM_PILOTS];
    float *Htrue_r = new float[total_channel_all];
    float *Htrue_i = new float[total_channel_all];
    float *Ypilot_r = new float[total_pilot_all];
    float *Ypilot_i = new float[total_pilot_all];
    float *Xdata_r = new float[total_data_all];
    float *Xdata_i = new float[total_data_all];
    float *Ydata_r = new float[total_data_all];
    float *Ydata_i = new float[total_data_all];

    float h_base_r[NUM_SUBCARRIERS];
    float h_base_i[NUM_SUBCARRIERS];
    init_reference_channel(h_base_r, h_base_i);

    for (int p = 0; p < NUM_PILOTS; ++p){
        pilot_w[p] = 1.0f / static_cast<float>(NUM_PILOTS);
    }

    for (int pattern = 0; pattern < num_patterns; ++pattern){
        mt19937 rng(12345u + static_cast<unsigned int>(pattern * 97));
        normal_distribution<float> awgn_dist(0.0f, NOISE_STD);

        for (int k = 0; k < NUM_SUBCARRIERS; ++k){
            int h_idx = multi_channel_index(pattern, k);
            Htrue_r[h_idx] = h_base_r[k];
            Htrue_i[h_idx] = h_base_i[k];
        }

        for (int p = 0; p < NUM_PILOTS; ++p){
            for (int k = 0; k < NUM_SUBCARRIERS; ++k){
                int h_idx = multi_channel_index(pattern, k);
                int idx = multi_pilot_index(pattern, k, p);
                Ypilot_r[idx] = Htrue_r[h_idx] + awgn_dist(rng);
                Ypilot_i[idx] = Htrue_i[h_idx] + awgn_dist(rng);
            }
        }

        for (int s = 0; s < NUM_DATA_SYMBOLS; ++s){
            for (int k = 0; k < NUM_SUBCARRIERS; ++k){
                int h_idx = multi_channel_index(pattern, k);
                int idx = multi_data_index(pattern, s, k);

                float xr, xi;
                float yr, yi;

                make_qpsk_symbol(s + pattern * NUM_DATA_SYMBOLS,
                                 k + pattern * 17,
                                 &xr,
                                 &xi);
                complex_mul(Htrue_r[h_idx], Htrue_i[h_idx], xr, xi, &yr, &yi);

                Xdata_r[idx] = xr;
                Xdata_i[idx] = xi;
                Ydata_r[idx] = yr + awgn_dist(rng);
                Ydata_i[idx] = yi + awgn_dist(rng);
            }
        }
    }

    FILE *fp = fopen(filename, "wb");

    if (!fp){
        printf("Cannot open output file: %s\n", filename);
        return 1;
    }

    OfdmMultiInputHeader header = {
        OFDM_MULTI_INPUT_MAGIC,
        num_patterns,
        NUM_SUBCARRIERS,
        NUM_PILOTS,
        NUM_DATA_SYMBOLS
    };

    write_exact(fp, &header, sizeof(header), "header");
    write_exact(fp, pilot_w, sizeof(float) * NUM_PILOTS, "pilot_w");
    write_exact(fp, Htrue_r, sizeof(float) * total_channel_all, "Htrue_r");
    write_exact(fp, Htrue_i, sizeof(float) * total_channel_all, "Htrue_i");
    write_exact(fp, Ypilot_r, sizeof(float) * total_pilot_all, "Ypilot_r");
    write_exact(fp, Ypilot_i, sizeof(float) * total_pilot_all, "Ypilot_i");
    write_exact(fp, Xdata_r, sizeof(float) * total_data_all, "Xdata_r");
    write_exact(fp, Xdata_i, sizeof(float) * total_data_all, "Xdata_i");
    write_exact(fp, Ydata_r, sizeof(float) * total_data_all, "Ydata_r");
    write_exact(fp, Ydata_i, sizeof(float) * total_data_all, "Ydata_i");
    fclose(fp);

    printf("Generated multi-pattern OFDM input: %s\n", filename);
    printf("NUM_PATTERNS    = %d\n", num_patterns);
    printf("NUM_SUBCARRIERS = %d\n", NUM_SUBCARRIERS);
    printf("NUM_PILOTS      = %d\n", NUM_PILOTS);
    printf("NUM_DATA_SYMBOLS= %d\n", NUM_DATA_SYMBOLS);
    printf("NOISE_STD       = %.8f\n", static_cast<double>(NOISE_STD));

    delete[] pilot_w;
    delete[] Htrue_r;
    delete[] Htrue_i;
    delete[] Ypilot_r;
    delete[] Ypilot_i;
    delete[] Xdata_r;
    delete[] Xdata_i;
    delete[] Ydata_r;
    delete[] Ydata_i;
    return 0;
}
