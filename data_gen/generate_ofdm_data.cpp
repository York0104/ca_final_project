#include <cstdio>
#include <cstdlib>
#include <random>

#include "../common/ofdm_params.h"
#include "../common/ofdm_model.h"
#include "../common/ofdm_io.h"

using namespace std;

alignas(64) static float htrue_r[NUM_SUBCARRIERS];
alignas(64) static float htrue_i[NUM_SUBCARRIERS];
alignas(64) static float pilot_w_local[NUM_PILOTS];
alignas(64) static float ypilot_r[TOTAL_PILOT];
alignas(64) static float ypilot_i[TOTAL_PILOT];
alignas(64) static float xdata_r[TOTAL_DATA];
alignas(64) static float xdata_i[TOTAL_DATA];
alignas(64) static float ydata_r[TOTAL_DATA];
alignas(64) static float ydata_i[TOTAL_DATA];

static mt19937 rng(12345);
static normal_distribution<float> awgn_dist(0.0f, NOISE_STD);

static inline float awgn_noise(){
    return awgn_dist(rng);
}

int main(int argc, char **argv){
    const char *filename = (argc >= 2) ? argv[1] : "../data/ofdm_input.bin";

    init_reference_channel(htrue_r, htrue_i);

    for (int p = 0; p < NUM_PILOTS; ++p){
        pilot_w_local[p] = 1.0f / static_cast<float>(NUM_PILOTS);
    }

    for (int p = 0; p < NUM_PILOTS; ++p){
        for (int k = 0; k < NUM_SUBCARRIERS; ++k){
            int idx = pilot_index(k, p);
            ypilot_r[idx] = htrue_r[k] + awgn_noise();
            ypilot_i[idx] = htrue_i[k] + awgn_noise();
        }
    }

    for (int s = 0; s < NUM_DATA_SYMBOLS; ++s){
        for (int k = 0; k < NUM_SUBCARRIERS; ++k){
            int idx = data_index(s, k);

            float xr, xi;
            float yr, yi;

            make_qpsk_symbol(s, k, &xr, &xi);
            complex_mul(htrue_r[k], htrue_i[k], xr, xi, &yr, &yi);

            xdata_r[idx] = xr;
            xdata_i[idx] = xi;
            ydata_r[idx] = yr + awgn_noise();
            ydata_i[idx] = yi + awgn_noise();
        }
    }

    FILE *fp = fopen(filename, "wb");

    if (!fp){
        printf("Cannot open output file: %s\n", filename);
        return 1;
    }

    OfdmInputHeader header = {
        OFDM_INPUT_MAGIC,
        NUM_SUBCARRIERS,
        NUM_PILOTS,
        NUM_DATA_SYMBOLS
    };

    fwrite(&header, sizeof(header), 1, fp);
    fwrite(htrue_r, sizeof(float), NUM_SUBCARRIERS, fp);
    fwrite(htrue_i, sizeof(float), NUM_SUBCARRIERS, fp);
    fwrite(pilot_w_local, sizeof(float), NUM_PILOTS, fp);
    fwrite(ypilot_r, sizeof(float), TOTAL_PILOT, fp);
    fwrite(ypilot_i, sizeof(float), TOTAL_PILOT, fp);
    fwrite(xdata_r, sizeof(float), TOTAL_DATA, fp);
    fwrite(xdata_i, sizeof(float), TOTAL_DATA, fp);
    fwrite(ydata_r, sizeof(float), TOTAL_DATA, fp);
    fwrite(ydata_i, sizeof(float), TOTAL_DATA, fp);
    fclose(fp);

    printf("Generated OFDM input: %s\n", filename);
    printf("NUM_SUBCARRIERS = %d\n", NUM_SUBCARRIERS);
    printf("NUM_PILOTS      = %d\n", NUM_PILOTS);
    printf("NUM_DATA_SYMBOLS= %d\n", NUM_DATA_SYMBOLS);
    printf("NOISE_STD       = %.8f\n", static_cast<double>(NOISE_STD));

    return 0;
}
