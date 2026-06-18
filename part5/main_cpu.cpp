#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "../common/ofdm_params.h"
#include "ofdm_multi_common.h"

using namespace std;

struct HostBuffers{
    int num_patterns;
    size_t total_channel_all;
    size_t total_pilot_all;
    size_t total_data_all;
    float *pilot_w;
    float *Htrue_r;
    float *Htrue_i;
    float *Ypilot_r;
    float *Ypilot_i;
    float *Xdata_r;
    float *Xdata_i;
    float *Ydata_r;
    float *Ydata_i;
    float *Hhat_r;
    float *Hhat_i;
    float *Xmmse_r;
    float *Xmmse_i;
};

static volatile float g_sink = 0.0f;

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

static void allocate_host_buffers(HostBuffers *host){
    host->total_channel_all = static_cast<size_t>(host->num_patterns) * NUM_SUBCARRIERS;
    host->total_pilot_all = static_cast<size_t>(host->num_patterns) * TOTAL_PILOT;
    host->total_data_all = static_cast<size_t>(host->num_patterns) * TOTAL_DATA;

    host->pilot_w = new float[NUM_PILOTS];
    host->Htrue_r = new float[host->total_channel_all];
    host->Htrue_i = new float[host->total_channel_all];
    host->Ypilot_r = new float[host->total_pilot_all];
    host->Ypilot_i = new float[host->total_pilot_all];
    host->Xdata_r = new float[host->total_data_all];
    host->Xdata_i = new float[host->total_data_all];
    host->Ydata_r = new float[host->total_data_all];
    host->Ydata_i = new float[host->total_data_all];
    host->Hhat_r = new float[host->total_channel_all];
    host->Hhat_i = new float[host->total_channel_all];
    host->Xmmse_r = new float[host->total_data_all];
    host->Xmmse_i = new float[host->total_data_all];
}

static void free_host_buffers(HostBuffers *host){
    delete[] host->pilot_w;
    delete[] host->Htrue_r;
    delete[] host->Htrue_i;
    delete[] host->Ypilot_r;
    delete[] host->Ypilot_i;
    delete[] host->Xdata_r;
    delete[] host->Xdata_i;
    delete[] host->Ydata_r;
    delete[] host->Ydata_i;
    delete[] host->Hhat_r;
    delete[] host->Hhat_i;
    delete[] host->Xmmse_r;
    delete[] host->Xmmse_i;
}

static void load_input_binary(HostBuffers *host, const char *filename){
    FILE *fp = fopen(filename, "rb");

    if (!fp){
        printf("Cannot open input file: %s\n", filename);
        exit(1);
    }

    OfdmMultiInputHeader header = {};
    read_exact(fp, &header, sizeof(header), "header");

    if (header.magic != OFDM_MULTI_INPUT_MAGIC ||
        header.num_patterns <= 0 ||
        header.num_subcarriers != NUM_SUBCARRIERS ||
        header.num_pilots != NUM_PILOTS ||
        header.num_data_symbols != NUM_DATA_SYMBOLS){
        printf("Input header mismatch\n");
        exit(1);
    }

    host->num_patterns = header.num_patterns;
    allocate_host_buffers(host);

    read_exact(fp, host->pilot_w, sizeof(float) * NUM_PILOTS, "pilot_w");
    read_exact(fp, host->Htrue_r, sizeof(float) * host->total_channel_all, "Htrue_r");
    read_exact(fp, host->Htrue_i, sizeof(float) * host->total_channel_all, "Htrue_i");
    read_exact(fp, host->Ypilot_r, sizeof(float) * host->total_pilot_all, "Ypilot_r");
    read_exact(fp, host->Ypilot_i, sizeof(float) * host->total_pilot_all, "Ypilot_i");
    read_exact(fp, host->Xdata_r, sizeof(float) * host->total_data_all, "Xdata_r");
    read_exact(fp, host->Xdata_i, sizeof(float) * host->total_data_all, "Xdata_i");
    read_exact(fp, host->Ydata_r, sizeof(float) * host->total_data_all, "Ydata_r");
    read_exact(fp, host->Ydata_i, sizeof(float) * host->total_data_all, "Ydata_i");
    fclose(fp);

    for (size_t i = 0; i < host->total_channel_all; ++i){
        host->Hhat_r[i] = 0.0f;
        host->Hhat_i[i] = 0.0f;
    }

    for (size_t i = 0; i < host->total_data_all; ++i){
        host->Xmmse_r[i] = 0.0f;
        host->Xmmse_i[i] = 0.0f;
    }
}

static void estimate_channel_scalar_multi(HostBuffers *host){
    for (int pattern = 0; pattern < host->num_patterns; ++pattern){
        for (int k = 0; k < NUM_SUBCARRIERS; ++k){
            float acc_r = 0.0f;
            float acc_i = 0.0f;

            for (int p = 0; p < NUM_PILOTS; ++p){
                int idx = multi_pilot_index(pattern, k, p);
                float w = host->pilot_w[p];
                acc_r += host->Ypilot_r[idx] * w;
                acc_i += host->Ypilot_i[idx] * w;
            }

            int h_idx = multi_channel_index(pattern, k);
            host->Hhat_r[h_idx] = acc_r;
            host->Hhat_i[h_idx] = acc_i;
        }
    }
}

static void equalize_lmmse_scalar_multi(HostBuffers *host){
    for (int pattern = 0; pattern < host->num_patterns; ++pattern){
        for (int s = 0; s < NUM_DATA_SYMBOLS; ++s){
            for (int k = 0; k < NUM_SUBCARRIERS; ++k){
                int idx = multi_data_index(pattern, s, k);
                int h_idx = multi_channel_index(pattern, k);

                float yr = host->Ydata_r[idx];
                float yi = host->Ydata_i[idx];
                float hr = host->Hhat_r[h_idx];
                float hi = host->Hhat_i[h_idx];
                float denom = hr * hr + hi * hi + NOISE_VAR_OVER_SYMBOL_POWER + EPSILON;

                host->Xmmse_r[idx] = (yr * hr + yi * hi) / denom;
                host->Xmmse_i[idx] = (yi * hr - yr * hi) / denom;
            }
        }
    }
}

static float compute_channel_mse_multi(const HostBuffers *host){
    float acc = 0.0f;

    for (size_t i = 0; i < host->total_channel_all; ++i){
        float er = host->Hhat_r[i] - host->Htrue_r[i];
        float ei = host->Hhat_i[i] - host->Htrue_i[i];
        acc += er * er + ei * ei;
    }

    return acc / static_cast<float>(host->total_channel_all);
}

static float compute_rx_mse_before_equalization_multi(const HostBuffers *host){
    float acc = 0.0f;

    for (size_t i = 0; i < host->total_data_all; ++i){
        float er = host->Ydata_r[i] - host->Xdata_r[i];
        float ei = host->Ydata_i[i] - host->Xdata_i[i];
        acc += er * er + ei * ei;
    }

    return acc / static_cast<float>(host->total_data_all);
}

static float compute_lmmse_mse_multi(const HostBuffers *host){
    float acc = 0.0f;

    for (size_t i = 0; i < host->total_data_all; ++i){
        float er = host->Xmmse_r[i] - host->Xdata_r[i];
        float ei = host->Xmmse_i[i] - host->Xdata_i[i];
        acc += er * er + ei * ei;
    }

    return acc / static_cast<float>(host->total_data_all);
}

static float checksum_lmmse_results_multi(const HostBuffers *host){
    float sum = 0.0f;

    for (size_t i = 0; i < host->total_channel_all; ++i){
        sum += host->Hhat_r[i] + host->Hhat_i[i];
    }

    for (size_t i = 0; i < host->total_data_all; ++i){
        sum += 0.5f * host->Xmmse_r[i] + 0.5f * host->Xmmse_i[i];
    }

    return sum;
}

static void print_results(const HostBuffers *host,
                          const char *title,
                          double elapsed_ms,
                          float h_mse,
                          float mse_rx_before_eq,
                          float mse_lmmse,
                          float checksum){
    int verification_pass =
        (mse_lmmse < mse_rx_before_eq) &&
        (h_mse < 0.01f) &&
        (abs_float(checksum) > 1.0e-6f);

    printf("%s\n", title);
    printf("NUM_PATTERNS                  = %d\n", host->num_patterns);
    printf("NUM_SUBCARRIERS              = %d\n", NUM_SUBCARRIERS);
    printf("NUM_PILOTS                   = %d\n", NUM_PILOTS);
    printf("NUM_DATA_SYMBOLS             = %d\n", NUM_DATA_SYMBOLS);
    printf("TOTAL_PILOT_ALL              = %lu\n",
           static_cast<unsigned long>(host->total_pilot_all));
    printf("TOTAL_DATA_ALL               = %lu\n",
           static_cast<unsigned long>(host->total_data_all));
    printf("CPU_PIPELINE_MS              = %.6f\n", elapsed_ms);
    printf("H_MSE                        = %.8f\n", static_cast<double>(h_mse));
    printf("MSE_RX_BEFORE_EQ             = %.8f\n", static_cast<double>(mse_rx_before_eq));
    printf("MSE_LMMSE                    = %.8f\n", static_cast<double>(mse_lmmse));
    printf("Xmmse[0]                     = %.8f + j%.8f\n",
           static_cast<double>(host->Xmmse_r[0]),
           static_cast<double>(host->Xmmse_i[0]));
    printf("checksum                     = %.8f\n", static_cast<double>(checksum));
    printf("guard                        = %.8f\n", static_cast<double>(g_sink));
    printf("Verification                 = %s\n", verification_pass ? "PASS" : "FAIL");
}

int main(int argc, char **argv){
    const char *filename = (argc >= 2) ? argv[1] : "../data/ofdm_input_multi.bin";

    HostBuffers host = {};
    load_input_binary(&host, filename);

    auto start = chrono::high_resolution_clock::now();
    estimate_channel_scalar_multi(&host);
    equalize_lmmse_scalar_multi(&host);
    auto stop = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = stop - start;

    float h_mse = compute_channel_mse_multi(&host);
    float mse_rx_before_eq = compute_rx_mse_before_equalization_multi(&host);
    float mse_lmmse = compute_lmmse_mse_multi(&host);
    float checksum = checksum_lmmse_results_multi(&host);

    g_sink = h_mse + mse_rx_before_eq + mse_lmmse + checksum +
             host.Xmmse_r[0] + host.Xmmse_i[host.total_data_all - 1];

    print_results(&host,
                  "Part 5 CPU Baseline - Multi-pattern LS Channel Estimation + Scalar LMMSE Equalization",
                  elapsed.count() * 1000.0,
                  h_mse,
                  mse_rx_before_eq,
                  mse_lmmse,
                  checksum);

    free_host_buffers(&host);
    return 0;
}
