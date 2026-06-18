#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

#include "../common/ofdm_params.h"
#include "ofdm_multi_common.h"

using namespace std;

#ifndef PART5_TPB_LS
#define PART5_TPB_LS 256
#endif

#ifndef PART5_TPB_EQ
#define PART5_TPB_EQ 256
#endif

#ifndef PART5_DEFAULT_WARMUP_ITERS
#define PART5_DEFAULT_WARMUP_ITERS 10
#endif

#ifndef PART5_DEFAULT_TIMED_ITERS
#define PART5_DEFAULT_TIMED_ITERS 100
#endif

static constexpr int TPB_LS = PART5_TPB_LS;
static constexpr int TPB_EQ = PART5_TPB_EQ;
static constexpr int DEFAULT_WARMUP_ITERS = PART5_DEFAULT_WARMUP_ITERS;
static constexpr int DEFAULT_TIMED_ITERS = PART5_DEFAULT_TIMED_ITERS;

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

struct DeviceBuffers{
    float *pilot_w;
    float *Ypilot_r;
    float *Ypilot_i;
    float *Ydata_r;
    float *Ydata_i;
    float *Hhat_r;
    float *Hhat_i;
    float *Xmmse_r;
    float *Xmmse_i;
};

static volatile float g_sink = 0.0f;

static inline void cuda_check(cudaError_t err, const char *what){
    if (err != cudaSuccess){
        printf("CUDA error at %s: %s\n", what, cudaGetErrorString(err));
        exit(1);
    }
}

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

// blockIdx.x -> subcarrier k
// blockIdx.y -> pattern index
// threadIdx.x -> pilot contributions within one (pattern, k) reduction
__global__ void ls_channel_estimation_multi_kernel(const float *Ypilot_r_dev,
                                                   const float *Ypilot_i_dev,
                                                   const float *pilot_w_dev,
                                                   float *Hhat_r_dev,
                                                   float *Hhat_i_dev){
    int k = blockIdx.x;
    int pattern = blockIdx.y;
    int tid = threadIdx.x;

    extern __shared__ float shmem[];
    float *sh_r = shmem;
    float *sh_i = shmem + blockDim.x;

    float partial_r = 0.0f;
    float partial_i = 0.0f;

    for (int p = tid; p < NUM_PILOTS; p += blockDim.x){
        int idx = multi_pilot_index(pattern, k, p);
        float w = pilot_w_dev[p];
        partial_r += Ypilot_r_dev[idx] * w;
        partial_i += Ypilot_i_dev[idx] * w;
    }

    sh_r[tid] = partial_r;
    sh_i[tid] = partial_i;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1){
        if (tid < offset){
            sh_r[tid] += sh_r[tid + offset];
            sh_i[tid] += sh_i[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0){
        int h_idx = multi_channel_index(pattern, k);
        Hhat_r_dev[h_idx] = sh_r[0];
        Hhat_i_dev[h_idx] = sh_i[0];
    }
}

// blockIdx.x / threadIdx.x -> flattened output index within one pattern
// blockIdx.y -> pattern index
__global__ void lmmse_equalization_multi_kernel(const float *Ydata_r_dev,
                                                const float *Ydata_i_dev,
                                                const float *Hhat_r_dev,
                                                const float *Hhat_i_dev,
                                                float *Xmmse_r_dev,
                                                float *Xmmse_i_dev){
    int data_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pattern = blockIdx.y;

    if (data_idx >= TOTAL_DATA){
        return;
    }

    int k = data_idx % NUM_SUBCARRIERS;
    int idx = pattern * TOTAL_DATA + data_idx;
    int h_idx = multi_channel_index(pattern, k);

    float yr = Ydata_r_dev[idx];
    float yi = Ydata_i_dev[idx];
    float hr = Hhat_r_dev[h_idx];
    float hi = Hhat_i_dev[h_idx];
    float denom = hr * hr + hi * hi + NOISE_VAR_OVER_SYMBOL_POWER + EPSILON;

    Xmmse_r_dev[idx] = (yr * hr + yi * hi) / denom;
    Xmmse_i_dev[idx] = (yi * hr - yr * hi) / denom;
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

static void allocate_device_buffers(const HostBuffers *host, DeviceBuffers *dev){
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev->pilot_w),
                          sizeof(float) * NUM_PILOTS),
               "cudaMalloc pilot_w");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev->Ypilot_r),
                          sizeof(float) * host->total_pilot_all),
               "cudaMalloc Ypilot_r");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev->Ypilot_i),
                          sizeof(float) * host->total_pilot_all),
               "cudaMalloc Ypilot_i");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev->Ydata_r),
                          sizeof(float) * host->total_data_all),
               "cudaMalloc Ydata_r");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev->Ydata_i),
                          sizeof(float) * host->total_data_all),
               "cudaMalloc Ydata_i");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev->Hhat_r),
                          sizeof(float) * host->total_channel_all),
               "cudaMalloc Hhat_r");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev->Hhat_i),
                          sizeof(float) * host->total_channel_all),
               "cudaMalloc Hhat_i");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev->Xmmse_r),
                          sizeof(float) * host->total_data_all),
               "cudaMalloc Xmmse_r");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev->Xmmse_i),
                          sizeof(float) * host->total_data_all),
               "cudaMalloc Xmmse_i");
}

static void free_device_buffers(DeviceBuffers *dev){
    cuda_check(cudaFree(dev->pilot_w), "cudaFree pilot_w");
    cuda_check(cudaFree(dev->Ypilot_r), "cudaFree Ypilot_r");
    cuda_check(cudaFree(dev->Ypilot_i), "cudaFree Ypilot_i");
    cuda_check(cudaFree(dev->Ydata_r), "cudaFree Ydata_r");
    cuda_check(cudaFree(dev->Ydata_i), "cudaFree Ydata_i");
    cuda_check(cudaFree(dev->Hhat_r), "cudaFree Hhat_r");
    cuda_check(cudaFree(dev->Hhat_i), "cudaFree Hhat_i");
    cuda_check(cudaFree(dev->Xmmse_r), "cudaFree Xmmse_r");
    cuda_check(cudaFree(dev->Xmmse_i), "cudaFree Xmmse_i");
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

static void copy_inputs_to_device(const HostBuffers *host, const DeviceBuffers *dev){
    cuda_check(cudaMemcpy(dev->pilot_w,
                          host->pilot_w,
                          sizeof(float) * NUM_PILOTS,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy pilot_w H2D");
    cuda_check(cudaMemcpy(dev->Ypilot_r,
                          host->Ypilot_r,
                          sizeof(float) * host->total_pilot_all,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy Ypilot_r H2D");
    cuda_check(cudaMemcpy(dev->Ypilot_i,
                          host->Ypilot_i,
                          sizeof(float) * host->total_pilot_all,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy Ypilot_i H2D");
    cuda_check(cudaMemcpy(dev->Ydata_r,
                          host->Ydata_r,
                          sizeof(float) * host->total_data_all,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy Ydata_r H2D");
    cuda_check(cudaMemcpy(dev->Ydata_i,
                          host->Ydata_i,
                          sizeof(float) * host->total_data_all,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy Ydata_i H2D");
}

static void copy_outputs_to_host(const HostBuffers *host, const DeviceBuffers *dev){
    cuda_check(cudaMemcpy(host->Hhat_r,
                          dev->Hhat_r,
                          sizeof(float) * host->total_channel_all,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy Hhat_r D2H");
    cuda_check(cudaMemcpy(host->Hhat_i,
                          dev->Hhat_i,
                          sizeof(float) * host->total_channel_all,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy Hhat_i D2H");
    cuda_check(cudaMemcpy(host->Xmmse_r,
                          dev->Xmmse_r,
                          sizeof(float) * host->total_data_all,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy Xmmse_r D2H");
    cuda_check(cudaMemcpy(host->Xmmse_i,
                          dev->Xmmse_i,
                          sizeof(float) * host->total_data_all,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy Xmmse_i D2H");
}

static void launch_pipeline_once(const HostBuffers *host,
                                 const DeviceBuffers *dev,
                                 dim3 grid_ls,
                                 dim3 block_ls,
                                 size_t shmem_bytes,
                                 dim3 grid_eq,
                                 dim3 block_eq){
    (void)host;
    ls_channel_estimation_multi_kernel<<<grid_ls, block_ls, shmem_bytes>>>(
        dev->Ypilot_r,
        dev->Ypilot_i,
        dev->pilot_w,
        dev->Hhat_r,
        dev->Hhat_i);
    cuda_check(cudaGetLastError(), "launch ls_channel_estimation_multi_kernel");

    lmmse_equalization_multi_kernel<<<grid_eq, block_eq>>>(
        dev->Ydata_r,
        dev->Ydata_i,
        dev->Hhat_r,
        dev->Hhat_i,
        dev->Xmmse_r,
        dev->Xmmse_i);
    cuda_check(cudaGetLastError(), "launch lmmse_equalization_multi_kernel");
}

static float measure_ls_kernel_ms(const DeviceBuffers *dev,
                                  dim3 grid_ls,
                                  dim3 block_ls,
                                  size_t shmem_bytes,
                                  int timed_iters){
    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "cudaEventCreate ls start");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate ls stop");

    cuda_check(cudaEventRecord(start), "cudaEventRecord ls start");
    for (int iter = 0; iter < timed_iters; ++iter){
        ls_channel_estimation_multi_kernel<<<grid_ls, block_ls, shmem_bytes>>>(
            dev->Ypilot_r,
            dev->Ypilot_i,
            dev->pilot_w,
            dev->Hhat_r,
            dev->Hhat_i);
    }
    cuda_check(cudaGetLastError(), "timed ls launch");
    cuda_check(cudaEventRecord(stop), "cudaEventRecord ls stop");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize ls stop");

    float elapsed_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop),
               "cudaEventElapsedTime ls");
    cuda_check(cudaEventDestroy(start), "cudaEventDestroy ls start");
    cuda_check(cudaEventDestroy(stop), "cudaEventDestroy ls stop");
    return elapsed_ms / static_cast<float>(timed_iters);
}

static float measure_eq_kernel_ms(const DeviceBuffers *dev,
                                  dim3 grid_eq,
                                  dim3 block_eq,
                                  int timed_iters){
    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "cudaEventCreate eq start");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate eq stop");

    cuda_check(cudaEventRecord(start), "cudaEventRecord eq start");
    for (int iter = 0; iter < timed_iters; ++iter){
        lmmse_equalization_multi_kernel<<<grid_eq, block_eq>>>(
            dev->Ydata_r,
            dev->Ydata_i,
            dev->Hhat_r,
            dev->Hhat_i,
            dev->Xmmse_r,
            dev->Xmmse_i);
    }
    cuda_check(cudaGetLastError(), "timed eq launch");
    cuda_check(cudaEventRecord(stop), "cudaEventRecord eq stop");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize eq stop");

    float elapsed_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop),
               "cudaEventElapsedTime eq");
    cuda_check(cudaEventDestroy(start), "cudaEventDestroy eq start");
    cuda_check(cudaEventDestroy(stop), "cudaEventDestroy eq stop");
    return elapsed_ms / static_cast<float>(timed_iters);
}

static float measure_pipeline_ms(const HostBuffers *host,
                                 const DeviceBuffers *dev,
                                 dim3 grid_ls,
                                 dim3 block_ls,
                                 size_t shmem_bytes,
                                 dim3 grid_eq,
                                 dim3 block_eq,
                                 int timed_iters){
    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "cudaEventCreate pipeline start");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate pipeline stop");

    cuda_check(cudaEventRecord(start), "cudaEventRecord pipeline start");
    for (int iter = 0; iter < timed_iters; ++iter){
        launch_pipeline_once(host,
                             dev,
                             grid_ls,
                             block_ls,
                             shmem_bytes,
                             grid_eq,
                             block_eq);
    }
    cuda_check(cudaEventRecord(stop), "cudaEventRecord pipeline stop");
    cuda_check(cudaEventSynchronize(stop), "cudaEventSynchronize pipeline stop");

    float elapsed_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, start, stop),
               "cudaEventElapsedTime pipeline");
    cuda_check(cudaEventDestroy(start), "cudaEventDestroy pipeline start");
    cuda_check(cudaEventDestroy(stop), "cudaEventDestroy pipeline stop");
    return elapsed_ms / static_cast<float>(timed_iters);
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
                          float h_mse,
                          float mse_rx_before_eq,
                          float mse_lmmse,
                          float checksum,
                          float ls_ms,
                          float eq_ms,
                          float pipeline_ms){
    int verification_pass =
        (mse_lmmse < mse_rx_before_eq) &&
        (h_mse < 0.01f) &&
        (abs_float(checksum) > 1.0e-6f);

    printf("Part 5 - Multi-pattern CUDA LS Channel Estimation + CUDA LMMSE Equalization\n");
    printf("NUM_PATTERNS                  = %d\n", host->num_patterns);
    printf("NUM_SUBCARRIERS              = %d\n", NUM_SUBCARRIERS);
    printf("NUM_PILOTS                   = %d\n", NUM_PILOTS);
    printf("NUM_DATA_SYMBOLS             = %d\n", NUM_DATA_SYMBOLS);
    printf("TOTAL_PILOT_ALL              = %lu\n",
           static_cast<unsigned long>(host->total_pilot_all));
    printf("TOTAL_DATA_ALL               = %lu\n",
           static_cast<unsigned long>(host->total_data_all));
    printf("NOISE_STD                    = %.8f\n", static_cast<double>(NOISE_STD));
    printf("NOISE_VAR                    = %.8f\n", static_cast<double>(NOISE_VAR));
    printf("NOISE_VAR_OVER_SYMBOL_POWER  = %.8f\n",
           static_cast<double>(NOISE_VAR_OVER_SYMBOL_POWER));
    printf("H_MSE                        = %.8f\n", static_cast<double>(h_mse));
    printf("MSE_RX_BEFORE_EQ             = %.8f\n", static_cast<double>(mse_rx_before_eq));
    printf("MSE_LMMSE                    = %.8f\n", static_cast<double>(mse_lmmse));
    printf("Xmmse[0]                     = %.8f + j%.8f\n",
           static_cast<double>(host->Xmmse_r[0]),
           static_cast<double>(host->Xmmse_i[0]));
    printf("checksum                     = %.8f\n", static_cast<double>(checksum));
    printf("guard                        = %.8f\n", static_cast<double>(g_sink));
    printf("TPB_LS                       = %d\n", TPB_LS);
    printf("TPB_EQ                       = %d\n", TPB_EQ);
    printf("LS_KERNEL_MS                 = %.6f\n", static_cast<double>(ls_ms));
    printf("LMMSE_KERNEL_MS              = %.6f\n", static_cast<double>(eq_ms));
    printf("PIPELINE_KERNEL_MS           = %.6f\n", static_cast<double>(pipeline_ms));
    printf("Verification                 = %s\n", verification_pass ? "PASS" : "FAIL");
}

int main(int argc, char **argv){
    const char *filename = (argc >= 2) ? argv[1] : "../data/ofdm_input_multi.bin";
    int warmup_iters = (argc >= 3) ? atoi(argv[2]) : DEFAULT_WARMUP_ITERS;
    int timed_iters = (argc >= 4) ? atoi(argv[3]) : DEFAULT_TIMED_ITERS;

    if (warmup_iters < 0 || timed_iters <= 0){
        printf("Invalid iteration counts: warmup=%d timed=%d\n", warmup_iters, timed_iters);
        return 1;
    }

    HostBuffers host = {};
    DeviceBuffers dev = {};

    load_input_binary(&host, filename);
    allocate_device_buffers(&host, &dev);
    copy_inputs_to_device(&host, &dev);

    dim3 block_ls(TPB_LS);
    dim3 grid_ls(NUM_SUBCARRIERS, host.num_patterns);
    size_t shmem_bytes = sizeof(float) * TPB_LS * 2;

    dim3 block_eq(TPB_EQ);
    dim3 grid_eq((TOTAL_DATA + TPB_EQ - 1) / TPB_EQ, host.num_patterns);

    for (int iter = 0; iter < warmup_iters; ++iter){
        launch_pipeline_once(&host,
                             &dev,
                             grid_ls,
                             block_ls,
                             shmem_bytes,
                             grid_eq,
                             block_eq);
    }
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup");

    float ls_ms = measure_ls_kernel_ms(&dev,
                                       grid_ls,
                                       block_ls,
                                       shmem_bytes,
                                       timed_iters);
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize after ls timing");

    ls_channel_estimation_multi_kernel<<<grid_ls, block_ls, shmem_bytes>>>(
        dev.Ypilot_r,
        dev.Ypilot_i,
        dev.pilot_w,
        dev.Hhat_r,
        dev.Hhat_i);
    cuda_check(cudaGetLastError(), "prep eq ls launch");
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize prep eq");

    float eq_ms = measure_eq_kernel_ms(&dev, grid_eq, block_eq, timed_iters);
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize after eq timing");

    float pipeline_ms = measure_pipeline_ms(&host,
                                            &dev,
                                            grid_ls,
                                            block_ls,
                                            shmem_bytes,
                                            grid_eq,
                                            block_eq,
                                            timed_iters);
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize after pipeline timing");

    launch_pipeline_once(&host,
                         &dev,
                         grid_ls,
                         block_ls,
                         shmem_bytes,
                         grid_eq,
                         block_eq);
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize final pipeline");

    copy_outputs_to_host(&host, &dev);

    float h_mse = compute_channel_mse_multi(&host);
    float mse_rx_before_eq = compute_rx_mse_before_equalization_multi(&host);
    float mse_lmmse = compute_lmmse_mse_multi(&host);
    float checksum = checksum_lmmse_results_multi(&host);

    g_sink = h_mse + mse_rx_before_eq + mse_lmmse + checksum +
             host.Xmmse_r[0] + host.Xmmse_i[host.total_data_all - 1];

    print_results(&host,
                  h_mse,
                  mse_rx_before_eq,
                  mse_lmmse,
                  checksum,
                  ls_ms,
                  eq_ms,
                  pipeline_ms);

    free_device_buffers(&dev);
    free_host_buffers(&host);
    return 0;
}
