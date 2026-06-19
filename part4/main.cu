#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

#include "../common/ofdm_params.h"
#include "../common/ofdm_data.h"
#include "../common/ofdm_io.h"
#include "../common/ofdm_verify.h"

using namespace std;

#ifndef PART4_TPB_LS
#define PART4_TPB_LS 256
#endif

#ifndef PART4_TPB_EQ
#define PART4_TPB_EQ 256
#endif

#ifndef PART4_DEFAULT_WARMUP_ITERS
#define PART4_DEFAULT_WARMUP_ITERS 10
#endif

#ifndef PART4_DEFAULT_TIMED_ITERS
#define PART4_DEFAULT_TIMED_ITERS 100
#endif

static constexpr int TPB_LS = PART4_TPB_LS;
static constexpr int TPB_EQ = PART4_TPB_EQ;
static constexpr int DEFAULT_WARMUP_ITERS = PART4_DEFAULT_WARMUP_ITERS;
static constexpr int DEFAULT_TIMED_ITERS = PART4_DEFAULT_TIMED_ITERS;

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

static inline void cuda_check(cudaError_t err, const char *what){
    if (err != cudaSuccess){
        printf("CUDA error at %s: %s\n", what, cudaGetErrorString(err));
        exit(1);
    }
}

// One block reduces one subcarrier k.
// Threads cooperate through shared memory because the pilot sum is a
// block-level reduction.
__global__ void ls_channel_estimation_shared_kernel(const float *Ypilot_r_dev,
                                                    const float *Ypilot_i_dev,
                                                    const float *pilot_w_dev,
                                                    float *Hhat_r_dev,
                                                    float *Hhat_i_dev){
    int k = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float shmem[];
    float *sh_r = shmem;
    float *sh_i = shmem + blockDim.x;

    float partial_r = 0.0f;
    float partial_i = 0.0f;

    for (int p = tid; p < NUM_PILOTS; p += blockDim.x){
        int idx = pilot_index(k, p);
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
        Hhat_r_dev[k] = sh_r[0];
        Hhat_i_dev[k] = sh_i[0];
    }
}

// Simple Stage 1 baseline for the shared-memory comparison.
// One thread handles one subcarrier and runs the pilot loop serially.
__global__ void ls_channel_estimation_serial_kernel(const float *Ypilot_r_dev,
                                                    const float *Ypilot_i_dev,
                                                    const float *pilot_w_dev,
                                                    float *Hhat_r_dev,
                                                    float *Hhat_i_dev){
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k >= NUM_SUBCARRIERS){
        return;
    }

    float acc_r = 0.0f;
    float acc_i = 0.0f;

    for (int p = 0; p < NUM_PILOTS; ++p){
        int idx = pilot_index(k, p);
        float w = pilot_w_dev[p];
        acc_r += Ypilot_r_dev[idx] * w;
        acc_i += Ypilot_i_dev[idx] * w;
    }

    Hhat_r_dev[k] = acc_r;
    Hhat_i_dev[k] = acc_i;
}

// Stage 2 uses one thread per flattened output index.
__global__ void lmmse_equalization_kernel(const float *Ydata_r_dev,
                                          const float *Ydata_i_dev,
                                          const float *Hhat_r_dev,
                                          const float *Hhat_i_dev,
                                          float *Xmmse_r_dev,
                                          float *Xmmse_i_dev){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= TOTAL_DATA){
        return;
    }

    int k = idx % NUM_SUBCARRIERS;

    float yr = Ydata_r_dev[idx];
    float yi = Ydata_i_dev[idx];
    float hr = Hhat_r_dev[k];
    float hi = Hhat_i_dev[k];

    float denom = hr * hr + hi * hi + NOISE_VAR_OVER_SYMBOL_POWER + EPSILON;

    Xmmse_r_dev[idx] = (yr * hr + yi * hi) / denom;
    Xmmse_i_dev[idx] = (yi * hr - yr * hi) / denom;
}

static void allocate_device_buffers(DeviceBuffers *dev){
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev->pilot_w),
                          sizeof(float) * NUM_PILOTS),
               "cudaMalloc pilot_w");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev->Ypilot_r),
                          sizeof(float) * TOTAL_PILOT),
               "cudaMalloc Ypilot_r");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev->Ypilot_i),
                          sizeof(float) * TOTAL_PILOT),
               "cudaMalloc Ypilot_i");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev->Ydata_r),
                          sizeof(float) * TOTAL_DATA),
               "cudaMalloc Ydata_r");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev->Ydata_i),
                          sizeof(float) * TOTAL_DATA),
               "cudaMalloc Ydata_i");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev->Hhat_r),
                          sizeof(float) * NUM_SUBCARRIERS),
               "cudaMalloc Hhat_r");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev->Hhat_i),
                          sizeof(float) * NUM_SUBCARRIERS),
               "cudaMalloc Hhat_i");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev->Xmmse_r),
                          sizeof(float) * TOTAL_DATA),
               "cudaMalloc Xmmse_r");
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&dev->Xmmse_i),
                          sizeof(float) * TOTAL_DATA),
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

static void copy_inputs_to_device(const DeviceBuffers *dev){
    cuda_check(cudaMemcpy(dev->pilot_w,
                          pilot_w,
                          sizeof(float) * NUM_PILOTS,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy pilot_w H2D");
    cuda_check(cudaMemcpy(dev->Ypilot_r,
                          Ypilot_r,
                          sizeof(float) * TOTAL_PILOT,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy Ypilot_r H2D");
    cuda_check(cudaMemcpy(dev->Ypilot_i,
                          Ypilot_i,
                          sizeof(float) * TOTAL_PILOT,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy Ypilot_i H2D");
    cuda_check(cudaMemcpy(dev->Ydata_r,
                          Ydata_r,
                          sizeof(float) * TOTAL_DATA,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy Ydata_r H2D");
    cuda_check(cudaMemcpy(dev->Ydata_i,
                          Ydata_i,
                          sizeof(float) * TOTAL_DATA,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy Ydata_i H2D");
}

static void copy_outputs_to_host(const DeviceBuffers *dev){
    cuda_check(cudaMemcpy(Hhat_r,
                          dev->Hhat_r,
                          sizeof(float) * NUM_SUBCARRIERS,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy Hhat_r D2H");
    cuda_check(cudaMemcpy(Hhat_i,
                          dev->Hhat_i,
                          sizeof(float) * NUM_SUBCARRIERS,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy Hhat_i D2H");
    cuda_check(cudaMemcpy(Xmmse_r,
                          dev->Xmmse_r,
                          sizeof(float) * TOTAL_DATA,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy Xmmse_r D2H");
    cuda_check(cudaMemcpy(Xmmse_i,
                          dev->Xmmse_i,
                          sizeof(float) * TOTAL_DATA,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy Xmmse_i D2H");
}

static void launch_pipeline_once(const DeviceBuffers *dev,
                                 bool use_shared_ls,
                                 dim3 grid_ls,
                                 dim3 block_ls,
                                 size_t shmem_bytes,
                                 dim3 grid_eq,
                                 dim3 block_eq){
    if (use_shared_ls){
        ls_channel_estimation_shared_kernel<<<grid_ls, block_ls, shmem_bytes>>>(
            dev->Ypilot_r,
            dev->Ypilot_i,
            dev->pilot_w,
            dev->Hhat_r,
            dev->Hhat_i);
        cuda_check(cudaGetLastError(), "launch ls_channel_estimation_shared_kernel");
    } else {
        ls_channel_estimation_serial_kernel<<<grid_ls, block_ls>>>(
            dev->Ypilot_r,
            dev->Ypilot_i,
            dev->pilot_w,
            dev->Hhat_r,
            dev->Hhat_i);
        cuda_check(cudaGetLastError(), "launch ls_channel_estimation_serial_kernel");
    }

    lmmse_equalization_kernel<<<grid_eq, block_eq>>>(
        dev->Ydata_r,
        dev->Ydata_i,
        dev->Hhat_r,
        dev->Hhat_i,
        dev->Xmmse_r,
        dev->Xmmse_i);
    cuda_check(cudaGetLastError(), "launch lmmse_equalization_kernel");
}

static float measure_ls_kernel_ms(const DeviceBuffers *dev,
                                  bool use_shared_ls,
                                  dim3 grid_ls,
                                  dim3 block_ls,
                                  size_t shmem_bytes,
                                  int timed_iters){
    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "cudaEventCreate ls start");
    cuda_check(cudaEventCreate(&stop), "cudaEventCreate ls stop");

    cuda_check(cudaEventRecord(start), "cudaEventRecord ls start");
    for (int iter = 0; iter < timed_iters; ++iter){
        if (use_shared_ls){
            ls_channel_estimation_shared_kernel<<<grid_ls, block_ls, shmem_bytes>>>(
                dev->Ypilot_r,
                dev->Ypilot_i,
                dev->pilot_w,
                dev->Hhat_r,
                dev->Hhat_i);
        } else {
            ls_channel_estimation_serial_kernel<<<grid_ls, block_ls>>>(
                dev->Ypilot_r,
                dev->Ypilot_i,
                dev->pilot_w,
                dev->Hhat_r,
                dev->Hhat_i);
        }
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
        lmmse_equalization_kernel<<<grid_eq, block_eq>>>(
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

static float measure_pipeline_ms(const DeviceBuffers *dev,
                                 bool use_shared_ls,
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
        launch_pipeline_once(dev,
                             use_shared_ls,
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

int main(int argc, char **argv){
    const char *filename = (argc >= 2) ? argv[1] : "../data/ofdm_input.bin";
    int warmup_iters = (argc >= 3) ? atoi(argv[2]) : DEFAULT_WARMUP_ITERS;
    int timed_iters = (argc >= 4) ? atoi(argv[3]) : DEFAULT_TIMED_ITERS;
    bool use_shared_ls = (argc >= 5) ? (strcmp(argv[4], "serial") != 0) : true;

    if ((TPB_LS & (TPB_LS - 1)) != 0){
        printf("TPB_LS must be a power of two.\n");
        return 1;
    }

    if (warmup_iters < 0 || timed_iters <= 0){
        printf("Invalid iteration counts: warmup=%d timed=%d\n", warmup_iters, timed_iters);
        return 1;
    }

    load_input_binary(filename);

    DeviceBuffers dev = {};
    allocate_device_buffers(&dev);
    copy_inputs_to_device(&dev);

    dim3 block_ls(TPB_LS);
    dim3 grid_ls = use_shared_ls
        ? dim3(NUM_SUBCARRIERS)
        : dim3((NUM_SUBCARRIERS + TPB_LS - 1) / TPB_LS);
    size_t shmem_bytes = use_shared_ls ? sizeof(float) * TPB_LS * 2 : 0;

    dim3 block_eq(TPB_EQ);
    dim3 grid_eq((TOTAL_DATA + TPB_EQ - 1) / TPB_EQ);

    for (int iter = 0; iter < warmup_iters; ++iter){
        launch_pipeline_once(&dev,
                             use_shared_ls,
                             grid_ls,
                             block_ls,
                             shmem_bytes,
                             grid_eq,
                             block_eq);
    }
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup");

    float ls_ms = measure_ls_kernel_ms(&dev,
                                       use_shared_ls,
                                       grid_ls,
                                       block_ls,
                                       shmem_bytes,
                                       timed_iters);
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize after ls timing");

    if (use_shared_ls){
        ls_channel_estimation_shared_kernel<<<grid_ls, block_ls, shmem_bytes>>>(
            dev.Ypilot_r,
            dev.Ypilot_i,
            dev.pilot_w,
            dev.Hhat_r,
            dev.Hhat_i);
        cuda_check(cudaGetLastError(), "prep eq shared launch");
    } else {
        ls_channel_estimation_serial_kernel<<<grid_ls, block_ls>>>(
            dev.Ypilot_r,
            dev.Ypilot_i,
            dev.pilot_w,
            dev.Hhat_r,
            dev.Hhat_i);
        cuda_check(cudaGetLastError(), "prep eq serial launch");
    }
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize prep eq");

    float eq_ms = measure_eq_kernel_ms(&dev, grid_eq, block_eq, timed_iters);
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize after eq timing");

    float pipeline_ms = measure_pipeline_ms(&dev,
                                            use_shared_ls,
                                            grid_ls,
                                            block_ls,
                                            shmem_bytes,
                                            grid_eq,
                                            block_eq,
                                            timed_iters);
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize after pipeline timing");

    launch_pipeline_once(&dev,
                         use_shared_ls,
                         grid_ls,
                         block_ls,
                         shmem_bytes,
                         grid_eq,
                         block_eq);
    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize final pipeline");

    copy_outputs_to_host(&dev);

    float h_mse = compute_channel_mse();
    float mse_rx_before_eq = compute_rx_mse_before_equalization();
    float mse_lmmse = compute_lmmse_mse();
    float checksum = checksum_lmmse_results();

    g_sink = h_mse + mse_rx_before_eq + mse_lmmse + checksum +
             Xmmse_r[0] + Xmmse_i[TOTAL_DATA - 1];

    print_lmmse_results("Part 4 - CUDA SIMT LS Channel Estimation + CUDA LMMSE Equalization",
                        h_mse,
                        mse_rx_before_eq,
                        mse_lmmse,
                        checksum);

    printf("TPB_LS                       = %d\n", TPB_LS);
    printf("TPB_EQ                       = %d\n", TPB_EQ);
    printf("LS_KERNEL_MODE               = %s\n", use_shared_ls ? "shared" : "serial");
    printf("WARMUP_ITERS                 = %d\n", warmup_iters);
    printf("TIMED_ITERS                  = %d\n", timed_iters);
    printf("LS_KERNEL_MS                 = %.6f\n", static_cast<double>(ls_ms));
    printf("LMMSE_KERNEL_MS              = %.6f\n", static_cast<double>(eq_ms));
    printf("PIPELINE_KERNEL_MS           = %.6f\n", static_cast<double>(pipeline_ms));

    free_device_buffers(&dev);
    return 0;
}
