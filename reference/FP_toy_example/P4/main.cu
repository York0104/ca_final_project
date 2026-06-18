#include <cstdio>
#include <cuda_runtime.h>

#define SEQ_LENGTH 500000

// ============================================================
// CUDA kernel
// ============================================================

__global__ void mrbayes_cuda(
    const float *tiPL,
    const float *tiPR,
    const float *clL,
    const float *clR,
    float *clP,
    int seq_length
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k >= seq_length)
        return;

    // ------------------------------------------------
    // shared memory
    // ------------------------------------------------

    __shared__ float s_tiPL[16];
    __shared__ float s_tiPR[16];

    if (threadIdx.x < 16) {
        s_tiPL[threadIdx.x] = tiPL[threadIdx.x];
        s_tiPR[threadIdx.x] = tiPR[threadIdx.x];
    }

    __syncthreads();

    const float *L = clL + 4*k;
    const float *R = clR + 4*k;
    float *P = clP + 4*k;

    for (int i = 0; i < 4; i++) {

        float left  = s_tiPL[4*i+0] * L[0] + s_tiPL[4*i+1] * L[1] + s_tiPL[4*i+2] * L[2] + s_tiPL[4*i+3] * L[3];
        float right = s_tiPR[4*i+0] * R[0] + s_tiPR[4*i+1] * R[1] + s_tiPR[4*i+2] * R[2] + s_tiPR[4*i+3] * R[3];

        P[i] = left * right;
    }
}

// ============================================================
// main
// ============================================================

int main() {
    // ------------------------------------------------
    // host memory
    // ------------------------------------------------

    float *clL = new float[4 * SEQ_LENGTH];
    float *clR = new float[4 * SEQ_LENGTH];
    float *clP_gpu = new float[4 * SEQ_LENGTH];

    // ------------------------------------------------
    // generate test pattern
    // ------------------------------------------------
    
    float tiPL[16] = {
        0.958069059f, 0.006029767f, 0.029245846f, 0.006655327f,
        0.009044650f, 0.945192414f, 0.008469012f, 0.037293924f,
        0.035095016f, 0.006775209f, 0.948784534f, 0.009345241f,
        0.007986393f, 0.029835139f, 0.009345241f, 0.952833227f
    };

    float tiPR[16] = {
        0.926617330f, 0.010718877f, 0.050752068f, 0.011911724f,
        0.016078316f, 0.904407809f, 0.015038731f, 0.064475144f,
        0.060902481f, 0.012030985f, 0.910558280f, 0.016508253f,
        0.014294069f, 0.051580115f, 0.016508253f, 0.917617562f
    };

    for (int i = 0; i < 4 * SEQ_LENGTH; i++) {
        clL[i] = 0.000731f * ((i * 13 + 7) % 271 + 1);
        clR[i] = 0.000913f * ((i * 17 + 11) % 319 + 1);
        clP_gpu[i] = 0.0f;
    }
    
    // ------------------------------------------------
    // device memory
    // ------------------------------------------------

    float *d_tiPL;
    float *d_tiPR;
    float *d_clL;
    float *d_clR;
    float *d_clP;

    cudaMalloc(&d_tiPL, 16 * sizeof(float));
    cudaMalloc(&d_tiPR, 16 * sizeof(float));
    cudaMalloc(&d_clL, 4 * SEQ_LENGTH * sizeof(float));
    cudaMalloc(&d_clR, 4 * SEQ_LENGTH * sizeof(float));
    cudaMalloc(&d_clP, 4 * SEQ_LENGTH * sizeof(float));

    // ------------------------------------------------
    // copy host -> device
    // ------------------------------------------------

    cudaMemcpy(d_tiPL, tiPL, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tiPR, tiPR, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_clL, clL, 4 * SEQ_LENGTH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_clR, clR, 4 * SEQ_LENGTH * sizeof(float), cudaMemcpyHostToDevice);

    // ------------------------------------------------
    // launch kernel
    // ------------------------------------------------

    int threadsPerBlock = 128;
    int blocks = (SEQ_LENGTH + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    mrbayes_cuda<<<blocks, threadsPerBlock>>>(d_tiPL, d_tiPR, d_clL, d_clR, d_clP, SEQ_LENGTH);

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU time = %f ms\n", ms);


    // ------------------------------------------------
    // copy device -> host
    // ------------------------------------------------

    cudaMemcpy(clP_gpu, d_clP, 4 * SEQ_LENGTH * sizeof(float), cudaMemcpyDeviceToHost);

    // ------------------------------------------------
    // verify
    // ------------------------------------------------
    float sum = 0.0f;
    for (int i = 0; i < 4 * SEQ_LENGTH; i++) {
        sum += clP_gpu[i];
    }

    printf("GPU sum = %f\n", sum);


    return 0;
}
