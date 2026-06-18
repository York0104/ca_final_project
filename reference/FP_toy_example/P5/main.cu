#include <cstdio>
#include <cuda_runtime.h>

#define SEQ_LENGTH 500000
#define NUM_TOTAL_NODES 18

__global__
void mrbayes_cuda_nodes(
    const float *tiPL,
    const float *tiPR,
    const float *clL_all,
    const float *clR_all,
    float *clP_all,
    int seq_length
) {
    __shared__ float s_tiPL[16];
    __shared__ float s_tiPR[16];

    if (threadIdx.x < 16) {
        s_tiPL[threadIdx.x] = tiPL[threadIdx.x];
        s_tiPR[threadIdx.x] = tiPR[threadIdx.x];
    }

    __syncthreads();

    int node = blockIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k >= seq_length) return;

    int base = node * 4 * seq_length;

    const float *L = clL_all + base + 4 * k;
    const float *R = clR_all + base + 4 * k;
    float *P = clP_all + base + 4 * k;

    for (int i = 0; i < 4; i++) {
        float left  = s_tiPL[4*i+0] * L[0] + s_tiPL[4*i+1] * L[1] + s_tiPL[4*i+2] * L[2] + s_tiPL[4*i+3] * L[3];
        float right = s_tiPR[4*i+0] * R[0] + s_tiPR[4*i+1] * R[1] + s_tiPR[4*i+2] * R[2] + s_tiPR[4*i+3] * R[3];
        P[i] = left * right;
    }
}

int main() {
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

    size_t tableCount = NUM_TOTAL_NODES * 4 * SEQ_LENGTH;
    size_t tableBytes = tableCount * sizeof(float);

    float *clL = new float[tableCount];
    float *clR = new float[tableCount];
    float *clP_gpu = new float[tableCount];

    for (size_t i = 0; i < tableCount; i++){
        clL[i] = 0.000731f * ((i * 13 + 7) % 271 + 1);
        clR[i] = 0.000913f * ((i * 17 + 11) % 319 + 1);
        clP_gpu[i] = 0.0f;
    }

    float *d_tiPL, *d_tiPR, *d_clL, *d_clR, *d_clP;

    cudaMalloc(&d_tiPL, 16 * sizeof(float));
    cudaMalloc(&d_tiPR, 16 * sizeof(float));
    cudaMalloc(&d_clL, tableBytes);
    cudaMalloc(&d_clR, tableBytes);
    cudaMalloc(&d_clP, tableBytes);

    cudaMemcpy(d_tiPL, tiPL, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tiPR, tiPR, 16 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_clL, clL, tableBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_clR, clR, tableBytes, cudaMemcpyHostToDevice);
    cudaMemset(d_clP, 0, tableBytes);

    int threadsPerBlock = 128;
    int gridX = (SEQ_LENGTH + threadsPerBlock - 1) / threadsPerBlock;

    dim3 grid(gridX, NUM_TOTAL_NODES);
    dim3 block(threadsPerBlock);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    mrbayes_cuda_nodes<<<grid, block>>>(d_tiPL, d_tiPR, d_clL, d_clR, d_clP, SEQ_LENGTH);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU time = %f ms\n", ms);

    cudaMemcpy(clP_gpu, d_clP, tableBytes, cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (size_t i = 0; i < tableCount; i++){
        sum += clP_gpu[i];
    }
    printf("GPU sum: %f\n", sum);

    return 0;
}