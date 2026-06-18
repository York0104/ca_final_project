#include <stdio.h>
#include <chrono>
#define SEQ_LENGTH 500000

void mrbayes_cpu(
    const float *tiPL,
    const float *tiPR,
    const float *clL,
    const float *clR,
    float *clP,
    int seq_length
) {
    for (int k = 0; k < seq_length; k++) {
        const float *L = clL + 4*k;
        const float *R = clR + 4*k;
        float *P = clP + 4*k;

        for (int i = 0; i < 4; i++) {
            float left  = tiPL[4*i+0] * L[0] + tiPL[4*i+1] * L[1] + tiPL[4*i+2] * L[2] + tiPL[4*i+3] * L[3];
            float right = tiPR[4*i+0] * R[0] + tiPR[4*i+1] * R[1] + tiPR[4*i+2] * R[2] + tiPR[4*i+3] * R[3];
            P[i] = left * right;
        }
    }
}

int main() {
    const int seq_length = SEQ_LENGTH;

    // -------------------
    //     Test data
    // -------------------
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

    float *clL = new float [4 * seq_length];
    float *clR = new float [4 * seq_length];
    float *clP = new float [4 * seq_length];

    for (int i = 0; i < 4 * seq_length; i++) {
        clL[i] = 0.000731f * ((i * 13 + 7) % 271 + 1);
        clR[i] = 0.000913f * ((i * 17 + 11) % 319 + 1);
        clP[i] = 0.0f;
    }

    // -------------------
    //    Computation  
    // -------------------
    auto st = std::chrono::high_resolution_clock::now();
    mrbayes_cpu(tiPL, tiPR, clL, clR, clP, seq_length);
    auto ed = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = ed - st;
    printf("CPU time: %f ms\n", elapsed.count() * 1000);

    // -------------------
    //     Verify
    // -------------------
    float sum = 0.0f;
    for (int i = 0; i < 4 * seq_length; i++) {
        sum += clP[i];
    }
    printf("sum = %f\n", sum);


    return 0;
}


