#include <stdio.h>
#define SEQ_LENGTH 500000

void mrbayes_rvv(
    const float *tiPL,
    const float *tiPR,
    const float *clL,
    const float *clR,
    float *clP,
    int seq_length
) {
    for (int k = 0; k < seq_length; k++) {
        const float *L = clL + 4 * k;
        const float *R = clR + 4 * k;
        float *P = clP + 4 * k;

        for (int i = 0; i < 4; i++) {
            const float *rowPL = tiPL + 4 * i;
            const float *rowPR = tiPR + 4 * i;
            float *outP = P + i;

            asm volatile(
                // VL = 4, SEW = 32-bit, LMUL = 1
                "li t0, 4\n\t"
                "vsetvli zero, t0, e32, m1, ta, ma\n\t"

                // v1 = clL[k][0,1,2,3]
                // v2 = tiPL[i][0,1,2,3]
                "vle32.v v1, (%[L])\n\t"
                "vle32.v v2, (%[PL])\n\t"

                // v3 = v1 * v2
                "vfmul.vv v3, v1, v2\n\t"

                // v0 = 0.0f vector for reduction initial value
                "vmv.v.i v0, 0\n\t"

                // v4 = sum(v3)
                "vfredusum.vs v4, v3, v0\n\t"

                // v5 = clR[k][0,1,2,3]
                // v6 = tiPR[i][0,1,2,3]
                "vle32.v v5, (%[R])\n\t"
                "vle32.v v6, (%[PR])\n\t"

                // v7 = v5 * v6
                "vfmul.vv v7, v5, v6\n\t"

                // v8 = sum(v7)
                "vfredusum.vs v8, v7, v0\n\t"

                // v4 * v8 (left * right)
                "vfmul.vv v9, v4, v8\n\t"

                // ft1 = v9[0]
                "vfmv.f.s ft1, v9\n\t"
                "fsw ft1, (%[out])\n\t"

                : 
                : [L] "r"(L),
                  [PL] "r"(rowPL),
                  [R] "r"(R),
                  [PR] "r"(rowPR),
                  [out] "r"(outP)
                : "t0", "ft0", "ft1", "ft2", "ft3", "memory"
            );

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
    mrbayes_rvv(tiPL, tiPR, clL, clR, clP, seq_length);

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
