#include <stdio.h>
#define SEQ_LENGTH 500000

void mrbayes_rvv_across_k(
    const float *tiPL,
    const float *tiPR,
    const float *clL,
    const float *clR,
    float *clP,
    int seq_length
) {
    int k = 0;

    while (k < seq_length) {
        int remaining = seq_length - k;
        int vl = remaining < 8 ? remaining : 8;

        const float *baseL = clL + 4 * k;
        const float *baseR = clR + 4 * k;
        float *baseP = clP + 4 * k;

        for (int i = 0; i < 4; i++) {
            const float *rowPL = tiPL + 4 * i;
            const float *rowPR = tiPR + 4 * i;
            float *outP = baseP + i;

            asm volatile(
                "mv t0, %[VL]\n\t"
                "vsetvli zero, t0, e32, m1, ta, ma\n\t"

                // stride = 16 bytes = 4 floats
                "li t1, 16\n\t"

                // -----------------------------
                // Load clL 0,1,2,3 across k
                // -----------------------------
                "vlse32.v v0, (%[L]), t1\n\t"

                "addi t2, %[L], 4\n\t"
                "vlse32.v v1, (t2), t1\n\t"

                "addi t2, %[L], 8\n\t"
                "vlse32.v v2, (t2), t1\n\t"

                "addi t2, %[L], 12\n\t"
                "vlse32.v v3, (t2), t1\n\t"

                // -----------------------------
                // Load clR 0,1,2,3 across k
                // -----------------------------
                "vlse32.v v4, (%[R]), t1\n\t"

                "addi t2, %[R], 4\n\t"
                "vlse32.v v5, (t2), t1\n\t"

                "addi t2, %[R], 8\n\t"
                "vlse32.v v6, (t2), t1\n\t"

                "addi t2, %[R], 12\n\t"
                "vlse32.v v7, (t2), t1\n\t"

                // -----------------------------
                // left = rowPL[0,1,2,3] dot clL[k][0,1,2,3]
                // -----------------------------
                "flw ft0, 0(%[PL])\n\t"
                "vfmul.vf v8, v0, ft0\n\t"

                "flw ft0, 4(%[PL])\n\t"
                "vfmacc.vf v8, ft0, v1\n\t"

                "flw ft0, 8(%[PL])\n\t"
                "vfmacc.vf v8, ft0, v2\n\t"

                "flw ft0, 12(%[PL])\n\t"
                "vfmacc.vf v8, ft0, v3\n\t"

                // -----------------------------
                // right = rowPR[0,1,2,3] dot clR[k][0,1,2,3]
                // -----------------------------
                "flw ft1, 0(%[PR])\n\t"
                "vfmul.vf v9, v4, ft1\n\t"

                "flw ft1, 4(%[PR])\n\t"
                "vfmacc.vf v9, ft1, v5\n\t"

                "flw ft1, 8(%[PR])\n\t"
                "vfmacc.vf v9, ft1, v6\n\t"

                "flw ft1, 12(%[PR])\n\t"
                "vfmacc.vf v9, ft1, v7\n\t"

                // out = left * right
                "vfmul.vv v10, v8, v9\n\t"

                // store clP[k...k+vl-1][i]
                "vsse32.v v10, (%[out]), t1\n\t"

                :
                : [VL] "r"(vl), [L] "r"(baseL), [R] "r"(baseR),
                  [PL] "r"(rowPL), [PR] "r"(rowPR), [out] "r"(outP)
                : "t0", "t1", "t2", "ft0", "ft1", "memory"
            );
        }

        k += vl;
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
    mrbayes_rvv_across_k(tiPL, tiPR, clL, clR, clP, seq_length);

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
