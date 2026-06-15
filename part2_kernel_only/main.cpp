#include <cstdio>
#include <cstdint>

// ============================================================
// Part 2 Kernel-only - RVV Vector Reduction
// Pilot-based OFDM MMSE Channel Equalizer
//
// Kernel:
//   H_hat[k] = sum_p Ypilot[k][p] * pilot_w[p]
//
// Only the channel estimation kernel is measured here.
// ============================================================

static constexpr int NUM_SUBCARRIERS = 512;
static constexpr int NUM_PILOTS = 256;
static constexpr int KERNEL_REPEAT = 100;

static constexpr int TOTAL_PILOT = NUM_SUBCARRIERS * NUM_PILOTS;

alignas(64) static float Ypilot_r[TOTAL_PILOT];
alignas(64) static float Ypilot_i[TOTAL_PILOT];
alignas(64) static float pilot_w[NUM_PILOTS];

alignas(64) static float Htrue_r[NUM_SUBCARRIERS];
alignas(64) static float Htrue_i[NUM_SUBCARRIERS];
alignas(64) static float Hhat_r[NUM_SUBCARRIERS];
alignas(64) static float Hhat_i[NUM_SUBCARRIERS];

static volatile float g_sink = 0.0f;

static inline int pilot_index(int k, int p)
{
    return k * NUM_PILOTS + p;
}

static inline float abs_float(float v)
{
    return (v < 0.0f) ? -v : v;
}

static inline float pseudo_noise(int a, int b, int salt, float scale)
{
    uint32_t x = static_cast<uint32_t>(a) * 1664525u + static_cast<uint32_t>(b) * 1013904223u + static_cast<uint32_t>(salt) * 747796405u + 2891336453u;

    x ^= (x >> 16);
    x *= 2246822519u;
    x ^= (x >> 13);

    int v = static_cast<int>(x & 1023u) - 512;
    return scale * static_cast<float>(v) / 512.0f;
}

static void init_channel()
{
    for (int k = 0; k < NUM_SUBCARRIERS; ++k)
    {
        float mag = 0.55f + 0.004f * static_cast<float>((k * 37) % 100);

        if ((k % 64) == 0 || (k % 64) == 1 || (k % 64) == 2)
        {
            mag = 0.14f + 0.015f * static_cast<float>(k % 3);
        }

        float shape_r = 0.85f + 0.012f * static_cast<float>(k % 9);
        float shape_i = 0.20f - 0.010f * static_cast<float>(k % 7);

        Htrue_r[k] = mag * shape_r;
        Htrue_i[k] = mag * shape_i;

        Hhat_r[k] = 0.0f;
        Hhat_i[k] = 0.0f;
    }

    for (int p = 0; p < NUM_PILOTS; ++p)
    {
        pilot_w[p] = 1.0f / static_cast<float>(NUM_PILOTS);
    }
}

static void init_pilots_only()
{
    const float noise_scale = 0.070f;

    for (int p = 0; p < NUM_PILOTS; ++p)
    {
        for (int k = 0; k < NUM_SUBCARRIERS; ++k)
        {
            int idx = pilot_index(k, p);
            Ypilot_r[idx] = Htrue_r[k] + pseudo_noise(p, k, 11, noise_scale);
            Ypilot_i[idx] = Htrue_i[k] + pseudo_noise(p, k, 23, noise_scale);
        }
    }
}

static inline float rvv_dot_product_reduction_f32(const float *a,
                                                  const float *b,
                                                  int n)
{
#if defined(__riscv) && defined(__riscv_vector)
    float total = 0.0f;
    int remaining = n;

    while (remaining > 0)
    {
        int vl = 0;
        float partial = 0.0f;

        asm volatile(
            "vsetvli %[vl], %[remaining], e32, m1, ta, ma\n\t"
            "vle32.v v1, (%[a])\n\t"
            "vle32.v v2, (%[b])\n\t"
            "vfmul.vv v3, v1, v2\n\t"
            "vmv.v.i v0, 0\n\t"
            "vfredusum.vs v4, v3, v0\n\t"
            "vfmv.f.s ft0, v4\n\t"
            "fsw ft0, (%[partial])\n\t"
            : [vl] "=r"(vl)
            : [a] "r"(a),
              [b] "r"(b),
              [remaining] "r"(remaining),
              [partial] "r"(&partial)
            : "ft0", "memory"
        );

        total += partial;
        a += vl;
        b += vl;
        remaining -= vl;
    }

    return total;
#else
    float total = 0.0f;

    for (int i = 0; i < n; ++i)
    {
        total += a[i] * b[i];
    }

    return total;
#endif
}

static void estimate_channel_rvv_reduction()
{
    for (int k = 0; k < NUM_SUBCARRIERS; ++k)
    {
        const float *yr = &Ypilot_r[pilot_index(k, 0)];
        const float *yi = &Ypilot_i[pilot_index(k, 0)];

        Hhat_r[k] = rvv_dot_product_reduction_f32(yr, pilot_w, NUM_PILOTS);
        Hhat_i[k] = rvv_dot_product_reduction_f32(yi, pilot_w, NUM_PILOTS);
    }
}

static float compute_channel_mse()
{
    float acc = 0.0f;

    for (int k = 0; k < NUM_SUBCARRIERS; ++k)
    {
        float er = Hhat_r[k] - Htrue_r[k];
        float ei = Hhat_i[k] - Htrue_i[k];
        acc += er * er + ei * ei;
    }

    return acc / static_cast<float>(NUM_SUBCARRIERS);
}

static float checksum_Hhat_only()
{
    float sum = 0.0f;

    for (int k = 0; k < NUM_SUBCARRIERS; ++k)
    {
        sum += Hhat_r[k] + Hhat_i[k];
    }

    return sum;
}

int main()
{
    init_channel();
    init_pilots_only();

    for (int repeat = 0; repeat < KERNEL_REPEAT; ++repeat)
    {
        asm volatile("" ::: "memory");
        estimate_channel_rvv_reduction();
        g_sink += Hhat_i[repeat & (NUM_SUBCARRIERS - 1)] * 1.0e-6f;
    }

    float h_mse = compute_channel_mse();
    float checksum = checksum_Hhat_only();

    g_sink += h_mse + checksum + Hhat_r[0] + Hhat_i[NUM_SUBCARRIERS - 1];

    std::printf("Part 2 Kernel-only - RVV Vector Reduction\n");
    std::printf("NUM_SUBCARRIERS  = %d\n", NUM_SUBCARRIERS);
    std::printf("NUM_PILOTS       = %d\n", NUM_PILOTS);
    std::printf("KERNEL_REPEAT    = %d\n", KERNEL_REPEAT);
    std::printf("H_MSE            = %.8f\n", static_cast<double>(h_mse));
    std::printf("checksum_Hhat    = %.8f\n", static_cast<double>(checksum));
    std::printf("guard            = %.8f\n", static_cast<double>(g_sink));

    if ((h_mse < 0.01f) && (abs_float(checksum) > 1.0e-6f))
    {
        std::printf("Verification: PASS\n");
    }
    else
    {
        std::printf("Verification: FAIL\n");
    }

    return 0;
}
