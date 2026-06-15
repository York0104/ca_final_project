#include <cstdio>
#include <cstdint>

// ============================================================
// Part 1 - Scalar Baseline
// Pilot-based OFDM MMSE Channel Equalizer
//
// Fixed pilot version:
//   Xpilot[p][k] = 1 + j0
//
// Therefore:
//   Ypilot[k][p] = H[k] + N[k][p]
//
// Channel estimation:
//   H_hat[k] = sum_p Ypilot[k][p] * w[p]
//
// MMSE equalization:
//   X_hat[s][k] = Ydata[s][k] * conj(H_hat[k])
//                 / (|H_hat[k]|^2 + NOISE_VAR)
//
// This is a scalar C++ baseline.
// The channel estimation loop is the main nested-loop reduction kernel.
// ============================================================

static constexpr int NUM_SUBCARRIERS = 512;
static constexpr int NUM_PILOTS = 256;
static constexpr int NUM_DATA_SYMBOLS = 512;

static constexpr int TOTAL_PILOT = NUM_SUBCARRIERS * NUM_PILOTS;
static constexpr int TOTAL_DATA = NUM_SUBCARRIERS * NUM_DATA_SYMBOLS;

static constexpr float NOISE_VAR = 0.0025f;
static constexpr float EPSILON = 1.0e-6f;

alignas(64) static float Ypilot_r[TOTAL_PILOT];
alignas(64) static float Ypilot_i[TOTAL_PILOT];
alignas(64) static float pilot_w[NUM_PILOTS];

alignas(64) static float Xdata_r[TOTAL_DATA];
alignas(64) static float Xdata_i[TOTAL_DATA];
alignas(64) static float Ydata_r[TOTAL_DATA];
alignas(64) static float Ydata_i[TOTAL_DATA];

alignas(64) static float Htrue_r[NUM_SUBCARRIERS];
alignas(64) static float Htrue_i[NUM_SUBCARRIERS];
alignas(64) static float Hhat_r[NUM_SUBCARRIERS];
alignas(64) static float Hhat_i[NUM_SUBCARRIERS];

alignas(64) static float Xhat_r[TOTAL_DATA];
alignas(64) static float Xhat_i[TOTAL_DATA];

static volatile float g_sink = 0.0f;

static inline int pilot_index(int k, int p)
{
    return k * NUM_PILOTS + p;
}

static inline int data_index(int s, int k)
{
    return s * NUM_SUBCARRIERS + k;
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

static inline void make_qpsk_symbol(int a, int b, float *out_r, float *out_i)
{
    int pattern = (a * 17 + b * 31 + 7) & 3;
    *out_r = (pattern & 1) ? 1.0f : -1.0f;
    *out_i = (pattern & 2) ? 1.0f : -1.0f;
}

static inline void complex_mul(float ar, float ai,
                               float br, float bi,
                               float *cr, float *ci)
{
    *cr = ar * br - ai * bi;
    *ci = ar * bi + ai * br;
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

static void init_pilots_and_data()
{
    const float noise_scale = 0.070f;

    for (int p = 0; p < NUM_PILOTS; ++p)
    {
        for (int k = 0; k < NUM_SUBCARRIERS; ++k)
        {
            int idx = pilot_index(k, p);

            // Fixed pilot: Xpilot[p][k] = 1 + j0
            // So Ypilot[k][p] = H[k] + noise.
            Ypilot_r[idx] = Htrue_r[k] + pseudo_noise(p, k, 11, noise_scale);
            Ypilot_i[idx] = Htrue_i[k] + pseudo_noise(p, k, 23, noise_scale);
        }
    }

    for (int s = 0; s < NUM_DATA_SYMBOLS; ++s)
    {
        for (int k = 0; k < NUM_SUBCARRIERS; ++k)
        {
            int idx = data_index(s, k);

            float xr, xi;
            make_qpsk_symbol(s + 100, k + 200, &xr, &xi);

            Xdata_r[idx] = xr;
            Xdata_i[idx] = xi;

            float yr, yi;
            complex_mul(Htrue_r[k], Htrue_i[k], xr, xi, &yr, &yi);

            Ydata_r[idx] = yr + pseudo_noise(s, k, 37, noise_scale);
            Ydata_i[idx] = yi + pseudo_noise(s, k, 41, noise_scale);

            Xhat_r[idx] = 0.0f;
            Xhat_i[idx] = 0.0f;
        }
    }
}

// Main nested-loop reduction kernel:
// For each subcarrier k, compute a weighted reduction over all pilot observations.
static void estimate_channel_scalar()
{
    for (int k = 0; k < NUM_SUBCARRIERS; ++k)
    {
        float acc_r = 0.0f;
        float acc_i = 0.0f;

        for (int p = 0; p < NUM_PILOTS; ++p)
        {
            int idx = pilot_index(k, p);
            float w = pilot_w[p];
            acc_r += Ypilot_r[idx] * w;
            acc_i += Ypilot_i[idx] * w;
        }

        Hhat_r[k] = acc_r;
        Hhat_i[k] = acc_i;
    }
}

static void equalize_mmse_scalar()
{
    for (int s = 0; s < NUM_DATA_SYMBOLS; ++s)
    {
        for (int k = 0; k < NUM_SUBCARRIERS; ++k)
        {
            int idx = data_index(s, k);

            float yr = Ydata_r[idx];
            float yi = Ydata_i[idx];

            float hr = Hhat_r[k];
            float hi = Hhat_i[k];

            float denom = hr * hr + hi * hi + NOISE_VAR + EPSILON;

            Xhat_r[idx] = (yr * hr + yi * hi) / denom;
            Xhat_i[idx] = (yi * hr - yr * hi) / denom;
        }
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

static float compute_received_mse()
{
    float acc = 0.0f;

    for (int idx = 0; idx < TOTAL_DATA; ++idx)
    {
        float er = Ydata_r[idx] - Xdata_r[idx];
        float ei = Ydata_i[idx] - Xdata_i[idx];
        acc += er * er + ei * ei;
    }

    return acc / static_cast<float>(TOTAL_DATA);
}

static float compute_mmse_mse()
{
    float acc = 0.0f;

    for (int idx = 0; idx < TOTAL_DATA; ++idx)
    {
        float er = Xhat_r[idx] - Xdata_r[idx];
        float ei = Xhat_i[idx] - Xdata_i[idx];
        acc += er * er + ei * ei;
    }

    return acc / static_cast<float>(TOTAL_DATA);
}

static float checksum_results()
{
    float sum = 0.0f;

    for (int k = 0; k < NUM_SUBCARRIERS; ++k)
    {
        sum += Hhat_r[k] + Hhat_i[k];
    }

    for (int idx = 0; idx < TOTAL_DATA; ++idx)
    {
        sum += 0.5f * Xhat_r[idx] + 0.5f * Xhat_i[idx];
    }

    return sum;
}

int main()
{
    init_channel();
    init_pilots_and_data();

    estimate_channel_scalar();
    equalize_mmse_scalar();

    float h_mse = compute_channel_mse();
    float mse_rx = compute_received_mse();
    float mse_mmse = compute_mmse_mse();
    float checksum = checksum_results();

    g_sink = h_mse + mse_rx + mse_mmse + checksum + Xhat_r[0] + Xhat_i[TOTAL_DATA - 1];

    std::printf("Part 1 - Scalar Baseline: Pilot-based OFDM MMSE Channel Equalizer\n");
    std::printf("NUM_SUBCARRIERS  = %d\n", NUM_SUBCARRIERS);
    std::printf("NUM_PILOTS       = %d\n", NUM_PILOTS);
    std::printf("NUM_DATA_SYMBOLS = %d\n", NUM_DATA_SYMBOLS);
    std::printf("TOTAL_DATA       = %d\n", TOTAL_DATA);
    std::printf("NOISE_VAR        = %.8f\n", static_cast<double>(NOISE_VAR));

    std::printf("H_MSE            = %.8f\n", static_cast<double>(h_mse));
    std::printf("MSE_RX           = %.8f\n", static_cast<double>(mse_rx));
    std::printf("MSE_MMSE         = %.8f\n", static_cast<double>(mse_mmse));

    std::printf("Xhat[0]          = %.8f + j%.8f\n",
                static_cast<double>(Xhat_r[0]),
                static_cast<double>(Xhat_i[0]));

    std::printf("checksum         = %.8f\n", static_cast<double>(checksum));
    std::printf("guard            = %.8f\n", static_cast<double>(g_sink));

    if ((mse_mmse < mse_rx) &&
        (h_mse < 0.01f) &&
        (abs_float(checksum) > 1.0e-6f))
    {
        std::printf("Verification: PASS\n");
    }
    else
    {
        std::printf("Verification: FAIL\n");
    }

    return 0;
}
