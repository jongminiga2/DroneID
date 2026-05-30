/* remove_turbo.cpp — DJI DroneID LTE Turbo decoder (standalone C++11)
 *
 * No external library dependencies.
 * Compiles and runs on x86-64, ARM (32/64-bit), and any C++11 platform.
 *
 * Algorithm: max-log MAP (BCJR) turbo decoder with 6 iterations.
 * LTE RSC constituent code: g0 = 1+D^2+D^3 (feedback), g1 = 1+D+D^3 (forward).
 *
 * Build:
 *   g++ -O2 -std=c++11 remove_turbo.cpp -o remove_turbo
 *   arm-linux-gnueabihf-g++ -O2 -std=c++11 remove_turbo.cpp -o remove_turbo
 *   aarch64-linux-gnu-g++ -O2 -std=c++11 remove_turbo.cpp -o remove_turbo
 *   cl /O2 /std:c++14 remove_turbo.cpp /Fe:remove_turbo.exe
 *
 * Usage: remove_turbo <bits_file>
 *   bits_file : exactly 7200 bytes, each 0x00 or 0x01 (hard-decision bits)
 *   stdout    : hex-encoded decoded DroneID frame (176 bytes)
 */

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

// ============================================================================
// Constants
// ============================================================================

static const int E_BITS    = 7200;   // rate-matched input bit count
static const int TURBO_D   = 1412;   // per-stream coded length (K + 4 tail bits)
static const int TURBO_K   = 1408;   // turbo decoder output bits
static const int PAYLOAD_B = 176;    // TURBO_K / 8
static const int N_STREAMS = 3;
static const int N_STATES  = 8;      // 2^(k-1), k=4 constraint length
static const int N_ITER    = 6;      // turbo iterations

// Rate-matching sub-block interleaver (3GPP TS 36.212 §5.1.4.1)
static const int  C_TC  = 32;
static const float DMVAL = 1e9f;

// Column permutation (3GPP TS 36.212 Table 5.1.4-1)
static const int IC_PERM[32] = {
     0,16, 8,24, 4,20,12,28, 2,18,10,26, 6,22,14,30,
     1,17, 9,25, 5,21,13,29, 3,19,11,27, 7,23,15,31
};

// 3GPP TS 36.212 Table 5.1.3-3: turbo internal interleaver (188 entries)
static const int K_TBL[188] = {
    40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,
    168,176,184,192,200,208,216,224,232,240,248,256,264,272,
    280,288,296,304,312,320,328,336,344,352,360,368,376,384,
    392,400,408,416,424,432,440,448,456,464,472,480,488,496,
    504,512,528,544,560,576,592,608,624,640,656,672,688,704,
    720,736,752,768,784,800,816,832,848,864,880,896,912,928,
    944,960,976,992,1008,1024,1056,1088,1120,1152,1184,1216,
    1248,1280,1312,1344,1376,1408,1440,1472,1504,1536,1568,
    1600,1632,1664,1696,1728,1760,1792,1824,1856,1888,1920,
    1952,1984,2016,2048,2112,2176,2240,2304,2368,2432,2496,
    2560,2624,2688,2752,2816,2880,2944,3008,3072,3136,3200,
    3264,3328,3392,3456,3520,3584,3648,3712,3776,3840,3904,
    3968,4032,4096,4160,4224,4288,4352,4416,4480,4544,4608,
    4672,4736,4800,4864,4928,4992,5056,5120,5184,5248,5312,
    5376,5440,5504,5568,5632,5696,5760,5824,5888,5952,6016,
    6080,6144
};
static const int F1_TBL[188] = {
    3,7,19,7,7,11,5,11,7,41,103,15,9,17,9,21,101,21,57,23,
    13,27,11,27,85,29,33,15,17,33,103,19,19,37,19,21,21,115,
    193,21,133,81,45,23,243,151,155,25,51,47,91,29,29,247,29,
    89,91,157,55,31,17,35,227,65,19,37,41,39,185,43,21,155,79,
    139,23,217,25,17,127,25,239,17,137,215,29,15,147,29,59,65,
    55,31,17,171,67,35,19,39,19,199,21,211,21,43,149,45,49,71,
    13,17,25,183,55,127,27,29,29,57,45,31,59,185,113,31,17,171,
    209,253,367,265,181,39,27,127,143,43,29,45,157,47,13,111,
    443,51,51,451,257,57,313,271,179,331,363,375,127,31,33,43,
    33,477,35,233,357,337,37,71,71,37,39,127,39,39,31,113,41,
    251,43,21,43,45,45,161,89,323,47,23,47,263
};
static const int F2_TBL[188] = {
    10,12,42,16,18,20,22,24,26,84,90,32,34,108,38,120,84,44,
    46,48,50,52,36,56,58,60,62,32,198,68,210,36,74,76,78,120,
    82,84,86,44,90,46,94,48,98,40,102,52,106,72,110,168,114,
    58,118,180,122,62,84,64,66,68,420,96,74,76,234,80,82,252,
    86,44,120,92,94,48,98,80,102,52,106,48,110,112,114,58,118,
    60,122,124,84,64,66,204,140,72,74,76,78,240,82,252,86,88,
    60,92,846,48,28,80,102,104,954,96,110,112,114,116,354,120,
    610,124,420,64,66,136,420,216,444,456,468,80,164,504,172,
    88,300,92,188,96,28,240,204,104,212,192,220,336,228,232,
    236,120,244,248,168,64,130,264,134,408,138,280,142,480,146,
    444,120,152,462,234,158,80,96,902,166,336,170,86,174,176,
    178,120,182,184,186,94,190,480
};

// ============================================================================
// CRC-24A  (3GPP TS 36.212, polynomial 0x864CFB, init=0)
// crc24a(data || CRC_bytes) == 0  ↔  valid frame
// ============================================================================

static uint32_t crc24a(const uint8_t *data, size_t len)
{
    uint32_t crc = 0;
    for (size_t i = 0; i < len; ++i) {
        crc ^= (uint32_t)data[i] << 16;
        for (int b = 0; b < 8; ++b) {
            crc <<= 1;
            if (crc & 0x01000000u) crc ^= 0x864CFBu;
        }
    }
    return crc & 0x00FFFFFFu;
}

// ============================================================================
// LTE turbo internal interleaver  (3GPP TS 36.212 §5.1.3.2.3)
// ============================================================================

static void get_f1_f2(int Kval, int &f1, int &f2)
{
    for (int i = 0; i < 188; ++i) {
        if (K_TBL[i] == Kval) { f1 = F1_TBL[i]; f2 = F2_TBL[i]; return; }
    }
    f1 = f2 = 0;
}

// pi(n) = (f1*n + f2*n^2) % K  — LTE turbo internal interleaver permutation
static int lte_pi(int n, int f1, int f2, int K)
{
    return (int)(((long long)f1 * n + (long long)f2 * n * n) % K);
}

static std::vector<float> lte_interleave_f(const std::vector<float> &in, int K)
{
    int f1, f2; get_f1_f2(K, f1, f2);
    std::vector<float> out(K);
    for (int n = 0; n < K; ++n) out[n] = in[lte_pi(n, f1, f2, K)];
    return out;
}

static std::vector<float> lte_deinterleave_f(const std::vector<float> &in, int K)
{
    int f1, f2; get_f1_f2(K, f1, f2);
    std::vector<float> out(K, 0.0f);
    for (int n = 0; n < K; ++n) out[lte_pi(n, f1, f2, K)] = in[n];
    return out;
}

// ============================================================================
// LTE RSC constituent encoder trellis
//
// State S (3 bits): S = D1 | (D2<<1) | (D3<<2)  (D1 newest)
// g0 = 1+D^2+D^3  →  feedback: v = u XOR D2 XOR D3
// g1 = 1+D  +D^3  →  parity:  d1 = v XOR D1 XOR D3
// Next state: ns = v | (D1<<1) | (D2<<2)
// ============================================================================

struct TrellisEdge {
    int ns;   // next state
    int d1;   // parity output bit
};
static TrellisEdge TRELLIS[N_STATES][2];  // [state][input_u]

static void build_trellis()
{
    for (int s = 0; s < N_STATES; ++s) {
        int D1 = (s >> 0) & 1, D2 = (s >> 1) & 1, D3 = (s >> 2) & 1;
        for (int u = 0; u < 2; ++u) {
            int v  = u ^ D2 ^ D3;
            TRELLIS[s][u].d1 = v ^ D1 ^ D3;
            TRELLIS[s][u].ns = v | (D1 << 1) | (D2 << 2);
        }
    }
}

// ============================================================================
// Max-log MAP (BCJR) decoder for one LTE RSC constituent encoder
//
// Convention: soft values positive = likely bit-0 (same as turbofec input).
// sys[0..total-1] : systematic soft LLRs  (total = K + 4 tail bits)
// par[0..total-1] : parity soft LLRs
// La[0..K-1]      : a priori LLRs (from other constituent decoder)
// K               : number of information bits
// Returns extrinsic LLR [0..K-1] in the same sign convention.
// ============================================================================

static std::vector<float> map_decode_one(
    const float *sys, const float *par,
    const std::vector<float> &La, int K)
{
    static const float NEG_INF = -1e10f;
    const int total = K + 4;  // K info bits + 4 termination tail bits

    // ── Forward pass (alpha) ─────────────────────────────────────────────────
    // alpha[t][s] stored as two alternating rows to save memory
    std::vector<float> alpha_cur(N_STATES, NEG_INF), alpha_nxt(N_STATES, NEG_INF);
    // Store ALL alpha values for backward pass combination
    std::vector<std::vector<float>> alpha_all(total + 1, std::vector<float>(N_STATES, NEG_INF));
    alpha_cur[0] = 0.0f;
    alpha_all[0] = alpha_cur;

    for (int t = 0; t < total; ++t) {
        std::fill(alpha_nxt.begin(), alpha_nxt.end(), NEG_INF);
        float la = (t < K) ? La[t] : 0.0f;
        for (int s = 0; s < N_STATES; ++s) {
            if (alpha_cur[s] == NEG_INF) continue;
            for (int u = 0; u < 2; ++u) {
                const auto &e = TRELLIS[s][u];
                // Branch metric: (sys + La) * (1-2u)/2 + par * (1-2d1)/2
                // positive soft → bit 0 → (1-2*0)=+1 rewards u=0 paths
                float bm = (sys[t] + la) * (1 - 2*u) * 0.5f
                         + par[t]         * (1 - 2*e.d1) * 0.5f;
                float a = alpha_cur[s] + bm;
                if (a > alpha_nxt[e.ns]) alpha_nxt[e.ns] = a;
            }
        }
        alpha_cur = alpha_nxt;
        alpha_all[t + 1] = alpha_cur;
    }

    // ── Backward pass (beta) + soft output ───────────────────────────────────
    std::vector<float> beta(N_STATES, NEG_INF);
    beta[0] = 0.0f;  // terminated to state 0

    std::vector<float> extrinsic(K);

    for (int t = total - 1; t >= 0; --t) {
        float la = (t < K) ? La[t] : 0.0f;

        // Compute soft output at time t (information bits only)
        if (t < K) {
            float max0 = NEG_INF, max1 = NEG_INF;
            for (int s = 0; s < N_STATES; ++s) {
                if (alpha_all[t][s] == NEG_INF) continue;
                for (int u = 0; u < 2; ++u) {
                    const auto &e = TRELLIS[s][u];
                    if (beta[e.ns] == NEG_INF) continue;
                    float bm = (sys[t] + la) * (1 - 2*u) * 0.5f
                             + par[t]         * (1 - 2*e.d1) * 0.5f;
                    float v = alpha_all[t][s] + bm + beta[e.ns];
                    if (u == 0) { if (v > max0) max0 = v; }
                    else        { if (v > max1) max1 = v; }
                }
            }
            // L_out = max0 - max1  (positive = bit 0)
            // Extrinsic = L_out - sys[t] - La[t]  (strip what was already known)
            float L_out = (max0 > NEG_INF && max1 > NEG_INF) ? (max0 - max1) :
                          (max0 > NEG_INF ? 1e4f : -1e4f);
            extrinsic[t] = L_out - sys[t] - la;
        }

        // Update beta backward
        std::vector<float> beta_new(N_STATES, NEG_INF);
        for (int s = 0; s < N_STATES; ++s) {
            for (int u = 0; u < 2; ++u) {
                const auto &e = TRELLIS[s][u];
                if (beta[e.ns] == NEG_INF) continue;
                float bm = (sys[t] + la) * (1 - 2*u) * 0.5f
                         + par[t]         * (1 - 2*e.d1) * 0.5f;
                float b = beta[e.ns] + bm;
                if (b > beta_new[s]) beta_new[s] = b;
            }
        }
        beta = beta_new;
    }

    return extrinsic;
}

// ============================================================================
// Iterative turbo decoder
//
// streams[3][TURBO_D]: de-rate-matched soft values
//   [0] = systematic + encoder-1 tail (positive = likely bit-0)
//   [1] = parity-1   + encoder-1 tail
//   [2] = parity-2   + encoder-2 tail
// Returns decoded bits [TURBO_K]
// ============================================================================

static std::vector<int> turbo_decode(
    const std::vector<std::vector<float>> &streams)
{
    const int K = TURBO_K;

    // Systematic stream for decoder 1: all K+4 entries from stream 0
    const float *sys1 = streams[0].data();
    const float *par1 = streams[1].data();
    const float *par2 = streams[2].data();

    // Systematic stream for decoder 2: interleave K info bits, pad tail with 0
    std::vector<float> sys_info(streams[0].begin(), streams[0].begin() + K);
    std::vector<float> sys2_full(K + 4, 0.0f);
    {
        auto intlv = lte_interleave_f(sys_info, K);
        for (int i = 0; i < K; ++i) sys2_full[i] = intlv[i];
        // tail bits for encoder 2 are unknown → leave as 0 (erasure)
    }

    std::vector<float> La1(K, 0.0f);  // a priori for decoder 1 (natural domain)

    for (int iter = 0; iter < N_ITER; ++iter) {
        // ── Decoder 1 (natural domain) ──
        auto e1 = map_decode_one(sys1, par1, La1, K);

        // Interleave e1 → a priori for decoder 2
        auto e1_vec = std::vector<float>(e1.begin(), e1.end());
        auto La2    = lte_interleave_f(e1_vec, K);

        // ── Decoder 2 (interleaved domain) ──
        auto e2 = map_decode_one(sys2_full.data(), par2, La2, K);

        // Deinterleave e2 → new a priori for decoder 1
        La1 = lte_deinterleave_f(e2, K);
    }

    // Final decision: run decoder 1 one last time, decide on L_out = sys + La + extrinsic
    auto e1_final = map_decode_one(sys1, par1, La1, K);

    std::vector<int> out(K);
    for (int t = 0; t < K; ++t) {
        float L_total = sys1[t] + La1[t] + e1_final[t];
        out[t] = (L_total >= 0.0f) ? 0 : 1;
    }
    return out;
}

// ============================================================================
// Rate de-matching  (3GPP TS 36.212 §5.1.4.1, rv_idx=0, single code block)
//
// soft_in[E_BITS]: received soft values (positive = likely bit-0)
// out[N_STREAMS][TURBO_D]: de-interleaved streams for turbo decoder
// ============================================================================

static void rate_dematch(const float *soft_in,
                         std::vector<std::vector<float>> &out)
{
    int R_tc = 0;
    while (TURBO_D > C_TC * R_tc) ++R_tc;         // = 45
    const int N_dum = C_TC * R_tc - TURBO_D;       // = 28
    const int K_pi  = C_TC * R_tc;                 // = 1440
    const int K_w   = N_STREAMS * K_pi;            // = 4320
    const int k_0   = R_tc * 2;                    // = 90  (rv_idx = 0)

    // Build dummy-position indicator by sub-block-interleaving a structure array
    auto build_v_structure = [&](int x) -> std::vector<float> {
        std::vector<float> tmp(K_pi);
        for (int i = 0; i < K_pi; ++i) tmp[i] = (i < N_dum) ? DMVAL : 0.0f;

        std::vector<std::vector<float>> sb(R_tc, std::vector<float>(C_TC));
        for (int n = 0; n < R_tc; ++n)
            for (int m = 0; m < C_TC; ++m)
                sb[n][m] = tmp[n * C_TC + m];

        std::vector<float> v(K_pi);
        if (x == 0 || x == 1) {
            std::vector<std::vector<float>> sbp(R_tc, std::vector<float>(C_TC));
            for (int n = 0; n < R_tc; ++n)
                for (int m = 0; m < C_TC; ++m)
                    sbp[n][m] = sb[n][IC_PERM[m]];
            int idx = 0;
            for (int m = 0; m < C_TC; ++m)
                for (int n = 0; n < R_tc; ++n)
                    v[idx++] = sbp[n][m];
        } else {
            for (int n = 0; n < K_pi; ++n) {
                int pi = (IC_PERM[n / R_tc] + C_TC * (n % R_tc) + 1) % K_pi;
                v[n] = tmp[pi];
            }
        }
        return v;
    };

    std::vector<std::vector<float>> v_str(N_STREAMS);
    for (int x = 0; x < N_STREAMS; ++x) v_str[x] = build_v_structure(x);

    std::vector<float> w_dum(K_w, DMVAL);
    for (int k = 0; k < K_pi; ++k) {
        w_dum[k]              = v_str[0][k];
        w_dum[K_pi + 2*k]     = v_str[1][k];
        w_dum[K_pi + 2*k + 1] = v_str[2][k];
    }

    std::vector<float> w(K_w, DMVAL);
    int k_idx = 0, j_idx = 0;
    while (k_idx < E_BITS) {
        int pos = (k_0 + j_idx) % K_w;
        if (w_dum[pos] != DMVAL) {
            if (w[pos] == DMVAL) w[pos]  = soft_in[k_idx];
            else                  w[pos] += soft_in[k_idx];
            ++k_idx;
        }
        ++j_idx;
    }
    for (auto &val : w) if (val == DMVAL) val = 0.0f;

    for (int k = 0; k < K_pi; ++k) {
        v_str[0][k] = w[k];
        v_str[1][k] = w[K_pi + 2*k];
        v_str[2][k] = w[K_pi + 2*k + 1];
    }

    out.assign(N_STREAMS, std::vector<float>(TURBO_D, 0.0f));
    for (int x = 0; x < N_STREAMS; ++x) {
        for (auto &val : v_str[x]) if (val == DMVAL) val = 0.0f;

        if (x == 0 || x == 1) {
            std::vector<std::vector<float>> sbp(R_tc, std::vector<float>(C_TC));
            int idx = 0;
            for (int m = 0; m < C_TC; ++m)
                for (int n = 0; n < R_tc; ++n)
                    sbp[n][m] = v_str[x][idx++];
            std::vector<std::vector<float>> sb(R_tc, std::vector<float>(C_TC));
            for (int n = 0; n < R_tc; ++n)
                for (int m = 0; m < C_TC; ++m)
                    sb[n][IC_PERM[m]] = sbp[n][m];
            int real = 0;
            for (int i = N_dum; i < K_pi && real < TURBO_D; ++i)
                out[x][real++] = sb[i / C_TC][i % C_TC];
        } else {
            std::vector<float> y(K_pi, 0.0f);
            for (int n = 0; n < K_pi; ++n) {
                int pi = (IC_PERM[n / R_tc] + C_TC * (n % R_tc) + 1) % K_pi;
                y[pi] = v_str[x][n];
            }
            for (int i = 0; i < TURBO_D; ++i) out[x][i] = y[N_dum + i];
        }
    }
}

// ============================================================================
// main
// ============================================================================

static void usage(const char *prog)
{
    fprintf(stderr, "\nUsage: %s <bits_file>\n\n", prog);
    fprintf(stderr, "  bits_file  path to file containing %d bytes, "
                    "each 0x00 or 0x01\n\n", E_BITS);
    exit(1);
}

int main(int argc, const char **argv)
{
    if (argc != 2) usage(argv[0]);

    build_trellis();

    // ── Read input file ──────────────────────────────────────────────────────
    FILE *fp = fopen(argv[1], "rb");
    if (!fp) {
        fprintf(stderr, "[ERROR] Cannot open '%s'\n", argv[1]);
        usage(argv[0]);
    }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (fsize != E_BITS) {
        fprintf(stderr, "[ERROR] Expected %d bytes in '%s', got %ld\n",
                E_BITS, argv[1], fsize);
        fclose(fp); usage(argv[0]);
    }

    std::vector<uint8_t> raw(E_BITS);
    if ((long)fread(raw.data(), 1, E_BITS, fp) != E_BITS) {
        fprintf(stderr, "[ERROR] Read error on '%s'\n", argv[1]);
        fclose(fp); return 1;
    }
    fclose(fp);

    // ── Convert hard bits to soft LLRs ───────────────────────────────────────
    // Sign convention: positive = likely bit-0 (matches turbofec's soft channel)
    // bit 0 → +63, bit 1 → -63
    std::vector<float> soft(E_BITS);
    for (int i = 0; i < E_BITS; ++i) {
        if (raw[i] != 0 && raw[i] != 1) {
            fprintf(stderr, "[ERROR] Invalid bit value 0x%02x at offset %d\n",
                    raw[i], i);
            return 1;
        }
        soft[i] = (raw[i] == 0) ? 63.0f : -63.0f;
    }

    // ── Rate de-matching: 7200 → 3 × 1412 ──────────────────────────────────
    std::vector<std::vector<float>> streams;
    rate_dematch(soft.data(), streams);

    // ── Turbo decoding: 3 × 1412 → 1408 bits ────────────────────────────────
    auto decoded_bits = turbo_decode(streams);

    // ── Pack bits → bytes (MSB first) ────────────────────────────────────────
    std::vector<uint8_t> decoded_bytes(PAYLOAD_B, 0);
    for (int i = 0; i < TURBO_K; ++i)
        decoded_bytes[i / 8] |= (uint8_t)(decoded_bits[i] << (7 - (i % 8)));

    // ── CRC-24A check (residue of full 176 bytes must be 0) ──────────────────
    uint32_t crc = crc24a(decoded_bytes.data(), PAYLOAD_B);
    if (crc != 0) {
        fprintf(stderr, "[ERROR] CRC-24A failed (residue 0x%06x)\n", crc);
        return 1;
    }

    // ── Print decoded frame in hex ────────────────────────────────────────────
    for (int i = 0; i < PAYLOAD_B; ++i)
        fprintf(stdout, "%02x", decoded_bytes[i]);
    fprintf(stdout, "\n");

    return 0;
}
