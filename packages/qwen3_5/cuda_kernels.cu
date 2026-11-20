/* Qwen3.5 CUDA kernels. sm_70+ (Tesla V100). fp32 accumulate. No bf16, no
   cp.async, no ldmatrix (that is sm_75). Prefill GEMM (T>1) can use Volta
   WMMA 16x16x16. Decode GEMV (T==1) has a bandwidth path: 128-bit loads, dp4a against an
   activation quantized to q8. Tall matrices use four rows per warp; short
   ones use gemv2r1_* (one row per warp). The original one-warp GEMV stays
   compiled for ODINFER_CUDA_GEMV=old. g_decode_pos is the token index when
   a kernel is launched with pos=-1 (CUDA graph replay).
   NVRTC does not ship mma.h or cuda_fp16.h, so WMMA and dp4a are inline
   PTX. NVRTC is given --gpu-architecture=compute_XX (CUDA 13 NVRTC
   defaults to sm_75 and will not JIT on sm_70).

   Quant layouts match packages/ggml/quant.odin. GEMV is one warp per output
   row. MMQ reuses the same accumulator over a tile of up to 8 tokens and
   writes token-major [T, rows]. */

/* Half conversion is inline so NVRTC does not need cuda_fp16.h. The pip
   wheel (nvidia-cuda-nvrtc) does not ship that header, and CUDA 12.9's
   libnvrtc-builtins does not register it as a builtin include. Round-to-
   nearest-even matches CUDA __float2half and Odin's f16 cast. */

/* Engine decode stores the token position here and passes pos=-1 so a CUDA
   graph can replay every token. A non-negative argument is the position.
   extern "C" keeps the PTX name `g_decode_pos` (no C++ mangling). NVRTC emits
   a plain `.global`; the host rewrites it to `.visible .global` and links
   the PTX so cuModuleGetGlobal_v2 can see that name. volatile: the host
   writes the symbol between replays. pos_slot is the fallback device int
   when the driver still will not export the symbol (error 500). */
extern "C" {
__device__ volatile int g_decode_pos;
}

__device__ __forceinline__ int live_pos(int pos, const int* pos_slot) {
    if (pos >= 0) return pos;
    if (pos_slot != 0) {
        int v;
        asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(v) : "l"(pos_slot));
        return v;
    }
    return g_decode_pos;
}

__device__ __forceinline__ float ld_f16(const unsigned char* p) {
    unsigned short u = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
    unsigned sign = ((unsigned)u >> 15) & 1u;
    unsigned exp = ((unsigned)u >> 10) & 0x1fu;
    unsigned mant = (unsigned)u & 0x3ffu;
    unsigned out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign << 31;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3ffu;
            out = (sign << 31) | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = (sign << 31) | (0xffu << 23) | (mant << 13);
    } else {
        out = (sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    return __uint_as_float(out);
}

__device__ __forceinline__ void st_f16(unsigned char* p, float v) {
    unsigned x = __float_as_uint(v);
    unsigned sign = (x >> 16) & 0x8000u;
    unsigned exp_f = (x >> 23) & 0xffu;
    unsigned mant = x & 0x7fffffu;
    unsigned short h;
    if (exp_f == 0xffu) {
        h = (unsigned short)(sign | 0x7c00u | (mant ? 0x200u : 0u));
    } else if (exp_f == 0) {
        h = (unsigned short)sign;
    } else {
        int exp = (int)exp_f - 127 + 15;
        if (exp >= 31) {
            h = (unsigned short)(sign | 0x7c00u);
        } else if (exp <= 0) {
            if (exp < -10) {
                h = (unsigned short)sign;
            } else {
                mant |= 0x800000u;
                int shift = 14 - exp;
                unsigned t = mant >> shift;
                unsigned rem = mant & ((1u << shift) - 1u);
                unsigned halfway = 1u << (shift - 1);
                if (rem > halfway || (rem == halfway && (t & 1u))) t++;
                h = (unsigned short)(sign | t);
            }
        } else {
            unsigned t = ((unsigned)exp << 10) | (mant >> 13);
            unsigned rem = mant & 0x1fffu;
            if (rem > 0x1000u || (rem == 0x1000u && (t & 1u))) t++;
            if (t >= 0x7c00u) h = (unsigned short)(sign | 0x7c00u);
            else h = (unsigned short)(sign | t);
        }
    }
    p[0] = (unsigned char)(h & 0xff);
    p[1] = (unsigned char)(h >> 8);
}

__device__ __forceinline__ float ld_f32(const unsigned char* p) {
    unsigned u = (unsigned)p[0] | ((unsigned)p[1] << 8) | ((unsigned)p[2] << 16) | ((unsigned)p[3] << 24);
    return __uint_as_float(u);
}

__device__ __forceinline__ unsigned ld_u32(const unsigned char* p) {
    return (unsigned)p[0] | ((unsigned)p[1] << 8) | ((unsigned)p[2] << 16) | ((unsigned)p[3] << 24);
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

/* blockDim.x must be 128. Returns the sum to every thread. */
__device__ float block_sum_128(float v) {
    __shared__ float sm[128];
    int tid = threadIdx.x;
    sm[tid] = v;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    float r = sm[0];
    __syncthreads();
    return r;
}

__device__ float block_max_128(float v) {
    __shared__ float sm[128];
    int tid = threadIdx.x;
    sm[tid] = v;
    __syncthreads();
    for (int s = 64; s > 0; s >>= 1) {
        if (tid < s) sm[tid] = fmaxf(sm[tid], sm[tid + s]);
        __syncthreads();
    }
    float r = sm[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float sigmoid_stable(float x) {
    if (x >= 0.f) {
        float z = expf(-x);
        return 1.f / (1.f + z);
    }
    float z = expf(x);
    return z / (1.f + z);
}

__device__ __forceinline__ float softplus_stable(float x) {
    if (x > 20.f) return x;
    if (x < -20.f) return 0.f;
    return logf(1.f + expf(x));
}

__device__ void scale_min_k4(int j, const unsigned char* q, int* sc, int* m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

/* ---- per-kind accumulators. lane is 0..31. acc[0..nt) is this lane's partial. ---- */

__device__ void acc_f32(const unsigned char* wb, int row, const float* x, int n, int nt, int lane, float* acc) {
    const unsigned char* wr = wb + (size_t)row * (size_t)n * 4;
    for (int j = lane; j < n; j += 32) {
        float wv = ld_f32(wr + (size_t)j * 4);
        for (int a = 0; a < nt; ++a) acc[a] += wv * x[(size_t)a * (size_t)n + j];
    }
}

__device__ void acc_f16(const unsigned char* wb, int row, const float* x, int n, int nt, int lane, float* acc) {
    const unsigned char* wr = wb + (size_t)row * (size_t)n * 2;
    for (int j = lane; j < n; j += 32) {
        float wv = ld_f16(wr + (size_t)j * 2);
        for (int a = 0; a < nt; ++a) acc[a] += wv * x[(size_t)a * (size_t)n + j];
    }
}

__device__ void acc_q8_0(const unsigned char* wb, int row, const float* x, int n, int nt, int lane, float* acc) {
    int nb = n >> 5;
    const unsigned char* wr = wb + (size_t)row * (size_t)nb * 34;
    for (int b = 0; b < nb; ++b) {
        const unsigned char* blk = wr + (size_t)b * 34;
        float d = ld_f16(blk);
        float wv = d * (float)(signed char)blk[2 + lane];
        int k = (b << 5) + lane;
        for (int a = 0; a < nt; ++a) acc[a] += wv * x[(size_t)a * (size_t)n + k];
    }
}

__device__ void acc_q4_0(const unsigned char* wb, int row, const float* x, int n, int nt, int lane, float* acc) {
    int nb = n >> 5;
    const unsigned char* wr = wb + (size_t)row * (size_t)nb * 18;
    int j = lane & 15;
    int hi = lane >> 4;
    for (int b = 0; b < nb; ++b) {
        const unsigned char* blk = wr + (size_t)b * 18;
        float d = ld_f16(blk);
        unsigned char q = blk[2 + j];
        int qv = hi ? (int)(q >> 4) - 8 : (int)(q & 0xF) - 8;
        int k = (b << 5) + j + hi * 16;
        float wv = d * (float)qv;
        for (int a = 0; a < nt; ++a) acc[a] += wv * x[(size_t)a * (size_t)n + k];
    }
}

__device__ void acc_q4_1(const unsigned char* wb, int row, const float* x, int n, int nt, int lane, float* acc) {
    int nb = n >> 5;
    const unsigned char* wr = wb + (size_t)row * (size_t)nb * 20;
    int j = lane & 15;
    int hi = lane >> 4;
    for (int b = 0; b < nb; ++b) {
        const unsigned char* blk = wr + (size_t)b * 20;
        float d = ld_f16(blk);
        float m = ld_f16(blk + 2);
        unsigned char q = blk[4 + j];
        int qv = hi ? (int)(q >> 4) : (int)(q & 0xF);
        int k = (b << 5) + j + hi * 16;
        float wv = d * (float)qv + m;
        for (int a = 0; a < nt; ++a) acc[a] += wv * x[(size_t)a * (size_t)n + k];
    }
}

__device__ void acc_q5_0(const unsigned char* wb, int row, const float* x, int n, int nt, int lane, float* acc) {
    int nb = n >> 5;
    const unsigned char* wr = wb + (size_t)row * (size_t)nb * 22;
    int j = lane & 15;
    int hi = lane >> 4;
    for (int b = 0; b < nb; ++b) {
        const unsigned char* blk = wr + (size_t)b * 22;
        float d = ld_f16(blk);
        unsigned qh = ld_u32(blk + 2);
        unsigned char qs = blk[6 + j];
        int qv;
        if (hi == 0) {
            unsigned xh = ((qh >> j) << 4) & 0x10;
            qv = (int)((qs & 0xF) | xh) - 16;
        } else {
            unsigned xh = (qh >> (j + 12)) & 0x10;
            qv = (int)((qs >> 4) | xh) - 16;
        }
        int k = (b << 5) + j + hi * 16;
        float wv = d * (float)qv;
        for (int a = 0; a < nt; ++a) acc[a] += wv * x[(size_t)a * (size_t)n + k];
    }
}

__device__ void acc_q5_1(const unsigned char* wb, int row, const float* x, int n, int nt, int lane, float* acc) {
    int nb = n >> 5;
    const unsigned char* wr = wb + (size_t)row * (size_t)nb * 24;
    int j = lane & 15;
    int hi = lane >> 4;
    for (int b = 0; b < nb; ++b) {
        const unsigned char* blk = wr + (size_t)b * 24;
        float d = ld_f16(blk);
        float m = ld_f16(blk + 2);
        unsigned qh = ld_u32(blk + 4);
        unsigned char qs = blk[8 + j];
        int qv;
        if (hi == 0) {
            unsigned xh = ((qh >> j) << 4) & 0x10;
            qv = (int)((qs & 0xF) | xh);
        } else {
            unsigned xh = (qh >> (j + 12)) & 0x10;
            qv = (int)((qs >> 4) | xh);
        }
        int k = (b << 5) + j + hi * 16;
        float wv = d * (float)qv + m;
        for (int a = 0; a < nt; ++a) acc[a] += wv * x[(size_t)a * (size_t)n + k];
    }
}

__device__ void acc_q4_k(const unsigned char* wb, int row, const float* x, int n, int nt, int lane, float* acc) {
    int nb = n >> 8;
    const unsigned char* wr = wb + (size_t)row * (size_t)nb * 144;
    for (int b = 0; b < nb; ++b) {
        const unsigned char* blk = wr + (size_t)b * 144;
        float d = ld_f16(blk);
        float dmin = ld_f16(blk + 2);
        const unsigned char* scales = blk + 4;
        const unsigned char* qs = blk + 16;
        for (int p = 0; p < 4; ++p) {
            int sc0, m0, sc1, m1;
            scale_min_k4(p * 2, scales, &sc0, &m0);
            scale_min_k4(p * 2 + 1, scales, &sc1, &m1);
            float d1 = d * (float)sc0, mm1 = dmin * (float)m0;
            float d2 = d * (float)sc1, mm2 = dmin * (float)m1;
            unsigned char q = qs[p * 32 + lane];
            int klo = b * 256 + p * 64 + lane;
            int khi = klo + 32;
            float wl = (float)(q & 0xF) * d1 - mm1;
            float wh = (float)(q >> 4) * d2 - mm2;
            for (int a = 0; a < nt; ++a) {
                const float* xa = x + (size_t)a * (size_t)n;
                acc[a] += wl * xa[klo] + wh * xa[khi];
            }
        }
    }
}

__device__ void acc_q6_k(const unsigned char* wb, int row, const float* x, int n, int nt, int lane, float* acc) {
    int nb = n >> 8;
    const unsigned char* wr = wb + (size_t)row * (size_t)nb * 210;
    for (int b = 0; b < nb; ++b) {
        const unsigned char* blk = wr + (size_t)b * 210;
        const unsigned char* ql = blk;
        const unsigned char* qh = blk + 128;
        const unsigned char* sc = blk + 192;
        float d = ld_f16(blk + 208);
        for (int half = 0; half < 2; ++half) {
            int l = lane;
            int is = l >> 4;
            int qloff = half * 64;
            int qhoff = half * 32;
            int scoff = half * 8;
            int yoff = half * 128;
            unsigned char qh1 = qh[qhoff + l];
            int q1 = (int)((ql[qloff + l] & 0xF) | (((qh1 >> 0) & 3) << 4)) - 32;
            int q2 = (int)((ql[qloff + l + 32] & 0xF) | (((qh1 >> 2) & 3) << 4)) - 32;
            int q3 = (int)((ql[qloff + l] >> 4) | (((qh1 >> 4) & 3) << 4)) - 32;
            int q4 = (int)((ql[qloff + l + 32] >> 4) | (((qh1 >> 6) & 3) << 4)) - 32;
            float s0 = d * (float)(signed char)sc[scoff + is + 0];
            float s1 = d * (float)(signed char)sc[scoff + is + 2];
            float s2 = d * (float)(signed char)sc[scoff + is + 4];
            float s3 = d * (float)(signed char)sc[scoff + is + 6];
            int k0 = b * 256 + yoff + l;
            for (int a = 0; a < nt; ++a) {
                const float* xa = x + (size_t)a * (size_t)n;
                acc[a] += s0 * (float)q1 * xa[k0]
                        + s1 * (float)q2 * xa[k0 + 32]
                        + s2 * (float)q3 * xa[k0 + 64]
                        + s3 * (float)q4 * xa[k0 + 96];
            }
        }
    }
}

__device__ void dequant_q2k(const unsigned char* blk, float* dst) {
    float d = ld_f16(blk + 80);
    float dmin = ld_f16(blk + 82);
    int is = 0, qoff = 0, yoff = 0;
    for (int half = 0; half < 2; ++half) {
        int shift = 0;
        for (int s = 0; s < 4; ++s) {
            unsigned char sc = blk[is++];
            float dl = d * (float)(sc & 0xF);
            float ml = dmin * (float)(sc >> 4);
            for (int l = 0; l < 16; ++l)
                dst[yoff + l] = dl * (float)((blk[16 + qoff + l] >> shift) & 3) - ml;
            sc = blk[is++];
            dl = d * (float)(sc & 0xF);
            ml = dmin * (float)(sc >> 4);
            for (int l = 0; l < 16; ++l)
                dst[yoff + 16 + l] = dl * (float)((blk[16 + qoff + 16 + l] >> shift) & 3) - ml;
            shift += 2;
            yoff += 32;
        }
        qoff += 32;
    }
}

__device__ void dequant_q3k(const unsigned char* blk, float* dst) {
    const unsigned KMASK1 = 0x03030303u;
    const unsigned KMASK2 = 0x0f0f0f0fu;
    const unsigned char* hm = blk;
    const unsigned char* qs = blk + 32;
    const unsigned char* scales = blk + 96;
    float d = ld_f16(blk + 108);
    unsigned aux0 = ld_u32(scales);
    unsigned aux1 = ld_u32(scales + 4);
    unsigned tmp = ld_u32(scales + 8);
    unsigned aux2 = ((aux0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    unsigned aux3 = ((aux1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux0 = (aux0 & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
    aux1 = (aux1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    unsigned aux[4];
    aux[0] = aux0; aux[1] = aux1; aux[2] = aux2; aux[3] = aux3;
    int yoff = 0, qoff = 0, is = 0;
    unsigned m = 1;
    for (int half = 0; half < 2; ++half) {
        int shift = 0;
        for (int s = 0; s < 4; ++s) {
            int sc = (int)(signed char)((aux[is / 4] >> (8 * (is & 3))) & 0xff);
            is += 1;
            float dl = d * (float)(sc - 32);
            for (int l = 0; l < 16; ++l) {
                int q = (int)((qs[qoff + l] >> shift) & 3) - ((hm[l] & m) == 0 ? 4 : 0);
                dst[yoff + l] = dl * (float)q;
            }
            sc = (int)(signed char)((aux[is / 4] >> (8 * (is & 3))) & 0xff);
            is += 1;
            dl = d * (float)(sc - 32);
            for (int l = 0; l < 16; ++l) {
                int q = (int)((qs[qoff + 16 + l] >> shift) & 3) - ((hm[l + 16] & m) == 0 ? 4 : 0);
                dst[yoff + 16 + l] = dl * (float)q;
            }
            shift += 2;
            m <<= 1;
            yoff += 32;
        }
        qoff += 32;
    }
}

__device__ void dequant_q5k(const unsigned char* blk, float* dst) {
    float d = ld_f16(blk);
    float dmin = ld_f16(blk + 2);
    const unsigned char* scales = blk + 4;
    const unsigned char* qh = blk + 16;
    const unsigned char* ql = blk + 48;
    int yoff = 0, is = 0, qloff = 0;
    unsigned u1 = 1, u2 = 2;
    for (int p = 0; p < 4; ++p) {
        int sc0, m0, sc1, m1;
        scale_min_k4(is, scales, &sc0, &m0);
        scale_min_k4(is + 1, scales, &sc1, &m1);
        float d1 = d * (float)sc0, mm1 = dmin * (float)m0;
        float d2 = d * (float)sc1, mm2 = dmin * (float)m1;
        for (int l = 0; l < 32; ++l) {
            int v = (int)(ql[qloff + l] & 0xF) + ((qh[l] & u1) == 0 ? 0 : 16);
            dst[yoff + l] = d1 * (float)v - mm1;
        }
        for (int l = 0; l < 32; ++l) {
            int v = (int)(ql[qloff + l] >> 4) + ((qh[l] & u2) == 0 ? 0 : 16);
            dst[yoff + 32 + l] = d2 * (float)v - mm2;
        }
        qloff += 32;
        is += 2;
        u1 <<= 2;
        u2 <<= 2;
        yoff += 64;
    }
}

__device__ void dequant_q8k(const unsigned char* blk, float* dst) {
    float d = ld_f32(blk);
    for (int j = 0; j < 256; ++j) dst[j] = d * (float)(signed char)blk[4 + j];
}

__device__ void acc_dequant(const unsigned char* wb, int row, const float* x, int n, int nt, int lane, float* acc, int block_elems, int block_bytes, int kind) {
    __shared__ float w[256];
    int nb = n / block_elems;
    const unsigned char* wr = wb + (size_t)row * (size_t)nb * (size_t)block_bytes;
    for (int b = 0; b < nb; ++b) {
        if (lane == 0) {
            const unsigned char* blk = wr + (size_t)b * (size_t)block_bytes;
            if (kind == 0) dequant_q2k(blk, w);
            else if (kind == 1) dequant_q3k(blk, w);
            else if (kind == 2) dequant_q5k(blk, w);
            else dequant_q8k(blk, w);
        }
        __syncthreads();
        for (int j = lane; j < block_elems; j += 32) {
            float wv = w[j];
            int k = b * block_elems + j;
            for (int a = 0; a < nt; ++a) acc[a] += wv * x[(size_t)a * (size_t)n + k];
        }
        __syncthreads();
    }
}

__device__ void acc_q2_k(const unsigned char* wb, int row, const float* x, int n, int nt, int lane, float* acc) {
    acc_dequant(wb, row, x, n, nt, lane, acc, 256, 84, 0);
}
__device__ void acc_q3_k(const unsigned char* wb, int row, const float* x, int n, int nt, int lane, float* acc) {
    acc_dequant(wb, row, x, n, nt, lane, acc, 256, 110, 1);
}
__device__ void acc_q5_k(const unsigned char* wb, int row, const float* x, int n, int nt, int lane, float* acc) {
    acc_dequant(wb, row, x, n, nt, lane, acc, 256, 176, 2);
}
__device__ void acc_q8_k(const unsigned char* wb, int row, const float* x, int n, int nt, int lane, float* acc) {
    acc_dequant(wb, row, x, n, nt, lane, acc, 256, 292, 3);
}

#define DEF_ROW(tag) \
extern "C" __global__ void gemv_##tag(const unsigned char* wb, const float* x, float* out, int n, int d) { \
    int row = (int)blockIdx.x; \
    int lane = (int)threadIdx.x; \
    if (row >= d) return; \
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0}; \
    acc_##tag(wb, row, x, n, 1, lane, acc); \
    float s = warp_sum(acc[0]); \
    if (lane == 0) out[row] = s; \
} \
extern "C" __global__ void mmq_##tag(const unsigned char* wb, const float* x, float* out, int n, int d, int T) { \
    int row = (int)blockIdx.x; \
    int lane = (int)threadIdx.x; \
    if (row >= d) return; \
    for (int t0 = 0; t0 < T; t0 += 8) { \
        int nt = T - t0; \
        if (nt > 8) nt = 8; \
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0}; \
        acc_##tag(wb, row, x + (size_t)t0 * (size_t)n, n, nt, lane, acc); \
        float red0 = warp_sum(acc[0]); \
        float red1 = warp_sum(acc[1]); \
        float red2 = warp_sum(acc[2]); \
        float red3 = warp_sum(acc[3]); \
        float red4 = warp_sum(acc[4]); \
        float red5 = warp_sum(acc[5]); \
        float red6 = warp_sum(acc[6]); \
        float red7 = warp_sum(acc[7]); \
        if (lane == 0) { \
            float red[8] = {red0, red1, red2, red3, red4, red5, red6, red7}; \
            for (int a = 0; a < nt; ++a) out[((size_t)t0 + a) * (size_t)d + row] = red[a]; \
        } \
    } \
}

DEF_ROW(f32)
DEF_ROW(f16)
DEF_ROW(q8_0)
DEF_ROW(q4_0)
DEF_ROW(q4_1)
DEF_ROW(q5_0)
DEF_ROW(q5_1)
DEF_ROW(q4_k)
DEF_ROW(q6_k)
DEF_ROW(q2_k)
DEF_ROW(q3_k)
DEF_ROW(q5_k)
DEF_ROW(q8_k)

/* ---- elementwise / attention. block 128 where a reduction is used. ---- */

extern "C" __global__ void rmsnorm_k(const float* inp, float* outp, const float* w, int size, float eps) {
    int grp = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    const float* xi = inp + (size_t)grp * (size_t)size;
    float* oi = outp + (size_t)grp * (size_t)size;
    float ss = 0.f;
    for (int j = tid; j < size; j += 128) ss += xi[j] * xi[j];
    ss = block_sum_128(ss);
    float inv = rsqrtf(ss / (float)size + eps);
    for (int j = tid; j < size; j += 128) oi[j] = w[j] * (inv * xi[j]);
}

extern "C" __global__ void qnorm_extract(const float* qproj, float* q, const float* qn, int n, float eps) {
    int h = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    const float* src = qproj + (size_t)h * (size_t)(2 * n);
    float* dst = q + (size_t)h * (size_t)n;
    float ss = 0.f;
    for (int j = tid; j < n; j += 128) ss += src[j] * src[j];
    ss = block_sum_128(ss);
    float inv = rsqrtf(ss / (float)n + eps);
    for (int j = tid; j < n; j += 128) dst[j] = qn[j] * (inv * src[j]);
}

/* qnorm_extract, then the partial RoPE that rope_partial applies. One block
   is one head of one token (block = token * n_heads + head). rotary_dim < 2
   leaves the vector unrotated. Pairs do not overlap, so the only extra
   barrier is the one before the rotate reads another thread's store. */
extern "C" __global__ void qnorm_rope(const float* qproj, float* q, const float* qn, int n, float eps,
                                      int pos, float rope_freq, int rotary_dim, int n_heads, const int* pos_slot) {
    int h = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    const float* src = qproj + (size_t)h * (size_t)(2 * n);
    float* dst = q + (size_t)h * (size_t)n;
    float ss = 0.f;
    for (int j = tid; j < n; j += 128) ss += src[j] * src[j];
    ss = block_sum_128(ss);
    float inv = rsqrtf(ss / (float)n + eps);
    for (int j = tid; j < n; j += 128) dst[j] = qn[j] * (inv * src[j]);
    __syncthreads();
    if (rotary_dim < 2 || n_heads < 1) return;
    int half = rotary_dim >> 1;
    int token = h / n_heads;
    int p = live_pos(pos, pos_slot) + token;
    for (int i = tid; i < half; i += 128) {
        float freq = 1.f / powf(rope_freq, (float)i / (float)half);
        float ang = (float)p * freq;
        float c = cosf(ang), s = sinf(ang);
        float x0 = dst[i], y0 = dst[i + half];
        dst[i] = x0 * c - y0 * s;
        dst[i + half] = x0 * s + y0 * c;
    }
}

/* In-place RMSNorm plus the same RoPE. Block = token * n_heads + head.
   The weight vector is one head, shared by every block. */
extern "C" __global__ void rmsnorm_rope(float* inout, const float* w, int size, float eps,
                                        int pos, float rope_freq, int rotary_dim, int n_heads, const int* pos_slot) {
    int h = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    float* oi = inout + (size_t)h * (size_t)size;
    float ss = 0.f;
    for (int j = tid; j < size; j += 128) ss += oi[j] * oi[j];
    ss = block_sum_128(ss);
    float inv = rsqrtf(ss / (float)size + eps);
    for (int j = tid; j < size; j += 128) oi[j] = w[j] * (inv * oi[j]);
    __syncthreads();
    if (rotary_dim < 2 || n_heads < 1) return;
    int half = rotary_dim >> 1;
    int token = h / n_heads;
    int p = live_pos(pos, pos_slot) + token;
    for (int i = tid; i < half; i += 128) {
        float freq = 1.f / powf(rope_freq, (float)i / (float)half);
        float ang = (float)p * freq;
        float c = cosf(ang), s = sinf(ang);
        float x0 = oi[i], y0 = oi[i + half];
        oi[i] = x0 * c - y0 * s;
        oi[i + half] = x0 * s + y0 * c;
    }
}

extern "C" __global__ void rope_partial(float* vec, int head_dim, int pos, float rope_freq, int rotary_dim, int n_heads, const int* pos_slot) {
    int half = rotary_dim >> 1;
    int total = n_heads * half;
    int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (gid >= total) return;
    int h = gid / half;
    int i = gid - h * half;
    float* q = vec + (size_t)h * (size_t)head_dim;
    float freq = 1.f / powf(rope_freq, (float)i / (float)half);
    float ang = (float)live_pos(pos, pos_slot) * freq;
    float c = cosf(ang), s = sinf(ang);
    float x0 = q[i], y0 = q[i + half];
    q[i] = x0 * c - y0 * s;
    q[i + half] = x0 * s + y0 * c;
}

/* Same groups as quant_q8: scale amax/127, rintf, sum of each 16-float half. */
__device__ void quant_store_group(const float* xi, signed char* qs, float* scale, float2* sum16, int group) {
    float amax = 0.f, s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        float a = xi[j];
        float b = xi[j + 16];
        amax = fmaxf(amax, fmaxf(fabsf(a), fabsf(b)));
        s0 += a;
        s1 += b;
    }
    float d = amax / 127.f;
    float id = d > 0.f ? 1.f / d : 0.f;
    scale[group] = d;
    sum16[group] = make_float2(s0, s1);
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        int q = (int)rintf(xi[j] * id);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        qs[(size_t)group * 32 + j] = (signed char)q;
    }
}

extern "C" __global__ void attn_gate(float* xb3, const float* qproj, int head_dim, int n_heads) {
    int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int n = n_heads * head_dim;
    if (gid >= n) return;
    int h = gid / head_dim;
    int i = gid - h * head_dim;
    float g = qproj[(size_t)h * (size_t)(2 * head_dim) + head_dim + i];
    xb3[(size_t)h * (size_t)head_dim + i] *= sigmoid_stable(g);
}

/* Decode attention gate plus the q8 cache the output GEMV reads. n % 32 == 0. */
extern "C" __global__ void attn_gate_quant(float* xb3, const float* qproj, signed char* qs, float* scale, float2* sum16,
                                           int head_dim, int n_heads) {
    int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int lane = (int)threadIdx.x & 31;
    int n = n_heads * head_dim;
    if (gid < n) {
        int h = gid / head_dim;
        int i = gid - h * head_dim;
        float g = qproj[(size_t)h * (size_t)(2 * head_dim) + head_dim + i];
        xb3[gid] *= sigmoid_stable(g);
    }
    __syncwarp();
    int group = gid >> 5;
    if (lane == 0 && (group << 5) < n) quant_store_group(xb3 + (group << 5), qs, scale, sum16, group);
}

extern "C" __global__ void swiglu(float* hb, const float* hb2, int n) {
    int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (gid >= n) return;
    float v = hb[gid];
    hb[gid] = (v * sigmoid_stable(v)) * hb2[gid];
}

/* Decode SwiGLU plus the q8 cache the down GEMV reads. n % 32 == 0. */
extern "C" __global__ void swiglu_quant(float* hb, const float* hb2, signed char* qs, float* scale, float2* sum16, int n) {
    int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int lane = (int)threadIdx.x & 31;
    if (gid < n) {
        float v = hb[gid];
        hb[gid] = (v * sigmoid_stable(v)) * hb2[gid];
    }
    __syncwarp();
    int group = gid >> 5;
    if (lane == 0 && (group << 5) < n) quant_store_group(hb + (group << 5), qs, scale, sum16, group);
}

extern "C" __global__ void residual(float* x, const float* y, int n) {
    int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (gid >= n) return;
    x[gid] += y[gid];
}

extern "C" __global__ void axpy_k(float* y, const float* x, float a, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) y[i] += a * x[i];
}

/* f16 KV: head-major [kvh, seq_len, head_dim], little-endian half. */
extern "C" __global__ void store_kv_f16(const float* k, const float* v, unsigned char* kc, unsigned char* vc,
                                        int head_dim, int seq_len, int pos, int n_kv, const int* pos_slot) {
    int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int n = n_kv * head_dim;
    if (gid >= n) return;
    int ti = (int)blockIdx.y;
    k += (size_t)ti * (size_t)n;
    v += (size_t)ti * (size_t)n;
    int kvh = gid / head_dim;
    int i = gid - kvh * head_dim;
    size_t dst = (((size_t)kvh * (size_t)seq_len + (size_t)(live_pos(pos, pos_slot) + ti)) * (size_t)head_dim + (size_t)i) * 2;
    st_f16(kc + dst, k[gid]);
    st_f16(vc + dst, v[gid]);
}

/* Full attention is quadratic in the prompt: query t scores and mixes
   npos = t+1 keys. Prefill used to launch that once per token (one block
   per head, so a 16-head layer filled 16 of 80 SMs). blockIdx.y is the
   token in the chunk. The host stores every token's KV first, then one
   launch covers the chunk. Flops stay O(N^2). tok_stride is the float
   stride of q and xb3. gridDim.y == 1 is the one-token path. */
extern "C" __global__ void attention_f16(const float* q, const unsigned char* kc, const unsigned char* vc, float* xb3, float* scores,
                                         int head_dim, int seq_len, int kv_mul, int pos, const int* pos_slot, int tok_stride) {
    int ti = (int)blockIdx.y;
    int h = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int kvh = h / kv_mul;
    if (ti != 0) {
        q += (size_t)ti * (size_t)tok_stride;
        xb3 += (size_t)ti * (size_t)tok_stride;
    }
    const float* qh = q + (size_t)h * (size_t)head_dim;
    const unsigned char* kbase = kc + (size_t)kvh * (size_t)seq_len * (size_t)head_dim * 2;
    const unsigned char* vbase = vc + (size_t)kvh * (size_t)seq_len * (size_t)head_dim * 2;
    float scale = rsqrtf((float)head_dim);
    int npos = live_pos(pos, pos_slot) + 1 + ti;
    float* sc = scores + ((size_t)ti * (size_t)gridDim.x + (size_t)h) * (size_t)seq_len;
    for (int t = tid; t < npos; t += 128) {
        const unsigned char* kk = kbase + (size_t)t * (size_t)head_dim * 2;
        float s = 0.f;
        for (int i = 0; i < head_dim; ++i) s += qh[i] * ld_f16(kk + (size_t)i * 2);
        sc[t] = s * scale;
    }
    __syncthreads();
    float lm = -1e30f;
    for (int t = tid; t < npos; t += 128) lm = fmaxf(lm, sc[t]);
    float m = block_max_128(lm);
    float ls = 0.f;
    for (int t = tid; t < npos; t += 128) {
        float e = expf(sc[t] - m);
        sc[t] = e;
        ls += e;
    }
    float sum = block_sum_128(ls);
    float inv = 1.f / sum;
    for (int i = tid; i < head_dim; i += 128) {
        float acc = 0.f;
        for (int t = 0; t < npos; ++t) {
            const unsigned char* vv = vbase + ((size_t)t * (size_t)head_dim + (size_t)i) * 2;
            acc += sc[t] * ld_f16(vv);
        }
        xb3[(size_t)h * (size_t)head_dim + i] = acc * inv;
    }
}

extern "C" __global__ void store_kv_q8(const float* k, const float* v, unsigned char* kc, unsigned char* vc,
                                       int nblocks, int pos_bytes, int pos, const int* pos_slot) {
    int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (gid >= nblocks) return;
    int ti = (int)blockIdx.y;
    k += (size_t)ti * (size_t)nblocks * 32;
    v += (size_t)ti * (size_t)nblocks * 32;
    const float* k32 = k + (size_t)gid * 32;
    const float* v32 = v + (size_t)gid * 32;
    float kamax = 0.f, vamax = 0.f;
    for (int j = 0; j < 32; ++j) {
        kamax = fmaxf(kamax, fabsf(k32[j]));
        vamax = fmaxf(vamax, fabsf(v32[j]));
    }
    float kd = kamax / 127.f, vd = vamax / 127.f;
    float kid = kd != 0.f ? 1.f / kd : 0.f;
    float vid = vd != 0.f ? 1.f / vd : 0.f;
    int pslot = live_pos(pos, pos_slot) + (int)blockIdx.y;
    unsigned char* kb = kc + (size_t)pslot * (size_t)pos_bytes + (size_t)gid * 34;
    unsigned char* vb = vc + (size_t)pslot * (size_t)pos_bytes + (size_t)gid * 34;
    st_f16(kb, kd);
    st_f16(vb, vd);
    for (int j = 0; j < 32; ++j) {
        kb[2 + j] = (unsigned char)(signed char)(int)roundf(k32[j] * kid);
        vb[2 + j] = (unsigned char)(signed char)(int)roundf(v32[j] * vid);
    }
}

extern "C" __global__ void attention_q8(const float* q, const unsigned char* kc, const unsigned char* vc, float* xb3, float* scores,
                                        int head_dim, int pos_bytes, int kv_mul, int pos, int seq_len, const int* pos_slot, int tok_stride) {
    int ti = (int)blockIdx.y;
    int h = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int nb = head_dim >> 5;
    int kvh = h / kv_mul;
    int hb = kvh * nb;
    if (ti != 0) {
        q += (size_t)ti * (size_t)tok_stride;
        xb3 += (size_t)ti * (size_t)tok_stride;
    }
    const float* qh = q + (size_t)h * (size_t)head_dim;
    float scale = rsqrtf((float)head_dim);
    int npos = live_pos(pos, pos_slot) + 1 + ti;
    float* sc = scores + ((size_t)ti * (size_t)gridDim.x + (size_t)h) * (size_t)seq_len;
    for (int t = tid; t < npos; t += 128) {
        const unsigned char* kb = kc + (size_t)t * (size_t)pos_bytes + (size_t)hb * 34;
        float s = 0.f;
        for (int b = 0; b < nb; ++b) {
            const unsigned char* bb = kb + (size_t)b * 34;
            float d = ld_f16(bb);
            for (int j = 0; j < 32; ++j) s += qh[b * 32 + j] * (d * (float)(signed char)bb[2 + j]);
        }
        sc[t] = s * scale;
    }
    __syncthreads();
    float lm = -1e30f;
    for (int t = tid; t < npos; t += 128) lm = fmaxf(lm, sc[t]);
    float m = block_max_128(lm);
    float ls = 0.f;
    for (int t = tid; t < npos; t += 128) {
        float e = expf(sc[t] - m);
        sc[t] = e;
        ls += e;
    }
    float sum = block_sum_128(ls);
    float inv = 1.f / sum;
    for (int i = tid; i < head_dim; i += 128) {
        int elem = kvh * head_dim + i;
        int bb = elem >> 5;
        int jj = elem - (bb << 5);
        float acc = 0.f;
        for (int t = 0; t < npos; ++t) {
            const unsigned char* vb = vc + (size_t)t * (size_t)pos_bytes + (size_t)bb * 34;
            float d = ld_f16(vb);
            acc += sc[t] * (d * (float)(signed char)vb[2 + jj]);
        }
        xb3[(size_t)h * (size_t)head_dim + i] = acc * inv;
    }
}

extern "C" __global__ void store_kv_q8v(const float* k, const float* v, unsigned char* kc, unsigned char* vc,
                                        int nblocks, int k_pos_bytes, int v_pos_bytes, int pos, const int* pos_slot) {
    int gid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (gid >= nblocks) return;
    int ti = (int)blockIdx.y;
    k += (size_t)ti * (size_t)nblocks * 32;
    v += (size_t)ti * (size_t)nblocks * 32;
    const float* k32 = k + (size_t)gid * 32;
    const float* v32 = v + (size_t)gid * 32;
    float kamax = 0.f;
    for (int j = 0; j < 32; ++j) kamax = fmaxf(kamax, fabsf(k32[j]));
    float kd = kamax / 127.f;
    float kid = kd != 0.f ? 1.f / kd : 0.f;
    int pslot = live_pos(pos, pos_slot) + (int)blockIdx.y;
    unsigned char* kb = kc + (size_t)pslot * (size_t)k_pos_bytes + (size_t)gid * 34;
    st_f16(kb, kd);
    for (int j = 0; j < 32; ++j) kb[2 + j] = (unsigned char)(signed char)(int)roundf(k32[j] * kid);

    float vamax = fabsf(v32[0]);
    int jmax = 0;
    for (int j = 1; j < 32; ++j) {
        float a = fabsf(v32[j]);
        if (a > vamax) { vamax = a; jmax = j; }
    }
    float amax2 = 0.f;
    for (int j = 0; j < 32; ++j) if (j != jmax) amax2 = fmaxf(amax2, fabsf(v32[j]));
    float vd = amax2 / 127.f;
    float vid = vd != 0.f ? 1.f / vd : 0.f;
    unsigned char* vb = vc + (size_t)pslot * (size_t)v_pos_bytes + (size_t)gid * 40;
    for (int j = 0; j < 40; ++j) vb[j] = 0;
    st_f16(vb, vd);
    vb[2] = (unsigned char)jmax;
    st_f16(vb + 4, v32[jmax]);
    for (int j = 0; j < 32; ++j) {
        if (j != jmax) vb[6 + j] = (unsigned char)(signed char)(int)roundf(v32[j] * vid);
    }
}

extern "C" __global__ void attention_q8v(const float* q, const unsigned char* kc, const unsigned char* vc, float* xb3, float* scores,
                                         int head_dim, int k_pos_bytes, int v_pos_bytes, int kv_mul, int pos, int seq_len, const int* pos_slot, int tok_stride) {
    int ti = (int)blockIdx.y;
    int h = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int nb = head_dim >> 5;
    int kvh = h / kv_mul;
    int hb = kvh * nb;
    if (ti != 0) {
        q += (size_t)ti * (size_t)tok_stride;
        xb3 += (size_t)ti * (size_t)tok_stride;
    }
    const float* qh = q + (size_t)h * (size_t)head_dim;
    float scale = rsqrtf((float)head_dim);
    int npos = live_pos(pos, pos_slot) + 1 + ti;
    float* sc = scores + ((size_t)ti * (size_t)gridDim.x + (size_t)h) * (size_t)seq_len;
    for (int t = tid; t < npos; t += 128) {
        const unsigned char* kb = kc + (size_t)t * (size_t)k_pos_bytes + (size_t)hb * 34;
        float s = 0.f;
        for (int b = 0; b < nb; ++b) {
            const unsigned char* bb = kb + (size_t)b * 34;
            float d = ld_f16(bb);
            for (int j = 0; j < 32; ++j) s += qh[b * 32 + j] * (d * (float)(signed char)bb[2 + j]);
        }
        sc[t] = s * scale;
    }
    __syncthreads();
    float lm = -1e30f;
    for (int t = tid; t < npos; t += 128) lm = fmaxf(lm, sc[t]);
    float m = block_max_128(lm);
    float ls = 0.f;
    for (int t = tid; t < npos; t += 128) {
        float e = expf(sc[t] - m);
        sc[t] = e;
        ls += e;
    }
    float sum = block_sum_128(ls);
    float inv = 1.f / sum;
    for (int i = tid; i < head_dim; i += 128) {
        int elem = kvh * head_dim + i;
        int bb = elem >> 5;
        int jj = elem - (bb << 5);
        float acc = 0.f;
        for (int t = 0; t < npos; ++t) {
            const unsigned char* vb = vc + (size_t)t * (size_t)v_pos_bytes + (size_t)bb * 40;
            float d = ld_f16(vb);
            int idx = vb[2];
            float ov = ld_f16(vb + 4);
            float vv = (jj == idx) ? ov : d * (float)(signed char)vb[6 + jj];
            acc += sc[t] * vv;
        }
        xb3[(size_t)h * (size_t)head_dim + i] = acc * inv;
    }
}

/* Split decode attention for the mixed-V cache. Prefill does not use
   these kernels: running the parts once per token of a T=128 chunk was
   slower on the V100S (about 10% on 905 and 1805 token prefills).
   Prefill is attention_q8v_flash, or attention_q8v when
   ODINFER_CUDA_PREFILL_FLASH=0.
   attention_q8v is one block per query head. Its V loop walks every
   position on each thread, so a long context uses a handful of SMs and
   the time grows linearly with npos even though the KV row is a few
   kilobytes. These two kernels split the sequence into 32-token parts.
   One block owns one KV head and one part, and the query heads in that
   GQA group share the staged K/V. Each lane owns one position and dots
   it in the same j-then-block order as attention_q8v. A second kernel
   merges the partial softmaxes (max, sum, unnormalized weighted V).
   In exact arithmetic that merge is the global softmax. In fp32 it is
   not bit-exact: exp(s - m_part) * exp(m_part - m_global) rounds once
   more than exp(s - m_global). cuda_selftest allows 2e-2 + 1e-3*|ref|
   against attention_q8v on the same cache. block is 128 threads and
   `part` is 32, so one lane owns one position. head_dim is a multiple
   of 32 and at most 256. kv_mul is at most 16. */
__device__ float attn_k_dot(const float* qh, const unsigned char* krow, int nb) {
    float s = 0.f;
    for (int b = 0; b < nb; ++b) {
        const unsigned char* bb = krow + (size_t)b * 34;
        float d = ld_f16(bb);
        int base = b * 32;
        for (int j = 0; j < 32; ++j) s += qh[base + j] * (d * (float)(signed char)bb[2 + j]);
    }
    return s;
}

__device__ float attn_v_at(const unsigned char* vbase, int nb, int t, int i) {
    int b = i >> 5;
    int j = i & 31;
    const unsigned char* vb = vbase + ((size_t)t * (size_t)nb + (size_t)b) * 40;
    float d = ld_f16(vb);
    int idx = (int)vb[2];
    float ov = ld_f16(vb + 4);
    return (j == idx) ? ov : d * (float)(signed char)vb[6 + j];
}

__device__ float warp_max_all(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    return v;
}

__device__ float warp_sum_all(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

/* 16-lane group. Lanes 0-15 and 16-31 are separate rows, so the mask is one half. */
__device__ float half_max(float v, unsigned mask) {
    v = fmaxf(v, __shfl_xor_sync(mask, v, 8));
    v = fmaxf(v, __shfl_xor_sync(mask, v, 4));
    v = fmaxf(v, __shfl_xor_sync(mask, v, 2));
    v = fmaxf(v, __shfl_xor_sync(mask, v, 1));
    return v;
}

__device__ float half_sum(float v, unsigned mask) {
    v += __shfl_xor_sync(mask, v, 8);
    v += __shfl_xor_sync(mask, v, 4);
    v += __shfl_xor_sync(mask, v, 2);
    v += __shfl_xor_sync(mask, v, 1);
    return v;
}

extern "C" __global__ void attention_q8v_part(const float* q, const unsigned char* kc, const unsigned char* vc,
                                              float* part_m, float* part_l, float* part_o,
                                              int head_dim, int k_pos_bytes, int v_pos_bytes, int kv_mul, int n_heads,
                                              int pos, int part, const int* pos_slot) {
    int kvh = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int nb = head_dim >> 5;
    int npos = live_pos(pos, pos_slot) + 1;
    int begin = (int)blockIdx.y * part;
    extern __shared__ __align__(16) unsigned char smem[];
    float* sq = (float*)smem;
    unsigned char* sk = smem + (size_t)kv_mul * (size_t)head_dim * 4;
    unsigned char* sv = sk + (size_t)part * (size_t)nb * 34;
    if (begin >= npos || part != 32 || head_dim > 256 || nb <= 0 || kv_mul > 16) {
        if (begin < npos && warp < kv_mul) {
            for (int qi = warp; qi < kv_mul; qi += 4) {
                int h = kvh * kv_mul + qi;
                size_t si = (size_t)blockIdx.y * (size_t)n_heads + (size_t)h;
                if (lane == 0) {
                    part_m[si] = -1e30f;
                    part_l[si] = 0.f;
                }
                for (int i = lane; i < head_dim; i += 32) part_o[si * (size_t)head_dim + i] = 0.f;
            }
        }
        return;
    }
    int nlocal = npos - begin;
    if (nlocal > part) nlocal = part;
    int qn = kv_mul * head_dim;
    const float* qg = q + (size_t)kvh * (size_t)kv_mul * (size_t)head_dim;
    for (int i = tid; i < qn; i += 128) sq[i] = qg[i];
    int kbytes = nb * 34;
    int vbytes = nb * 40;
    for (int t = 0; t < nlocal; ++t) {
        const unsigned char* ks = kc + ((size_t)(begin + t) * (size_t)k_pos_bytes + (size_t)kvh * (size_t)kbytes);
        const unsigned char* vs = vc + ((size_t)(begin + t) * (size_t)v_pos_bytes + (size_t)kvh * (size_t)vbytes);
        unsigned char* kd = sk + (size_t)t * (size_t)kbytes;
        unsigned char* vd = sv + (size_t)t * (size_t)vbytes;
        for (int i = tid; i < kbytes; i += 128) kd[i] = ks[i];
        for (int i = tid; i < vbytes; i += 128) vd[i] = vs[i];
    }
    __syncthreads();
    float scale = rsqrtf((float)head_dim);
    for (int qi = warp; qi < kv_mul; qi += 4) {
        const float* qh = sq + (size_t)qi * (size_t)head_dim;
        float s = -1e30f;
        if (lane < nlocal) s = attn_k_dot(qh, sk + (size_t)lane * (size_t)kbytes, nb) * scale;
        float m = warp_max_all(s);
        float e = expf(s - m);
        float lsum = warp_sum_all(e);
        int h = kvh * kv_mul + qi;
        size_t si = (size_t)blockIdx.y * (size_t)n_heads + (size_t)h;
        float* po = part_o + si * (size_t)head_dim;
        for (int i = lane; i < head_dim; i += 32) {
            float acc = 0.f;
            for (int src = 0; src < nlocal; ++src) {
                float wt = __shfl_sync(0xffffffff, e, src);
                acc += wt * attn_v_at(sv, nb, src, i);
            }
            po[i] = acc;
        }
        if (lane == 0) {
            part_m[si] = m;
            part_l[si] = lsum;
        }
    }
}

extern "C" __global__ void attention_q8v_reduce(float* xb3, const float* part_m, const float* part_l, const float* part_o,
                                                int head_dim, int n_heads, int n_splits) {
    int h = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    if (h >= n_heads || head_dim > 256) return;
    int i0 = tid;
    int i1 = tid + 128;
    bool has0 = i0 < head_dim;
    bool has1 = i1 < head_dim;
    float acc0 = 0.f, acc1 = 0.f;
    float m = -1e30f;
    float l = 0.f;
    for (int s = 0; s < n_splits; ++s) {
        size_t si = (size_t)s * (size_t)n_heads + (size_t)h;
        float ms = part_m[si];
        float ls = part_l[si];
        float m2 = fmaxf(m, ms);
        float e0 = expf(m - m2);
        float e1 = expf(ms - m2);
        const float* po = part_o + si * (size_t)head_dim;
        if (has0) acc0 = acc0 * e0 + po[i0] * e1;
        if (has1) acc1 = acc1 * e0 + po[i1] * e1;
        l = l * e0 + ls * e1;
        m = m2;
    }
    float inv = l > 0.f ? 1.f / l : 0.f;
    if (has0) xb3[(size_t)h * (size_t)head_dim + i0] = acc0 * inv;
    if (has1) xb3[(size_t)h * (size_t)head_dim + i1] = acc1 * inv;
}

__device__ __forceinline__ uint4 ld_u4_aligned(const unsigned char* p);

/* Long-context decode attention. One block owns one KV head and 128
   positions (one thread, one position). K is dequantized once per position
   and dotted with every query head in the GQA group. V is dequantized once
   per position into smem and reused across those heads. The block softmax
   is one partial; attention_q8v_reduce merges partials. head_dim is 256.
   MUL is 4 (9B) or 6 (27B). Shorter contexts stay on attention_q8v_part
   (32 positions, one per lane). ODINFER_CUDA_ATTN_LONG=0 keeps that path
   at every length. blockDim is 128. */
template<int MUL>
__device__ void attention_q8v_long_run(const float* q, const unsigned char* kc, const unsigned char* vc,
                                       float* part_m, float* part_l, float* part_o,
                                       int head_dim, int k_pos_bytes, int v_pos_bytes, int n_heads,
                                       int pos, int part, const int* pos_slot) {
    int kvh = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int npos = live_pos(pos, pos_slot) + 1;
    int begin = (int)blockIdx.y * 128;
    extern __shared__ __align__(16) unsigned char long_smem[];
    float* sq = (float*)long_smem;
    float* scores = sq + MUL * 256;
    float* meta = scores + MUL * 128;
    if (head_dim != 256 || part != 128) return;
    if (begin >= npos) {
        if (tid < MUL) {
            int h = kvh * MUL + tid;
            size_t si = (size_t)blockIdx.y * (size_t)n_heads + (size_t)h;
            part_m[si] = -1e30f;
            part_l[si] = 0.f;
        }
        for (int qi = 0; qi < MUL; ++qi) {
            int h = kvh * MUL + qi;
            size_t si = (size_t)blockIdx.y * (size_t)n_heads + (size_t)h;
            for (int i = tid; i < 256; i += 128) part_o[si * 256 + i] = 0.f;
        }
        return;
    }
    int nlocal = npos - begin;
    if (nlocal > 128) nlocal = 128;
    const float* qg = q + (size_t)kvh * (size_t)MUL * 256;
    for (int i = tid; i < MUL * 256; i += 128) sq[i] = qg[i];
    __syncthreads();
    float dots[MUL];
#pragma unroll
    for (int qi = 0; qi < MUL; ++qi) dots[qi] = 0.f;
    if (tid < nlocal) {
        const unsigned char* krow = kc + ((size_t)(begin + tid) * (size_t)k_pos_bytes + (size_t)kvh * 272);
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            const unsigned char* bb = krow + b * 34;
            float d = ld_f16(bb);
            /* 32 q bytes, two aligned 128-bit windows. The FMA order stays
               j = 0..31 so the partial matches the scalar load. */
            uint4 q0 = ld_u4_aligned(bb + 2);
            uint4 q1 = ld_u4_aligned(bb + 18);
            unsigned qw[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
            int col = b * 32;
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                unsigned v = qw[w];
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    float kv = d * (float)(signed char)(unsigned char)(v & 0xffu);
                    v >>= 8;
#pragma unroll
                    for (int qi = 0; qi < MUL; ++qi) dots[qi] += sq[qi * 256 + col] * kv;
                    col += 1;
                }
            }
        }
    }
    float scale = rsqrtf(256.f);
#pragma unroll
    for (int qi = 0; qi < MUL; ++qi) {
        float s = (tid < nlocal) ? dots[qi] * scale : -1e30f;
        float m = block_max_128(s);
        float e = (tid < nlocal) ? expf(s - m) : 0.f;
        float lsum = block_sum_128(e);
        scores[qi * 128 + tid] = e;
        if (tid == 0) {
            meta[qi * 2] = m;
            meta[qi * 2 + 1] = lsum;
        }
    }
    __syncthreads();
    float a0[MUL], a1[MUL];
#pragma unroll
    for (int qi = 0; qi < MUL; ++qi) {
        a0[qi] = 0.f;
        a1[qi] = 0.f;
    }
    /* Each thread writes sv[tid] and sv[tid+128] and reads only those.
       The scores were published before this loop. No cross-thread read,
       so the old per-position barriers are not required. */
    for (int t = 0; t < nlocal; ++t) {
        const unsigned char* vrow = vc + ((size_t)(begin + t) * (size_t)v_pos_bytes + (size_t)kvh * 320);
        float v0, v1;
#pragma unroll
        for (int rep = 0; rep < 2; ++rep) {
            int i = tid + rep * 128;
            int b = i >> 5;
            int j = i & 31;
            const unsigned char* vb = vrow + b * 40;
            float d = ld_f16(vb);
            int idx = (int)vb[2];
            float ov = ld_f16(vb + 4);
            float vv = (j == idx) ? ov : d * (float)(signed char)vb[6 + j];
            if (rep == 0) v0 = vv;
            else v1 = vv;
        }
#pragma unroll
        for (int qi = 0; qi < MUL; ++qi) {
            float w = scores[qi * 128 + t];
            a0[qi] += w * v0;
            a1[qi] += w * v1;
        }
    }
#pragma unroll
    for (int qi = 0; qi < MUL; ++qi) {
        int h = kvh * MUL + qi;
        size_t si = (size_t)blockIdx.y * (size_t)n_heads + (size_t)h;
        part_o[si * 256 + tid] = a0[qi];
        part_o[si * 256 + tid + 128] = a1[qi];
        if (tid == 0) {
            part_m[si] = meta[qi * 2];
            part_l[si] = meta[qi * 2 + 1];
        }
    }
}

extern "C" __global__ void attention_q8v_long4(const float* q, const unsigned char* kc, const unsigned char* vc,
                                               float* part_m, float* part_l, float* part_o,
                                               int head_dim, int k_pos_bytes, int v_pos_bytes, int kv_mul, int n_heads,
                                               int pos, int part, const int* pos_slot) {
    (void)kv_mul;
    attention_q8v_long_run<4>(q, kc, vc, part_m, part_l, part_o, head_dim, k_pos_bytes, v_pos_bytes, n_heads, pos, part, pos_slot);
}

extern "C" __global__ void attention_q8v_long6(const float* q, const unsigned char* kc, const unsigned char* vc,
                                               float* part_m, float* part_l, float* part_o,
                                               int head_dim, int k_pos_bytes, int v_pos_bytes, int kv_mul, int n_heads,
                                               int pos, int part, const int* pos_slot) {
    (void)kv_mul;
    attention_q8v_long_run<6>(q, kc, vc, part_m, part_l, part_o, head_dim, k_pos_bytes, v_pos_bytes, n_heads, pos, part, pos_slot);
}

/* Prefill flash attention for the mixed-V cache. One block owns one KV
   head and `br` query tokens (the last tile of the chunk may be shorter).
   It stages 32-token K and V tiles once and reuses them across every query
   head in the GQA group, so a 27B group of 6 query heads does not reload
   that KV six times.    Scores stay in registers: each warp keeps an online
   softmax (max, sum, unnormalized V) and merges the next tile with
   exp(m_old - m_new). That rescale is the same one attention_q8v_reduce
   applies once at the end; here it runs as each 32-token tile arrives,
   so it is not bit-exact against attention_q8v. cuda_selftest allows
   2e-2 + 1e-3*|ref|. Dots are fp32 warp dots in attn_k_dot's j-then-block
   order. WMMA fp16 is not used: it would move the softmax enough to risk
   the greedy-text check, and there is no V100 timing that says it is faster.
   head_dim is a multiple of 32, at most 256. kv_mul is at most 16. br is
   1, 2, 3, or 4. Shared memory stays under 48 KB. ODINFER_CUDA_PREFILL_FLASH=0
   keeps attention_q8v. Decode (T==1) does not launch this kernel. */
extern "C" __global__ void attention_q8v_flash(const float* q, const unsigned char* kc, const unsigned char* vc, float* xb3,
                                               int head_dim, int k_pos_bytes, int v_pos_bytes, int kv_mul, int n_heads,
                                               int pos0, int T, int tok_stride, int br) {
    int kvh = (int)blockIdx.x;
    int tile0 = (int)blockIdx.y * br;
    int tid = (int)threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    if (tile0 >= T || br < 1 || br > 4 || head_dim > 256 || head_dim < 32 || kv_mul < 1 || kv_mul > 16) return;
    if (kvh * kv_mul >= n_heads) return;
    int ntok = br;
    if (tile0 + ntok > T) ntok = T - tile0;
    int nb = head_dim >> 5;
    if (nb <= 0) return;
    int cap = br * kv_mul;
    int njob = ntok * kv_mul;
    int max_npos = pos0 + tile0 + ntok;
    extern __shared__ __align__(16) unsigned char flash_smem[];
    float* sq = (float*)flash_smem;
    float* acc = sq + (size_t)cap * (size_t)head_dim;
    float* ms = acc + (size_t)cap * (size_t)head_dim;
    float* ls = ms + cap;
    int fbytes = cap * head_dim * 4 * 2 + cap * 8;
    fbytes = (fbytes + 15) & ~15;
    unsigned char* sk = flash_smem + fbytes;
    unsigned char* sv = sk + (size_t)32 * (size_t)nb * 34;
    for (int job = 0; job < njob; ++job) {
        int tok = job / kv_mul;
        int qi = job - tok * kv_mul;
        int h = kvh * kv_mul + qi;
        const float* qh = q + ((size_t)(tile0 + tok) * (size_t)tok_stride + (size_t)h * (size_t)head_dim);
        float* dst = sq + (size_t)job * (size_t)head_dim;
        for (int i = tid; i < head_dim; i += 128) {
            dst[i] = qh[i];
            acc[(size_t)job * (size_t)head_dim + i] = 0.f;
        }
        if (tid == 0) {
            ms[job] = -1e30f;
            ls[job] = 0.f;
        }
    }
    __syncthreads();
    int kbytes = nb * 34;
    int vbytes = nb * 40;
    float scale = rsqrtf((float)head_dim);
    int ntiles = (max_npos + 31) >> 5;
    for (int tile = 0; tile < ntiles; ++tile) {
        int begin = tile << 5;
        int ncopy = max_npos - begin;
        if (ncopy > 32) ncopy = 32;
        for (int t = 0; t < ncopy; ++t) {
            const unsigned char* ks = kc + ((size_t)(begin + t) * (size_t)k_pos_bytes + (size_t)kvh * (size_t)kbytes);
            const unsigned char* vs = vc + ((size_t)(begin + t) * (size_t)v_pos_bytes + (size_t)kvh * (size_t)vbytes);
            unsigned char* kd = sk + (size_t)t * (size_t)kbytes;
            unsigned char* vd = sv + (size_t)t * (size_t)vbytes;
            for (int i = tid; i < kbytes; i += 128) kd[i] = ks[i];
            for (int i = tid; i < vbytes; i += 128) vd[i] = vs[i];
        }
        __syncthreads();
        for (int job = warp; job < njob; job += 4) {
            int tok = job / kv_mul;
            int npos = pos0 + tile0 + tok + 1;
            if (begin >= npos) continue;
            int nlocal = npos - begin;
            if (nlocal > 32) nlocal = 32;
            const float* qh = sq + (size_t)job * (size_t)head_dim;
            float s = -1e30f;
            if (lane < nlocal) s = attn_k_dot(qh, sk + (size_t)lane * (size_t)kbytes, nb) * scale;
            float mt = warp_max_all(s);
            float e = (lane < nlocal) ? expf(s - mt) : 0.f;
            float lt = warp_sum_all(e);
            float m_old = ms[job];
            float l_old = ls[job];
            float m_new = fmaxf(m_old, mt);
            float alpha = expf(m_old - m_new);
            float beta = expf(mt - m_new);
            float* ao = acc + (size_t)job * (size_t)head_dim;
            for (int i = lane; i < head_dim; i += 32) {
                float pv = 0.f;
                for (int src = 0; src < nlocal; ++src) {
                    float wt = __shfl_sync(0xffffffff, e, src);
                    pv += wt * attn_v_at(sv, nb, src, i);
                }
                ao[i] = ao[i] * alpha + pv * beta;
            }
            if (lane == 0) {
                ms[job] = m_new;
                ls[job] = l_old * alpha + lt * beta;
            }
        }
        __syncthreads();
    }
    for (int job = 0; job < njob; ++job) {
        int tok = job / kv_mul;
        int qi = job - tok * kv_mul;
        int h = kvh * kv_mul + qi;
        float inv = ls[job] > 0.f ? 1.f / ls[job] : 0.f;
        float* ao = acc + (size_t)job * (size_t)head_dim;
        float* out = xb3 + ((size_t)(tile0 + tok) * (size_t)tok_stride + (size_t)h * (size_t)head_dim);
        for (int i = tid; i < head_dim; i += 128) out[i] = ao[i] * inv;
    }
}

/* Tensor-core prefill attention for the mixed-V cache. head_dim is 256.
   One block owns 32 query tokens (two of the old 16-token tiles) and one
   KV head. A 512-token chunk of K is 256 KB and does not fit in shared
   memory, so each step dequantizes 64 keys to fp16 once and reuses that
   tile for every query head in the group (at most 8). Eight warps each
   own one 16x16 score tile and walk all 256 K in the accumulator, so the
   dot is not split across warps. V then replaces K and all 16 warps do PV.
   A tile that starts past the last query in the block is not loaded.
   Softmax is fp32 and online. attention_q8v_wmma_reduce merges partitions.
   fp16 Q/K/V rounding is not the fp32 kernel. cuda_selftest allows
   1e-1 + 5e-2*|ref|. ODINFER_CUDA_PREFILL_FLASH=0 keeps attention_q8v.
   Shared memory is 91136 bytes (ATTN_WMMA_SMEM). */
__device__ __forceinline__ unsigned short f32_to_f16(float v);
__device__ __forceinline__ void wmma_load_a(unsigned& a0, unsigned& a1, unsigned& a2, unsigned& a3,
                                            unsigned& a4, unsigned& a5, unsigned& a6, unsigned& a7,
                                            const unsigned short* p, int stride);
__device__ __forceinline__ void wmma_load_b(unsigned& b0, unsigned& b1, unsigned& b2, unsigned& b3,
                                            unsigned& b4, unsigned& b5, unsigned& b6, unsigned& b7,
                                            const unsigned short* p, int stride);
__device__ __forceinline__ void wmma_load_b_row(unsigned& b0, unsigned& b1, unsigned& b2, unsigned& b3,
                                                unsigned& b4, unsigned& b5, unsigned& b6, unsigned& b7,
                                                const unsigned short* p, int stride);
__device__ __forceinline__ void wmma_mma(float& c0, float& c1, float& c2, float& c3,
                                         float& c4, float& c5, float& c6, float& c7,
                                         unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                         unsigned a4, unsigned a5, unsigned a6, unsigned a7,
                                         unsigned b0, unsigned b1, unsigned b2, unsigned b3,
                                         unsigned b4, unsigned b5, unsigned b6, unsigned b7);
__device__ __forceinline__ void wmma_mma_rr(float& c0, float& c1, float& c2, float& c3,
                                            float& c4, float& c5, float& c6, float& c7,
                                            unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                            unsigned a4, unsigned a5, unsigned a6, unsigned a7,
                                            unsigned b0, unsigned b1, unsigned b2, unsigned b3,
                                            unsigned b4, unsigned b5, unsigned b6, unsigned b7);
__device__ __forceinline__ void wmma_store(float* p, int stride,
                                           float c0, float c1, float c2, float c3,
                                           float c4, float c5, float c6, float c7);

__device__ void dequant_k256(unsigned short* dst, const unsigned char* kc, int k_pos, int kvh, int nb,
                             int nrows, int pos, int nvalid, int tid, int nthreads) {
    /* One scale for 32 values. The old loop reloaded that half 32 times. */
    for (int e = tid; e < nrows * 8; e += nthreads) {
        int r = e >> 3;
        int b = e & 7;
        unsigned short* row = dst + ((size_t)r << 8) + ((size_t)b << 5);
        if (r >= nvalid) {
            for (int j = 0; j < 32; ++j) row[j] = 0;
            continue;
        }
        const unsigned char* bb = kc + ((size_t)(pos + r) * (size_t)k_pos + (size_t)(kvh * nb + b) * 34);
        float d = ld_f16(bb);
        for (int j = 0; j < 32; ++j) row[j] = f32_to_f16(d * (float)(signed char)bb[2 + j]);
    }
}

__device__ void dequant_v256(unsigned short* dst, const unsigned char* vc, int v_pos, int kvh, int nb,
                             int nrows, int pos, int nvalid, int tid, int nthreads) {
    for (int e = tid; e < nrows * 8; e += nthreads) {
        int r = e >> 3;
        int b = e & 7;
        unsigned short* row = dst + ((size_t)r << 8) + ((size_t)b << 5);
        if (r >= nvalid) {
            for (int j = 0; j < 32; ++j) row[j] = 0;
            continue;
        }
        const unsigned char* vb = vc + ((size_t)(pos + r) * (size_t)v_pos + (size_t)(kvh * nb + b) * 40);
        float d = ld_f16(vb);
        int idx = (int)vb[2];
        float ov = ld_f16(vb + 4);
        for (int j = 0; j < 32; ++j) {
            float v = (j == idx) ? ov : d * (float)(signed char)vb[6 + j];
            row[j] = f32_to_f16(v);
        }
    }
}

__device__ size_t wmma_mi(int part, int q_tiles, int q_tile, int n_heads, int h, int bq, int r) {
    return (((size_t)part * (size_t)q_tiles + (size_t)q_tile) * (size_t)n_heads + (size_t)h) * (size_t)bq + (size_t)r;
}

#define WM_Z(p) p##0 = p##1 = p##2 = p##3 = p##4 = p##5 = p##6 = p##7 = 0.f
#define WM_LA(p, ptr, st) wmma_load_a(p##0, p##1, p##2, p##3, p##4, p##5, p##6, p##7, ptr, st)
#define WM_LB(p, ptr, st) wmma_load_b(p##0, p##1, p##2, p##3, p##4, p##5, p##6, p##7, ptr, st)
#define WM_LBR(p, ptr, st) wmma_load_b_row(p##0, p##1, p##2, p##3, p##4, p##5, p##6, p##7, ptr, st)
#define WM_M(c, a, b) wmma_mma(c##0, c##1, c##2, c##3, c##4, c##5, c##6, c##7, a##0, a##1, a##2, a##3, a##4, a##5, a##6, a##7, b##0, b##1, b##2, b##3, b##4, b##5, b##6, b##7)
#define WM_MR(c, a, b) wmma_mma_rr(c##0, c##1, c##2, c##3, c##4, c##5, c##6, c##7, a##0, a##1, a##2, a##3, a##4, a##5, a##6, a##7, b##0, b##1, b##2, b##3, b##4, b##5, b##6, b##7)
#define WM_ST(ptr, st, c) wmma_store(ptr, st, c##0, c##1, c##2, c##3, c##4, c##5, c##6, c##7)

extern "C" __global__ void attention_q8v_wmma(const float* q, const unsigned char* kc, const unsigned char* vc,
                                              float* partial_o, float* partial_m, float* partial_l,
                                              int head_dim, int k_pos_bytes, int v_pos_bytes, int kv_mul, int n_heads,
                                              int pos0, int T, int tok_stride, int q_tiles, int part_len) {
    const int BQ = 32;
    const int BC = 64;
    int q_tile = (int)blockIdx.x;
    int kvh = (int)blockIdx.y;
    int part = (int)blockIdx.z;
    int tid = (int)threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int q0 = q_tile * BQ;
    int ntok = BQ;
    if (q0 + ntok > T) ntok = T - q0;
    if (ntok < 0) ntok = 0;
    int kv_len = pos0 + T;
    int part_begin = part * part_len;
    int part_end = part_begin + part_len;
    if (part_end > kv_len) part_end = kv_len;
    if (head_dim != 256 || blockDim.x != 512 || kv_mul < 1 || kv_mul > 8 || ntok <= 0 || q0 >= T) return;
    int nb = head_dim >> 5;
    extern __shared__ __align__(16) unsigned char wmma_smem[];
    // 64*256 halves, then P for 8 heads, then Q, scores, alpha. 91136 bytes.
    unsigned short* Ks = (unsigned short*)wmma_smem;
    unsigned short* Ps = Ks + BC * 256;
    unsigned short* Qs = Ps + 8 * BQ * BC;
    float* S = (float*)(Qs + BQ * 256);
    float* alpha = S + BQ * BC;
    float* scratch = (float*)Qs;
    float scale = rsqrtf(256.f);
    int causal_end = pos0 + q0 + ntok;
    for (int qi = 0; qi < kv_mul; ++qi) {
        int h = kvh * kv_mul + qi;
        for (int r = 0; r < ntok; ++r) {
            size_t mi = wmma_mi(part, q_tiles, q_tile, n_heads, h, BQ, r);
            if (tid == 0) {
                partial_m[mi] = -1e30f;
                partial_l[mi] = 0.f;
            }
            for (int i = tid; i < 256; i += (int)blockDim.x) partial_o[mi * 256 + i] = 0.f;
        }
    }
    __syncthreads();
    if (part_begin >= part_end || part_begin >= causal_end) return;
    for (int kv = part_begin; kv < part_end; kv += BC) {
        if (kv >= causal_end) break;
        int nvalid = part_end - kv;
        if (nvalid > BC) nvalid = BC;
        int room = causal_end - kv;
        if (nvalid > room) nvalid = room;
        if (nvalid <= 0) break;
        dequant_k256(Ks, kc, k_pos_bytes, kvh, nb, BC, kv, nvalid, tid, (int)blockDim.x);
        __syncthreads();
        for (int qi = 0; qi < kv_mul; ++qi) {
            int h = kvh * kv_mul + qi;
            for (int e = tid; e < BQ * 256; e += (int)blockDim.x) {
                int r = e >> 8;
                int dim = e & 255;
                float v = 0.f;
                if (r < ntok) v = q[((size_t)(q0 + r) * (size_t)tok_stride + (size_t)h * 256) + dim];
                Qs[e] = f32_to_f16(v);
            }
            __syncthreads();
            if (warp < 8) {
                int row_group = warp >> 2;
                int col_group = warp & 3;
                float s0, s1, s2, s3, s4, s5, s6, s7;
                WM_Z(s);
                for (int kk = 0; kk < 256; kk += 16) {
                    unsigned a0, a1, a2, a3, a4, a5, a6, a7;
                    unsigned b0, b1, b2, b3, b4, b5, b6, b7;
                    WM_LA(a, Qs + row_group * 16 * 256 + kk, 256);
                    WM_LB(b, Ks + col_group * 16 * 256 + kk, 256);
                    WM_M(s, a, b);
                }
                WM_ST(S + row_group * 16 * BC + col_group * 16, BC, s);
            }
            __syncthreads();
            {
                int row = tid >> 4;
                int sub = tid & 15;
                unsigned half = (lane < 16) ? 0x0000ffffu : 0xffff0000u;
                int src = lane & 16;
                float mloc = -1e30f;
                for (int c = sub; c < BC; c += 16) {
                    float sv = S[row * BC + c];
                    int pos = kv + c;
                    int npos = pos0 + q0 + row + 1;
                    if (row >= ntok || c >= nvalid || pos >= npos) sv = -1e30f;
                    else sv *= scale;
                    S[row * BC + c] = sv;
                    mloc = fmaxf(mloc, sv);
                }
                float m_tile = half_max(mloc, half);
                size_t mi = wmma_mi(part, q_tiles, q_tile, n_heads, h, BQ, row);
                float m_old = -1e30f;
                float l_old = 0.f;
                if (sub == 0 && row < ntok) {
                    m_old = partial_m[mi];
                    l_old = partial_l[mi];
                }
                m_old = __shfl_sync(half, m_old, src);
                l_old = __shfl_sync(half, l_old, src);
                int live = m_tile >= -1e20f;
                float m_new = m_tile;
                float alpha_v = 1.f;
                if (live) {
                    m_new = fmaxf(m_old, m_tile);
                    alpha_v = expf(m_old - m_new);
                }
                unsigned short* Ph = Ps + qi * BQ * BC;
                float ploc = 0.f;
                for (int c = sub; c < BC; c += 16) {
                    float sv = S[row * BC + c];
                    float p = 0.f;
                    if (live && sv >= -1e20f) {
                        p = expf(sv - m_new);
                        ploc += p;
                    }
                    Ph[row * BC + c] = f32_to_f16(p);
                }
                ploc = half_sum(ploc, half);
                if (sub == 0 && row < ntok && live) {
                    partial_m[mi] = m_new;
                    partial_l[mi] = l_old * alpha_v + ploc;
                }
                if (sub == 0) alpha[qi * BQ + row] = alpha_v;
            }
            __syncthreads();
        }
        __syncthreads();
        dequant_v256(Ks, vc, v_pos_bytes, kvh, nb, BC, kv, nvalid, tid, (int)blockDim.x);
        __syncthreads();
        for (int qi = 0; qi < kv_mul; ++qi) {
            int h = kvh * kv_mul + qi;
            unsigned short* Ph = Ps + qi * BQ * BC;
            for (int rg = 0; rg < 2; ++rg) {
                float c0, c1, c2, c3, c4, c5, c6, c7;
                WM_Z(c);
                for (int kk = 0; kk < BC; kk += 16) {
                    unsigned a0, a1, a2, a3, a4, a5, a6, a7;
                    unsigned b0, b1, b2, b3, b4, b5, b6, b7;
                    WM_LA(a, Ph + rg * 16 * BC + kk, BC);
                    WM_LBR(b, Ks + kk * 256 + warp * 16, 256);
                    WM_MR(c, a, b);
                }
                WM_ST(scratch + warp * 256, 16, c);
                __syncthreads();
                for (int e = lane; e < 256; e += 32) {
                    int r = e >> 4;
                    int col = e & 15;
                    int qr = rg * 16 + r;
                    if (qr < ntok) {
                        size_t mi = wmma_mi(part, q_tiles, q_tile, n_heads, h, BQ, qr);
                        float* o = partial_o + mi * 256 + warp * 16 + col;
                        *o = *o * alpha[qi * BQ + qr] + scratch[warp * 256 + e];
                    }
                }
                __syncthreads();
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void attention_q8v_wmma_reduce(float* xb3, const float* partial_o, const float* partial_m,
                                                     const float* partial_l, int n_heads, int q_tiles, int nparts,
                                                     int pos0, int T, int tok_stride) {
    int h = (int)blockIdx.x;
    int q_tile = (int)blockIdx.y;
    int tid = (int)threadIdx.x;
    int q0 = q_tile * 32;
    int ntok = 32;
    if (q0 + ntok > T) ntok = T - q0;
    if (h >= n_heads || ntok <= 0 || q0 < 0) return;
    for (int r = 0; r < ntok; ++r) {
        float m = -1e30f;
        for (int p = 0; p < nparts; ++p) {
            m = fmaxf(m, partial_m[wmma_mi(p, q_tiles, q_tile, n_heads, h, 32, r)]);
        }
        float l = 0.f;
        for (int p = 0; p < nparts; ++p) {
            size_t mi = wmma_mi(p, q_tiles, q_tile, n_heads, h, 32, r);
            l += partial_l[mi] * expf(partial_m[mi] - m);
        }
        float inv = (l > 0.f) ? 1.f / l : 0.f;
        float* out = xb3 + ((size_t)(q0 + r) * (size_t)tok_stride + (size_t)h * 256);
        for (int i = tid; i < 256; i += (int)blockDim.x) {
            float acc = 0.f;
            for (int p = 0; p < nparts; ++p) {
                size_t mi = wmma_mi(p, q_tiles, q_tile, n_heads, h, 32, r);
                acc += partial_o[mi * 256 + i] * expf(partial_m[mi] - m);
            }
            out[i] = acc * inv;
        }
    }
    (void)pos0;
}

#undef WM_Z
#undef WM_LA
#undef WM_LB
#undef WM_LBR
#undef WM_M
#undef WM_MR
#undef WM_ST

extern "C" __global__ void conv1d_step(float* qkv_out, const float* qkv_in, float* conv_state, const float* weight,
                                       int conv_dim, int kernel) {
    int c = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= conv_dim) return;
    int km1 = kernel - 1;
    float acc = weight[c * kernel + km1] * qkv_in[c];
    for (int i = 0; i < km1; ++i) acc += weight[c * kernel + i] * conv_state[c * km1 + i];
    qkv_out[c] = acc * sigmoid_stable(acc);
    for (int i = 0; i + 1 < km1; ++i) conv_state[c * km1 + i] = conv_state[c * km1 + i + 1];
    if (km1 > 0) conv_state[c * km1 + km1 - 1] = qkv_in[c];
}

/* T steps of conv1d_step on one channel. Token t+1 reads the state token t
   just wrote, so the T loop stays on the thread. One launch per layer. */
extern "C" __global__ void conv1d_chunk(float* qkv_out, const float* qkv_in, float* conv_state, const float* weight,
                                        int conv_dim, int kernel, int T) {
    int c = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= conv_dim || T <= 0) return;
    int km1 = kernel - 1;
    for (int t = 0; t < T; ++t) {
        const float* in = qkv_in + (size_t)t * (size_t)conv_dim;
        float* out = qkv_out + (size_t)t * (size_t)conv_dim;
        float acc = weight[c * kernel + km1] * in[c];
        for (int i = 0; i < km1; ++i) acc += weight[c * kernel + i] * conv_state[c * km1 + i];
        out[c] = acc * sigmoid_stable(acc);
        for (int i = 0; i + 1 < km1; ++i) conv_state[c * km1 + i] = conv_state[c * km1 + i + 1];
        if (km1 > 0) conv_state[c * km1 + km1 - 1] = in[c];
    }
}

/* blockIdx.y is the token. q_in and k_in advance by q_stride (the conv
   width). q_out and k_out advance by head_k * n_heads, which the host
   passes as out_stride. gridDim.y == 1 is one token. */
extern "C" __global__ void l2norm_chunk(float* q_out, float* k_out, const float* q_in, const float* k_in,
                                        int head_k_dim, float eps, float scale, int q_stride, int out_stride) {
    int kh = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int t = (int)blockIdx.y;
    int n = head_k_dim;
    const float* qi = q_in + (size_t)t * (size_t)q_stride + (size_t)kh * (size_t)n;
    const float* ki = k_in + (size_t)t * (size_t)q_stride + (size_t)kh * (size_t)n;
    float* qo = q_out + (size_t)t * (size_t)out_stride + (size_t)kh * (size_t)n;
    float* ko = k_out + (size_t)t * (size_t)out_stride + (size_t)kh * (size_t)n;
    float sq = 0.f, sk = 0.f;
    for (int j = tid; j < n; j += 128) {
        sq += qi[j] * qi[j];
        sk += ki[j] * ki[j];
    }
    sq = block_sum_128(sq);
    sk = block_sum_128(sk);
    float iq = 1.f / fmaxf(sqrtf(sq), eps);
    float ik = 1.f / fmaxf(sqrtf(sk), eps);
    for (int j = tid; j < n; j += 128) {
        qo[j] = qi[j] * iq * scale;
        ko[j] = ki[j] * ik;
    }
}

extern "C" __global__ void l2norm_scale(float* q_out, float* k_out, const float* q_in, const float* k_in,
                                        int head_k_dim, float eps, float scale) {
    int kh = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int n = head_k_dim;
    const float* qi = q_in + (size_t)kh * (size_t)n;
    const float* ki = k_in + (size_t)kh * (size_t)n;
    float* qo = q_out + (size_t)kh * (size_t)n;
    float* ko = k_out + (size_t)kh * (size_t)n;
    float sq = 0.f, sk = 0.f;
    for (int j = tid; j < n; j += 128) {
        sq += qi[j] * qi[j];
        sk += ki[j] * ki[j];
    }
    sq = block_sum_128(sq);
    sk = block_sum_128(sk);
    float iq = 1.f / fmaxf(sqrtf(sq), eps);
    float ik = 1.f / fmaxf(sqrtf(sk), eps);
    for (int j = tid; j < n; j += 128) {
        qo[j] = qi[j] * iq * scale;
        ko[j] = ki[j] * ik;
    }
}

extern "C" __global__ void delta_recurrent(float* out, float* state_all, const float* q_lin, const float* k_lin,
                                           const float* v_vec, const float* b_vec, const float* a_vec,
                                           const float* a_decay, const float* dt_bias,
                                           int head_k_dim, int head_v_dim, int n_k_heads) {
    int vh = (int)blockIdx.x;
    int hkd = head_k_dim, hvd = head_v_dim;
    int kh = vh % n_k_heads;
    float* state = state_all + (size_t)vh * (size_t)hkd * (size_t)hvd;
    const float* qt = q_lin + (size_t)kh * (size_t)hkd;
    const float* kt = k_lin + (size_t)kh * (size_t)hkd;
    const float* vt = v_vec + (size_t)vh * (size_t)hvd;
    float beta = sigmoid_stable(b_vec[vh]);
    float g_decay = expf(a_decay[vh] * softplus_stable(a_vec[vh] + dt_bias[vh]));
    for (int j = (int)threadIdx.x; j < hvd; j += (int)blockDim.x) {
        for (int i = 0; i < hkd; ++i) state[(size_t)i * (size_t)hvd + j] *= g_decay;
        float kv_mem = 0.f;
        for (int i = 0; i < hkd; ++i) kv_mem += state[(size_t)i * (size_t)hvd + j] * kt[i];
        float delta = (vt[j] - kv_mem) * beta;
        for (int i = 0; i < hkd; ++i) state[(size_t)i * (size_t)hvd + j] += kt[i] * delta;
        float o = 0.f;
        for (int i = 0; i < hkd; ++i) o += state[(size_t)i * (size_t)hvd + j] * qt[i];
        out[(size_t)vh * (size_t)hvd + j] = o;
    }
}

/* One launch for a whole prefill chunk. Token t depends on the gated state
   left by token t-1, so the T loop stays inside the block. A parallel scan
   would be a different algorithm, not a local change. The work is O(T) per
   head: each token touches the head_k by head_v state a constant number of
   times. Heads stay parallel (one block per value head).
   delta_recurrent_chunk_sm is the same math with that matrix resident in
   dynamic shared memory for the chunk. The host uses it when the matrix is
   at most 64 KiB (Qwen3.5 is 128x128, 64 KiB) and cuFuncSetAttribute raised
   the dynamic-smem cap. Columns do not cross threads, so the only barriers
   are the load and the store. */
extern "C" __global__ void delta_recurrent_chunk(float* out, float* state_all, const float* q_lin, const float* k_lin,
                                                 const float* v_vec, const float* b_vec, const float* a_vec,
                                                 const float* a_decay, const float* dt_bias,
                                                 int head_k_dim, int head_v_dim, int n_k_heads, int n_v_heads, int T,
                                                 int q_stride, int v_stride) {
    int vh = (int)blockIdx.x;
    int hkd = head_k_dim, hvd = head_v_dim;
    int kh = vh % n_k_heads;
    float* state = state_all + (size_t)vh * (size_t)hkd * (size_t)hvd;
    for (int t = 0; t < T; ++t) {
        const float* qt = q_lin + (size_t)t * (size_t)q_stride + (size_t)kh * (size_t)hkd;
        const float* kt = k_lin + (size_t)t * (size_t)q_stride + (size_t)kh * (size_t)hkd;
        const float* vt = v_vec + (size_t)t * (size_t)v_stride + (size_t)vh * (size_t)hvd;
        float beta = sigmoid_stable(b_vec[(size_t)t * (size_t)n_v_heads + vh]);
        float g_decay = expf(a_decay[vh] * softplus_stable(a_vec[(size_t)t * (size_t)n_v_heads + vh] + dt_bias[vh]));
        float* oout = out + ((size_t)t * (size_t)n_v_heads + (size_t)vh) * (size_t)hvd;
        for (int j = (int)threadIdx.x; j < hvd; j += (int)blockDim.x) {
            for (int i = 0; i < hkd; ++i) state[(size_t)i * (size_t)hvd + j] *= g_decay;
            float kv_mem = 0.f;
            for (int i = 0; i < hkd; ++i) kv_mem += state[(size_t)i * (size_t)hvd + j] * kt[i];
            float delta = (vt[j] - kv_mem) * beta;
            for (int i = 0; i < hkd; ++i) state[(size_t)i * (size_t)hvd + j] += kt[i] * delta;
            float o = 0.f;
            for (int i = 0; i < hkd; ++i) o += state[(size_t)i * (size_t)hvd + j] * qt[i];
            oout[j] = o;
        }
        __syncthreads();
    }
}

extern "C" __global__ void delta_recurrent_chunk_sm(float* out, float* state_all, const float* q_lin, const float* k_lin,
                                                    const float* v_vec, const float* b_vec, const float* a_vec,
                                                    const float* a_decay, const float* dt_bias,
                                                    int head_k_dim, int head_v_dim, int n_k_heads, int n_v_heads, int T,
                                                    int q_stride, int v_stride) {
    extern __shared__ __align__(16) unsigned char delta_sm[];
    int vh = (int)blockIdx.x;
    int hkd = head_k_dim, hvd = head_v_dim;
    int tid = (int)threadIdx.x;
    int kh = vh % n_k_heads;
    float* sst = (float*)delta_sm;
    float* gst = state_all + (size_t)vh * (size_t)hkd * (size_t)hvd;
    int cells = hkd * hvd;
    for (int e = tid; e < cells; e += (int)blockDim.x) sst[e] = gst[e];
    __syncthreads();
    for (int t = 0; t < T; ++t) {
        const float* qt = q_lin + (size_t)t * (size_t)q_stride + (size_t)kh * (size_t)hkd;
        const float* kt = k_lin + (size_t)t * (size_t)q_stride + (size_t)kh * (size_t)hkd;
        const float* vt = v_vec + (size_t)t * (size_t)v_stride + (size_t)vh * (size_t)hvd;
        float beta = sigmoid_stable(b_vec[(size_t)t * (size_t)n_v_heads + vh]);
        float g_decay = expf(a_decay[vh] * softplus_stable(a_vec[(size_t)t * (size_t)n_v_heads + vh] + dt_bias[vh]));
        float* oout = out + ((size_t)t * (size_t)n_v_heads + (size_t)vh) * (size_t)hvd;
        for (int j = tid; j < hvd; j += (int)blockDim.x) {
            for (int i = 0; i < hkd; ++i) sst[(size_t)i * (size_t)hvd + j] *= g_decay;
            float kv_mem = 0.f;
            for (int i = 0; i < hkd; ++i) kv_mem += sst[(size_t)i * (size_t)hvd + j] * kt[i];
            float delta = (vt[j] - kv_mem) * beta;
            for (int i = 0; i < hkd; ++i) sst[(size_t)i * (size_t)hvd + j] += kt[i] * delta;
            float o = 0.f;
            for (int i = 0; i < hkd; ++i) o += sst[(size_t)i * (size_t)hvd + j] * qt[i];
            oout[j] = o;
        }
    }
    __syncthreads();
    for (int e = tid; e < cells; e += (int)blockDim.x) gst[e] = sst[e];
}

extern "C" __global__ void rmsnorm_gated(float* out, const float* inp, const float* weight, const float* gate, int size, float eps) {
    int vh = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    int t = (int)blockIdx.y;
    int stride = (int)gridDim.x * size;
    const float* xi = inp + (size_t)t * (size_t)stride + (size_t)vh * (size_t)size;
    float* oi = out + (size_t)t * (size_t)stride + (size_t)vh * (size_t)size;
    const float* zi = gate + (size_t)t * (size_t)stride + (size_t)vh * (size_t)size;
    float ss = 0.f;
    for (int j = tid; j < size; j += 128) ss += xi[j] * xi[j];
    ss = block_sum_128(ss);
    float inv = rsqrtf(ss / (float)size + eps);
    for (int j = tid; j < size; j += 128) {
        float normed = weight[j] * (inv * xi[j]);
        float gz = zi[j] * sigmoid_stable(zi[j]);
        oi[j] = normed * gz;
    }
}

/* Same norm, then q8 of this head. size % 32 == 0 so groups do not cross heads. */
extern "C" __global__ void rmsnorm_gated_quant(float* out, const float* inp, const float* weight, const float* gate,
                                               signed char* qs, float* scale, float2* sum16, int size, float eps) {
    int vh = (int)blockIdx.x;
    int tid = (int)threadIdx.x;
    const float* xi = inp + (size_t)vh * (size_t)size;
    float* oi = out + (size_t)vh * (size_t)size;
    const float* zi = gate + (size_t)vh * (size_t)size;
    float ss = 0.f;
    for (int j = tid; j < size; j += 128) ss += xi[j] * xi[j];
    ss = block_sum_128(ss);
    float inv = rsqrtf(ss / (float)size + eps);
    for (int j = tid; j < size; j += 128) {
        float normed = weight[j] * (inv * xi[j]);
        float gz = zi[j] * sigmoid_stable(zi[j]);
        oi[j] = normed * gz;
    }
    __syncthreads();
    int ng = size >> 5;
    for (int g = tid; g < ng; g += 128) {
        quant_store_group(oi + (g << 5), qs, scale, sum16, vh * ng + g);
    }
}

/* n <= 256. One thread. Matches moe_ffn: softmax, top-k, clamp, expert_w_scale. */
extern "C" __global__ void moe_softmax_topk(const float* logits, int* sel, float* weights, int n, int k, float clamp_v, float scale) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float s[256];
    int picked[256];
    float m = logits[0];
    for (int i = 1; i < n; ++i) m = fmaxf(m, logits[i]);
    float sum = 0.f;
    for (int i = 0; i < n; ++i) {
        s[i] = expf(logits[i] - m);
        sum += s[i];
        picked[i] = 0;
    }
    float inv = 1.f / sum;
    for (int i = 0; i < n; ++i) s[i] *= inv;
    for (int t = 0; t < k; ++t) {
        int best = 0;
        float bv = -1e30f;
        for (int j = 0; j < n; ++j) {
            if (!picked[j] && s[j] > bv) { bv = s[j]; best = j; }
        }
        sel[t] = best;
        picked[best] = 1;
        weights[t] = s[best];
    }
    float wsum = 0.f;
    for (int t = 0; t < k; ++t) wsum += weights[t];
    if (wsum < clamp_v) wsum = clamp_v;
    for (int t = 0; t < k; ++t) {
        weights[t] /= wsum;
        if (scale != 1.f && scale != 0.f) weights[t] *= scale;
    }
}

/* ---- Volta tensor-core GEMM -------------------------------------------------
   Y[T, d] = X[T, n] * W[d, n], fp32 accumulate. W is dequantized into fp16
   tiles in shared memory, X is narrowed to fp16, then one warp issues
   wmma 16x16x16. A block owns 64 tokens by 64 rows. T==1 never launches this.
   The hardware op under the WMMA is HMMA.884; the PTX we emit is the sm_70
   WMMA form, which NVRTC 12.9 accepts for compute_70. */

__device__ __forceinline__ unsigned short f32_to_f16(float v) {
    unsigned short h;
    st_f16((unsigned char*)&h, v);
    return h;
}

__device__ __forceinline__ void wmma_load_a(unsigned& a0, unsigned& a1, unsigned& a2, unsigned& a3,
                                            unsigned& a4, unsigned& a5, unsigned& a6, unsigned& a7,
                                            const unsigned short* p, int stride) {
    asm volatile(
        "wmma.load.a.sync.aligned.row.m16n16k16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;\n"
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3), "=r"(a4), "=r"(a5), "=r"(a6), "=r"(a7)
        : "l"(p), "r"(stride));
}

__device__ __forceinline__ void wmma_load_b(unsigned& b0, unsigned& b1, unsigned& b2, unsigned& b3,
                                            unsigned& b4, unsigned& b5, unsigned& b6, unsigned& b7,
                                            const unsigned short* p, int stride) {
    asm volatile(
        "wmma.load.b.sync.aligned.col.m16n16k16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;\n"
        : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3), "=r"(b4), "=r"(b5), "=r"(b6), "=r"(b7)
        : "l"(p), "r"(stride));
}

__device__ __forceinline__ void wmma_mma(float& c0, float& c1, float& c2, float& c3,
                                         float& c4, float& c5, float& c6, float& c7,
                                         unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                         unsigned a4, unsigned a5, unsigned a6, unsigned a7,
                                         unsigned b0, unsigned b1, unsigned b2, unsigned b3,
                                         unsigned b4, unsigned b5, unsigned b6, unsigned b7) {
    asm volatile(
        "wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, "
        "{%8,%9,%10,%11,%12,%13,%14,%15}, "
        "{%16,%17,%18,%19,%20,%21,%22,%23}, "
        "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3), "+f"(c4), "+f"(c5), "+f"(c6), "+f"(c7)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5), "r"(a6), "r"(a7),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(b4), "r"(b5), "r"(b6), "r"(b7));
}

__device__ __forceinline__ void wmma_store(float* p, int stride,
                                           float c0, float c1, float c2, float c3,
                                           float c4, float c5, float c6, float c7) {
    asm volatile(
        "wmma.store.d.sync.aligned.row.m16n16k16.f32 [%0], "
        "{%1,%2,%3,%4,%5,%6,%7,%8}, %9;\n"
        :: "l"(p), "f"(c0), "f"(c1), "f"(c2), "f"(c3), "f"(c4), "f"(c5), "f"(c6), "f"(c7), "r"(stride));
}

__device__ __forceinline__ void wmma_load_b_row(unsigned& b0, unsigned& b1, unsigned& b2, unsigned& b3,
                                                unsigned& b4, unsigned& b5, unsigned& b6, unsigned& b7,
                                                const unsigned short* p, int stride) {
    asm volatile(
        "wmma.load.b.sync.aligned.row.m16n16k16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8], %9;\n"
        : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3), "=r"(b4), "=r"(b5), "=r"(b6), "=r"(b7)
        : "l"(p), "r"(stride));
}

__device__ __forceinline__ void wmma_mma_rr(float& c0, float& c1, float& c2, float& c3,
                                            float& c4, float& c5, float& c6, float& c7,
                                            unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                            unsigned a4, unsigned a5, unsigned a6, unsigned a7,
                                            unsigned b0, unsigned b1, unsigned b2, unsigned b3,
                                            unsigned b4, unsigned b5, unsigned b6, unsigned b7) {
    asm volatile(
        "wmma.mma.sync.aligned.row.row.m16n16k16.f32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7}, "
        "{%8,%9,%10,%11,%12,%13,%14,%15}, "
        "{%16,%17,%18,%19,%20,%21,%22,%23}, "
        "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3), "+f"(c4), "+f"(c5), "+f"(c6), "+f"(c7)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5), "r"(a6), "r"(a7),
          "r"(b0), "r"(b1), "r"(b2), "r"(b3), "r"(b4), "r"(b5), "r"(b6), "r"(b7));
}

/* Chunked gated delta rule for a 128x128 state. One block per value head,
   256 threads. A chunk of 16 tokens is the WY form used by the flash-linear-
   attention reference (strictly-lower key system, solved in fp32, then the
   state products). Those products are WMMA 16x16x16: fp16 operands, fp32
   accumulate. The caller l2-normalizes k (k·k ≤ 1) and scales q by
   1/sqrt(head_k); an unnormalized key makes the fp32 WY substitution
   overflow. The same recurrence in fp32 is ODINFER_CUDA_DELTA_WMMA=0.
   cuda_selftest allows 2e-3 against delta_recurrent_step on 128x128,
   including a partial chunk, with those normalized inputs. Head dims other
   than 128 stay on delta_recurrent_chunk_sm. */
extern "C" __global__ void delta_chunk_wmma(float* out, float* state_all, const float* q_lin, const float* k_lin,
                                            const float* v_vec, const float* b_vec, const float* a_vec,
                                            const float* a_decay, const float* dt_bias,
                                            int head_k_dim, int head_v_dim, int n_k_heads, int n_v_heads, int T,
                                            int q_stride, int v_stride) {
    if (head_k_dim != 128 || head_v_dim != 128) return;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int vh = (int)blockIdx.x;
    const int kh = vh % n_k_heads;
    float* state = state_all + (size_t)vh * 128 * 128;
    extern __shared__ __align__(16) unsigned char delta_wmma_sm[];
    float* kbuf = (float*)delta_wmma_sm;
    float* vbuf = kbuf + 2048;
    float* qbuf = vbuf + 2048;
    float* cdbuf = qbuf + 2048;
    float* att = cdbuf + 2048;
    float* attq = att + 256;
    float* decay = attq + 256;
    float* beta = decay + 16;
    unsigned short* As = (unsigned short*)(delta_wmma_sm + 35456);
    unsigned short* Bs = As + 512;
    float* store = (float*)(delta_wmma_sm + 40576);

    for (int base = 0; base < T; base += 16) {
        int nloc = T - base;
        if (nloc > 16) nloc = 16;
        for (int e = tid; e < 2048; e += 256) {
            kbuf[e] = 0.f;
            qbuf[e] = 0.f;
            vbuf[e] = 0.f;
        }
        if (tid < 16) {
            beta[tid] = 0.f;
            decay[tid] = 1.f;
            if (tid < nloc) {
                int t = base + tid;
                beta[tid] = sigmoid_stable(b_vec[(size_t)t * (size_t)n_v_heads + vh]);
                float gd = expf(a_decay[vh] * softplus_stable(a_vec[(size_t)t * (size_t)n_v_heads + vh] + dt_bias[vh]));
                if (gd < 1e-20f) gd = 1e-20f;
                if (gd > 1.f) gd = 1.f;
                decay[tid] = gd;
            }
        }
        for (int e = tid; e < nloc * 128; e += 256) {
            int tok = e >> 7;
            int d = e & 127;
            int t = base + tok;
            kbuf[e] = k_lin[(size_t)t * (size_t)q_stride + (size_t)kh * 128 + d];
            qbuf[e] = q_lin[(size_t)t * (size_t)q_stride + (size_t)kh * 128 + d];
            vbuf[e] = v_vec[(size_t)t * (size_t)v_stride + (size_t)vh * 128 + d];
        }
        __syncthreads();
        {
            int i = tid >> 4;
            int j = tid & 15;
            float dk = 0.f, dq = 0.f;
            for (int d = 0; d < 128; ++d) {
                float kd = kbuf[j * 128 + d];
                dk += kbuf[i * 128 + d] * kd;
                dq += qbuf[i * 128 + d] * kd;
            }
            att[i * 16 + j] = dk;
            attq[i * 16 + j] = dq;
        }
        __syncthreads();
        if (tid == 0) {
            float dec[16];
            float sum = 0.f;
            for (int i = 0; i < 16; ++i) {
                sum += logf(decay[i]);
                dec[i] = sum;
            }
            float L[256], M[256];
            for (int i = 0; i < 16; ++i) {
                for (int j = 0; j < 16; ++j) {
                    float ell = (i >= j) ? expf(dec[i] - dec[j]) : 0.f;
                    L[i * 16 + j] = ell;
                    float kk = att[i * 16 + j];
                    M[i * 16 + j] = (j < i) ? -beta[i] * kk * ell : 0.f;
                }
            }
            for (int i = 1; i < 16; ++i) {
                float old[16];
                for (int p = 0; p < i; ++p) old[p] = M[i * 16 + p];
                for (int j = 0; j < i; ++j) {
                    float s = old[j];
                    for (int p = 0; p < i; ++p) s += old[p] * M[p * 16 + j];
                    M[i * 16 + j] = s;
                }
            }
            for (int i = 0; i < 16; ++i) {
                for (int j = 0; j < 16; ++j) {
                    float a = M[i * 16 + j];
                    if (i == j) a += 1.f;
                    att[i * 16 + j] = a;
                    float qk = attq[i * 16 + j];
                    attq[i * 16 + j] = (j <= i) ? qk * L[i * 16 + j] : 0.f;
                }
            }
            for (int i = 0; i < 16; ++i) decay[i] = dec[i];
        }
        __syncthreads();
        /* The per-column scale does not depend on the row i, and exp(decay)
           was being recomputed 16 times. tid and tid+128 own the same d and
           split the 16 rows. Each thread writes only its own i, so the old
           per-row syncthreads around row[] is not required. One sync publishes
           cdbuf and qbuf before kbuf is overwritten. */
        {
            int d = tid & 127;
            int i0 = (tid < 128) ? 0 : 8;
            float wk[16];
#pragma unroll
            for (int p = 0; p < 16; ++p) {
                wk[p] = kbuf[p * 128 + d] * beta[p] * expf(decay[p]);
            }
#pragma unroll
            for (int i = i0; i < i0 + 8; ++i) {
                float s = 0.f;
#pragma unroll
                for (int p = 0; p < 16; ++p) s += att[i * 16 + p] * wk[p];
                cdbuf[i * 128 + d] = s;
                qbuf[i * 128 + d] *= expf(decay[i]);
            }
        }
        __syncthreads();
        {
            int d = tid & 127;
            int i0 = (tid < 128) ? 0 : 8;
            float vv[16];
#pragma unroll
            for (int p = 0; p < 16; ++p) vv[p] = vbuf[p * 128 + d] * beta[p];
#pragma unroll
            for (int i = i0; i < i0 + 8; ++i) {
                float s = 0.f;
#pragma unroll
                for (int p = 0; p < 16; ++p) s += att[i * 16 + p] * vv[p];
                kbuf[i * 128 + d] = s;
            }
        }
        __syncthreads();
        float vp0 = 0.f, vp1 = 0.f, vp2 = 0.f, vp3 = 0.f, vp4 = 0.f, vp5 = 0.f, vp6 = 0.f, vp7 = 0.f;
        float oi0 = 0.f, oi1 = 0.f, oi2 = 0.f, oi3 = 0.f, oi4 = 0.f, oi5 = 0.f, oi6 = 0.f, oi7 = 0.f;
        int v0 = warp * 16;
        for (int kk = 0; kk < 8; ++kk) {
            int k0 = kk * 16;
            if (warp == 0) {
                for (int e = lane; e < 256; e += 32) {
                    int r = e >> 4;
                    int c = e & 15;
                    As[r * 16 + c] = f32_to_f16(cdbuf[r * 128 + k0 + c]);
                    As[256 + r * 16 + c] = f32_to_f16(qbuf[r * 128 + k0 + c]);
                }
            }
            unsigned short* Bt = Bs + warp * 256;
            for (int e = lane; e < 256; e += 32) {
                int r = e >> 4;
                int c = e & 15;
                Bt[r * 16 + c] = f32_to_f16(state[((size_t)k0 + r) * 128 + v0 + c]);
            }
            __syncthreads();
            unsigned a0, a1, a2, a3, a4, a5, a6, a7;
            unsigned b0, b1, b2, b3, b4, b5, b6, b7;
            unsigned aq0, aq1, aq2, aq3, aq4, aq5, aq6, aq7;
            wmma_load_a(a0, a1, a2, a3, a4, a5, a6, a7, As, 16);
            wmma_load_a(aq0, aq1, aq2, aq3, aq4, aq5, aq6, aq7, As + 256, 16);
            wmma_load_b_row(b0, b1, b2, b3, b4, b5, b6, b7, Bt, 16);
            wmma_mma_rr(vp0, vp1, vp2, vp3, vp4, vp5, vp6, vp7, a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7);
            wmma_mma_rr(oi0, oi1, oi2, oi3, oi4, oi5, oi6, oi7, aq0, aq1, aq2, aq3, aq4, aq5, aq6, aq7, b0, b1, b2, b3, b4, b5, b6, b7);
            __syncthreads();
        }
        float* mine = store + warp * 256;
        wmma_store(mine, 16, vp0, vp1, vp2, vp3, vp4, vp5, vp6, vp7);
        for (int e = lane; e < 256; e += 32) {
            int r = e >> 4;
            int c = e & 15;
            vbuf[r * 128 + v0 + c] = mine[r * 16 + c];
        }
        wmma_store(mine, 16, oi0, oi1, oi2, oi3, oi4, oi5, oi6, oi7);
        for (int e = lane; e < 256; e += 32) {
            int r = e >> 4;
            int c = e & 15;
            cdbuf[r * 128 + v0 + c] = mine[r * 16 + c];
        }
        __syncthreads();
        for (int e = tid; e < 2048; e += 256) vbuf[e] = kbuf[e] - vbuf[e];
        __syncthreads();
        {
            int d = tid & 127;
            int i0 = (tid < 128) ? 0 : 8;
            float vv[16];
#pragma unroll
            for (int p = 0; p < 16; ++p) vv[p] = vbuf[p * 128 + d];
#pragma unroll
            for (int i = i0; i < i0 + 8; ++i) {
                float s = cdbuf[i * 128 + d];
#pragma unroll
                for (int p = 0; p < 16; ++p) s += attq[i * 16 + p] * vv[p];
                cdbuf[i * 128 + d] = s;
                if (i < nloc) {
                    float* oout = out + ((size_t)(base + i) * (size_t)n_v_heads + (size_t)vh) * 128;
                    oout[d] = s;
                }
            }
        }
        __syncthreads();
        float last = expf(decay[15]);
        for (int nt = 0; nt < 8; ++nt) {
            int col = nt * 16;
            int k0 = warp * 16;
            unsigned short* tile = Bs + warp * 256;
            for (int e = lane; e < 256; e += 32) {
                int r = e >> 4;
                int c = e & 15;
                float ks = 0.f;
                if (c < nloc) {
                    int t = base + c;
                    float kd = k_lin[(size_t)t * (size_t)q_stride + (size_t)kh * 128 + (k0 + r)];
                    ks = kd * expf(decay[15] - decay[c]);
                }
                tile[r * 16 + c] = f32_to_f16(ks);
            }
            __syncwarp();
            unsigned a0, a1, a2, a3, a4, a5, a6, a7;
            unsigned b0, b1, b2, b3, b4, b5, b6, b7;
            float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f, c4 = 0.f, c5 = 0.f, c6 = 0.f, c7 = 0.f;
            wmma_load_a(a0, a1, a2, a3, a4, a5, a6, a7, tile, 16);
            for (int e = lane; e < 256; e += 32) {
                int r = e >> 4;
                int c = e & 15;
                tile[r * 16 + c] = f32_to_f16(vbuf[r * 128 + col + c]);
            }
            __syncwarp();
            wmma_load_b_row(b0, b1, b2, b3, b4, b5, b6, b7, tile, 16);
            wmma_mma_rr(c0, c1, c2, c3, c4, c5, c6, c7, a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7);
            wmma_store(mine, 16, c0, c1, c2, c3, c4, c5, c6, c7);
            for (int e = lane; e < 256; e += 32) {
                int r = e >> 4;
                int c = e & 15;
                float* sp = state + ((size_t)k0 + r) * 128 + col + c;
                *sp = *sp * last + mine[r * 16 + c];
            }
            __syncthreads();
        }
    }
}

__device__ float w_f32(const unsigned char* wb, int row, int k, int n) {
    return ld_f32(wb + ((size_t)row * (size_t)n + (size_t)k) * 4);
}
__device__ float w_f16(const unsigned char* wb, int row, int k, int n) {
    return ld_f16(wb + ((size_t)row * (size_t)n + (size_t)k) * 2);
}
__device__ float w_q8_0(const unsigned char* wb, int row, int k, int n) {
    int nb = n >> 5;
    const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)(k >> 5)) * 34;
    return ld_f16(blk) * (float)(signed char)blk[2 + (k & 31)];
}
__device__ float w_q4_0(const unsigned char* wb, int row, int k, int n) {
    int nb = n >> 5;
    int inner = k & 31;
    int j = inner & 15;
    int hi = inner >> 4;
    const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)(k >> 5)) * 18;
    unsigned char q = blk[2 + j];
    int qv = hi ? (int)(q >> 4) - 8 : (int)(q & 0xF) - 8;
    return ld_f16(blk) * (float)qv;
}
__device__ float w_q4_1(const unsigned char* wb, int row, int k, int n) {
    int nb = n >> 5;
    int inner = k & 31;
    int j = inner & 15;
    int hi = inner >> 4;
    const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)(k >> 5)) * 20;
    unsigned char q = blk[4 + j];
    int qv = hi ? (int)(q >> 4) : (int)(q & 0xF);
    return ld_f16(blk) * (float)qv + ld_f16(blk + 2);
}
__device__ float w_q5_0(const unsigned char* wb, int row, int k, int n) {
    int nb = n >> 5;
    int inner = k & 31;
    int j = inner & 15;
    int hi = inner >> 4;
    const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)(k >> 5)) * 22;
    unsigned qh = ld_u32(blk + 2);
    unsigned char qs = blk[6 + j];
    int qv;
    if (hi == 0) {
        unsigned xh = ((qh >> j) << 4) & 0x10u;
        qv = (int)((qs & 0xF) | xh) - 16;
    } else {
        unsigned xh = (qh >> (j + 12)) & 0x10u;
        qv = (int)((qs >> 4) | xh) - 16;
    }
    return ld_f16(blk) * (float)qv;
}
__device__ float w_q5_1(const unsigned char* wb, int row, int k, int n) {
    int nb = n >> 5;
    int inner = k & 31;
    int j = inner & 15;
    int hi = inner >> 4;
    const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)(k >> 5)) * 24;
    unsigned qh = ld_u32(blk + 4);
    unsigned char qs = blk[8 + j];
    int qv;
    if (hi == 0) {
        unsigned xh = ((qh >> j) << 4) & 0x10u;
        qv = (int)((qs & 0xF) | xh);
    } else {
        unsigned xh = (qh >> (j + 12)) & 0x10u;
        qv = (int)((qs >> 4) | xh);
    }
    return ld_f16(blk) * (float)qv + ld_f16(blk + 2);
}
__device__ float w_q4_k(const unsigned char* wb, int row, int k, int n) {
    int nb = n >> 8;
    const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)(k >> 8)) * 144;
    int inner = k & 255;
    int p = inner >> 6;
    int within = inner & 63;
    int hi = within >> 5;
    int lane = within & 31;
    int sc, m;
    scale_min_k4(p * 2 + hi, blk + 4, &sc, &m);
    unsigned char q = blk[16 + p * 32 + lane];
    int qv = hi ? (int)(q >> 4) : (int)(q & 0xF);
    float d = ld_f16(blk);
    float dmin = ld_f16(blk + 2);
    return (float)qv * (d * (float)sc) - (dmin * (float)m);
}
__device__ float w_q6_k(const unsigned char* wb, int row, int k, int n) {
    int nb = n >> 8;
    const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)(k >> 8)) * 210;
    int inner = k & 255;
    int half = inner >> 7;
    int rem = inner & 127;
    int group = rem >> 5;
    int l = rem & 31;
    const unsigned char* ql = blk;
    const unsigned char* qh = blk + 128;
    const unsigned char* scb = blk + 192;
    int qloff = half * 64;
    int qhoff = half * 32;
    int scoff = half * 8;
    unsigned char qh1 = qh[qhoff + l];
    int qv, sc_i;
    int is = l >> 4;
    if (group == 0) {
        qv = (int)((ql[qloff + l] & 0xF) | (((qh1 >> 0) & 3) << 4)) - 32;
        sc_i = scoff + is;
    } else if (group == 1) {
        qv = (int)((ql[qloff + l + 32] & 0xF) | (((qh1 >> 2) & 3) << 4)) - 32;
        sc_i = scoff + is + 2;
    } else if (group == 2) {
        qv = (int)((ql[qloff + l] >> 4) | (((qh1 >> 4) & 3) << 4)) - 32;
        sc_i = scoff + is + 4;
    } else {
        qv = (int)((ql[qloff + l + 32] >> 4) | (((qh1 >> 6) & 3) << 4)) - 32;
        sc_i = scoff + is + 6;
    }
    float d = ld_f16(blk + 208);
    return d * (float)(signed char)scb[sc_i] * (float)qv;
}
__device__ float w_q8_k(const unsigned char* wb, int row, int k, int n) {
    int nb = n >> 8;
    const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)(k >> 8)) * 292;
    return ld_f32(blk) * (float)(signed char)blk[4 + (k & 255)];
}
__device__ float w_q2_k(const unsigned char* wb, int row, int k, int n) {
    int nb = n >> 8;
    const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)(k >> 8)) * 84;
    int inner = k & 255;
    int group = inner >> 5;
    int half = group >> 2;
    int s = group & 3;
    int shift = s * 2;
    int qoff = half * 32;
    int within = inner & 31;
    int l = within & 15;
    int hi = within >> 4;
    unsigned char sc = blk[group * 2 + hi];
    float d = ld_f16(blk + 80);
    float dmin = ld_f16(blk + 82);
    float dl = d * (float)(sc & 0xF);
    float ml = dmin * (float)(sc >> 4);
    unsigned char byte = blk[16 + qoff + hi * 16 + l];
    return dl * (float)((byte >> shift) & 3) - ml;
}
__device__ float w_q3_k(const unsigned char* wb, int row, int k, int n) {
    const unsigned KMASK1 = 0x03030303u;
    const unsigned KMASK2 = 0x0f0f0f0fu;
    int nb = n >> 8;
    const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)(k >> 8)) * 110;
    int inner = k & 255;
    unsigned aux0 = ld_u32(blk + 96);
    unsigned aux1 = ld_u32(blk + 100);
    unsigned tmp = ld_u32(blk + 104);
    unsigned aux2 = ((aux0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    unsigned aux3 = ((aux1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux0 = (aux0 & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
    aux1 = (aux1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    unsigned aux[4];
    aux[0] = aux0; aux[1] = aux1; aux[2] = aux2; aux[3] = aux3;
    int group = inner >> 5;
    int half = group >> 2;
    int s = group & 3;
    int shift = s * 2;
    int qoff = half * 32;
    int within = inner & 31;
    int l = within & 15;
    int hi = within >> 4;
    int is = group * 2 + hi;
    int sc = (int)(signed char)((aux[is / 4] >> (8 * (is & 3))) & 0xff);
    float dl = ld_f16(blk + 108) * (float)(sc - 32);
    unsigned mbit = 1u << group;
    const unsigned char* hm = blk + (hi ? 16 : 0);
    const unsigned char* qs = blk + 32;
    int q = (int)((qs[qoff + hi * 16 + l] >> shift) & 3) - ((hm[l] & mbit) == 0 ? 4 : 0);
    return dl * (float)q;
}
__device__ float w_q5_k(const unsigned char* wb, int row, int k, int n) {
    int nb = n >> 8;
    const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)(k >> 8)) * 176;
    int inner = k & 255;
    int p = inner >> 6;
    int within = inner & 63;
    int hi = within >> 5;
    int l = within & 31;
    int sc, m;
    scale_min_k4(p * 2 + hi, blk + 4, &sc, &m);
    float d = ld_f16(blk);
    float dmin = ld_f16(blk + 2);
    const unsigned char* qh = blk + 16;
    const unsigned char* ql = blk + 48;
    unsigned u = 1u << (2 * p + hi);
    int nib = hi ? (int)(ql[p * 32 + l] >> 4) : (int)(ql[p * 32 + l] & 0xF);
    int v = nib + ((qh[l] & u) ? 16 : 0);
    return (d * (float)sc) * (float)v - (dmin * (float)m);
}

template<int KIND>
__device__ float tc_w(const unsigned char* wb, int row, int k, int n) {
    if (KIND == 0) return w_f32(wb, row, k, n);
    if (KIND == 1) return w_f16(wb, row, k, n);
    if (KIND == 2) return w_q4_0(wb, row, k, n);
    if (KIND == 3) return w_q4_1(wb, row, k, n);
    if (KIND == 4) return w_q5_0(wb, row, k, n);
    if (KIND == 5) return w_q5_1(wb, row, k, n);
    if (KIND == 6) return w_q8_0(wb, row, k, n);
    if (KIND == 7) return w_q2_k(wb, row, k, n);
    if (KIND == 8) return w_q3_k(wb, row, k, n);
    if (KIND == 9) return w_q4_k(wb, row, k, n);
    if (KIND == 10) return w_q5_k(wb, row, k, n);
    if (KIND == 11) return w_q6_k(wb, row, k, n);
    return w_q8_k(wb, row, k, n);
}

/* KIND matches the order in bind_kernels. Block: 512 threads, 16 warps.
   grid: (ceil(d/64), ceil(T/64)). */
template<int KIND>
__device__ void tc_run(const unsigned char* wb, const float* x, float* out, int n, int d, int T) {
    const int BM = 64;
    const int BN = 64;
    const int BK = 32;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int wm = warp >> 2;
    const int wn = warp & 3;
    const int t0 = (int)blockIdx.y * BM;
    const int n0 = (int)blockIdx.x * BN;
    if (t0 >= T || n0 >= d) return;

    __shared__ __align__(16) unsigned char sm[16 * 256 * 4];
    unsigned short* Xs = (unsigned short*)sm;
    unsigned short* Ws = (unsigned short*)(sm + BM * BK * 2);

    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f, c4 = 0.f, c5 = 0.f, c6 = 0.f, c7 = 0.f;
    for (int k0 = 0; k0 < n; k0 += BK) {
        for (int i = tid; i < BM * BK; i += 512) {
            int t = i / BK;
            int kk = i - t * BK;
            float v = 0.f;
            if (t0 + t < T && k0 + kk < n) v = x[((size_t)t0 + t) * (size_t)n + (size_t)k0 + kk];
            Xs[i] = f32_to_f16(v);
        }
        for (int i = tid; i < BN * BK; i += 512) {
            int row = i / BK;
            int kk = i - row * BK;
            float v = 0.f;
            if (n0 + row < d && k0 + kk < n) v = tc_w<KIND>(wb, n0 + row, k0 + kk, n);
            Ws[i] = f32_to_f16(v);
        }
        __syncthreads();
        for (int kk = 0; kk < BK; kk += 16) {
            unsigned a0, a1, a2, a3, a4, a5, a6, a7;
            unsigned b0, b1, b2, b3, b4, b5, b6, b7;
            wmma_load_a(a0, a1, a2, a3, a4, a5, a6, a7, Xs + (wm * 16) * BK + kk, BK);
            wmma_load_b(b0, b1, b2, b3, b4, b5, b6, b7, Ws + (wn * 16) * BK + kk, BK);
            wmma_mma(c0, c1, c2, c3, c4, c5, c6, c7, a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7);
        }
        __syncthreads();
    }
    float* tile = ((float*)sm) + warp * 256;
    wmma_store(tile, 16, c0, c1, c2, c3, c4, c5, c6, c7);
    __syncthreads();
    for (int idx = tid; idx < 16 * 256; idx += 512) {
        int w = idx >> 8;
        int e = idx & 255;
        int i = e >> 4;
        int j = e & 15;
        int tw = w >> 2;
        int nw = w & 3;
        int t = t0 + tw * 16 + i;
        int row = n0 + nw * 16 + j;
        if (t < T && row < d) out[(size_t)t * (size_t)d + row] = ((float*)sm)[idx];
    }
}

#define DEF_TC(tag, kind) \
extern "C" __global__ void tc_##tag(const unsigned char* wb, const float* x, float* out, int n, int d, int T) { \
    tc_run<kind>(wb, x, out, n, d, T); \
}

DEF_TC(f32, 0)
DEF_TC(f16, 1)
DEF_TC(q4_0, 2)
DEF_TC(q4_1, 3)
DEF_TC(q5_0, 4)
DEF_TC(q5_1, 5)
DEF_TC(q8_0, 6)
DEF_TC(q2_k, 7)
DEF_TC(q3_k, 8)
DEF_TC(q4_k, 9)
DEF_TC(q5_k, 10)
DEF_TC(q6_k, 11)
DEF_TC(q8_k, 12)

#undef DEF_TC

/* Q4_K / Q6_K prefill, wide tile. A block owns 32 output rows and 128
   tokens, in two 64-token WMMA chunks that share one fp16 weight tile.
   The host uses this for T>64. T<=64 uses tc2s (32 tokens, BK=256): at
   T=19 this tile still MMAs 64 tokens per K-step and measured 323 ms
   against tc2s at 176 ms. ODINFER_CUDA_TC2=0 selects the scalar tile. */
template<int Q6>
__device__ void tc2_dequant_row(const unsigned char* blk, int part, unsigned short* dst) {
    if (Q6 == 0) {
        int sc, m;
        scale_min_k4(part, blk + 4, &sc, &m);
        float ds = ld_f16(blk) * (float)sc;
        float dm = ld_f16(blk + 2) * (float)m;
        int p = part >> 1;
        int hi = part & 1;
        const unsigned char* qs = blk + 16 + p * 32;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            unsigned char q = qs[j];
            int qv = hi ? (int)(q >> 4) : (int)(q & 0xF);
            dst[j] = f32_to_f16((float)qv * ds - dm);
        }
        return;
    }
    int half = part >> 2;
    int group = part & 3;
    float d = ld_f16(blk + 208);
    const unsigned char* ql = blk + half * 64;
    const unsigned char* qh = blk + 128 + half * 32;
    const unsigned char* scb = blk + 192 + half * 8;
#pragma unroll
    for (int l = 0; l < 32; ++l) {
        unsigned char qh1 = qh[l];
        int is = l >> 4;
        int qv, sc_i;
        if (group == 0) {
            qv = (int)((ql[l] & 0xF) | (((qh1 >> 0) & 3) << 4)) - 32;
            sc_i = is;
        } else if (group == 1) {
            qv = (int)((ql[l + 32] & 0xF) | (((qh1 >> 2) & 3) << 4)) - 32;
            sc_i = is + 2;
        } else if (group == 2) {
            qv = (int)((ql[l] >> 4) | (((qh1 >> 4) & 3) << 4)) - 32;
            sc_i = is + 4;
        } else {
            qv = (int)((ql[l + 32] >> 4) | (((qh1 >> 6) & 3) << 4)) - 32;
            sc_i = is + 6;
        }
        dst[l] = f32_to_f16(d * (float)(signed char)scb[sc_i] * (float)qv);
    }
}

template<int Q6>
__device__ void tc2_mma_k(unsigned short* Xs, unsigned short* Ws, int BK, int warp,
                          float& c0, float& c1, float& c2, float& c3, float& c4, float& c5, float& c6, float& c7) {
    /* 8 warps cover a 64x32 tile: warp>>1 is the token group, warp&1 the row group. */
    int wm = warp >> 1;
    int wn = warp & 1;
#pragma unroll 1
    for (int kk = 0; kk < BK; kk += 16) {
        unsigned a0, a1, a2, a3, a4, a5, a6, a7;
        unsigned b0, b1, b2, b3, b4, b5, b6, b7;
        wmma_load_a(a0, a1, a2, a3, a4, a5, a6, a7, Xs + (wm * 16) * BK + kk, BK);
        wmma_load_b(b0, b1, b2, b3, b4, b5, b6, b7, Ws + (wn * 16) * BK + kk, BK);
        wmma_mma(c0, c1, c2, c3, c4, c5, c6, c7, a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7);
    }
}

__device__ void tc2_scatter(float* tile, float* out, int tid, int nwarps, int tbase, int n0, int T, int d) {
    for (int idx = tid; idx < nwarps * 256; idx += 256) {
        int w = idx >> 8;
        int e = idx & 255;
        int i = e >> 4;
        int j = e & 15;
        int wm = w >> 1;
        int wn = w & 1;
        int t = tbase + wm * 16 + i;
        int orow = n0 + wn * 16 + j;
        if (t < T && orow < d) out[(size_t)t * (size_t)d + orow] = tile[idx];
    }
}

template<int Q6>
__device__ void tc2_run(const unsigned char* wb, const float* x, float* out, int n, int d, int T) {
    const int BM = 64;
    const int BN = 32;
    const int BK = 128;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int t0 = (int)blockIdx.y * (BM * 2);
    const int n0 = (int)blockIdx.x * BN;
    if (t0 >= T || n0 >= d) return;
    int bbytes = Q6 ? 210 : 144;
    int nb = n >> 8;
    int second = (t0 + BM) < T;

    __shared__ __align__(16) unsigned char raw[64 * 128 * 2 + 32 * 128 * 2];
    unsigned short* Xs = (unsigned short*)raw;
    unsigned short* Ws = Xs + BM * BK;

    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f;
    float b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f, b4 = 0.f, b5 = 0.f, b6 = 0.f, b7 = 0.f;
    for (int k0 = 0; k0 < n; k0 += BK) {
        if (tid < BN * 4) {
            int row = tid & (BN - 1);
            int part = ((k0 & 255) >> 5) + (tid >> 5);
            unsigned short* dst = Ws + row * BK + (tid >> 5) * 32;
            if (n0 + row < d) {
                const unsigned char* blk = wb + ((size_t)(n0 + row) * (size_t)nb + (size_t)(k0 >> 8)) * (size_t)bbytes;
                tc2_dequant_row<Q6>(blk, part, dst);
            } else {
#pragma unroll
                for (int j = 0; j < 32; ++j) dst[j] = 0;
            }
        }
        for (int e = tid; e < (BM * BK) / 4; e += 256) {
            int elem = e << 2;
            int tr = elem >> 7;
            int kk = elem & 127;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            int tt = t0 + tr;
            int kkabs = k0 + kk;
            if (tt < T && kkabs + 3 < n) {
                const float* xp = x + (size_t)tt * (size_t)n + (size_t)kkabs;
                v = *reinterpret_cast<const float4*>(xp);
            }
            unsigned short* dp = Xs + elem;
            dp[0] = f32_to_f16(v.x);
            dp[1] = f32_to_f16(v.y);
            dp[2] = f32_to_f16(v.z);
            dp[3] = f32_to_f16(v.w);
        }
        __syncthreads();
        tc2_mma_k<Q6>(Xs, Ws, BK, warp, a0, a1, a2, a3, a4, a5, a6, a7);
        if (second) {
            __syncthreads();
            for (int e = tid; e < (BM * BK) / 4; e += 256) {
                int elem = e << 2;
                int tr = elem >> 7;
                int kk = elem & 127;
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                int tt = t0 + BM + tr;
                int kkabs = k0 + kk;
                if (tt < T && kkabs + 3 < n) {
                    const float* xp = x + (size_t)tt * (size_t)n + (size_t)kkabs;
                    v = *reinterpret_cast<const float4*>(xp);
                }
                unsigned short* dp = Xs + elem;
                dp[0] = f32_to_f16(v.x);
                dp[1] = f32_to_f16(v.y);
                dp[2] = f32_to_f16(v.z);
                dp[3] = f32_to_f16(v.w);
            }
            __syncthreads();
            tc2_mma_k<Q6>(Xs, Ws, BK, warp, b0, b1, b2, b3, b4, b5, b6, b7);
        }
        __syncthreads();
    }
    float* tile = (float*)raw;
    wmma_store(tile + warp * 256, 16, a0, a1, a2, a3, a4, a5, a6, a7);
    __syncthreads();
    tc2_scatter(tile, out, tid, 8, t0, n0, T, d);
    if (second) {
        __syncthreads();
        wmma_store(tile + warp * 256, 16, b0, b1, b2, b3, b4, b5, b6, b7);
        __syncthreads();
        tc2_scatter(tile, out, tid, 8, t0 + BM, n0, T, d);
    }
}

extern "C" __global__ void tc2_q4_k(const unsigned char* wb, const float* x, float* out, int n, int d, int T) {
    tc2_run<0>(wb, x, out, n, d, T);
}
extern "C" __global__ void tc2_q6_k(const unsigned char* wb, const float* x, float* out, int n, int d, int T) {
    tc2_run<1>(wb, x, out, n, d, T);
}

/* 32-token tile from 18202b0. One 256-K quant block is expanded once and
   reused for 32 tokens. At T=19 the 128-token tile above did twice the MMA
   and twice the X loads (a 64-token chunk, BK=128) and measured 323 ms vs
   176 ms on the 9B. The host uses this kernel for T<=64. */
template<int Q6>
__device__ void tc2s_run(const unsigned char* wb, const float* x, float* out, int n, int d, int T) {
    const int BM = 32;
    const int BN = 32;
    const int BK = 256;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int t0 = (int)blockIdx.y * BM;
    const int n0 = (int)blockIdx.x * BN;
    if (t0 >= T || n0 >= d) return;
    int sub = lane >> 3;
    int part = lane & 7;
    int row = warp * 4 + sub;
    int bbytes = Q6 ? 210 : 144;
    int nb = n >> 8;

    __shared__ __align__(16) unsigned char raw[32 * 256 * 4];
    unsigned short* Xs = (unsigned short*)raw;
    unsigned short* Ws = Xs + BM * BK;

    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f, c4 = 0.f, c5 = 0.f, c6 = 0.f, c7 = 0.f;
    for (int k0 = 0; k0 < n; k0 += BK) {
        if (row < BN) {
            unsigned short* dst = Ws + row * BK + part * 32;
            if (n0 + row < d) {
                const unsigned char* blk = wb + ((size_t)(n0 + row) * (size_t)nb + (size_t)(k0 >> 8)) * (size_t)bbytes;
                tc2_dequant_row<Q6>(blk, part, dst);
            } else {
#pragma unroll
                for (int j = 0; j < 32; ++j) dst[j] = 0;
            }
        }
        for (int i = tid; i < BM * BK; i += 256) {
            int tr = i >> 8;
            int kk = i & 255;
            float v = 0.f;
            int tt = t0 + tr;
            int kkabs = k0 + kk;
            if (tt < T && kkabs < n) v = x[(size_t)tt * (size_t)n + (size_t)kkabs];
            Xs[i] = f32_to_f16(v);
        }
        __syncthreads();
        if (warp < 4) {
            int wm = warp >> 1;
            int wn = warp & 1;
            for (int kk = 0; kk < BK; kk += 16) {
                unsigned a0, a1, a2, a3, a4, a5, a6, a7;
                unsigned b0, b1, b2, b3, b4, b5, b6, b7;
                wmma_load_a(a0, a1, a2, a3, a4, a5, a6, a7, Xs + (wm * 16) * BK + kk, BK);
                wmma_load_b(b0, b1, b2, b3, b4, b5, b6, b7, Ws + (wn * 16) * BK + kk, BK);
                wmma_mma(c0, c1, c2, c3, c4, c5, c6, c7, a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7);
            }
        }
        __syncthreads();
    }
    if (warp < 4) {
        float* tile = (float*)raw;
        wmma_store(tile + warp * 256, 16, c0, c1, c2, c3, c4, c5, c6, c7);
    }
    __syncthreads();
    float* tile = (float*)raw;
    for (int idx = tid; idx < 4 * 256; idx += 256) {
        int w = idx >> 8;
        int e = idx & 255;
        int i = e >> 4;
        int j = e & 15;
        int wm = w >> 1;
        int wn = w & 1;
        int t = t0 + wm * 16 + i;
        int orow = n0 + wn * 16 + j;
        if (t < T && orow < d) out[(size_t)t * (size_t)d + orow] = tile[idx];
    }
}

extern "C" __global__ void tc2s_q4_k(const unsigned char* wb, const float* x, float* out, int n, int d, int T) {
    tc2s_run<0>(wb, x, out, n, d, T);
}
extern "C" __global__ void tc2s_q6_k(const unsigned char* wb, const float* x, float* out, int n, int d, int T) {
    tc2s_run<1>(wb, x, out, n, d, T);
}

/* One 32-wide slice of a Q4_K or Q6_K superblock. Lane `lane` writes element
   `lane`. The scale is read on lane 0 and broadcast. The scalar tc kernel
   repeats that scale decode on every element; a warp on one row was doing
   it 32 times. */
template<int Q6>
__device__ __forceinline__ unsigned short tc2m_elem(const unsigned char* blk, int part, int lane) {
    if (Q6 == 0) {
        int sc = 0, m = 0;
        float ds = 0.f, dm = 0.f;
        if (lane == 0) {
            scale_min_k4(part, blk + 4, &sc, &m);
            ds = ld_f16(blk) * (float)sc;
            dm = ld_f16(blk + 2) * (float)m;
        }
        ds = __shfl_sync(0xffffffff, ds, 0);
        dm = __shfl_sync(0xffffffff, dm, 0);
        int p = part >> 1;
        int hi = part & 1;
        unsigned char q = blk[16 + p * 32 + lane];
        int qv = hi ? (int)(q >> 4) : (int)(q & 0xF);
        return f32_to_f16((float)qv * ds - dm);
    }
    float d = 0.f;
    if (lane == 0) d = ld_f16(blk + 208);
    d = __shfl_sync(0xffffffff, d, 0);
    int half = part >> 2;
    int group = part & 3;
    const unsigned char* ql = blk + half * 64;
    const unsigned char* qh = blk + 128 + half * 32;
    const unsigned char* scb = blk + 192 + half * 8;
    unsigned char qh1 = qh[lane];
    int is = lane >> 4;
    int qv, sc_i;
    if (group == 0) {
        qv = (int)((ql[lane] & 0xF) | (((qh1 >> 0) & 3) << 4)) - 32;
        sc_i = is;
    } else if (group == 1) {
        qv = (int)((ql[lane + 32] & 0xF) | (((qh1 >> 2) & 3) << 4)) - 32;
        sc_i = is + 2;
    } else if (group == 2) {
        qv = (int)((ql[lane] >> 4) | (((qh1 >> 4) & 3) << 4)) - 32;
        sc_i = is + 4;
    } else {
        qv = (int)((ql[lane + 32] >> 4) | (((qh1 >> 6) & 3) << 4)) - 32;
        sc_i = is + 6;
    }
    return f32_to_f16(d * (float)(signed char)scb[sc_i] * (float)qv);
}

/* Large-T Q4_K / Q6_K tile. Same 64 tokens x 64 rows and 16 warps as the
   scalar tc kernel (the one that beat tc2w at 581 and 2000). BK is 64: one
   step expands both nibble planes of a shared 32-byte qs group, and the
   fp16 tile is 16 KB, which is the same buffer the scalar kernel uses to
   store the WMMA result, so the SM warp cap does not drop. T<=64 stays on
   tc2s. This kernel is ODINFER_CUDA_TC2_BLOCK=1 only. On the V100S it was
   slower than the scalar tile (9B 581 tokens 151 vs 208 tok/s) and it did
   not match that tile's tokens. Leave it off. */
template<int Q6>
__device__ void tc2m_run(const unsigned char* wb, const float* x, float* out, int n, int d, int T) {
    const int BM = 64;
    const int BN = 64;
    const int BK = 64;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp >> 2;
    const int wn = warp & 3;
    const int t0 = (int)blockIdx.y * BM;
    const int n0 = (int)blockIdx.x * BN;
    if (t0 >= T || n0 >= d) return;
    int bbytes = Q6 ? 210 : 144;
    int nb = n >> 8;

    __shared__ __align__(16) unsigned char sm[16 * 256 * 4];
    unsigned short* Xs = (unsigned short*)sm;
    unsigned short* Ws = Xs + BM * BK;

    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f, c4 = 0.f, c5 = 0.f, c6 = 0.f, c7 = 0.f;
    for (int k0 = 0; k0 < n; k0 += BK) {
        for (int i = tid; i < BM * BK; i += 512) {
            int t = i / BK;
            int kk = i - t * BK;
            float v = 0.f;
            if (t0 + t < T && k0 + kk < n) v = x[((size_t)t0 + t) * (size_t)n + (size_t)k0 + kk];
            Xs[i] = f32_to_f16(v);
        }
        int part = (k0 >> 5) & 7;
        for (int r = warp; r < BN; r += 16) {
            if (n0 + r < d && k0 + 63 < n) {
                const unsigned char* blk = wb + ((size_t)(n0 + r) * (size_t)nb + (size_t)(k0 >> 8)) * (size_t)bbytes;
                Ws[r * BK + lane] = tc2m_elem<Q6>(blk, part, lane);
                Ws[r * BK + 32 + lane] = tc2m_elem<Q6>(blk, part + 1, lane);
            } else {
                Ws[r * BK + lane] = 0;
                Ws[r * BK + 32 + lane] = 0;
            }
        }
        __syncthreads();
        for (int kk = 0; kk < BK; kk += 16) {
            unsigned a0, a1, a2, a3, a4, a5, a6, a7;
            unsigned b0, b1, b2, b3, b4, b5, b6, b7;
            wmma_load_a(a0, a1, a2, a3, a4, a5, a6, a7, Xs + (wm * 16) * BK + kk, BK);
            wmma_load_b(b0, b1, b2, b3, b4, b5, b6, b7, Ws + (wn * 16) * BK + kk, BK);
            wmma_mma(c0, c1, c2, c3, c4, c5, c6, c7, a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7);
        }
        __syncthreads();
    }
    float* tile = ((float*)sm) + warp * 256;
    wmma_store(tile, 16, c0, c1, c2, c3, c4, c5, c6, c7);
    __syncthreads();
    for (int idx = tid; idx < 16 * 256; idx += 512) {
        int w = idx >> 8;
        int e = idx & 255;
        int i = e >> 4;
        int j = e & 15;
        int tw = w >> 2;
        int nw = w & 3;
        int t = t0 + tw * 16 + i;
        int row = n0 + nw * 16 + j;
        if (t < T && row < d) out[(size_t)t * (size_t)d + row] = ((float*)sm)[idx];
    }
}

extern "C" __global__ void tc2m_q4_k(const unsigned char* wb, const float* x, float* out, int n, int d, int T) {
    tc2m_run<0>(wb, x, out, n, d, T);
}
extern "C" __global__ void tc2m_q6_k(const unsigned char* wb, const float* x, float* out, int n, int d, int T) {
    tc2m_run<1>(wb, x, out, n, d, T);
}

/* Same fill and the same WMMA order as tc_run. Two 64-token halves share one
   weight tile, so a 128-token chunk dequants each weight once instead of
   twice. No scale broadcast: that path (tc2m) was slower on the V100S and
   disagreed with this dequant. */
template<int KIND>
__device__ void tc_fill_x(unsigned short* Xs, const float* x, int tid, int tbase, int k0, int n, int T) {
    const int BM = 64;
    const int BK = 32;
    for (int i = tid; i < BM * BK; i += 512) {
        int t = i / BK;
        int kk = i - t * BK;
        float v = 0.f;
        if (tbase + t < T && k0 + kk < n) v = x[((size_t)tbase + t) * (size_t)n + (size_t)k0 + kk];
        Xs[i] = f32_to_f16(v);
    }
}

template<int KIND>
__device__ void tc_fill_w(unsigned short* Ws, const unsigned char* wb, int tid, int n0, int k0, int n, int d) {
    const int BN = 64;
    const int BK = 32;
    for (int i = tid; i < BN * BK; i += 512) {
        int row = i / BK;
        int kk = i - row * BK;
        float v = 0.f;
        if (n0 + row < d && k0 + kk < n) v = tc_w<KIND>(wb, n0 + row, k0 + kk, n);
        Ws[i] = f32_to_f16(v);
    }
}

__device__ void tc_mma_64(unsigned short* Xs, unsigned short* Ws, int warp,
                          float& c0, float& c1, float& c2, float& c3, float& c4, float& c5, float& c6, float& c7) {
    const int BK = 32;
    int wm = warp >> 2;
    int wn = warp & 3;
    for (int kk = 0; kk < BK; kk += 16) {
        unsigned a0, a1, a2, a3, a4, a5, a6, a7;
        unsigned b0, b1, b2, b3, b4, b5, b6, b7;
        wmma_load_a(a0, a1, a2, a3, a4, a5, a6, a7, Xs + (wm * 16) * BK + kk, BK);
        wmma_load_b(b0, b1, b2, b3, b4, b5, b6, b7, Ws + (wn * 16) * BK + kk, BK);
        wmma_mma(c0, c1, c2, c3, c4, c5, c6, c7, a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7);
    }
}

__device__ void tc_flush(float* sm, float* out, int tid, int warp, int tbase, int n0, int T, int d,
                         float c0, float c1, float c2, float c3, float c4, float c5, float c6, float c7) {
    wmma_store(sm + warp * 256, 16, c0, c1, c2, c3, c4, c5, c6, c7);
    __syncthreads();
    for (int idx = tid; idx < 16 * 256; idx += 512) {
        int w = idx >> 8;
        int e = idx & 255;
        int i = e >> 4;
        int j = e & 15;
        int tw = w >> 2;
        int nw = w & 3;
        int t = tbase + tw * 16 + i;
        int row = n0 + nw * 16 + j;
        if (t < T && row < d) out[(size_t)t * (size_t)d + row] = sm[idx];
    }
    __syncthreads();
}

/* Opt-in. T>64 only. ODINFER_CUDA_TC2_REUSE=1. A block owns 128 tokens by
   64 rows and runs the scalar WMMA twice on one weight tile. */
template<int KIND>
__device__ void tc2r_run(const unsigned char* wb, const float* x, float* out, int n, int d, int T) {
    const int BM = 64;
    const int BN = 64;
    const int BK = 32;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int t0 = (int)blockIdx.y * (BM * 2);
    const int n0 = (int)blockIdx.x * BN;
    if (t0 >= T || n0 >= d) return;

    __shared__ __align__(16) unsigned char sm[16 * 256 * 4];
    unsigned short* Xs = (unsigned short*)sm;
    unsigned short* Xs2 = Xs + BM * BK;
    unsigned short* Ws = Xs2 + BM * BK;

    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, a4 = 0.f, a5 = 0.f, a6 = 0.f, a7 = 0.f;
    float b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f, b4 = 0.f, b5 = 0.f, b6 = 0.f, b7 = 0.f;
    for (int k0 = 0; k0 < n; k0 += BK) {
        tc_fill_x<KIND>(Xs, x, tid, t0, k0, n, T);
        tc_fill_x<KIND>(Xs2, x, tid, t0 + BM, k0, n, T);
        tc_fill_w<KIND>(Ws, wb, tid, n0, k0, n, d);
        __syncthreads();
        tc_mma_64(Xs, Ws, warp, a0, a1, a2, a3, a4, a5, a6, a7);
        tc_mma_64(Xs2, Ws, warp, b0, b1, b2, b3, b4, b5, b6, b7);
        __syncthreads();
    }
    tc_flush((float*)sm, out, tid, warp, t0, n0, T, d, a0, a1, a2, a3, a4, a5, a6, a7);
    tc_flush((float*)sm, out, tid, warp, t0 + BM, n0, T, d, b0, b1, b2, b3, b4, b5, b6, b7);
}

extern "C" __global__ void tc2r_q4_k(const unsigned char* wb, const float* x, float* out, int n, int d, int T) {
    tc2r_run<9>(wb, x, out, n, d, T);
}
extern "C" __global__ void tc2r_q6_k(const unsigned char* wb, const float* x, float* out, int n, int d, int T) {
    tc2r_run<11>(wb, x, out, n, d, T);
}

/* ---- bandwidth GEMV (T == 1) ------------------------------------------------
   Four rows per warp, eight threads per row. Each of those eight threads
   issues one ld.global.v4 (16 bytes) so a row's quant payload is one
   coalesced 128-byte read. The activation was quantized to q8 (block 32)
   once per vector; the dot is dp4a.s32.s32. K-quant rows require n % 256
   == 0. Q8_0 / Q5_0 require n % 32 == 0. No per-element tail branch.

   Lane split: sub = lane >> 3 (row within the four), t = lane & 7.
   Block: (warps * 32) threads, grid ceil(rows / (warps * 4)). */

__device__ __forceinline__ int dp4a_s32(int a, int b, int c) {
    int d;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

__device__ __forceinline__ int pack_lo_nib(unsigned v) {
    return (int)(v & 0x0f0f0f0fu);
}

__device__ __forceinline__ int pack_hi_nib(unsigned v) {
    return (int)((v >> 4) & 0x0f0f0f0fu);
}

__device__ __forceinline__ unsigned ld_u32_any(const unsigned char* p) {
    unsigned long long a = (unsigned long long)p;
    if ((a & 3u) == 0u) return *(const unsigned*)p;
    unsigned short a0 = *(const unsigned short*)p;
    unsigned short a1 = *(const unsigned short*)(p + 2);
    return (unsigned)a0 | ((unsigned)a1 << 16);
}

__device__ __forceinline__ uint4 ld_u4_any(const unsigned char* p) {
    unsigned long long a = (unsigned long long)p;
    if ((a & 15u) == 0u) return *(const uint4*)p;
    uint4 v;
    v.x = ld_u32_any(p);
    v.y = ld_u32_any(p + 4);
    v.z = ld_u32_any(p + 8);
    v.w = ld_u32_any(p + 12);
    return v;
}

/* Shift two aligned uint4 windows by `mis` bytes (0..15). mis==0 returns lo. */
__device__ __forceinline__ uint4 u4_from_mis(uint4 lo, uint4 hi, unsigned mis) {
    if (mis == 0u) return lo;
    unsigned long long L0 = (unsigned long long)lo.x | ((unsigned long long)lo.y << 32);
    unsigned long long L1 = (unsigned long long)lo.z | ((unsigned long long)lo.w << 32);
    unsigned long long H0 = (unsigned long long)hi.x | ((unsigned long long)hi.y << 32);
    unsigned long long H1 = (unsigned long long)hi.z | ((unsigned long long)hi.w << 32);
    unsigned long long A, B;
    if (mis < 8u) {
        unsigned sh = mis * 8u;
        A = (L0 >> sh) | (L1 << (64u - sh));
        B = (L1 >> sh) | (H0 << (64u - sh));
    } else if (mis == 8u) {
        A = L1;
        B = H0;
    } else {
        unsigned sh = (mis - 8u) * 8u;
        A = (L1 >> sh) | (H0 << (64u - sh));
        B = (H0 >> sh) | (H1 << (64u - sh));
    }
    uint4 o;
    o.x = (unsigned)A;
    o.y = (unsigned)(A >> 32);
    o.z = (unsigned)B;
    o.w = (unsigned)(B >> 32);
    return o;
}

/* 16 bytes at p, using only aligned ld.global.v4. Misaligned p (Q6 blocks are
   210 bytes) is a shift of two aligned windows. The caller must pad the
   allocation by 32 bytes so the second window stays in range. */
__device__ __forceinline__ uint4 ld_u4_aligned(const unsigned char* p) {
    unsigned long long a = (unsigned long long)p;
    unsigned mis = (unsigned)(a & 15ull);
    if (mis == 0u) return *(const uint4*)p;
    const uint4* base = (const uint4*)(a & ~15ull);
    return u4_from_mis(base[0], base[1], mis);
}

/* Read-only cache. Slower than a plain ld.global on the V100S (9B q4
   6.01 -> 6.50 ms, 27B 19.57 -> 22.4). Only gemv2_q4nc_k uses it. */
__device__ __forceinline__ uint4 ldg_u4(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

/* Coherent 128-bit load. Not .nc. p is 16-byte aligned. */
__device__ __forceinline__ uint4 ld_global_v4(const void* p) {
    uint4 v;
    asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}


/* Same shift, from a 16-aligned shared tile. `at` is the byte offset. */
__device__ __forceinline__ uint4 ld_u4_staged(const unsigned char* staged, unsigned at) {
    unsigned base = at & ~15u;
    const uint4* p = (const uint4*)(staged + base);
    return u4_from_mis(p[0], p[1], at & 15u);
}

/* Subtract 32 from each byte. No vsub4: NVRTC 12.9 emits
   vsub4.s32.s32.s32, and the sm_70 PTX JIT rejects that opcode
   (CUDA_ERROR_INVALID_PTX), which rejects the whole module. */
__device__ __forceinline__ int sub_bias32_bytes(unsigned x) {
    unsigned b0 = (x & 0xffu) - 32u;
    unsigned b1 = ((x >> 8) & 0xffu) - 32u;
    unsigned b2 = ((x >> 16) & 0xffu) - 32u;
    unsigned b3 = ((x >> 24) & 0xffu) - 32u;
    return (int)((b0 & 0xffu) | ((b1 & 0xffu) << 8) | ((b2 & 0xffu) << 16) | ((b3 & 0xffu) << 24));
}

/* Four Q6 codes from one ql word and the matching qh word. hi_pair 0 is
   q1/q3 (qh bits 0 and 4); hi_pair 1 is q2/q4 (qh bits 2 and 6). */
__device__ __forceinline__ void q6_pack(unsigned ql, unsigned qh, int hi_pair, int* qa, int* qb) {
    unsigned lo = ql & 0x0f0f0f0fu;
    unsigned hi = (ql >> 4) & 0x0f0f0f0fu;
    unsigned hs = hi_pair ? 2u : 0u;
    unsigned hb = hi_pair ? 6u : 4u;
    *qa = sub_bias32_bytes(lo | (((qh >> hs) & 0x03030303u) << 4));
    *qb = sub_bias32_bytes(hi | (((qh >> hb) & 0x03030303u) << 4));
}

__device__ __forceinline__ unsigned hdr_scale_byte(uint4 hdr, int i) {
    unsigned w = (i < 4) ? hdr.y : (i < 8) ? hdr.z : hdr.w;
    return (w >> ((i & 3) * 8)) & 0xffu;
}

__device__ __forceinline__ void scale_min_hdr(int j, uint4 hdr, int* sc, int* m) {
    if (j < 4) {
        *sc = (int)(hdr_scale_byte(hdr, j) & 63u);
        *m = (int)(hdr_scale_byte(hdr, j + 4) & 63u);
    } else {
        unsigned qj4 = hdr_scale_byte(hdr, j + 4);
        unsigned qjm = hdr_scale_byte(hdr, j - 4);
        unsigned qj = hdr_scale_byte(hdr, j);
        *sc = (int)((qj4 & 0xFu) | ((qjm >> 6) << 4));
        *m = (int)((qj4 >> 4) | ((qj >> 6) << 4));
    }
}

__device__ __forceinline__ int scv_i8(uint4 scv, int idx) {
    unsigned w = idx < 4 ? scv.x : idx < 8 ? scv.y : idx < 12 ? scv.z : scv.w;
    return (int)(signed char)((w >> ((idx & 3) * 8)) & 0xffu);
}

__device__ __forceinline__ void k_range(int nb, int* b0, int* b1) {
    int splits = (int)gridDim.y;
    int split = (int)blockIdx.y;
    if (splits <= 1) {
        *b0 = 0;
        *b1 = nb;
        return;
    }
    *b0 = (int)(((long long)nb * split) / splits);
    *b1 = (int)(((long long)nb * (split + 1)) / splits);
}

__device__ __forceinline__ void store_row(float* out, int row, int d, int writer, float acc, int accumulate) {
    if (writer && row < d) {
        /* gridDim.y > 1 is split-K. The caller zeros `out` unless accumulate
           was already set, and every split adds its partial. */
        if ((int)gridDim.y > 1) atomicAdd(out + row, acc);
        else if (accumulate) out[row] += acc;
        else out[row] = acc;
    }
}

__device__ __forceinline__ int dot_nib(uint4 q, uint4 x, int hi) {
    unsigned qw[4] = {q.x, q.y, q.z, q.w};
    unsigned xw[4] = {x.x, x.y, x.z, x.w};
    int acc = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int packed = hi ? pack_hi_nib(qw[i]) : pack_lo_nib(qw[i]);
        acc = dp4a_s32(packed, (int)xw[i], acc);
    }
    return acc;
}

__device__ __forceinline__ float f16_bits(unsigned bits) {
    unsigned char p[2];
    p[0] = (unsigned char)(bits & 0xff);
    p[1] = (unsigned char)((bits >> 8) & 0xff);
    return ld_f16(p);
}

__device__ __forceinline__ uint4 shfl_u4(uint4 v, int src) {
    uint4 o;
    o.x = (unsigned)__shfl_sync(0xffffffff, (int)v.x, src);
    o.y = (unsigned)__shfl_sync(0xffffffff, (int)v.y, src);
    o.z = (unsigned)__shfl_sync(0xffffffff, (int)v.z, src);
    o.w = (unsigned)__shfl_sync(0xffffffff, (int)v.w, src);
    return o;
}

__device__ __forceinline__ float warp8_sum(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 4);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    v += __shfl_xor_sync(0xffffffff, v, 1);
    return v;
}

/* x[n] -> q8 groups of 32. scale[g], sum16[g] = (sum of first 16, sum of last 16)
   of the original floats, so the Q4_K / Q5_K min term stays exact. */
extern "C" __global__ void quant_q8(const float* x, signed char* qs, float* scale, float2* sum16, int n) {
    int g = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int ng = n >> 5;
    if (g >= ng) return;
    const float* xi = x + (size_t)g * 32;
    float amax = 0.f, s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        float a = xi[j];
        float b = xi[j + 16];
        amax = fmaxf(amax, fmaxf(fabsf(a), fabsf(b)));
        s0 += a;
        s1 += b;
    }
    float d = amax / 127.f;
    float id = d > 0.f ? 1.f / d : 0.f;
    scale[g] = d;
    sum16[g] = make_float2(s0, s1);
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        int q = (int)rintf(xi[j] * id);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        qs[(size_t)g * 32 + j] = (signed char)q;
    }
}

/* RMSNorm one vector and quantize it. One block, 128 threads. n % 32 == 0, n <= 32768. */
/* x += y, then the same RMSNorm and q8 store as rmsnorm_quant_q8.
   One block, 128 threads. n % 32 == 0. The add is finished before the
   sum of squares, matching a separate residual kernel followed by the norm. */
extern "C" __global__ void residual_rmsnorm_q8(float* x, const float* y, float* out, const float* weight,
                                               signed char* qs, float* scale, float2* sum16, int n, float eps) {
    int tid = (int)threadIdx.x;
    for (int i = tid; i < n; i += 128) x[i] += y[i];
    __syncthreads();
    float ss = 0.f;
    for (int i = tid; i < n; i += 128) {
        float v = x[i];
        ss += v * v;
    }
    ss = block_sum_128(ss);
    float inv = rsqrtf(ss / (float)n + eps);
    for (int i = tid; i < n; i += 128) out[i] = weight[i] * (x[i] * inv);
    __syncthreads();
    int ng = n >> 5;
    for (int g = tid; g < ng; g += 128) {
        const float* xi = out + (size_t)g * 32;
        float amax = 0.f, s0 = 0.f, s1 = 0.f;
        for (int j = 0; j < 16; ++j) {
            float a = xi[j];
            float b = xi[j + 16];
            amax = fmaxf(amax, fmaxf(fabsf(a), fabsf(b)));
            s0 += a;
            s1 += b;
        }
        float d = amax / 127.f;
        float id = d > 0.f ? 1.f / d : 0.f;
        scale[g] = d;
        sum16[g] = make_float2(s0, s1);
        for (int j = 0; j < 32; ++j) {
            int q = (int)rintf(xi[j] * id);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            qs[(size_t)g * 32 + j] = (signed char)q;
        }
    }
}

extern "C" __global__ void rmsnorm_quant_q8(const float* inp, float* out, const float* weight,
                                            signed char* qs, float* scale, float2* sum16, int n, float eps) {
    int tid = (int)threadIdx.x;
    float ss = 0.f;
    for (int i = tid; i < n; i += 128) {
        float v = inp[i];
        ss += v * v;
    }
    ss = block_sum_128(ss);
    float inv = rsqrtf(ss / (float)n + eps);
    for (int i = tid; i < n; i += 128) out[i] = weight[i] * (inp[i] * inv);
    __syncthreads();
    int ng = n >> 5;
    for (int g = tid; g < ng; g += 128) {
        const float* xi = out + (size_t)g * 32;
        float amax = 0.f, s0 = 0.f, s1 = 0.f;
        for (int j = 0; j < 16; ++j) {
            float a = xi[j];
            float b = xi[j + 16];
            amax = fmaxf(amax, fmaxf(fabsf(a), fabsf(b)));
            s0 += a;
            s1 += b;
        }
        float d = amax / 127.f;
        float id = d > 0.f ? 1.f / d : 0.f;
        scale[g] = d;
        sum16[g] = make_float2(s0, s1);
        for (int j = 0; j < 32; ++j) {
            int q = (int)rintf(xi[j] * id);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            qs[(size_t)g * 32 + j] = (signed char)q;
        }
    }
}

/* Four threads own one row (lanes slot .. slot+3). xor 4 would cross rows. */
__device__ __forceinline__ float warp4_sum(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 2);
    v += __shfl_xor_sync(0xffffffff, v, 1);
    return v;
}

/* One Q4 thread-slot (t in 0..7): two 128-bit qs groups, two q8 groups. */
__device__ __forceinline__ float q4_pair(uint4 hdr, uint4 qv, const signed char* xq, const float* xd, const float2* sum16, int b, int t) {
    int p = t >> 1;
    int half = t & 1;
    float dW = f16_bits(hdr.x & 0xffff);
    float dmin = f16_bits(hdr.x >> 16);
    int sc0, m0, sc1, m1;
    scale_min_hdr(p * 2, hdr, &sc0, &m0);
    scale_min_hdr(p * 2 + 1, hdr, &sc1, &m1);
    int group = b * 8 + p * 2;
    int off = half * 16;
    uint4 xlo = ld_global_v4(xq + (size_t)group * 32 + off);
    uint4 xhi = ld_global_v4(xq + (size_t)(group + 1) * 32 + off);
    int dot_lo = dot_nib(qv, xlo, 0);
    int dot_hi = dot_nib(qv, xhi, 1);
    float2 slo = sum16[group];
    float2 shi = sum16[group + 1];
    float sum_lo = half ? slo.y : slo.x;
    float sum_hi = half ? shi.y : shi.x;
    float acc = (dW * (float)sc0) * xd[group] * (float)dot_lo - (dmin * (float)m0) * sum_lo;
    acc += (dW * (float)sc1) * xd[group + 1] * (float)dot_hi - (dmin * (float)m1) * sum_hi;
    return acc;
}

/* LD is either a plain ld.global (the default, same as 8975985) or ld.global.nc. */
#define GEMV2_Q4_BODY(LD)                                                                 \
    int lane = (int)threadIdx.x & 31;                                                     \
    int warp = (int)threadIdx.x >> 5;                                                     \
    int sub = lane >> 3;                                                                  \
    int t = lane & 7;                                                                     \
    int row = ((int)blockIdx.x * ((int)blockDim.x >> 5) + warp) * 4 + sub;                \
    int p = t >> 1;                                                                       \
    int half = t & 1;                                                                     \
    int nb = n >> 8;                                                                      \
    int b0s, b1s;                                                                         \
    k_range(nb, &b0s, &b1s);                                                              \
    float acc = 0.f;                                                                      \
    for (int b = b0s; b < b1s; ++b) {                                                     \
        uint4 hdr = make_uint4(0, 0, 0, 0);                                               \
        uint4 qv = make_uint4(0, 0, 0, 0);                                                \
        if (row < d) {                                                                    \
            const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)b) * 144; \
            if (t == 0) hdr = LD(blk);                                                    \
            qv = LD(blk + 16 + t * 16);                                                   \
        }                                                                                 \
        int src = lane & ~7;                                                              \
        hdr = shfl_u4(hdr, src);                                                          \
        int group = b * 8 + p * 2;                                                        \
        int off = half * 16;                                                              \
        /* Four rows share x. Lane t loads it; the shuffle is uniform. */                 \
        uint4 xlo = make_uint4(0, 0, 0, 0);                                               \
        uint4 xhi = make_uint4(0, 0, 0, 0);                                               \
        if (sub == 0) {                                                                   \
            xlo = LD(xq + (size_t)group * 32 + off);                                      \
            xhi = LD(xq + (size_t)(group + 1) * 32 + off);                                \
        }                                                                                 \
        xlo = shfl_u4(xlo, t);                                                            \
        xhi = shfl_u4(xhi, t);                                                            \
        if (row < d) {                                                                    \
            float dW = f16_bits(hdr.x & 0xffff);                                          \
            float dmin = f16_bits(hdr.x >> 16);                                           \
            int sc0, m0, sc1, m1;                                                         \
            scale_min_hdr(p * 2, hdr, &sc0, &m0);                                         \
            scale_min_hdr(p * 2 + 1, hdr, &sc1, &m1);                                     \
            int dot_lo = dot_nib(qv, xlo, 0);                                             \
            int dot_hi = dot_nib(qv, xhi, 1);                                             \
            float2 slo = sum16[group];                                                    \
            float2 shi = sum16[group + 1];                                                \
            float sum_lo = half ? slo.y : slo.x;                                          \
            float sum_hi = half ? shi.y : shi.x;                                          \
            acc += (dW * (float)sc0) * xd[group] * (float)dot_lo - (dmin * (float)m0) * sum_lo; \
            acc += (dW * (float)sc1) * xd[group + 1] * (float)dot_hi - (dmin * (float)m1) * sum_hi; \
        }                                                                                 \
    }                                                                                     \
    acc = warp8_sum(acc);                                                                 \
    store_row(out, row, d, t == 0, acc, accumulate);

#define LD_PLAIN(p) (*(const uint4*)(p))
#define LD_NC(p) (ldg_u4((p)))

extern "C" __global__ void gemv2_q4_k(const unsigned char* wb, const signed char* xq, const float* xd, const float2* sum16,
                                      float* out, int n, int d, int accumulate) {
    GEMV2_Q4_BODY(LD_PLAIN)
}

extern "C" __global__ void gemv2_q4nc_k(const unsigned char* wb, const signed char* xq, const float* xd, const float2* sum16,
                                        float* out, int n, int d, int accumulate) {
    GEMV2_Q4_BODY(LD_NC)
}

/* One K sweep, two Q4_K matrices that share x (gate and up). x is loaded
   once per super-block; both weight streams are issued before the dots.
   gy is 1: the host keeps split-K on the two-launch path. Same math as
   two gemv2_q4_k calls. */
__device__ __forceinline__ float q4_acc(uint4 hdr, uint4 qv, uint4 xlo, uint4 xhi, float2 slo, float2 shi,
                                        const float* xd, int b, int t) {
    int p = t >> 1;
    int half = t & 1;
    float dW = f16_bits(hdr.x & 0xffff);
    float dmin = f16_bits(hdr.x >> 16);
    int sc0, m0, sc1, m1;
    scale_min_hdr(p * 2, hdr, &sc0, &m0);
    scale_min_hdr(p * 2 + 1, hdr, &sc1, &m1);
    int group = b * 8 + p * 2;
    int dot_lo = dot_nib(qv, xlo, 0);
    int dot_hi = dot_nib(qv, xhi, 1);
    float sum_lo = half ? slo.y : slo.x;
    float sum_hi = half ? shi.y : shi.x;
    float acc = (dW * (float)sc0) * xd[group] * (float)dot_lo - (dmin * (float)m0) * sum_lo;
    acc += (dW * (float)sc1) * xd[group + 1] * (float)dot_hi - (dmin * (float)m1) * sum_hi;
    return acc;
}

extern "C" __global__ void gemv2_q4_pair_k(const unsigned char* w0, const unsigned char* w1,
                                           const signed char* xq, const float* xd, const float2* sum16,
                                           float* y0, float* y1, int n, int d) {
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    int sub = lane >> 3;
    int t = lane & 7;
    int row = ((int)blockIdx.x * ((int)blockDim.x >> 5) + warp) * 4 + sub;
    int nb = n >> 8;
    float acc0 = 0.f, acc1 = 0.f;
    for (int b = 0; b < nb; ++b) {
        uint4 h0 = make_uint4(0, 0, 0, 0);
        uint4 h1 = make_uint4(0, 0, 0, 0);
        uint4 q0 = make_uint4(0, 0, 0, 0);
        uint4 q1 = make_uint4(0, 0, 0, 0);
        if (row < d) {
            const unsigned char* blk0 = w0 + ((size_t)row * (size_t)nb + (size_t)b) * 144;
            const unsigned char* blk1 = w1 + ((size_t)row * (size_t)nb + (size_t)b) * 144;
            if (t == 0) {
                h0 = ld_global_v4(blk0);
                h1 = ld_global_v4(blk1);
            }
            q0 = ld_global_v4(blk0 + 16 + t * 16);
            q1 = ld_global_v4(blk1 + 16 + t * 16);
        }
        int src = lane & ~7;
        h0 = shfl_u4(h0, src);
        h1 = shfl_u4(h1, src);
        int p = t >> 1;
        int half = t & 1;
        int group = b * 8 + p * 2;
        int off = half * 16;
        uint4 xlo = make_uint4(0, 0, 0, 0);
        uint4 xhi = make_uint4(0, 0, 0, 0);
        if (sub == 0) {
            xlo = ld_global_v4(xq + (size_t)group * 32 + off);
            xhi = ld_global_v4(xq + (size_t)(group + 1) * 32 + off);
        }
        xlo = shfl_u4(xlo, t);
        xhi = shfl_u4(xhi, t);
        if (row < d) {
            float2 slo = sum16[group];
            float2 shi = sum16[group + 1];
            acc0 += q4_acc(h0, q0, xlo, xhi, slo, shi, xd, b, t);
            acc1 += q4_acc(h1, q1, xlo, xhi, slo, shi, xd, b, t);
        }
    }
    acc0 = warp8_sum(acc0);
    acc1 = warp8_sum(acc1);
    store_row(y0, row, d, t == 0, acc0, 0);
    store_row(y1, row, d, t == 0, acc1, 0);
}

#undef GEMV2_Q4_BODY
#undef LD_PLAIN
#undef LD_NC

/* Eight rows per warp (four threads per row, each thread owns two of the
   eight Q4 slots). Each iteration issues two super-blocks of coherent
   128-bit weight loads before the dots. Opt-in: ODINFER_CUDA_Q4_ILP=1.
   The default stays gemv2_q4_k. */
extern "C" __global__ void gemv2_q4x_k(const unsigned char* wb, const signed char* xq, const float* xd, const float2* sum16,
                                       float* out, int n, int d, int accumulate) {
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    int slot = lane & 3;
    int sub = lane >> 2;
    int row = ((int)blockIdx.x * ((int)blockDim.x >> 5) + warp) * 8 + sub;
    int nb = n >> 8;
    int b0s, b1s;
    k_range(nb, &b0s, &b1s);
    float acc = 0.f;
    int b = b0s;
    int src = lane & ~3;
    for (; b + 1 < b1s; b += 2) {
        uint4 h0 = make_uint4(0, 0, 0, 0);
        uint4 h1 = make_uint4(0, 0, 0, 0);
        uint4 q0a = make_uint4(0, 0, 0, 0);
        uint4 q0b = make_uint4(0, 0, 0, 0);
        uint4 q1a = make_uint4(0, 0, 0, 0);
        uint4 q1b = make_uint4(0, 0, 0, 0);
        if (row < d) {
            const unsigned char* blk0 = wb + ((size_t)row * (size_t)nb + (size_t)b) * 144;
            const unsigned char* blk1 = blk0 + 144;
            if (slot == 0) {
                h0 = ld_global_v4(blk0);
                h1 = ld_global_v4(blk1);
            }
            q0a = ld_global_v4(blk0 + 16 + slot * 16);
            q0b = ld_global_v4(blk0 + 16 + (slot + 4) * 16);
            q1a = ld_global_v4(blk1 + 16 + slot * 16);
            q1b = ld_global_v4(blk1 + 16 + (slot + 4) * 16);
        }
        h0 = shfl_u4(h0, src);
        h1 = shfl_u4(h1, src);
        if (row < d) {
            acc += q4_pair(h0, q0a, xq, xd, sum16, b, slot);
            acc += q4_pair(h0, q0b, xq, xd, sum16, b, slot + 4);
            acc += q4_pair(h1, q1a, xq, xd, sum16, b + 1, slot);
            acc += q4_pair(h1, q1b, xq, xd, sum16, b + 1, slot + 4);
        }
    }
    if (b < b1s) {
        uint4 h = make_uint4(0, 0, 0, 0);
        uint4 qa = make_uint4(0, 0, 0, 0);
        uint4 qb = make_uint4(0, 0, 0, 0);
        if (row < d) {
            const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)b) * 144;
            if (slot == 0) h = ld_global_v4(blk);
            qa = ld_global_v4(blk + 16 + slot * 16);
            qb = ld_global_v4(blk + 16 + (slot + 4) * 16);
        }
        h = shfl_u4(h, src);
        if (row < d) {
            acc += q4_pair(h, qa, xq, xd, sum16, b, slot);
            acc += q4_pair(h, qb, xq, xd, sum16, b, slot + 4);
        }
    }
    acc = warp4_sum(acc);
    store_row(out, row, d, slot == 0, acc, accumulate);
}

/* STAGE=1 copies each 210-byte block through aligned global loads into
   shared memory, then shifts aligned smem uint4s. A byte-wise smem reload
   made the 9B w2 slower (K=12288). The host uses STAGE=1 only for long K
   (n>=16384, the 27B w2) and for tall rows (d>8192, the logits). */
template<int STAGE>
__device__ void gemv2_q6_run(const unsigned char* wb, const signed char* xq, const float* xd,
                             float* out, int n, int d, int accumulate) {
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    int sub = lane >> 3;
    int t = lane & 7;
    int row = ((int)blockIdx.x * ((int)blockDim.x >> 5) + warp) * 4 + sub;
    int half = t >> 2;
    int piece = t & 3;
    int l0 = (piece & 1) * 16;
    int hi_pair = piece >> 1;
    int nb = n >> 8;
    int b0s, b1s;
    k_range(nb, &b0s, &b1s);
    __shared__ __align__(16) unsigned char q6sm[STAGE ? 16 * 4 * 240 : 4];
    float acc = 0.f;
    for (int b = b0s; b < b1s; ++b) {
        uint4 ql = make_uint4(0, 0, 0, 0);
        uint4 qh = make_uint4(0, 0, 0, 0);
        uint4 scv = make_uint4(0, 0, 0, 0);
        float dW = 0.f;
        if (STAGE) {
            unsigned char* staged = q6sm + (warp * 4 + sub) * 240;
            if (row < d) {
                const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)b) * 210;
                unsigned long long addr = (unsigned long long)blk;
                const unsigned char* base = (const unsigned char*)(addr & ~15ull);
                for (int v = t; v < 15; v += 8) {
                    *(uint4*)(staged + (size_t)v * 16) = *(const uint4*)(base + (size_t)v * 16);
                }
            }
            __syncwarp();
            if (row < d) {
                unsigned mis = (unsigned)(((unsigned long long)(wb + ((size_t)row * (size_t)nb + (size_t)b) * 210)) & 15ull);
                ql = ld_u4_staged(staged, mis + (unsigned)t * 16u);
                qh = ld_u4_staged(staged, mis + 128u + (unsigned)(half * 32 + l0));
                if (t == 0) {
                    scv = ld_u4_staged(staged, mis + 192u);
                    dW = ld_f16(staged + mis + 208);
                }
            }
        } else if (row < d) {
            const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)b) * 210;
            ql = ld_u4_aligned(blk + t * 16);
            qh = ld_u4_aligned(blk + 128 + half * 32 + l0);
            if (t == 0) {
                scv = ld_u4_aligned(blk + 192);
                dW = ld_f16(blk + 208);
            }
        }
        int src = lane & ~7;
        scv = shfl_u4(scv, src);
        dW = __shfl_sync(0xffffffff, dW, src);
        int group0 = b * 8 + half * 4 + (hi_pair ? 1 : 0);
        int off = l0;
        uint4 xa = make_uint4(0, 0, 0, 0);
        uint4 xb = make_uint4(0, 0, 0, 0);
        if (sub == 0) {
            xa = *(const uint4*)(xq + (size_t)group0 * 32 + off);
            xb = *(const uint4*)(xq + (size_t)(group0 + 2) * 32 + off);
        }
        xa = shfl_u4(xa, t);
        xb = shfl_u4(xb, t);
        if (row < d) {
            int scoff = half * 8;
            int is = l0 >> 4;
            int s_a = scoff + is + (hi_pair ? 2 : 0);
            int s_b = scoff + is + (hi_pair ? 6 : 4);
            float sa = dW * (float)scv_i8(scv, s_a);
            float sb = dW * (float)scv_i8(scv, s_b);
            unsigned qw[4] = {ql.x, ql.y, ql.z, ql.w};
            unsigned hw[4] = {qh.x, qh.y, qh.z, qh.w};
            unsigned aw[4] = {xa.x, xa.y, xa.z, xa.w};
            unsigned bw[4] = {xb.x, xb.y, xb.z, xb.w};
            int dot_a = 0, dot_b = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int qa, qb;
                q6_pack(qw[i], hw[i], hi_pair, &qa, &qb);
                dot_a = dp4a_s32(qa, (int)aw[i], dot_a);
                dot_b = dp4a_s32(qb, (int)bw[i], dot_b);
            }
            acc += sa * xd[group0] * (float)dot_a;
            acc += sb * xd[group0 + 2] * (float)dot_b;
        }
    }
    acc = warp8_sum(acc);
    store_row(out, row, d, t == 0, acc, accumulate);
}

extern "C" __global__ void gemv2_q6_k(const unsigned char* wb, const signed char* xq, const float* xd,
                                      float* out, int n, int d, int accumulate) {
    gemv2_q6_run<0>(wb, xq, xd, out, n, d, accumulate);
}
extern "C" __global__ void gemv2_q6s_k(const unsigned char* wb, const signed char* xq, const float* xd,
                                       float* out, int n, int d, int accumulate) {
    gemv2_q6_run<1>(wb, xq, xd, out, n, d, accumulate);
}

extern "C" __global__ void gemv2_q5_k(const unsigned char* wb, const signed char* xq, const float* xd, const float2* sum16,
                                      float* out, int n, int d, int accumulate) {
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    int sub = lane >> 3;
    int t = lane & 7;
    int row = ((int)blockIdx.x * ((int)blockDim.x >> 5) + warp) * 4 + sub;
    int p = t >> 1;
    int half = t & 1;
    int nb = n >> 8;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        uint4 hdr = make_uint4(0, 0, 0, 0);
        uint4 qv = make_uint4(0, 0, 0, 0);
        uint4 hv = make_uint4(0, 0, 0, 0);
        if (row < d) {
            const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)b) * 176;
            if (t == 0) hdr = *(const uint4*)blk;
            qv = *(const uint4*)(blk + 48 + t * 16);
            hv = *(const uint4*)(blk + 16 + half * 16);
        }
        int src = lane & ~7;
        hdr = shfl_u4(hdr, src);
        if (row < d) {
            float dW = f16_bits(hdr.x & 0xffff);
            float dmin = f16_bits(hdr.x >> 16);
            int sc0, m0, sc1, m1;
            scale_min_hdr(p * 2, hdr, &sc0, &m0);
            scale_min_hdr(p * 2 + 1, hdr, &sc1, &m1);
            unsigned bit_lo = 1u << (2 * p);
            unsigned bit_hi = bit_lo << 1;
            int group = b * 8 + p * 2;
            int off = half * 16;
            uint4 xlo = *(const uint4*)(xq + (size_t)group * 32 + off);
            uint4 xhi = *(const uint4*)(xq + (size_t)(group + 1) * 32 + off);
            unsigned qw[4] = {qv.x, qv.y, qv.z, qv.w};
            unsigned hw[4] = {hv.x, hv.y, hv.z, hv.w};
            unsigned al[4] = {xlo.x, xlo.y, xlo.z, xlo.w};
            unsigned ah[4] = {xhi.x, xhi.y, xhi.z, xhi.w};
            int dot_lo = 0, dot_hi = 0;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int plo = 0, phi = 0;
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    unsigned byte = (qw[i] >> (8 * k)) & 0xff;
                    unsigned hb = (hw[i] >> (8 * k)) & 0xff;
                    int lo = (int)(byte & 0xf) + ((hb & bit_lo) ? 16 : 0);
                    int hi = (int)(byte >> 4) + ((hb & bit_hi) ? 16 : 0);
                    plo |= (lo & 0xff) << (8 * k);
                    phi |= (hi & 0xff) << (8 * k);
                }
                dot_lo = dp4a_s32(plo, (int)al[i], dot_lo);
                dot_hi = dp4a_s32(phi, (int)ah[i], dot_hi);
            }
            float2 slo = sum16[group];
            float2 shi = sum16[group + 1];
            float sum_lo = half ? slo.y : slo.x;
            float sum_hi = half ? shi.y : shi.x;
            acc += (dW * (float)sc0) * xd[group] * (float)dot_lo - (dmin * (float)m0) * sum_lo;
            acc += (dW * (float)sc1) * xd[group + 1] * (float)dot_hi - (dmin * (float)m1) * sum_hi;
        }
    }
    acc = warp8_sum(acc);
    store_row(out, row, d, t == 0, acc, accumulate);
}

extern "C" __global__ void gemv2_q8_0(const unsigned char* wb, const signed char* xq, const float* xd,
                                      float* out, int n, int d, int accumulate) {
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    int sub = lane >> 3;
    int t = lane & 7;
    int row = ((int)blockIdx.x * ((int)blockDim.x >> 5) + warp) * 4 + sub;
    int nb = n >> 5;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        if (row < d) {
            const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)b) * 34;
            float dW = ld_f16(blk);
            unsigned w = ld_u32_any(blk + 2 + t * 4);
            unsigned xv = *(const unsigned*)(xq + (size_t)b * 32 + t * 4);
            int dot = dp4a_s32((int)w, (int)xv, 0);
            acc += dW * xd[b] * (float)dot;
        }
    }
    acc = warp8_sum(acc);
    store_row(out, row, d, t == 0, acc, accumulate);
}

extern "C" __global__ void gemv2_q5_0(const unsigned char* wb, const signed char* xq, const float* xd,
                                      float* out, int n, int d, int accumulate) {
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    int sub = lane >> 3;
    int t = lane & 7;
    int row = ((int)blockIdx.x * ((int)blockDim.x >> 5) + warp) * 4 + sub;
    int nb = n >> 5;
    float acc = 0.f;
    for (int b = 0; b < nb; ++b) {
        uint4 qv = make_uint4(0, 0, 0, 0);
        unsigned qh = 0;
        float dW = 0.f;
        if (row < d) {
            const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)b) * 22;
            dW = ld_f16(blk);
            qh = ld_u32_any(blk + 2);
            if (t == 0) qv = ld_u4_any(blk + 6);
        }
        int src = lane & ~7;
        qv = shfl_u4(qv, src);
        if (row < d) {
            unsigned qw[4] = {qv.x, qv.y, qv.z, qv.w};
            int base = (t & 3) * 4;
            int hi = t >> 2;
            unsigned xv = *(const unsigned*)(xq + (size_t)b * 32 + t * 4);
            int packed = 0;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                int j = base + k;
                unsigned byte = (qw[j >> 2] >> (8 * (j & 3))) & 0xff;
                int nib = hi ? (int)(byte >> 4) : (int)(byte & 0xf);
                int elem = hi ? (j + 16) : j;
                int qv_i = (nib | (((qh >> elem) & 1) << 4)) - 16;
                packed |= (qv_i & 0xff) << (8 * k);
            }
            int dot = dp4a_s32(packed, (int)xv, 0);
            acc += dW * xd[b] * (float)dot;
        }
    }
    acc = warp8_sum(acc);
    store_row(out, row, d, t == 0, acc, accumulate);
}

/* One output row per warp. Used when the matrix is short (wo, w2, 9B wq):
   four rows per warp leaves the V100 under-occupied. One lane owns one
   element of each 32-wide group, so the loads are coalesced. The Q4 min
   term is the full 32-element sum and is subtracted once, on lane 0. */
extern "C" __global__ void gemv2r1_q4_k(const unsigned char* wb, const signed char* xq, const float* xd, const float2* sum16,
                                        float* out, int n, int d, int accumulate) {
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    int row = (int)blockIdx.x * ((int)blockDim.x >> 5) + warp;
    int nb = n >> 8;
    float acc = 0.f;
    if (row < d) {
        for (int b = 0; b < nb; ++b) {
            const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)b) * 144;
            uint4 hdr = make_uint4(0, 0, 0, 0);
            if (lane == 0) hdr = *(const uint4*)blk;
            hdr = shfl_u4(hdr, 0);
            float dW = f16_bits(hdr.x & 0xffff);
            float dmin = f16_bits(hdr.x >> 16);
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                int sc0, m0, sc1, m1;
                scale_min_hdr(p * 2, hdr, &sc0, &m0);
                scale_min_hdr(p * 2 + 1, hdr, &sc1, &m1);
                unsigned char q = blk[16 + p * 32 + lane];
                int g0 = b * 8 + p * 2;
                int xlo = (int)xq[(size_t)g0 * 32 + lane];
                int xhi = (int)xq[(size_t)(g0 + 1) * 32 + lane];
                acc += (dW * (float)sc0) * xd[g0] * (float)(q & 0xf) * (float)xlo;
                acc += (dW * (float)sc1) * xd[g0 + 1] * (float)(q >> 4) * (float)xhi;
                if (lane == 0) {
                    float2 s0 = sum16[g0];
                    float2 s1 = sum16[g0 + 1];
                    acc -= (dmin * (float)m0) * (s0.x + s0.y);
                    acc -= (dmin * (float)m1) * (s1.x + s1.y);
                }
            }
        }
    }
    acc = warp_sum(acc);
    store_row(out, row, d, lane == 0, acc, accumulate);
}

/* Q4_K / Q6_K prefill GEMM. One block owns 64 tokens by 64 rows, BK=128,
   512 threads (16 warps). Weights are expanded with tc2_dequant_row (one
   32-element slice per thread, the same expansion as tc2s) into fp16 shared
   memory. Activations are narrowed to fp16. WMMA 16x16x16 accumulates in
   fp32. K must be a multiple of 256 so every K step is a whole quant slice
   and the float4 activation load stays inside the row. T==1 never launches
   this. ODINFER_CUDA_PREFILL_GEMM=0 keeps tc / tc2. */
template<int Q6>
__device__ void pgemm_run(const unsigned char* wb, const float* x, float* out, int n, int d, int T) {
    const int BM = 64;
    const int BN = 64;
    const int BK = 128;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int wm = warp >> 2;
    const int wn = warp & 3;
    const int t0 = (int)blockIdx.y * BM;
    const int n0 = (int)blockIdx.x * BN;
    if (t0 >= T || n0 >= d) return;

    int bbytes = Q6 ? 210 : 144;
    int nb = n >> 8;

    __shared__ __align__(16) unsigned char raw[64 * 128 * 2 + 64 * 128 * 2];
    unsigned short* Xs = (unsigned short*)raw;
    unsigned short* Ws = Xs + BM * BK;

    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f, c4 = 0.f, c5 = 0.f, c6 = 0.f, c7 = 0.f;
    for (int k0 = 0; k0 < n; k0 += BK) {
        if (tid < BN * 4) {
            int row = tid & (BN - 1);
            int sub = tid >> 6;
            int part = ((k0 >> 5) & 7) + sub;
            unsigned short* dst = Ws + row * BK + sub * 32;
            if (n0 + row < d) {
                const unsigned char* blk = wb + ((size_t)(n0 + row) * (size_t)nb + (size_t)(k0 >> 8)) * (size_t)bbytes;
                tc2_dequant_row<Q6>(blk, part, dst);
            } else {
#pragma unroll
                for (int j = 0; j < 32; ++j) dst[j] = 0;
            }
        }
        for (int e = tid; e < (BM * BK) / 4; e += 512) {
            int elem = e << 2;
            int tr = elem >> 7;
            int kk = elem & 127;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            int tt = t0 + tr;
            int kkabs = k0 + kk;
            if (tt < T && kkabs + 3 < n) {
                const float* xp = x + (size_t)tt * (size_t)n + (size_t)kkabs;
                v = *reinterpret_cast<const float4*>(xp);
            }
            unsigned short* dp = Xs + elem;
            dp[0] = f32_to_f16(v.x);
            dp[1] = f32_to_f16(v.y);
            dp[2] = f32_to_f16(v.z);
            dp[3] = f32_to_f16(v.w);
        }
        __syncthreads();
#pragma unroll 1
        for (int kk = 0; kk < BK; kk += 16) {
            unsigned a0, a1, a2, a3, a4, a5, a6, a7;
            unsigned b0, b1, b2, b3, b4, b5, b6, b7;
            wmma_load_a(a0, a1, a2, a3, a4, a5, a6, a7, Xs + (wm * 16) * BK + kk, BK);
            wmma_load_b(b0, b1, b2, b3, b4, b5, b6, b7, Ws + (wn * 16) * BK + kk, BK);
            wmma_mma(c0, c1, c2, c3, c4, c5, c6, c7, a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7);
        }
        __syncthreads();
    }
    float* tile = (float*)raw;
    wmma_store(tile + warp * 256, 16, c0, c1, c2, c3, c4, c5, c6, c7);
    __syncthreads();
    for (int idx = tid; idx < 16 * 256; idx += 512) {
        int w = idx >> 8;
        int e = idx & 255;
        int i = e >> 4;
        int j = e & 15;
        int tw = w >> 2;
        int nw = w & 3;
        int t = t0 + tw * 16 + i;
        int row = n0 + nw * 16 + j;
        if (t < T && row < d) out[(size_t)t * (size_t)d + row] = tile[idx];
    }
}

extern "C" __global__ void pgemm_q4_k(const unsigned char* wb, const float* x, float* out, int n, int d, int T) {
    pgemm_run<0>(wb, x, out, n, d, T);
}
extern "C" __global__ void pgemm_q6_k(const unsigned char* wb, const float* x, float* out, int n, int d, int T) {
    pgemm_run<1>(wb, x, out, n, d, T);
}

/* wgemm: 128 tokens by 128 rows, BK=64, fp16 operands, fp32 accumulate.
   Each weight K-tile is dequantized once (pgemm's 64-token tile dequantized
   a 128-token chunk twice). The fp16 tile is row-major with stride 80.
   80*2 = 160 bytes, a multiple of 16 so wmma.load stays aligned, and
   consecutive rows land 8 banks apart instead of the 32-way conflict from
   pgemm's 256-byte stride. Eight warps, each holding 4x2 WMMA fragments
   (64 rows by 32 tokens). Activations are narrowed to fp16, not q8.
   The next K-tile's bytes sit in registers across the current MMA, then
   land in the other shared buffer. Shared memory is 81920 bytes, so the
   host raises the dynamic cap. Not timed. ODINFER_CUDA_PREFILL_LINEAR=mmqk
   keeps mmqk. pgemm is still not launched. */

__device__ __forceinline__ unsigned char wg_byte(uint4 v, int i) {
    unsigned w = i < 4 ? v.x : i < 8 ? v.y : i < 12 ? v.z : v.w;
    return (unsigned char)(w >> ((i & 3) * 8));
}

__device__ __forceinline__ uint4 wg_ld_u4(const unsigned char* p) {
    uint4 v;
    v.x = (unsigned)p[0] | ((unsigned)p[1] << 8) | ((unsigned)p[2] << 16) | ((unsigned)p[3] << 24);
    v.y = (unsigned)p[4] | ((unsigned)p[5] << 8) | ((unsigned)p[6] << 16) | ((unsigned)p[7] << 24);
    v.z = (unsigned)p[8] | ((unsigned)p[9] << 8) | ((unsigned)p[10] << 16) | ((unsigned)p[11] << 24);
    v.w = (unsigned)p[12] | ((unsigned)p[13] << 8) | ((unsigned)p[14] << 16) | ((unsigned)p[15] << 24);
    return v;
}

__device__ __forceinline__ void wg_dequant_q4(uint4 hdr, uint4 qs0, uint4 qs1, int part, unsigned short* dst) {
    unsigned char hb[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) hb[i] = wg_byte(hdr, i);
    int sc, m;
    scale_min_k4(part, hb + 4, &sc, &m);
    float ds = ld_f16(hb) * (float)sc;
    float dm = ld_f16(hb + 2) * (float)m;
    int hi = part & 1;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        unsigned char q = j < 16 ? wg_byte(qs0, j) : wg_byte(qs1, j - 16);
        int qv = hi ? (int)(q >> 4) : (int)(q & 0xF);
        dst[j] = f32_to_f16((float)qv * ds - dm);
    }
}

__device__ __forceinline__ void wg_dequant_q6(uint4 ql0, uint4 ql1, uint4 qh0, uint4 qh1,
                                              unsigned sc_lo, unsigned sc_hi, unsigned d_bits,
                                              int part, unsigned short* dst) {
    int group = part & 3;
    float d = f16_bits(d_bits);
#pragma unroll
    for (int l = 0; l < 32; ++l) {
        unsigned char qh = l < 16 ? wg_byte(qh0, l) : wg_byte(qh1, l - 16);
        unsigned char ql = l < 16 ? wg_byte(ql0, l) : wg_byte(ql1, l - 16);
        int is = l >> 4;
        int qv, sc_i;
        if (group == 0) {
            qv = (int)((ql & 0xF) | (((qh >> 0) & 3) << 4)) - 32;
            sc_i = is;
        } else if (group == 1) {
            qv = (int)((ql & 0xF) | (((qh >> 2) & 3) << 4)) - 32;
            sc_i = is + 2;
        } else if (group == 2) {
            qv = (int)((ql >> 4) | (((qh >> 4) & 3) << 4)) - 32;
            sc_i = is + 4;
        } else {
            qv = (int)((ql >> 4) | (((qh >> 6) & 3) << 4)) - 32;
            sc_i = is + 6;
        }
        unsigned scb = sc_i < 4 ? (sc_lo >> (sc_i * 8)) : (sc_hi >> ((sc_i - 4) * 8));
        dst[l] = f32_to_f16(d * (float)(signed char)(scb & 0xff) * (float)qv);
    }
}

template<int Q6, int X16>
__device__ __forceinline__ void wg_prefetch(const unsigned char* wb, const void* x, int n, int d, int T,
                                            int n0, int t0, int k0, int tid,
                                            float4& x0, float4& x1, float4& x2, float4& x3,
                                            float4& x4, float4& x5, float4& x6, float4& x7,
                                            uint4& h, uint4& q0, uint4& q1, uint4& q2, uint4& q3,
                                            unsigned& sc_lo, unsigned& sc_hi, unsigned& d_bits, int& row_ok) {
    float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 got[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int e = tid + i * 256;
        int elem = e << 2;
        int token = elem >> 6;
        int kk = elem & 63;
        int tt = t0 + token;
        int kkabs = k0 + kk;
        if (tt < T && kkabs + 3 < n) {
            if (X16) {
                /* Four fp16 values packed once per chunk. The bits ride in
                   the float4 channels and wg_commit writes them back out. */
                const unsigned short* hp = (const unsigned short*)x + ((size_t)tt * (size_t)n + (size_t)kkabs);
                uint2 h = *reinterpret_cast<const uint2*>(hp);
                got[i].x = __uint_as_float(h.x & 0xffffu);
                got[i].y = __uint_as_float(h.x >> 16);
                got[i].z = __uint_as_float(h.y & 0xffffu);
                got[i].w = __uint_as_float(h.y >> 16);
            } else {
                const float* p = (const float*)x + ((size_t)tt * (size_t)n + (size_t)kkabs);
                got[i] = *reinterpret_cast<const float4*>(p);
            }
        } else {
            got[i] = z;
        }
    }
    x0 = got[0]; x1 = got[1]; x2 = got[2]; x3 = got[3];
    x4 = got[4]; x5 = got[5]; x6 = got[6]; x7 = got[7];

    int row = tid & 127;
    int sub = tid >> 7;
    int part = ((k0 & 255) >> 5) + sub;
    row_ok = (n0 + row < d) ? 1 : 0;
    h = q0 = q1 = q2 = q3 = make_uint4(0, 0, 0, 0);
    sc_lo = sc_hi = d_bits = 0;
    if (!row_ok) return;
    int bbytes = Q6 ? 210 : 144;
    int nb = n >> 8;
    const unsigned char* blk = wb + ((size_t)(n0 + row) * (size_t)nb + (size_t)(k0 >> 8)) * (size_t)bbytes;
    if (Q6 == 0) {
        h = ld_global_v4(blk);
        const unsigned char* qs = blk + 16 + (part >> 1) * 32;
        q0 = ld_global_v4(qs);
        q1 = ld_global_v4(qs + 16);
        return;
    }
    int half = part >> 2;
    int group = part & 3;
    int ql_off = (group == 1 || group == 3) ? 32 : 0;
    const unsigned char* ql = blk + half * 64 + ql_off;
    const unsigned char* qh = blk + 128 + half * 32;
    const unsigned char* scb = blk + 192 + half * 8;
    q0 = wg_ld_u4(ql);
    q1 = wg_ld_u4(ql + 16);
    q2 = wg_ld_u4(qh);
    q3 = wg_ld_u4(qh + 16);
    sc_lo = (unsigned)scb[0] | ((unsigned)scb[1] << 8) | ((unsigned)scb[2] << 16) | ((unsigned)scb[3] << 24);
    sc_hi = (unsigned)scb[4] | ((unsigned)scb[5] << 8) | ((unsigned)scb[6] << 16) | ((unsigned)scb[7] << 24);
    d_bits = (unsigned)blk[208] | ((unsigned)blk[209] << 8);
}

template<int Q6, int X16>
__device__ __forceinline__ void wg_commit(int n, int T, int n0, int t0, int k0, int tid,
                                          float4 x0, float4 x1, float4 x2, float4 x3,
                                          float4 x4, float4 x5, float4 x6, float4 x7,
                                          uint4 h, uint4 q0, uint4 q1, uint4 q2, uint4 q3,
                                          unsigned sc_lo, unsigned sc_hi, unsigned d_bits, int row_ok,
                                          unsigned short* Xs, unsigned short* Ws) {
    float4 xv[8] = {x0, x1, x2, x3, x4, x5, x6, x7};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int e = tid + i * 256;
        int elem = e << 2;
        int token = elem >> 6;
        int kk = elem & 63;
        unsigned short* dst = Xs + token * 80 + kk;
        int tt = t0 + token;
        int kkabs = k0 + kk;
        float4 v = xv[i];
        if (X16) {
            dst[0] = (tt < T && kkabs < n) ? (unsigned short)__float_as_uint(v.x) : 0;
            dst[1] = (tt < T && kkabs + 1 < n) ? (unsigned short)__float_as_uint(v.y) : 0;
            dst[2] = (tt < T && kkabs + 2 < n) ? (unsigned short)__float_as_uint(v.z) : 0;
            dst[3] = (tt < T && kkabs + 3 < n) ? (unsigned short)__float_as_uint(v.w) : 0;
        } else {
            dst[0] = (tt < T && kkabs < n) ? f32_to_f16(v.x) : 0;
            dst[1] = (tt < T && kkabs + 1 < n) ? f32_to_f16(v.y) : 0;
            dst[2] = (tt < T && kkabs + 2 < n) ? f32_to_f16(v.z) : 0;
            dst[3] = (tt < T && kkabs + 3 < n) ? f32_to_f16(v.w) : 0;
        }
    }
    int row = tid & 127;
    int sub = tid >> 7;
    int part = ((k0 & 255) >> 5) + sub;
    unsigned short* wdst = Ws + row * 80 + sub * 32;
    if (row_ok) {
        if (Q6 == 0) wg_dequant_q4(h, q0, q1, part, wdst);
        else wg_dequant_q6(q0, q1, q2, q3, sc_lo, sc_hi, d_bits, part, wdst);
    } else {
#pragma unroll
        for (int j = 0; j < 32; ++j) wdst[j] = 0;
    }
}

/* Tile stays 128x128, BK=64, 256 threads, one block per SM. The double
   buffer is 81920 bytes, so a second resident block does not fit in 96 KB.
   A 27B MLP prefill is latency bound, not bandwidth bound: 16000 tokens
   read about 1.5 TB of weights in 85 s, ~18 GB/s on a 1134 GB/s V100, and
   about 10x off the tensor-core roof. X is fp32 and every row-tile reloads
   it (17408/128 tiles; one 128x5120 tile is 2.6 MB, ~7x the Q4 weights of
   that tile). Caching a Q4 panel as fp16 for the four token-tiles of a
   512 chunk increases traffic: 368 KB of Q4 versus 1.31 MB of fp16, and
   the first tile also writes the cache. A BK=32 panel plus this double
   buffer, or a 256-row tile, does not fit in 96 KB either (two weight
   panels + the current double buffer is 122880). A second accumulator
   set spills. The launched kernel is unchanged. */
template<int Q6, int X16>
__device__ void wgemm_run(const unsigned char* wb, const void* x, float* out, int n, int d, int T) {
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int t0 = (int)blockIdx.y * 128;
    const int n0 = (int)blockIdx.x * 128;
    if (t0 >= T || n0 >= d || n <= 0) return;

    extern __shared__ __align__(16) unsigned char wgemm_sm[];
    const int tw = warp & 3;
    const int rw = warp >> 2;

    float acc[8][8];
#pragma unroll
    for (int f = 0; f < 8; ++f) {
#pragma unroll
        for (int r = 0; r < 8; ++r) acc[f][r] = 0.f;
    }

    float4 x0, x1, x2, x3, x4, x5, x6, x7;
    uint4 h, q0, q1, q2, q3;
    unsigned sc_lo, sc_hi, d_bits;
    int row_ok;
    wg_prefetch<Q6, X16>(wb, x, n, d, T, n0, t0, 0, tid, x0, x1, x2, x3, x4, x5, x6, x7, h, q0, q1, q2, q3, sc_lo, sc_hi, d_bits, row_ok);
    {
        unsigned short* base = (unsigned short*)wgemm_sm;
        wg_commit<Q6, X16>(n, T, n0, t0, 0, tid, x0, x1, x2, x3, x4, x5, x6, x7, h, q0, q1, q2, q3, sc_lo, sc_hi, d_bits, row_ok, base, base + 128 * 80);
    }
    __syncthreads();

    int phase = 0;
    for (int k0 = 0; k0 < n; k0 += 64) {
        int nxt = k0 + 64;
        float4 nx0, nx1, nx2, nx3, nx4, nx5, nx6, nx7;
        uint4 nh, nq0, nq1, nq2, nq3;
        unsigned nsc_lo, nsc_hi, nd_bits;
        int nrow_ok = 0;
        if (nxt < n) {
            wg_prefetch<Q6, X16>(wb, x, n, d, T, n0, t0, nxt, tid, nx0, nx1, nx2, nx3, nx4, nx5, nx6, nx7, nh, nq0, nq1, nq2, nq3, nsc_lo, nsc_hi, nd_bits, nrow_ok);
        }
        unsigned short* cur = (unsigned short*)(wgemm_sm + phase * 40960);
        unsigned short* Xs = cur;
        unsigned short* Ws = cur + 128 * 80;
        /* A partial token tile still dequants the weight rows once. Skip the
           16-wide MMA fragments that start past T, so a 15-token tail does
           not run the other 112 columns. Full tiles take the first branch
           and the inner loop stays free of that test. Syncs stay outside. */
        int ntok = T - t0;
        if (ntok > 128) ntok = 128;
        int alive0 = (tw * 32) < ntok;
        int alive1 = (tw * 32 + 16) < ntok;
        if (alive0 && alive1) {
#pragma unroll
            for (int kk = 0; kk < 64; kk += 16) {
                unsigned a00, a01, a02, a03, a04, a05, a06, a07;
                unsigned a10, a11, a12, a13, a14, a15, a16, a17;
                unsigned b00, b01, b02, b03, b04, b05, b06, b07;
                unsigned b10, b11, b12, b13, b14, b15, b16, b17;
                unsigned b20, b21, b22, b23, b24, b25, b26, b27;
                unsigned b30, b31, b32, b33, b34, b35, b36, b37;
                wmma_load_a(a00, a01, a02, a03, a04, a05, a06, a07, Xs + (tw * 32) * 80 + kk, 80);
                wmma_load_a(a10, a11, a12, a13, a14, a15, a16, a17, Xs + (tw * 32 + 16) * 80 + kk, 80);
                wmma_load_b(b00, b01, b02, b03, b04, b05, b06, b07, Ws + (rw * 64) * 80 + kk, 80);
                wmma_load_b(b10, b11, b12, b13, b14, b15, b16, b17, Ws + (rw * 64 + 16) * 80 + kk, 80);
                wmma_load_b(b20, b21, b22, b23, b24, b25, b26, b27, Ws + (rw * 64 + 32) * 80 + kk, 80);
                wmma_load_b(b30, b31, b32, b33, b34, b35, b36, b37, Ws + (rw * 64 + 48) * 80 + kk, 80);
                wmma_mma(acc[0][0], acc[0][1], acc[0][2], acc[0][3], acc[0][4], acc[0][5], acc[0][6], acc[0][7], a00, a01, a02, a03, a04, a05, a06, a07, b00, b01, b02, b03, b04, b05, b06, b07);
                wmma_mma(acc[1][0], acc[1][1], acc[1][2], acc[1][3], acc[1][4], acc[1][5], acc[1][6], acc[1][7], a10, a11, a12, a13, a14, a15, a16, a17, b00, b01, b02, b03, b04, b05, b06, b07);
                wmma_mma(acc[2][0], acc[2][1], acc[2][2], acc[2][3], acc[2][4], acc[2][5], acc[2][6], acc[2][7], a00, a01, a02, a03, a04, a05, a06, a07, b10, b11, b12, b13, b14, b15, b16, b17);
                wmma_mma(acc[3][0], acc[3][1], acc[3][2], acc[3][3], acc[3][4], acc[3][5], acc[3][6], acc[3][7], a10, a11, a12, a13, a14, a15, a16, a17, b10, b11, b12, b13, b14, b15, b16, b17);
                wmma_mma(acc[4][0], acc[4][1], acc[4][2], acc[4][3], acc[4][4], acc[4][5], acc[4][6], acc[4][7], a00, a01, a02, a03, a04, a05, a06, a07, b20, b21, b22, b23, b24, b25, b26, b27);
                wmma_mma(acc[5][0], acc[5][1], acc[5][2], acc[5][3], acc[5][4], acc[5][5], acc[5][6], acc[5][7], a10, a11, a12, a13, a14, a15, a16, a17, b20, b21, b22, b23, b24, b25, b26, b27);
                wmma_mma(acc[6][0], acc[6][1], acc[6][2], acc[6][3], acc[6][4], acc[6][5], acc[6][6], acc[6][7], a00, a01, a02, a03, a04, a05, a06, a07, b30, b31, b32, b33, b34, b35, b36, b37);
                wmma_mma(acc[7][0], acc[7][1], acc[7][2], acc[7][3], acc[7][4], acc[7][5], acc[7][6], acc[7][7], a10, a11, a12, a13, a14, a15, a16, a17, b30, b31, b32, b33, b34, b35, b36, b37);
            }
        } else if (alive0 || alive1) {
#pragma unroll
            for (int kk = 0; kk < 64; kk += 16) {
                unsigned a00, a01, a02, a03, a04, a05, a06, a07;
                unsigned b00, b01, b02, b03, b04, b05, b06, b07;
                unsigned b10, b11, b12, b13, b14, b15, b16, b17;
                unsigned b20, b21, b22, b23, b24, b25, b26, b27;
                unsigned b30, b31, b32, b33, b34, b35, b36, b37;
                wmma_load_b(b00, b01, b02, b03, b04, b05, b06, b07, Ws + (rw * 64) * 80 + kk, 80);
                wmma_load_b(b10, b11, b12, b13, b14, b15, b16, b17, Ws + (rw * 64 + 16) * 80 + kk, 80);
                wmma_load_b(b20, b21, b22, b23, b24, b25, b26, b27, Ws + (rw * 64 + 32) * 80 + kk, 80);
                wmma_load_b(b30, b31, b32, b33, b34, b35, b36, b37, Ws + (rw * 64 + 48) * 80 + kk, 80);
                if (alive0) {
                    wmma_load_a(a00, a01, a02, a03, a04, a05, a06, a07, Xs + (tw * 32) * 80 + kk, 80);
                    wmma_mma(acc[0][0], acc[0][1], acc[0][2], acc[0][3], acc[0][4], acc[0][5], acc[0][6], acc[0][7], a00, a01, a02, a03, a04, a05, a06, a07, b00, b01, b02, b03, b04, b05, b06, b07);
                    wmma_mma(acc[2][0], acc[2][1], acc[2][2], acc[2][3], acc[2][4], acc[2][5], acc[2][6], acc[2][7], a00, a01, a02, a03, a04, a05, a06, a07, b10, b11, b12, b13, b14, b15, b16, b17);
                    wmma_mma(acc[4][0], acc[4][1], acc[4][2], acc[4][3], acc[4][4], acc[4][5], acc[4][6], acc[4][7], a00, a01, a02, a03, a04, a05, a06, a07, b20, b21, b22, b23, b24, b25, b26, b27);
                    wmma_mma(acc[6][0], acc[6][1], acc[6][2], acc[6][3], acc[6][4], acc[6][5], acc[6][6], acc[6][7], a00, a01, a02, a03, a04, a05, a06, a07, b30, b31, b32, b33, b34, b35, b36, b37);
                }
                if (alive1) {
                    wmma_load_a(a00, a01, a02, a03, a04, a05, a06, a07, Xs + (tw * 32 + 16) * 80 + kk, 80);
                    wmma_mma(acc[1][0], acc[1][1], acc[1][2], acc[1][3], acc[1][4], acc[1][5], acc[1][6], acc[1][7], a00, a01, a02, a03, a04, a05, a06, a07, b00, b01, b02, b03, b04, b05, b06, b07);
                    wmma_mma(acc[3][0], acc[3][1], acc[3][2], acc[3][3], acc[3][4], acc[3][5], acc[3][6], acc[3][7], a00, a01, a02, a03, a04, a05, a06, a07, b10, b11, b12, b13, b14, b15, b16, b17);
                    wmma_mma(acc[5][0], acc[5][1], acc[5][2], acc[5][3], acc[5][4], acc[5][5], acc[5][6], acc[5][7], a00, a01, a02, a03, a04, a05, a06, a07, b20, b21, b22, b23, b24, b25, b26, b27);
                    wmma_mma(acc[7][0], acc[7][1], acc[7][2], acc[7][3], acc[7][4], acc[7][5], acc[7][6], acc[7][7], a00, a01, a02, a03, a04, a05, a06, a07, b30, b31, b32, b33, b34, b35, b36, b37);
                }
            }
        }
        if (nxt < n) {
            int np = phase ^ 1;
            unsigned short* nb = (unsigned short*)(wgemm_sm + np * 40960);
            wg_commit<Q6, X16>(n, T, n0, t0, nxt, tid, nx0, nx1, nx2, nx3, nx4, nx5, nx6, nx7, nh, nq0, nq1, nq2, nq3, nsc_lo, nsc_hi, nd_bits, nrow_ok, nb, nb + 128 * 80);
            __syncthreads();
            phase = np;
        }
    }
    __syncthreads();

    float* tile = (float*)wgemm_sm;
#pragma unroll
    for (int f = 0; f < 8; ++f) {
        wmma_store(tile + (warp * 8 + f) * 256, 16,
                   acc[f][0], acc[f][1], acc[f][2], acc[f][3], acc[f][4], acc[f][5], acc[f][6], acc[f][7]);
    }
    __syncthreads();
    for (int idx = tid; idx < 8 * 8 * 256; idx += 256) {
        int frag = idx >> 8;
        int e = idx & 255;
        int w = frag >> 3;
        int fi = frag & 7;
        int ri = fi >> 1;
        int ci = fi & 1;
        int i = e >> 4;
        int j = e & 15;
        int t = t0 + (w & 3) * 32 + ci * 16 + i;
        int orow = n0 + (w >> 2) * 64 + ri * 16 + j;
        if (t < T && orow < d) out[(size_t)t * (size_t)d + orow] = tile[idx];
    }
}

extern "C" __global__ void wgemm_q4_k(const unsigned char* wb, const float* x, float* out, int n, int d, int T) {
    wgemm_run<0, 0>(wb, x, out, n, d, T);
}
extern "C" __global__ void wgemm_q6_k(const unsigned char* wb, const float* x, float* out, int n, int d, int T) {
    wgemm_run<1, 0>(wb, x, out, n, d, T);
}

/* Same tile. x is fp16, packed once per activation by pack_f16, so every
   row-tile reloads half the bytes and skips the in-kernel convert.
   ODINFER_CUDA_WGEMM_X=0 keeps wgemm_q4_k / wgemm_q6_k. */
extern "C" __global__ void wgemm_q4_x16(const unsigned char* wb, const unsigned short* x, float* out, int n, int d, int T) {
    wgemm_run<0, 1>(wb, x, out, n, d, T);
}
extern "C" __global__ void wgemm_q6_x16(const unsigned char* wb, const unsigned short* x, float* out, int n, int d, int T) {
    wgemm_run<1, 1>(wb, x, out, n, d, T);
}

extern "C" __global__ void pack_f16(const float* x, unsigned short* y, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) y[i] = f32_to_f16(x[i]);
}

/* pgemm (fp16 dequant + WMMA) was slower than the scalar tile on a V100
   and is not launched. mmqk quantizes activations to q8 once per GEMM and
   dots the stored nibbles with dp4a.

   The timed tile stages one q8 superblock and reuses it across 32 rows
   (4 per warp). That is the default. An 8-row Q4 tile (64 rows) compiled
   to 128 registers with no spills and was slower on a V100 (9B at 8000
   tokens, 333 vs 337 tok/s); ODINFER_CUDA_MMQK_ROWS=8 still selects it.
   Q6 at 8 rows spilled, so Q6 stays at 4 rows per warp. A pair kernel
   dots a second same-shaped matrix (gate+up, K and V) before the next
   superblock replaces X. Shared memory is 22528 bytes. Weights are
   broadcast inside the warp, so staging them does not cut device traffic.
   ODINFER_CUDA_PREFILL_GEMM=0 still selects tc2 / the scalar tile.
   ODINFER_CUDA_PREFILL_FUSE=1 launches the pair kernel. It was neutral
   on that V100, so the two matrices launch separately unless it is set. */

extern "C" __global__ void quant_q8_batch(const float* x, unsigned* xpack, float* scale, float2* sum16, int n, int T) {
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int ng = n >> 5;
    int total = ng * T;
    if (idx >= total) return;
    int g = idx / T;
    int t = idx - g * T;
    const float* xi = x + ((size_t)t * (size_t)n + (size_t)g * 32);
    float amax = 0.f, s0 = 0.f, s1 = 0.f;
    int qv[32];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        float a = xi[j];
        float b = xi[j + 16];
        amax = fmaxf(amax, fmaxf(fabsf(a), fabsf(b)));
        s0 += a;
        s1 += b;
    }
    float d = amax / 127.f;
    float id = d > 0.f ? 1.f / d : 0.f;
    scale[(size_t)g * (size_t)T + t] = d;
    sum16[(size_t)g * (size_t)T + t] = make_float2(s0, s1);
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        int q = (int)rintf(xi[j] * id);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        qv[j] = q;
    }
#pragma unroll
    for (int j = 0; j < 32; j += 4) {
        unsigned pack = (unsigned)(qv[j] & 0xff)
            | ((unsigned)(qv[j + 1] & 0xff) << 8)
            | ((unsigned)(qv[j + 2] & 0xff) << 16)
            | ((unsigned)(qv[j + 3] & 0xff) << 24);
        int k4 = (g * 32 + j) >> 2;
        xpack[(size_t)k4 * (size_t)T + t] = pack;
    }
}


/* Stage superblock b of the K-major q8 tile. xs is [64 k4][64 tok]. */
__device__ void mmqk_stage_x(unsigned* xs, float* scs, float2* sus,
                             const unsigned* xpack, const float* scale, const float2* sum16,
                             int b, int t0, int T) {
    const int BN = 64;
    const int tid = (int)threadIdx.x;
    const int nt = (int)blockDim.x;
    const int nxs = 64 * BN;
    if ((T & 3) == 0) {
        const int nvec = nxs >> 2;
        for (int i = tid; i < nvec; i += nt) {
            int elem = i << 2;
            int k4 = elem >> 6;
            int tok = elem & 63;
            int gt = t0 + tok;
            unsigned gk = (unsigned)(b * 64 + k4);
            uint4 v = make_uint4(0, 0, 0, 0);
            if (gt + 3 < T) {
                v = *(const uint4*)(xpack + (size_t)gk * (size_t)T + (size_t)gt);
            } else {
                unsigned s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                if (gt < T) s0 = xpack[(size_t)gk * (size_t)T + (size_t)gt];
                if (gt + 1 < T) s1 = xpack[(size_t)gk * (size_t)T + (size_t)(gt + 1)];
                if (gt + 2 < T) s2 = xpack[(size_t)gk * (size_t)T + (size_t)(gt + 2)];
                if (gt + 3 < T) s3 = xpack[(size_t)gk * (size_t)T + (size_t)(gt + 3)];
                v = make_uint4(s0, s1, s2, s3);
            }
            *(uint4*)(xs + elem) = v;
        }
    } else {
        for (int i = tid; i < nxs; i += nt) {
            int k4 = i >> 6;
            int tok = i & 63;
            int gt = t0 + tok;
            unsigned v = 0;
            if (gt < T) v = xpack[(size_t)(b * 64 + k4) * (size_t)T + (size_t)gt];
            xs[i] = v;
        }
    }
    const int nsc = 8 * BN;
    for (int i = tid; i < nsc; i += nt) {
        int g = i >> 6;
        int tok = i & 63;
        int gt = t0 + tok;
        float s = 0.f;
        float2 u = make_float2(0.f, 0.f);
        if (gt < T) {
            int gg = b * 8 + g;
            s = scale[(size_t)gg * (size_t)T + (size_t)gt];
            u = sum16[(size_t)gg * (size_t)T + (size_t)gt];
        }
        scs[i] = s;
        sus[i] = u;
    }
}

__device__ __forceinline__ unsigned u4_word(uint4 a, int i) {
    if (i == 0) return a.x;
    if (i == 1) return a.y;
    if (i == 2) return a.z;
    return a.w;
}

__device__ __forceinline__ unsigned mmqk_xs(const unsigned* xs, int k4, int tok) {
    return xs[k4 * 64 + tok];
}

/* Q4's opt-in 64-row kernel and the pair kernels use sm_70
   launch_bounds(256, 2). The default 4-row kernels do not: that bound caps
   them at 128 registers, and the 32-row tile that won on the V100 had none.
   Q6 at 8 rows spilled, so it stays at 4. PAIR=1 dots a second weight
   matrix while the staged q8 superblock is still in shared memory. */
template<int Q6, int ROWS, int PAIR>
__device__ void mmqk_warp(const unsigned char* wb, const unsigned char* wb2, const unsigned* xpack, const float* scale, const float2* sum16,
                          float* out, float* out2, int n, int d, int T) {
    const int BN = 64;
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int nwarps = (int)blockDim.x >> 5;
    const int BM = nwarps * ROWS;
    const int row_block = (int)blockIdx.x * BM;
    const int t0 = (int)blockIdx.y * BN;
    if (t0 >= T || row_block >= d) return;
    const int row0 = row_block + warp * ROWS;
    const int tok0 = t0 + lane;
    const int tok1 = t0 + 32 + lane;
    const int live0 = tok0 < T;
    const int live1 = tok1 < T;
    const int nb = n >> 8;
    extern __shared__ __align__(16) unsigned char mmqk_sm[];
    unsigned* xs = (unsigned*)mmqk_sm;
    float* scs = (float*)(xs + 64 * BN);
    float2* sus = (float2*)(scs + 8 * BN);
    float a[PAIR + 1][ROWS], b[PAIR + 1][ROWS];
#pragma unroll
    for (int m = 0; m < PAIR + 1; ++m) {
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            a[m][r] = 0.f;
            b[m][r] = 0.f;
        }
    }
    for (int sb = 0; sb < nb; ++sb) {
        mmqk_stage_x(xs, scs, sus, xpack, scale, sum16, sb, t0, T);
        __syncthreads();
        if (row0 < d) {
#pragma unroll
            for (int m = 0; m < PAIR + 1; ++m) {
                const unsigned char* wuse = (m == 0) ? wb : wb2;
                if (Q6 == 0) {
    #pragma unroll 1
                    for (int p = 0; p < 4; ++p) {
                        float s00 = 0.f, s10 = 0.f, s01 = 0.f, s11 = 0.f;
                        float2 u00, u10, u01, u11;
                        u00 = u10 = u01 = u11 = make_float2(0.f, 0.f);
                        int lg0 = p * 2;
                        int lg1 = lg0 + 1;
                        if (live0) {
                            s00 = scs[lg0 * BN + lane];
                            s10 = scs[lg1 * BN + lane];
                            u00 = sus[lg0 * BN + lane];
                            u10 = sus[lg1 * BN + lane];
                        }
                        if (live1) {
                            s01 = scs[lg0 * BN + 32 + lane];
                            s11 = scs[lg1 * BN + 32 + lane];
                            u01 = sus[lg0 * BN + 32 + lane];
                            u11 = sus[lg1 * BN + 32 + lane];
                        }
    #pragma unroll
                        for (int rg = 0; rg < ROWS; rg += 4) {
                            uint4 hdr[4], q0[4], q1[4];
    #pragma unroll
                            for (int r = 0; r < 4; ++r) {
                                hdr[r] = make_uint4(0, 0, 0, 0);
                                q0[r] = make_uint4(0, 0, 0, 0);
                                q1[r] = make_uint4(0, 0, 0, 0);
                                int rr = row0 + rg + r;
                                if (rr < d) {
                                    const unsigned char* blk = wuse + ((size_t)rr * (size_t)nb + (size_t)sb) * 144;
                                    hdr[r] = *(const uint4*)blk;
                                    q0[r] = *(const uint4*)(blk + 16 + p * 32);
                                    q1[r] = *(const uint4*)(blk + 32 + p * 32);
                                }
                            }
                            int dlo0[4], dlo1[4], dhi0[4], dhi1[4];
    #pragma unroll
                            for (int r = 0; r < 4; ++r) {
                                dlo0[r] = 0; dlo1[r] = 0; dhi0[r] = 0; dhi1[r] = 0;
                            }
    #pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                int klo = p * 16 + i;
                                int khi = klo + 8;
                                unsigned xlo0 = mmqk_xs(xs, klo, lane);
                                unsigned xlo1 = mmqk_xs(xs, klo, 32 + lane);
                                unsigned xhi0 = mmqk_xs(xs, khi, lane);
                                unsigned xhi1 = mmqk_xs(xs, khi, 32 + lane);
    #pragma unroll
                                for (int r = 0; r < 4; ++r) {
                                    unsigned qw = (i < 4) ? u4_word(q0[r], i) : u4_word(q1[r], i - 4);
                                    int lo = pack_lo_nib(qw);
                                    int hi = pack_hi_nib(qw);
                                    dlo0[r] = dp4a_s32(lo, (int)xlo0, dlo0[r]);
                                    dlo1[r] = dp4a_s32(lo, (int)xlo1, dlo1[r]);
                                    dhi0[r] = dp4a_s32(hi, (int)xhi0, dhi0[r]);
                                    dhi1[r] = dp4a_s32(hi, (int)xhi1, dhi1[r]);
                                }
                            }
    #pragma unroll
                            for (int r = 0; r < 4; ++r) {
                                if (row0 + rg + r >= d) continue;
                                int sc0, m0, sc1, m1;
                                scale_min_hdr(p * 2, hdr[r], &sc0, &m0);
                                scale_min_hdr(p * 2 + 1, hdr[r], &sc1, &m1);
                                float dW = f16_bits(hdr[r].x & 0xffff);
                                float dmin = f16_bits(hdr[r].x >> 16);
                                if (live0) {
                                    a[m][rg + r] += (dW * (float)sc0) * s00 * (float)dlo0[r] - (dmin * (float)m0) * (u00.x + u00.y);
                                    a[m][rg + r] += (dW * (float)sc1) * s10 * (float)dhi0[r] - (dmin * (float)m1) * (u10.x + u10.y);
                                }
                                if (live1) {
                                    b[m][rg + r] += (dW * (float)sc0) * s01 * (float)dlo1[r] - (dmin * (float)m0) * (u01.x + u01.y);
                                    b[m][rg + r] += (dW * (float)sc1) * s11 * (float)dhi1[r] - (dmin * (float)m1) * (u11.x + u11.y);
                                }
                            }
                        }
                    }
                } else {
    #pragma unroll 1
                    for (int slot = 0; slot < 8; ++slot) {
                        int half = slot >> 2;
                        int piece = slot & 3;
                        int l0 = (piece & 1) * 16;
                        int hi_pair = piece >> 1;
                        int scoff = half * 8;
                        int is = l0 >> 4;
                        int s_a = scoff + is + (hi_pair ? 2 : 0);
                        int s_b = scoff + is + (hi_pair ? 6 : 4);
                        int lg0 = half * 4 + (hi_pair ? 1 : 0);
                        int lg2 = lg0 + 2;
                        float sa0 = 0.f, sb0 = 0.f, sa1 = 0.f, sb1 = 0.f;
                        if (live0) {
                            sa0 = scs[lg0 * BN + lane];
                            sb0 = scs[lg2 * BN + lane];
                        }
                        if (live1) {
                            sa1 = scs[lg0 * BN + 32 + lane];
                            sb1 = scs[lg2 * BN + 32 + lane];
                        }
    #pragma unroll
                        for (int rg = 0; rg < ROWS; rg += 4) {
                            uint4 ql[4], qh[4];
                            int sca[4], scb[4];
                            float dW[4];
    #pragma unroll
                            for (int r = 0; r < 4; ++r) {
                                ql[r] = make_uint4(0, 0, 0, 0);
                                qh[r] = make_uint4(0, 0, 0, 0);
                                sca[r] = 0;
                                scb[r] = 0;
                                dW[r] = 0.f;
                                int rr = row0 + rg + r;
                                if (rr < d) {
                                    const unsigned char* blk = wuse + ((size_t)rr * (size_t)nb + (size_t)sb) * 210;
                                    ql[r] = ld_u4_aligned(blk + slot * 16);
                                    qh[r] = ld_u4_aligned(blk + 128 + half * 32 + l0);
                                    sca[r] = (int)(signed char)blk[192 + s_a];
                                    scb[r] = (int)(signed char)blk[192 + s_b];
                                    dW[r] = ld_f16(blk + 208);
                                }
                            }
                            int da0[4], da1[4], db0[4], db1[4];
    #pragma unroll
                            for (int r = 0; r < 4; ++r) {
                                da0[r] = 0; da1[r] = 0; db0[r] = 0; db1[r] = 0;
                            }
    #pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                int k0 = (lg0 * 32 + l0 + i * 4) >> 2;
                                int k2 = (lg2 * 32 + l0 + i * 4) >> 2;
                                unsigned xa0 = mmqk_xs(xs, k0, lane);
                                unsigned xa1 = mmqk_xs(xs, k0, 32 + lane);
                                unsigned xb0 = mmqk_xs(xs, k2, lane);
                                unsigned xb1 = mmqk_xs(xs, k2, 32 + lane);
    #pragma unroll
                                for (int r = 0; r < 4; ++r) {
                                    int qa, qb;
                                    q6_pack(u4_word(ql[r], i), u4_word(qh[r], i), hi_pair, &qa, &qb);
                                    da0[r] = dp4a_s32(qa, (int)xa0, da0[r]);
                                    da1[r] = dp4a_s32(qa, (int)xa1, da1[r]);
                                    db0[r] = dp4a_s32(qb, (int)xb0, db0[r]);
                                    db1[r] = dp4a_s32(qb, (int)xb1, db1[r]);
                                }
                            }
    #pragma unroll
                            for (int r = 0; r < 4; ++r) {
                                if (row0 + rg + r >= d) continue;
                                if (live0) {
                                    a[m][rg + r] += dW[r] * (float)sca[r] * sa0 * (float)da0[r] + dW[r] * (float)scb[r] * sb0 * (float)db0[r];
                                }
                                if (live1) {
                                    b[m][rg + r] += dW[r] * (float)sca[r] * sa1 * (float)da1[r] + dW[r] * (float)scb[r] * sb1 * (float)db1[r];
                                }
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int m = 0; m < PAIR + 1; ++m) {
        float* dst = (m == 0) ? out : out2;
        if (live0) {
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                int rr = row0 + r;
                if (rr < d) dst[(size_t)tok0 * (size_t)d + rr] = a[m][r];
            }
        }
        if (live1) {
#pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                int rr = row0 + r;
                if (rr < d) dst[(size_t)tok1 * (size_t)d + rr] = b[m][r];
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 2)
mmqk_q4_k(const unsigned char* wb, const unsigned* xpack, const float* scale, const float2* sum16,
          float* out, int n, int d, int T) {
    mmqk_warp<0, 8, 0>(wb, 0, xpack, scale, sum16, out, 0, n, d, T);
}
extern "C" __global__ void
mmqk_q4_r4(const unsigned char* wb, const unsigned* xpack, const float* scale, const float2* sum16,
           float* out, int n, int d, int T) {
    mmqk_warp<0, 4, 0>(wb, 0, xpack, scale, sum16, out, 0, n, d, T);
}
extern "C" __global__ void
mmqk_q6_k(const unsigned char* wb, const unsigned* xpack, const float* scale, const float2* sum16,
          float* out, int n, int d, int T) {
    mmqk_warp<1, 4, 0>(wb, 0, xpack, scale, sum16, out, 0, n, d, T);
}
extern "C" __global__ void __launch_bounds__(256, 2)
mmqk_q4_pair(const unsigned char* wb, const unsigned char* wb2, const unsigned* xpack, const float* scale, const float2* sum16,
             float* out, float* out2, int n, int d, int T) {
    mmqk_warp<0, 4, 1>(wb, wb2, xpack, scale, sum16, out, out2, n, d, T);
}
extern "C" __global__ void __launch_bounds__(256, 2)
mmqk_q6_pair(const unsigned char* wb, const unsigned char* wb2, const unsigned* xpack, const float* scale, const float2* sum16,
             float* out, float* out2, int n, int d, int T) {
    mmqk_warp<1, 4, 1>(wb, wb2, xpack, scale, sum16, out, out2, n, d, T);
}

extern "C" __global__ void gemv2r1_q6_k(const unsigned char* wb, const signed char* xq, const float* xd,
                                        float* out, int n, int d, int accumulate) {
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    int row = (int)blockIdx.x * ((int)blockDim.x >> 5) + warp;
    int nb = n >> 8;
    float acc = 0.f;
    if (row < d) {
        for (int b = 0; b < nb; ++b) {
            const unsigned char* blk = wb + ((size_t)row * (size_t)nb + (size_t)b) * 210;
            uint4 scv = make_uint4(0, 0, 0, 0);
            float dW = 0.f;
            if (lane == 0) {
                scv = ld_u4_aligned(blk + 192);
                dW = ld_f16(blk + 208);
            }
            scv = shfl_u4(scv, 0);
            dW = __shfl_sync(0xffffffff, dW, 0);
            const unsigned char* ql = blk;
            const unsigned char* qh = blk + 128;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                int is = lane >> 4;
                int qloff = half * 64;
                int qhoff = half * 32;
                int scoff = half * 8;
                int xelem = lane;
                unsigned char qhb = qh[qhoff + lane];
                int q1 = (int)((ql[qloff + lane] & 0xF) | (((qhb >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[qloff + lane + 32] & 0xF) | (((qhb >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[qloff + lane] >> 4) | (((qhb >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[qloff + lane + 32] >> 4) | (((qhb >> 6) & 3) << 4)) - 32;
                float s0 = dW * (float)scv_i8(scv, scoff + is + 0);
                float s1 = dW * (float)scv_i8(scv, scoff + is + 2);
                float s2 = dW * (float)scv_i8(scv, scoff + is + 4);
                float s3 = dW * (float)scv_i8(scv, scoff + is + 6);
                int g = b * 8 + half * 4;
                acc += s0 * xd[g + 0] * (float)q1 * (float)(int)xq[(size_t)(g + 0) * 32 + xelem];
                acc += s1 * xd[g + 1] * (float)q2 * (float)(int)xq[(size_t)(g + 1) * 32 + xelem];
                acc += s2 * xd[g + 2] * (float)q3 * (float)(int)xq[(size_t)(g + 2) * 32 + xelem];
                acc += s3 * xd[g + 3] * (float)q4 * (float)(int)xq[(size_t)(g + 3) * 32 + xelem];
            }
        }
    }
    acc = warp_sum(acc);
    store_row(out, row, d, lane == 0, acc, accumulate);
}
