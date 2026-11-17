#+build darwin

/* MMQ dequant vectorization sweep: the wide-N kernels (Q4 BM=32 BN=32
   BK=64 ~1470 GFLOPS, Q6 BM=16 BN=64 BK=64 ~800) are dequant-ALU limited.
   The A-load path dequantizes one scalar per loop iteration; these variants
   do uchar4 loads + vector bit math + half4 stores (values identical, float
   order unchanged -> same halves as the production kernels), and vectorize
   the B stage load to half4. BK=128 variants halve the barrier count.

   Timing is THERMAL-INTERLEAVED (see gemv_9b_ks.odin): every kernel runs
   once per round, rounds rotate; min + median reported. Correctness gate is
   ggml.dequant_row rounded to f16 then an f64 dot (same as qn_sweep).

   odin run tools/gemm_harness/mmq_v4_sweep.odin -file -o:speed \
     -collection:ggml=packages
*/

package mmq_v4_sweep

import "core:fmt"
import "core:math"
import "core:os"
import "core:strings"
import "core:time"
import ggml "ggml:ggml"
import NS "core:sys/darwin/Foundation"
import MTL "vendor:darwin/Metal"

GATE_TOL :: 5e-3
ROUNDS :: 10

Variant :: struct {
	name:  string,
	q6:    bool,
	bm, bn, bk: int,
	nt:    int,
}

// Baselines are the shipped kernels verbatim (names differ only to sit in
// one library). v4 variants vectorize the dequant; v128 additionally uses
// BK=128.
VARIANTS := []Variant{
	{"q4k_n32",     false, 32, 32, 64, 512},
	{"q4k_n32v",    false, 32, 32, 64, 512},
	{"q4k_n32v128", false, 32, 32, 128, 512},
	{"q6k_n64",     true,  16, 64, 64, 512},
	{"q6k_n64v",    true,  16, 64, 64, 512},
	{"q6k_n64v128", true,  16, 64, 128, 512},
}

Shape :: struct {
	m, n, k: int,
	q6:      bool,
	name:    string,
}

SHAPES := []Shape{
	{12288, 64, 4096, false, "up/gate 12288x64x4096"},
	{4096, 64, 12288, true, "down 4096x64x12288"},
	{8192, 64, 4096, true, "qkv 8192x64x4096"},
}

OUT_SHAPES := []Shape{
	{248320, 64, 4096, true, "output 248320x64x4096"},
}

g_dev: ^MTL.Device
g_queue: ^MTL.CommandQueue

MSL_HEAD :: `#include <metal_stdlib>
using namespace metal;
struct GemmDims { uint M; uint N; uint K; };
inline void k4_get_scale_min(int j, device const uchar * q, thread uchar & d, thread uchar & m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else { d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
`

// ---- production baseline: gemm_q4k_f32_n32 verbatim ----
MSL_Q4_BASE :: `kernel void q4k_n32(
    device const uchar * A [[buffer(0)]], device const half * B [[buffer(1)]],
    device float * C [[buffer(2)]], constant GemmDims & dims [[buffer(3)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiitg [[thread_index_in_threadgroup]]) {
    const uint M = dims.M, N = dims.N, K = dims.K;
    const uint r0 = tgpig.y * 32, c0 = tgpig.x * 32;
    const ulong row_bytes = (ulong)(K / 256) * 144;
    threadgroup half sa[32 * 64];
    threadgroup half sb[64 * 32];
    simdgroup_half8x8 a_tile, b_tile;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);
    const uint sg_row = (sgitg / 4) * 8;
    const uint sg_col = (sgitg % 4) * 8;
    for (uint k = 0; k < K; k += 64) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint row = tiitg / 8; row < 32; row += 64) {
            const uint col_base = (tiitg % 8) * 8;
            uint e_base = k + col_base;
            if (r0 + row >= M) { continue; }
            device const uchar * blk = A + (ulong)(r0 + row) * row_bytes + (uint)(e_base / 256) * 144;
            int ew = int(e_base & 255), sup = ew >> 6, hf = (ew >> 5) & 1, is = sup * 2 + hf;
            float d = (float)(*(device const half *)(blk));
            float dmin = (float)(*(device const half *)(blk + 2));
            device const uchar * scales = blk + 4;
            device const uchar * qs = blk + 16;
            uchar sc, m; k4_get_scale_min(is, scales, sc, m);
            float dl = d * (float)sc, ml = dmin * (float)m;
            int qoff = sup * 32, shift = 4 * hf;
            int l32 = ew & 31;
            for (uint j = 0; j < 8; j++) {
                uchar nib = (qs[qoff + l32 + int(j)] >> shift) & 0xF;
                sa[row * 64 + col_base + j] = (half)(dl * (float)nib - ml);
            }
        }
        for (uint t = tiitg; t < 64 * 32; t += 512) {
            uint kr = t / 32, nc = t % 32;
            sb[t] = B[(ulong)(k + kr) * N + c0 + nc];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ik = 0; ik < 8; ik++) {
            simdgroup_load(a_tile, sa + sg_row * 64 + ik * 8, 64);
            simdgroup_load(b_tile, sb + ik * 8 * 32 + sg_col, 32);
            simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (r0 + sg_row < M) simdgroup_store(acc, C + (ulong)(r0 + sg_row) * N + c0 + sg_col, N);
}
`

// ---- production baseline: gemm_q6k_f32_n64 verbatim ----
MSL_Q6_BASE :: `kernel void q6k_n64(
    device const uchar * A [[buffer(0)]], device const half * B [[buffer(1)]],
    device float * C [[buffer(2)]], constant GemmDims & dims [[buffer(3)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiitg [[thread_index_in_threadgroup]]) {
    const uint M = dims.M, N = dims.N, K = dims.K;
    const uint r0 = tgpig.y * 16, c0 = tgpig.x * 64;
    const ulong row_bytes = (ulong)(K / 256) * 210;
    threadgroup half sa[16 * 64];
    threadgroup half sb[64 * 64];
    simdgroup_half8x8 a_tile, b_tile;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);
    const uint sg_row = (sgitg / 8) * 8;
    const uint sg_col = (sgitg % 8) * 8;
    for (uint k = 0; k < K; k += 64) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint row = tiitg / 8; row < 16; row += 64) {
            const uint col_base = (tiitg % 8) * 8;
            uint e_base = k + col_base;
            if (r0 + row >= M) { continue; }
            device const uchar * blk = A + (ulong)(r0 + row) * row_bytes + (uint)(e_base / 256) * 210;
            int ew = int(e_base & 255);
            int hf = ew >> 7, sub = (ew >> 5) & 3, is = (ew >> 4) & 1;
            float d = (float)(*(device const half *)(blk + 208));
            device const uchar * ql = blk;
            device const uchar * qh = blk + 128;
            device const char  * sc = (device const char *)(blk + 192);
            float dl = d * (float)sc[hf * 8 + is + 2 * sub];
            int qloff = hf * 64, qhoff = hf * 32;
            int l32 = ew & 31;
            for (uint j = 0; j < 8; j++) {
                int l = l32 + int(j);
                int q;
                if (sub == 0)      q = ((ql[qloff + l]      & 0xF) | (((qh[qhoff + l] >> 0) & 3) << 4)) - 32;
                else if (sub == 1) q = ((ql[qloff + l + 32] & 0xF) | (((qh[qhoff + l] >> 2) & 3) << 4)) - 32;
                else if (sub == 2) q = ((ql[qloff + l] >> 4)       | (((qh[qhoff + l] >> 4) & 3) << 4)) - 32;
                else               q = ((ql[qloff + l + 32] >> 4)  | (((qh[qhoff + l] >> 6) & 3) << 4)) - 32;
                sa[row * 64 + col_base + j] = (half)(dl * (float)q);
            }
        }
        for (uint t = tiitg; t < 64 * 64; t += 512) {
            uint kr = t / 64, nc = t % 64;
            sb[t] = B[(ulong)(k + kr) * N + c0 + nc];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ik = 0; ik < 8; ik++) {
            simdgroup_load(a_tile, sa + sg_row * 64 + ik * 8, 64);
            simdgroup_load(b_tile, sb + ik * 8 * 64 + sg_col, 64);
            simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (r0 + sg_row < M) simdgroup_store(acc, C + (ulong)(r0 + sg_row) * N + c0 + sg_col, N);
}
`

// ---- Q4 v4: vectorized dequant + half4 B loads ----
MSL_Q4_V4 :: `kernel void q4k_n32v(
    device const uchar * A [[buffer(0)]], device const half * B [[buffer(1)]],
    device float * C [[buffer(2)]], constant GemmDims & dims [[buffer(3)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiitg [[thread_index_in_threadgroup]]) {
    const uint M = dims.M, N = dims.N, K = dims.K;
    const uint r0 = tgpig.y * 32, c0 = tgpig.x * 32;
    const ulong row_bytes = (ulong)(K / 256) * 144;
    threadgroup half sa[32 * 64];
    threadgroup half sb[64 * 32];
    simdgroup_half8x8 a_tile, b_tile;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);
    const uint sg_row = (sgitg / 4) * 8;
    const uint sg_col = (sgitg % 4) * 8;
    const uint4 sh_all = uint4(uint(0));
    for (uint k = 0; k < K; k += 64) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint row = tiitg / 8; row < 32; row += 64) {
            const uint col_base = (tiitg % 8) * 8;
            uint e_base = k + col_base;
            if (r0 + row >= M) { continue; }
            device const uchar * blk = A + (ulong)(r0 + row) * row_bytes + (uint)(e_base / 256) * 144;
            int ew = int(e_base & 255), sup = ew >> 6, hf = (ew >> 5) & 1, is = sup * 2 + hf;
            float d = (float)(*(device const half *)(blk));
            float dmin = (float)(*(device const half *)(blk + 2));
            device const uchar * scales = blk + 4;
            device const uchar * qs = blk + 16;
            uchar sc, m; k4_get_scale_min(is, scales, sc, m);
            float dl = d * (float)sc, ml = dmin * (float)m;
            int qoff = sup * 32;
            uint4 sh = uint4(uint(4 * hf)) + sh_all;
            int l32 = ew & 31;
            device const uchar4 * qv = (device const uchar4 *)(qs + qoff + l32);
            threadgroup half4 * sav = (threadgroup half4 *)(sa + row * 64 + col_base);
            uint4 v0 = uint4(qv[0]);
            uint4 v1 = uint4(qv[1]);
            sav[0] = (half4)(float4((v0 >> sh) & 15u) * dl - ml);
            sav[1] = (half4)(float4((v1 >> sh) & 15u) * dl - ml);
        }
        for (uint t4 = tiitg; t4 < 64 * 32 / 4; t4 += 512) {
            uint e = t4 * 4;
            uint kr = e / 32, nc = e % 32;
            *(threadgroup half4 *)(sb + e) = *(device const half4 *)(B + (ulong)(k + kr) * N + c0 + nc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ik = 0; ik < 8; ik++) {
            simdgroup_load(a_tile, sa + sg_row * 64 + ik * 8, 64);
            simdgroup_load(b_tile, sb + ik * 8 * 32 + sg_col, 32);
            simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (r0 + sg_row < M) simdgroup_store(acc, C + (ulong)(r0 + sg_row) * N + c0 + sg_col, N);
}
`

// ---- Q6 v4: vectorized dequant + half4 B loads ----
MSL_Q6_V4 :: `kernel void q6k_n64v(
    device const uchar * A [[buffer(0)]], device const half * B [[buffer(1)]],
    device float * C [[buffer(2)]], constant GemmDims & dims [[buffer(3)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiitg [[thread_index_in_threadgroup]]) {
    const uint M = dims.M, N = dims.N, K = dims.K;
    const uint r0 = tgpig.y * 16, c0 = tgpig.x * 64;
    const ulong row_bytes = (ulong)(K / 256) * 210;
    threadgroup half sa[16 * 64];
    threadgroup half sb[64 * 64];
    simdgroup_half8x8 a_tile, b_tile;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);
    const uint sg_row = (sgitg / 8) * 8;
    const uint sg_col = (sgitg % 8) * 8;
    for (uint k = 0; k < K; k += 64) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint row = tiitg / 8; row < 16; row += 64) {
            const uint col_base = (tiitg % 8) * 8;
            uint e_base = k + col_base;
            if (r0 + row >= M) { continue; }
            device const uchar * blk = A + (ulong)(r0 + row) * row_bytes + (uint)(e_base / 256) * 210;
            int ew = int(e_base & 255);
            int hf = ew >> 7, sub = (ew >> 5) & 3, is = (ew >> 4) & 1;
            float d = (float)(*(device const half *)(blk + 208));
            device const uchar * ql = blk;
            device const uchar * qh = blk + 128;
            device const char  * sc = (device const char *)(blk + 192);
            float dl = d * (float)sc[hf * 8 + is + 2 * sub];
            int qloff = hf * 64 + (sub & 1) * 32, qhoff = hf * 32;
            int l32 = ew & 31;
            device const uchar4 * qlv = (device const uchar4 *)(ql + qloff + l32);
            device const uchar4 * qhv = (device const uchar4 *)(qh + qhoff + l32);
            threadgroup half4 * sav = (threadgroup half4 *)(sa + row * 64 + col_base);
            uint4 qa = uint4(qlv[0]), qb = uint4(qlv[1]);
            uint4 h0 = uint4(qhv[0]), h1 = uint4(qhv[1]);
            uint4 q01, q23;
            if (sub == 0) {
                q01 = (qa & 15u) | ((h0 & 3u) << 4);
                q23 = (qb & 15u) | ((h1 & 3u) << 4);
            } else if (sub == 1) {
                q01 = (qa & 15u) | (((h0 >> 2u) & 3u) << 4);
                q23 = (qb & 15u) | (((h1 >> 2u) & 3u) << 4);
            } else if (sub == 2) {
                q01 = ((qa >> 4u) & 15u) | (((h0 >> 4u) & 3u) << 4);
                q23 = ((qb >> 4u) & 15u) | (((h1 >> 4u) & 3u) << 4);
            } else {
                q01 = ((qa >> 4u) & 15u) | (((h0 >> 6u) & 3u) << 4);
                q23 = ((qb >> 4u) & 15u) | (((h1 >> 6u) & 3u) << 4);
            }
            sav[0] = (half4)(float4(int4(q01) - 32) * dl);
            sav[1] = (half4)(float4(int4(q23) - 32) * dl);
        }
        for (uint t4 = tiitg; t4 < 64 * 64 / 4; t4 += 512) {
            uint e = t4 * 4;
            uint kr = e / 64, nc = e % 64;
            *(threadgroup half4 *)(sb + e) = *(device const half4 *)(B + (ulong)(k + kr) * N + c0 + nc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ik = 0; ik < 8; ik++) {
            simdgroup_load(a_tile, sa + sg_row * 64 + ik * 8, 64);
            simdgroup_load(b_tile, sb + ik * 8 * 64 + sg_col, 64);
            simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (r0 + sg_row < M) simdgroup_store(acc, C + (ulong)(r0 + sg_row) * N + c0 + sg_col, N);
}
`

// ---- BK=128 variants: same A/B stage code, double K step ----
MSL_Q4_V128 :: `kernel void q4k_n32v128(
    device const uchar * A [[buffer(0)]], device const half * B [[buffer(1)]],
    device float * C [[buffer(2)]], constant GemmDims & dims [[buffer(3)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiitg [[thread_index_in_threadgroup]]) {
    const uint M = dims.M, N = dims.N, K = dims.K;
    const uint r0 = tgpig.y * 32, c0 = tgpig.x * 32;
    const ulong row_bytes = (ulong)(K / 256) * 144;
    threadgroup half sa[32 * 128];
    threadgroup half sb[128 * 32];
    simdgroup_half8x8 a_tile, b_tile;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);
    const uint sg_row = (sgitg / 4) * 8;
    const uint sg_col = (sgitg % 4) * 8;
    for (uint k = 0; k < K; k += 128) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint row = tiitg / 16; row < 32; row += 32) {
            const uint col_base = (tiitg % 16) * 8;
            uint e_base = k + col_base;
            if (r0 + row >= M) { continue; }
            device const uchar * blk = A + (ulong)(r0 + row) * row_bytes + (uint)(e_base / 256) * 144;
            int ew = int(e_base & 255), sup = ew >> 6, hf = (ew >> 5) & 1, is = sup * 2 + hf;
            float d = (float)(*(device const half *)(blk));
            float dmin = (float)(*(device const half *)(blk + 2));
            device const uchar * scales = blk + 4;
            device const uchar * qs = blk + 16;
            uchar sc, m; k4_get_scale_min(is, scales, sc, m);
            float dl = d * (float)sc, ml = dmin * (float)m;
            int qoff = sup * 32;
            uint4 sh = uint4(uint(4 * hf));
            int l32 = ew & 31;
            device const uchar4 * qv = (device const uchar4 *)(qs + qoff + l32);
            threadgroup half4 * sav = (threadgroup half4 *)(sa + row * 128 + col_base);
            uint4 v0 = uint4(qv[0]);
            uint4 v1 = uint4(qv[1]);
            sav[0] = (half4)(float4((v0 >> sh) & 15u) * dl - ml);
            sav[1] = (half4)(float4((v1 >> sh) & 15u) * dl - ml);
        }
        for (uint t4 = tiitg; t4 < 128 * 32 / 4; t4 += 512) {
            uint e = t4 * 4;
            uint kr = e / 32, nc = e % 32;
            *(threadgroup half4 *)(sb + e) = *(device const half4 *)(B + (ulong)(k + kr) * N + c0 + nc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ik = 0; ik < 16; ik++) {
            simdgroup_load(a_tile, sa + sg_row * 128 + ik * 8, 128);
            simdgroup_load(b_tile, sb + ik * 8 * 32 + sg_col, 32);
            simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (r0 + sg_row < M) simdgroup_store(acc, C + (ulong)(r0 + sg_row) * N + c0 + sg_col, N);
}
`

MSL_Q6_V128 :: `kernel void q6k_n64v128(
    device const uchar * A [[buffer(0)]], device const half * B [[buffer(1)]],
    device float * C [[buffer(2)]], constant GemmDims & dims [[buffer(3)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiitg [[thread_index_in_threadgroup]]) {
    const uint M = dims.M, N = dims.N, K = dims.K;
    const uint r0 = tgpig.y * 16, c0 = tgpig.x * 64;
    const ulong row_bytes = (ulong)(K / 256) * 210;
    threadgroup half sa[16 * 128];
    threadgroup half sb[128 * 64];
    simdgroup_half8x8 a_tile, b_tile;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);
    const uint sg_row = (sgitg / 8) * 8;
    const uint sg_col = (sgitg % 8) * 8;
    for (uint k = 0; k < K; k += 128) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint row = tiitg / 16; row < 16; row += 32) {
            const uint col_base = (tiitg % 16) * 8;
            uint e_base = k + col_base;
            if (r0 + row >= M) { continue; }
            device const uchar * blk = A + (ulong)(r0 + row) * row_bytes + (uint)(e_base / 256) * 210;
            int ew = int(e_base & 255);
            int hf = ew >> 7, sub = (ew >> 5) & 3, is = (ew >> 4) & 1;
            float d = (float)(*(device const half *)(blk + 208));
            device const uchar * ql = blk;
            device const uchar * qh = blk + 128;
            device const char  * sc = (device const char *)(blk + 192);
            float dl = d * (float)sc[hf * 8 + is + 2 * sub];
            int qloff = hf * 64 + (sub & 1) * 32, qhoff = hf * 32;
            int l32 = ew & 31;
            device const uchar4 * qlv = (device const uchar4 *)(ql + qloff + l32);
            device const uchar4 * qhv = (device const uchar4 *)(qh + qhoff + l32);
            threadgroup half4 * sav = (threadgroup half4 *)(sa + row * 128 + col_base);
            uint4 qa = uint4(qlv[0]), qb = uint4(qlv[1]);
            uint4 h0 = uint4(qhv[0]), h1 = uint4(qhv[1]);
            uint4 q01, q23;
            if (sub == 0) {
                q01 = (qa & 15u) | ((h0 & 3u) << 4);
                q23 = (qb & 15u) | ((h1 & 3u) << 4);
            } else if (sub == 1) {
                q01 = (qa & 15u) | (((h0 >> 2u) & 3u) << 4);
                q23 = (qb & 15u) | (((h1 >> 2u) & 3u) << 4);
            } else if (sub == 2) {
                q01 = ((qa >> 4u) & 15u) | (((h0 >> 4u) & 3u) << 4);
                q23 = ((qb >> 4u) & 15u) | (((h1 >> 4u) & 3u) << 4);
            } else {
                q01 = ((qa >> 4u) & 15u) | (((h0 >> 6u) & 3u) << 4);
                q23 = ((qb >> 4u) & 15u) | (((h1 >> 6u) & 3u) << 4);
            }
            sav[0] = (half4)(float4(int4(q01) - 32) * dl);
            sav[1] = (half4)(float4(int4(q23) - 32) * dl);
        }
        for (uint t4 = tiitg; t4 < 128 * 64 / 4; t4 += 512) {
            uint e = t4 * 4;
            uint kr = e / 64, nc = e % 64;
            *(threadgroup half4 *)(sb + e) = *(device const half4 *)(B + (ulong)(k + kr) * N + c0 + nc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ik = 0; ik < 16; ik++) {
            simdgroup_load(a_tile, sa + sg_row * 128 + ik * 8, 128);
            simdgroup_load(b_tile, sb + ik * 8 * 64 + sg_col, 64);
            simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (r0 + sg_row < M) simdgroup_store(acc, C + (ulong)(r0 + sg_row) * N + c0 + sg_col, N);
}
`

rng_f :: proc(seed: ^u32) -> f32 {
	x := seed^
	x = x ~ (x << 13)
	x = x ~ (x >> 17)
	x = x ~ (x << 5)
	seed^ = x
	return f32((x >> 8) & 0xFFFFFF) / f32(0x1000000)
}

write_f16_le :: proc(buf: []u8, off: int, v: f32) {
	h := f16(v)
	u := transmute(u16)h
	buf[off] = u8(u & 0xFF)
	buf[off + 1] = u8((u >> 8) & 0xFF)
}

gen_quant :: proc(q6: bool, M, K: int, seed_in: u32) -> []u8 {
	nb := K / 256
	bs := q6 ? 210 : 144
	a := make([]u8, M * nb * bs)
	seed := seed_in
	for b in 0 ..< M * nb {
		base := b * bs
		if q6 {
			write_f16_le(a, base + 208, rng_f(&seed)*0.18 + 0.02)
			for i in 0 ..< 128 { a[base + i] = u8(rng_f(&seed) * 256.0) }
			for i in 0 ..< 64 { a[base + 128 + i] = u8(rng_f(&seed) * 256.0) }
			for i in 0 ..< 16 { a[base + 192 + i] = u8(rng_f(&seed) * 256.0) }
		} else {
			write_f16_le(a, base, rng_f(&seed)*0.18 + 0.02)
			write_f16_le(a, base + 2, rng_f(&seed)*0.05)
			for s in 0 ..< 12 { a[base + 4 + s] = u8(rng_f(&seed) * 256.0) }
			for q in 0 ..< 128 { a[base + 16 + q] = u8(rng_f(&seed) * 256.0) }
		}
	}
	return a
}

gen_b :: proc(K, N: int, seed_in: u32) -> []f16 {
	b := make([]f16, K * N)
	seed := seed_in
	for i in 0 ..< K * N {
		b[i] = f16(rng_f(&seed)*2.0 - 1.0)
	}
	return b
}

GemmDims :: struct { M, N, K: u32 }

make_pso :: proc(name: string) -> ^MTL.ComputePipelineState {
	ns_name := NS.String.alloc()->initWithOdinString(name)
	defer ns_name->release()
	fn := g_lib->newFunctionWithName(ns_name)
	if fn == nil {
		fmt.eprintfln("missing {}", name)
		os.exit(1)
	}
	defer fn->release()
	pso, perr := g_dev->newComputePipelineStateWithFunction(fn)
	if perr != nil {
		fmt.eprintfln("pso {}: {}", name, perr->localizedDescription()->odinString())
		os.exit(1)
	}
	return pso
}

g_lib: ^MTL.Library

run_once :: proc(pso: ^MTL.ComputePipelineState, bufA, bufB, bufC: ^MTL.Buffer, M, N, K, bm, bn, nt: int) {
	cmd := g_queue->commandBuffer()
	enc := cmd->computeCommandEncoder()
	enc->setComputePipelineState(pso)
	enc->setBuffer(bufA, 0, 0)
	enc->setBuffer(bufB, 0, 1)
	enc->setBuffer(bufC, 0, 2)
	dims := GemmDims{u32(M), u32(N), u32(K)}
	enc->setBytes(([^]u8)(&dims)[:size_of(GemmDims)], 3)
	enc->dispatchThreadgroups(MTL.Size{NS.Integer(N / bn), NS.Integer(M / bm), 1}, MTL.Size{NS.Integer(nt), 1, 1})
	enc->endEncoding()
	cmd->commit()
	cmd->waitUntilCompleted()
}

gate :: proc(v: Variant) -> bool {
	M, N, K := 192, 64, 256
	nb := K / 256
	bs := v.q6 ? 210 : 144
	a_bytes := gen_quant(v.q6, M, K, 0xBEEF)
	b_h := gen_b(K, N, 0xF00D)
	defer delete(a_bytes)
	defer delete(b_h)
	a_ref := make([]f32, M * K)
	defer delete(a_ref)
	ggml.dequant_row(v.q6 ? .Q6_K : .Q4_K, a_bytes, M * K, a_ref)
	for i in 0 ..< M * K {
		a_ref[i] = f32(f16(a_ref[i]))
	}
	c_ref := make([]f64, M * N)
	defer delete(c_ref)
	for m in 0 ..< M {
		for n in 0 ..< N {
			s: f64 = 0
			for k in 0 ..< K {
				s += f64(a_ref[m*K + k]) * f64(b_h[k*N + n])
			}
			c_ref[m*N + n] = s
		}
	}
	bufA := g_dev->newBufferWithLength(NS.UInteger(M * nb * bs), MTL.ResourceStorageModeShared)
	bufB := g_dev->newBufferWithLength(NS.UInteger(K * N * 2), MTL.ResourceStorageModeShared)
	bufC := g_dev->newBufferWithLength(NS.UInteger(M * N * 4), MTL.ResourceStorageModeShared)
	defer bufA->release()
	defer bufB->release()
	defer bufC->release()
	copy(bufA->contentsAsSlice([]u8), a_bytes)
	dstb := bufB->contentsAsSlice([]u8)
	sb := ([^]u8)(raw_data(b_h))[:len(b_h) * 2]
	copy(dstb, sb)
	pso := make_pso(v.name)
	defer pso->release()
	run_once(pso, bufA, bufB, bufC, M, N, K, v.bm, v.bn, v.nt)
	c_gpu := bufC->contentsAsSlice([]f32)[:M * N]
	max_abs, scale: f64 = 0, 0
	for i in 0 ..< M * N {
		d := math.abs(f64(c_gpu[i]) - c_ref[i])
		if d > max_abs { max_abs = d }
		a := math.abs(c_ref[i])
		if a > scale { scale = a }
	}
	scaled := scale > 1e-6 ? max_abs / scale : max_abs
	ok := scaled <= GATE_TOL
	fmt.eprintfln("gate {:12s}: max_rel {:.2e} {}", v.name, scaled, ok ? "OK" : "FAIL")
	return ok
}

Bench_Entry :: struct {
	v:    Variant,
	s:    Shape,
	pso:  ^MTL.ComputePipelineState,
	bufA: ^MTL.Buffer,
	bufB: ^MTL.Buffer,
	bufC: ^MTL.Buffer,
}

main :: proc() {
	g_dev = MTL.CreateSystemDefaultDevice()
	if g_dev == nil {
		fmt.eprintln("no metal device")
		os.exit(1)
	}
	defer g_dev->release()
	fmt.eprintfln("metal: {}", g_dev->name()->odinString())
	g_queue = g_dev->newCommandQueue()
	defer g_queue->release()

	src, _ := strings.concatenate({MSL_HEAD, MSL_Q4_BASE, MSL_Q6_BASE, MSL_Q4_V4, MSL_Q6_V4, MSL_Q4_V128, MSL_Q6_V128})
	src_ns := NS.String.alloc()->initWithOdinString(src)
	defer src_ns->release()
	opts := MTL.CompileOptions.alloc()->init()
	defer opts->release()
	opts->setFastMathEnabled(true)
	fmt.eprintln("compiling...")
	lib, err := g_dev->newLibraryWithSource(src_ns, opts)
	if err != nil {
		fmt.eprintfln("compile: {}", err->localizedDescription()->odinString())
		os.exit(1)
	}
	g_lib = lib
	defer lib->release()
	fmt.eprintln("compiled")

	all_ok := true
	for v in VARIANTS {
		if !gate(v) { all_ok = false }
	}
	if !all_ok {
		fmt.eprintln("gate failures; aborting timing")
		os.exit(1)
	}

	bench := proc(shapes: []Shape, label: string, rounds: int) {
		entries := make([dynamic]Bench_Entry, 0, len(VARIANTS) * len(shapes))
		defer delete(entries)
		for s in shapes {
			for v in VARIANTS {
				if v.q6 != s.q6 { continue }
				a_bytes := gen_quant(s.q6, s.m, s.k, 0xBEEF)
				b_h := gen_b(s.k, s.n, 0xF00D)
				e := Bench_Entry{
					v    = v,
					s    = s,
					pso  = make_pso(v.name),
					bufA = g_dev->newBufferWithLength(NS.UInteger(len(a_bytes)), MTL.ResourceStorageModeShared),
					bufB = g_dev->newBufferWithLength(NS.UInteger(s.k * s.n * 2), MTL.ResourceStorageModeShared),
					bufC = g_dev->newBufferWithLength(NS.UInteger(s.m * s.n * 4), MTL.ResourceStorageModeShared),
				}
				copy(e.bufA->contentsAsSlice([]u8), a_bytes)
				dstb := e.bufB->contentsAsSlice([]u8)
				sb := ([^]u8)(raw_data(b_h))[:len(b_h) * 2]
				copy(dstb, sb)
				delete(a_bytes)
				delete(b_h)
				append(&entries, e)
				run_once(e.pso, e.bufA, e.bufB, e.bufC, s.m, s.n, s.k, v.bm, v.bn, v.nt)
			}
		}
		samples := make([dynamic][dynamic]f64, len(entries))
		defer delete(samples)
		for r := 0; r < rounds; r += 1 {
			for e, i in entries {
				t0 := time.tick_now()
				run_once(e.pso, e.bufA, e.bufB, e.bufC, e.s.m, e.s.n, e.s.k, e.v.bm, e.v.bn, e.v.nt)
				append(&samples[i], f64(time.tick_since(t0)) / 1e6)
			}
		}
		fmt.eprintfln("--- {} (interleaved x{}) ---", label, rounds)
		for e, i in entries {
			best, sum := math.INF_F64, 0.0
			for ms in samples[i] {
				if ms < best { best = ms }
				sum += ms
			}
			med := sum / f64(len(samples[i]))
			flop := 2.0 * f64(e.s.m) * f64(e.s.n) * f64(e.s.k)
			fmt.eprintfln("{:12s} {:6.2f} ms  {:6.0f} GFLOPS   (med {:6.0f})", e.v.name, best, flop / (best * 1e-3) / 1e9, flop / (med * 1e-3) / 1e9)
		}
		for e in entries {
			e.pso->release(); e.bufA->release(); e.bufB->release(); e.bufC->release()
		}
	}

	bench(SHAPES, "prefill 64-token slabs", ROUNDS)
	bench(OUT_SHAPES, "output head", 4)
}
