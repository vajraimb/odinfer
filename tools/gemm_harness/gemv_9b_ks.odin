// Decode-GEMV A/B rig with THERMAL-INTERLEAVED timing (2026-11-17).
//
// Why interleaved: this passively-cooled M3 Air drifts GPU clocks ~2x within
// a session; sequentially timing each variant produces inverted conclusions
// (measured: q4k ks8r2 "won 44%" sequential, loses 20% interleaved). This
// rig rotates every job through one command buffer per round (12 rounds,
// 5 dispatches each) and reports min + median, so all variants sample the
// same thermal windows.
//
// Contents: production generic + Tier 3.5 constexpr spec twins vs the
// K-split / row-sharing / uchar4-vectorized variants, on the tensors that
// carry ~90% of the 9B per-token weight traffic (up/gate Q4_K 12288x4096,
// down Q6_K 4096x12288, qkv Q6_K 8192x4096, output Q6_K 248320x4096), plus
// the uncommitted IQ ks kernels on the 27B down shape and gemv_q8_0 as the
// cheap-dequant bandwidth-ceiling probe on identical shapes.
//
// Findings (see packages/qwen3_5/ROOFLINE.md 2026-11-17): Q4_K/Q6_K
// generic = spec = every variant (55-75 GB/s, Q8_0 probe 66-86) -> those
// kernels are at the memory wall; IQ ks kernels are real (+27% IQ2_XXS
// same-window). CHECK=1 verifies each variant against the production
// kernel on synthetic weights (all bit-identical or summation-order only).
//
// odin run tools/gemm_harness/gemv_9b_ks.odin -file -o:speed \
//   -collection:ggml=packages -collection:qwen3_5=packages
// CHECK=1 odin run ...   (correctness only)

package gemv_9b_ks

import q35 "qwen3_5:qwen3_5"

import NS "core:sys/darwin/Foundation"
import "core:fmt"
import "core:math"
import "core:os"
import "core:strings"
import "core:time"
import MTL "vendor:darwin/Metal"

Dims :: struct { n, d: u32 }

// Kernels with `nsb`/`d` as uniforms (buffer 3/4). tg_div: rows per TG.
EXTRA :: `

// ---- Q4_K: 16 lanes per row, one superblock per lane (nsb <= 16), 16 rows / TG ----
kernel void q4k_ks16(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant uint &nsb_in [[buffer(3)]], constant uint &din [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    uint nsb = nsb_in; uint d = din;
    uint row = ((tg * 8u + sgitg) * 2u) + (tiisg >> 4);
    uint lane = tiisg & 15u;
    float sum = 0.0;
    if (row < d && lane < nsb) {
        device const uchar *blk = wb + (ulong)row * nsb * 144 + (ulong)lane * 144;
        float dall = (float)(*(device const half*)blk); float dmin = (float)(*(device const half*)(blk + 2));
        device const uchar *scales = blk + 4; device const uchar *qs = blk + 16; uint ybase = lane * 256; int is = 0; uint qoff = 0;
        float slo = 0.0, shi = 0.0;
        for (uint j = 0; j < 256; j += 64) { uchar sc, m; get_scale_min_k4(is + 0, scales, sc, m); float d1 = dall * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, scales, sc, m); float d2 = dall * sc, m2 = dmin * m;
            for (uint l = 0; l < 32; l += 4) {
                float4 wl = float4((float)(qs[qoff+l]&0xF),(float)(qs[qoff+l+1]&0xF),(float)(qs[qoff+l+2]&0xF),(float)(qs[qoff+l+3]&0xF));
                float4 wh = float4((float)(qs[qoff+l]>>4),(float)(qs[qoff+l+1]>>4),(float)(qs[qoff+l+2]>>4),(float)(qs[qoff+l+3]>>4));
                float4 xl = float4(x[ybase+j+l],x[ybase+j+l+1],x[ybase+j+l+2],x[ybase+j+l+3]);
                float4 xh = float4(x[ybase+j+32+l],x[ybase+j+32+l+1],x[ybase+j+32+l+2],x[ybase+j+32+l+3]);
                slo += dot(wl * d1 - m1, xl);
                shi += dot(wh * d2 - m2, xh);
            }
            qoff += 32; is += 2; }
        sum = slo + shi;
    }
    sum += simd_shuffle_xor(sum, 8);
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if ((tiisg & 15u) == 0 && row < d) out[row] = sum;
}

// ---- Q4_K: two rows share an 8-lane group, x float4s loaded once, 64 rows / TG ----
kernel void q4k_ks8r2(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant uint &nsb_in [[buffer(3)]], constant uint &din [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    uint nsb = nsb_in; uint d = din;
    uint lane = tiisg & 7u;
    uint row0 = (((tg * 8u + sgitg) * 4u) + (tiisg >> 3)) * 2u;
    uint row1 = row0 + 1u;
    float slo0 = 0.0, shi0 = 0.0, slo1 = 0.0, shi1 = 0.0;
    if (row1 < d) {
        device const uchar *wr0 = wb + (ulong)row0 * nsb * 144;
        device const uchar *wr1 = wb + (ulong)row1 * nsb * 144;
        for (uint sb = lane; sb < nsb; sb += 8) {
            device const uchar *b0 = wr0 + sb * 144; device const uchar *b1 = wr1 + sb * 144;
            float da0 = (float)(*(device const half*)b0); float dm0 = (float)(*(device const half*)(b0 + 2));
            float da1 = (float)(*(device const half*)b1); float dm1_ = (float)(*(device const half*)(b1 + 2));
            device const uchar *sc0 = b0 + 4; device const uchar *qs0 = b0 + 16;
            device const uchar *sc1 = b1 + 4; device const uchar *qs1 = b1 + 16;
            uint ybase = sb * 256; int is = 0; uint qoff = 0;
            for (uint j = 0; j < 256; j += 64) { uchar s, m;
                get_scale_min_k4(is + 0, sc0, s, m); float d10 = da0 * s, m10 = dm0 * m;
                get_scale_min_k4(is + 1, sc0, s, m); float d20 = da0 * s, m20 = dm0 * m;
                get_scale_min_k4(is + 0, sc1, s, m); float d11 = da1 * s, m11 = dm1_ * m;
                get_scale_min_k4(is + 1, sc1, s, m); float d21 = da1 * s, m21 = dm1_ * m;
                for (uint l = 0; l < 32; l += 4) {
                    float4 xl = float4(x[ybase+j+l],x[ybase+j+l+1],x[ybase+j+l+2],x[ybase+j+l+3]);
                    float4 xh = float4(x[ybase+j+32+l],x[ybase+j+32+l+1],x[ybase+j+32+l+2],x[ybase+j+32+l+3]);
                    float4 wl0 = float4((float)(qs0[qoff+l]&0xF),(float)(qs0[qoff+l+1]&0xF),(float)(qs0[qoff+l+2]&0xF),(float)(qs0[qoff+l+3]&0xF));
                    float4 wh0 = float4((float)(qs0[qoff+l]>>4),(float)(qs0[qoff+l+1]>>4),(float)(qs0[qoff+l+2]>>4),(float)(qs0[qoff+l+3]>>4));
                    float4 wl1 = float4((float)(qs1[qoff+l]&0xF),(float)(qs1[qoff+l+1]&0xF),(float)(qs1[qoff+l+2]&0xF),(float)(qs1[qoff+l+3]&0xF));
                    float4 wh1 = float4((float)(qs1[qoff+l]>>4),(float)(qs1[qoff+l+1]>>4),(float)(qs1[qoff+l+2]>>4),(float)(qs1[qoff+l+3]>>4));
                    slo0 += dot(wl0 * d10 - m10, xl);
                    shi0 += dot(wh0 * d20 - m20, xh);
                    slo1 += dot(wl1 * d11 - m11, xl);
                    shi1 += dot(wh1 * d21 - m21, xh);
                }
                qoff += 32; is += 2; }
        }
    }
    float s0 = slo0 + shi0, s1 = slo1 + shi1;
    s0 += simd_shuffle_xor(s0, 4); s0 += simd_shuffle_xor(s0, 2); s0 += simd_shuffle_xor(s0, 1);
    s1 += simd_shuffle_xor(s1, 4); s1 += simd_shuffle_xor(s1, 2); s1 += simd_shuffle_xor(s1, 1);
    if (lane == 0 && row1 < d) { out[row0] = s0; out[row1] = s1; }
}

// ---- Q4_K: four rows share an 8-lane group, 128 rows / TG ----
kernel void q4k_ks8r4(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant uint &nsb_in [[buffer(3)]], constant uint &din [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    uint nsb = nsb_in; uint d = din;
    uint lane = tiisg & 7u;
    uint rowb = ((tg * 8u + sgitg) * 4u + (tiisg >> 3)) * 4u;
    float slo0 = 0.0, shi0 = 0.0, slo1 = 0.0, shi1 = 0.0, slo2 = 0.0, shi2 = 0.0, slo3 = 0.0, shi3 = 0.0;
    if (rowb + 3u < d) {
        device const uchar *wr0 = wb + (ulong)(rowb + 0u) * nsb * 144;
        device const uchar *wr1 = wb + (ulong)(rowb + 1u) * nsb * 144;
        device const uchar *wr2 = wb + (ulong)(rowb + 2u) * nsb * 144;
        device const uchar *wr3 = wb + (ulong)(rowb + 3u) * nsb * 144;
        for (uint sb = lane; sb < nsb; sb += 8) {
            device const uchar *b0 = wr0 + sb * 144; device const uchar *b1 = wr1 + sb * 144;
            device const uchar *b2 = wr2 + sb * 144; device const uchar *b3 = wr3 + sb * 144;
            float da0 = (float)(*(device const half*)b0); float dm0 = (float)(*(device const half*)(b0 + 2));
            float da1 = (float)(*(device const half*)b1); float dm1_ = (float)(*(device const half*)(b1 + 2));
            float da2 = (float)(*(device const half*)b2); float dm2 = (float)(*(device const half*)(b2 + 2));
            float da3 = (float)(*(device const half*)b3); float dm3 = (float)(*(device const half*)(b3 + 2));
            device const uchar *sc0 = b0 + 4; device const uchar *qs0 = b0 + 16;
            device const uchar *sc1 = b1 + 4; device const uchar *qs1 = b1 + 16;
            device const uchar *sc2 = b2 + 4; device const uchar *qs2 = b2 + 16;
            device const uchar *sc3 = b3 + 4; device const uchar *qs3 = b3 + 16;
            uint ybase = sb * 256; int is = 0; uint qoff = 0;
            for (uint j = 0; j < 256; j += 64) { uchar s, m;
                get_scale_min_k4(is + 0, sc0, s, m); float d10 = da0 * s, m10 = dm0 * m;
                get_scale_min_k4(is + 1, sc0, s, m); float d20 = da0 * s, m20 = dm0 * m;
                get_scale_min_k4(is + 0, sc1, s, m); float d11 = da1 * s, m11 = dm1_ * m;
                get_scale_min_k4(is + 1, sc1, s, m); float d21 = da1 * s, m21 = dm1_ * m;
                get_scale_min_k4(is + 0, sc2, s, m); float d12 = da2 * s, m12 = dm2 * m;
                get_scale_min_k4(is + 1, sc2, s, m); float d22 = da2 * s, m22 = dm2 * m;
                get_scale_min_k4(is + 0, sc3, s, m); float d13 = da3 * s, m13 = dm3 * m;
                get_scale_min_k4(is + 1, sc3, s, m); float d23 = da3 * s, m23 = dm3 * m;
                for (uint l = 0; l < 32; l += 4) {
                    float4 xl = float4(x[ybase+j+l],x[ybase+j+l+1],x[ybase+j+l+2],x[ybase+j+l+3]);
                    float4 xh = float4(x[ybase+j+32+l],x[ybase+j+32+l+1],x[ybase+j+32+l+2],x[ybase+j+32+l+3]);
                    float4 wl0 = float4((float)(qs0[qoff+l]&0xF),(float)(qs0[qoff+l+1]&0xF),(float)(qs0[qoff+l+2]&0xF),(float)(qs0[qoff+l+3]&0xF));
                    float4 wh0 = float4((float)(qs0[qoff+l]>>4),(float)(qs0[qoff+l+1]>>4),(float)(qs0[qoff+l+2]>>4),(float)(qs0[qoff+l+3]>>4));
                    float4 wl1 = float4((float)(qs1[qoff+l]&0xF),(float)(qs1[qoff+l+1]&0xF),(float)(qs1[qoff+l+2]&0xF),(float)(qs1[qoff+l+3]&0xF));
                    float4 wh1 = float4((float)(qs1[qoff+l]>>4),(float)(qs1[qoff+l+1]>>4),(float)(qs1[qoff+l+2]>>4),(float)(qs1[qoff+l+3]>>4));
                    float4 wl2 = float4((float)(qs2[qoff+l]&0xF),(float)(qs2[qoff+l+1]&0xF),(float)(qs2[qoff+l+2]&0xF),(float)(qs2[qoff+l+3]&0xF));
                    float4 wh2 = float4((float)(qs2[qoff+l]>>4),(float)(qs2[qoff+l+1]>>4),(float)(qs2[qoff+l+2]>>4),(float)(qs2[qoff+l+3]>>4));
                    float4 wl3 = float4((float)(qs3[qoff+l]&0xF),(float)(qs3[qoff+l+1]&0xF),(float)(qs3[qoff+l+2]&0xF),(float)(qs3[qoff+l+3]&0xF));
                    float4 wh3 = float4((float)(qs3[qoff+l]>>4),(float)(qs3[qoff+l+1]>>4),(float)(qs3[qoff+l+2]>>4),(float)(qs3[qoff+l+3]>>4));
                    slo0 += dot(wl0 * d10 - m10, xl);
                    shi0 += dot(wh0 * d20 - m20, xh);
                    slo1 += dot(wl1 * d11 - m11, xl);
                    shi1 += dot(wh1 * d21 - m21, xh);
                    slo2 += dot(wl2 * d12 - m12, xl);
                    shi2 += dot(wh2 * d22 - m22, xh);
                    slo3 += dot(wl3 * d13 - m13, xl);
                    shi3 += dot(wh3 * d23 - m23, xh);
                }
                qoff += 32; is += 2; }
        }
    }
    float s0 = slo0 + shi0, s1 = slo1 + shi1, s2 = slo2 + shi2, s3 = slo3 + shi3;
    s0 += simd_shuffle_xor(s0, 4); s0 += simd_shuffle_xor(s0, 2); s0 += simd_shuffle_xor(s0, 1);
    s1 += simd_shuffle_xor(s1, 4); s1 += simd_shuffle_xor(s1, 2); s1 += simd_shuffle_xor(s1, 1);
    s2 += simd_shuffle_xor(s2, 4); s2 += simd_shuffle_xor(s2, 2); s2 += simd_shuffle_xor(s2, 1);
    s3 += simd_shuffle_xor(s3, 4); s3 += simd_shuffle_xor(s3, 2); s3 += simd_shuffle_xor(s3, 1);
    if (lane == 0 && rowb + 3u < d) { out[rowb+0u] = s0; out[rowb+1u] = s1; out[rowb+2u] = s2; out[rowb+3u] = s3; }
}

// ---- Q6_K: two rows in every lane, superblock strided across all 32 lanes, 16 rows / TG ----
kernel void q6k_ks32(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant uint &nsb_in [[buffer(3)]], constant uint &din [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    uint nsb = nsb_in; uint d = din;
    uint row0 = (tg * 8u + sgitg) * 2u;
    uint row1 = row0 + 1u;
    float s0 = 0.0, s1 = 0.0;
    if (row1 < d) {
        device const uchar *wr0 = wb + (ulong)row0 * nsb * 210;
        device const uchar *wr1 = wb + (ulong)row1 * nsb * 210;
        for (uint sb = tiisg; sb < nsb; sb += 32) {
            device const uchar *b0 = wr0 + sb * 210; device const uchar *b1 = wr1 + sb * 210;
            float da0 = (float)(*(device const half*)(b0 + 208)); float da1 = (float)(*(device const half*)(b1 + 208));
            device const uchar *ql0 = b0; device const uchar *qh0 = b0 + 128; device const char *k0 = (device const char*)(b0 + 192);
            device const uchar *ql1 = b1; device const uchar *qh1 = b1 + 128; device const char *k1 = (device const char*)(b1 + 192);
            for (int hf = 0; hf < 2; ++hf) { uint qlo = hf * 64, qho = hf * 32, sco = hf * 8, yo = sb * 256 + hf * 128;
                float a0 = da0 * (float)k0[sco+0], a1 = da0 * (float)k0[sco+1], a2 = da0 * (float)k0[sco+2], a3 = da0 * (float)k0[sco+3];
                float a4 = da0 * (float)k0[sco+4], a5 = da0 * (float)k0[sco+5], a6 = da0 * (float)k0[sco+6], a7 = da0 * (float)k0[sco+7];
                float b0s = da1 * (float)k1[sco+0], b1s = da1 * (float)k1[sco+1], b2s = da1 * (float)k1[sco+2], b3s = da1 * (float)k1[sco+3];
                float b4s = da1 * (float)k1[sco+4], b5s = da1 * (float)k1[sco+5], b6s = da1 * (float)k1[sco+6], b7s = da1 * (float)k1[sco+7];
                for (int l = 0; l < 32; l += 4) {
                    float4 xv1=float4(x[yo+l],x[yo+l+1],x[yo+l+2],x[yo+l+3]); float4 xv2=float4(x[yo+l+32],x[yo+l+33],x[yo+l+34],x[yo+l+35]);
                    float4 xv3=float4(x[yo+l+64],x[yo+l+65],x[yo+l+66],x[yo+l+67]); float4 xv4=float4(x[yo+l+96],x[yo+l+97],x[yo+l+98],x[yo+l+99]);
                    float ds0 = (l < 16) ? a0 : a1; float ds1 = (l < 16) ? a2 : a3;
                    float ds2 = (l < 16) ? a4 : a5; float ds3 = (l < 16) ? a6 : a7;
                    float es0 = (l < 16) ? b0s : b1s; float es1 = (l < 16) ? b2s : b3s;
                    float es2 = (l < 16) ? b4s : b5s; float es3 = (l < 16) ? b6s : b7s;
                    float4 w1=float4((float)((int)((ql0[qlo+l+0]&0xF)|(((qh0[qho+l+0]>>0)&3)<<4))-32),(float)((int)((ql0[qlo+l+1]&0xF)|(((qh0[qho+l+1]>>0)&3)<<4))-32),(float)((int)((ql0[qlo+l+2]&0xF)|(((qh0[qho+l+2]>>0)&3)<<4))-32),(float)((int)((ql0[qlo+l+3]&0xF)|(((qh0[qho+l+3]>>0)&3)<<4))-32));
                    float4 w2=float4((float)((int)((ql0[qlo+l+32]&0xF)|(((qh0[qho+l+0]>>2)&3)<<4))-32),(float)((int)((ql0[qlo+l+33]&0xF)|(((qh0[qho+l+1]>>2)&3)<<4))-32),(float)((int)((ql0[qlo+l+34]&0xF)|(((qh0[qho+l+2]>>2)&3)<<4))-32),(float)((int)((ql0[qlo+l+35]&0xF)|(((qh0[qho+l+3]>>2)&3)<<4))-32));
                    float4 w3=float4((float)((int)((ql0[qlo+l+0]>>4)|(((qh0[qho+l+0]>>4)&3)<<4))-32),(float)((int)((ql0[qlo+l+1]>>4)|(((qh0[qho+l+1]>>4)&3)<<4))-32),(float)((int)((ql0[qlo+l+2]>>4)|(((qh0[qho+l+2]>>4)&3)<<4))-32),(float)((int)((ql0[qlo+l+3]>>4)|(((qh0[qho+l+3]>>4)&3)<<4))-32));
                    float4 w4=float4((float)((int)((ql0[qlo+l+32]>>4)|(((qh0[qho+l+0]>>6)&3)<<4))-32),(float)((int)((ql0[qlo+l+33]>>4)|(((qh0[qho+l+1]>>6)&3)<<4))-32),(float)((int)((ql0[qlo+l+34]>>4)|(((qh0[qho+l+2]>>6)&3)<<4))-32),(float)((int)((ql0[qlo+l+35]>>4)|(((qh0[qho+l+3]>>6)&3)<<4))-32));
                    float4 v1=float4((float)((int)((ql1[qlo+l+0]&0xF)|(((qh1[qho+l+0]>>0)&3)<<4))-32),(float)((int)((ql1[qlo+l+1]&0xF)|(((qh1[qho+l+1]>>0)&3)<<4))-32),(float)((int)((ql1[qlo+l+2]&0xF)|(((qh1[qho+l+2]>>0)&3)<<4))-32),(float)((int)((ql1[qlo+l+3]&0xF)|(((qh1[qho+l+3]>>0)&3)<<4))-32));
                    float4 v2=float4((float)((int)((ql1[qlo+l+32]&0xF)|(((qh1[qho+l+0]>>2)&3)<<4))-32),(float)((int)((ql1[qlo+l+33]&0xF)|(((qh1[qho+l+1]>>2)&3)<<4))-32),(float)((int)((ql1[qlo+l+34]&0xF)|(((qh1[qho+l+2]>>2)&3)<<4))-32),(float)((int)((ql1[qlo+l+35]&0xF)|(((qh1[qho+l+3]>>2)&3)<<4))-32));
                    float4 v3=float4((float)((int)((ql1[qlo+l+0]>>4)|(((qh1[qho+l+0]>>4)&3)<<4))-32),(float)((int)((ql1[qlo+l+1]>>4)|(((qh1[qho+l+1]>>4)&3)<<4))-32),(float)((int)((ql1[qlo+l+2]>>4)|(((qh1[qho+l+2]>>4)&3)<<4))-32),(float)((int)((ql1[qlo+l+3]>>4)|(((qh1[qho+l+3]>>4)&3)<<4))-32));
                    float4 v4=float4((float)((int)((ql1[qlo+l+32]>>4)|(((qh1[qho+l+0]>>6)&3)<<4))-32),(float)((int)((ql1[qlo+l+33]>>4)|(((qh1[qho+l+1]>>6)&3)<<4))-32),(float)((int)((ql1[qlo+l+34]>>4)|(((qh1[qho+l+2]>>6)&3)<<4))-32),(float)((int)((ql1[qlo+l+35]>>4)|(((qh1[qho+l+3]>>6)&3)<<4))-32));
                    s0 += ds0*dot(w1,xv1)+ds1*dot(w2,xv2)+ds2*dot(w3,xv3)+ds3*dot(w4,xv4);
                    s1 += es0*dot(v1,xv1)+es1*dot(v2,xv2)+es2*dot(v3,xv3)+es3*dot(v4,xv4);
                }
            }
        }
    }
    s0 = simd_sum(s0); s1 = simd_sum(s1);
    if (tiisg == 0 && row1 < d) { out[row0] = s0; out[row1] = s1; }
}

// ---- Q6_K: two rows share an 8-lane group, x float4s loaded once, 64 rows / TG ----
kernel void q6k_ks8r2(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant uint &nsb_in [[buffer(3)]], constant uint &din [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    uint nsb = nsb_in; uint d = din;
    uint lane = tiisg & 7u;
    uint row0 = (((tg * 8u + sgitg) * 4u) + (tiisg >> 3)) * 2u;
    uint row1 = row0 + 1u;
    float s0 = 0.0, s1 = 0.0;
    if (row1 < d) {
        device const uchar *wr0 = wb + (ulong)row0 * nsb * 210;
        device const uchar *wr1 = wb + (ulong)row1 * nsb * 210;
        for (uint sb = lane; sb < nsb; sb += 8) {
            device const uchar *b0 = wr0 + sb * 210; device const uchar *b1 = wr1 + sb * 210;
            float da0 = (float)(*(device const half*)(b0 + 208)); float da1 = (float)(*(device const half*)(b1 + 208));
            device const uchar *ql0 = b0; device const uchar *qh0 = b0 + 128; device const char *k0 = (device const char*)(b0 + 192);
            device const uchar *ql1 = b1; device const uchar *qh1 = b1 + 128; device const char *k1 = (device const char*)(b1 + 192);
            for (int hf = 0; hf < 2; ++hf) { uint qlo = hf * 64, qho = hf * 32, sco = hf * 8, yo = sb * 256 + hf * 128;
                float a0 = da0 * (float)k0[sco+0], a1 = da0 * (float)k0[sco+1], a2 = da0 * (float)k0[sco+2], a3 = da0 * (float)k0[sco+3];
                float a4 = da0 * (float)k0[sco+4], a5 = da0 * (float)k0[sco+5], a6 = da0 * (float)k0[sco+6], a7 = da0 * (float)k0[sco+7];
                float b0s = da1 * (float)k1[sco+0], b1s = da1 * (float)k1[sco+1], b2s = da1 * (float)k1[sco+2], b3s = da1 * (float)k1[sco+3];
                float b4s = da1 * (float)k1[sco+4], b5s = da1 * (float)k1[sco+5], b6s = da1 * (float)k1[sco+6], b7s = da1 * (float)k1[sco+7];
                for (int l = 0; l < 32; l += 4) {
                    float4 xv1=float4(x[yo+l],x[yo+l+1],x[yo+l+2],x[yo+l+3]); float4 xv2=float4(x[yo+l+32],x[yo+l+33],x[yo+l+34],x[yo+l+35]);
                    float4 xv3=float4(x[yo+l+64],x[yo+l+65],x[yo+l+66],x[yo+l+67]); float4 xv4=float4(x[yo+l+96],x[yo+l+97],x[yo+l+98],x[yo+l+99]);
                    float ds0 = (l < 16) ? a0 : a1; float ds1 = (l < 16) ? a2 : a3;
                    float ds2 = (l < 16) ? a4 : a5; float ds3 = (l < 16) ? a6 : a7;
                    float es0 = (l < 16) ? b0s : b1s; float es1 = (l < 16) ? b2s : b3s;
                    float es2 = (l < 16) ? b4s : b5s; float es3 = (l < 16) ? b6s : b7s;
                    float4 w1=float4((float)((int)((ql0[qlo+l+0]&0xF)|(((qh0[qho+l+0]>>0)&3)<<4))-32),(float)((int)((ql0[qlo+l+1]&0xF)|(((qh0[qho+l+1]>>0)&3)<<4))-32),(float)((int)((ql0[qlo+l+2]&0xF)|(((qh0[qho+l+2]>>0)&3)<<4))-32),(float)((int)((ql0[qlo+l+3]&0xF)|(((qh0[qho+l+3]>>0)&3)<<4))-32));
                    float4 w2=float4((float)((int)((ql0[qlo+l+32]&0xF)|(((qh0[qho+l+0]>>2)&3)<<4))-32),(float)((int)((ql0[qlo+l+33]&0xF)|(((qh0[qho+l+1]>>2)&3)<<4))-32),(float)((int)((ql0[qlo+l+34]&0xF)|(((qh0[qho+l+2]>>2)&3)<<4))-32),(float)((int)((ql0[qlo+l+35]&0xF)|(((qh0[qho+l+3]>>2)&3)<<4))-32));
                    float4 w3=float4((float)((int)((ql0[qlo+l+0]>>4)|(((qh0[qho+l+0]>>4)&3)<<4))-32),(float)((int)((ql0[qlo+l+1]>>4)|(((qh0[qho+l+1]>>4)&3)<<4))-32),(float)((int)((ql0[qlo+l+2]>>4)|(((qh0[qho+l+2]>>4)&3)<<4))-32),(float)((int)((ql0[qlo+l+3]>>4)|(((qh0[qho+l+3]>>4)&3)<<4))-32));
                    float4 w4=float4((float)((int)((ql0[qlo+l+32]>>4)|(((qh0[qho+l+0]>>6)&3)<<4))-32),(float)((int)((ql0[qlo+l+33]>>4)|(((qh0[qho+l+1]>>6)&3)<<4))-32),(float)((int)((ql0[qlo+l+34]>>4)|(((qh0[qho+l+2]>>6)&3)<<4))-32),(float)((int)((ql0[qlo+l+35]>>4)|(((qh0[qho+l+3]>>6)&3)<<4))-32));
                    float4 v1=float4((float)((int)((ql1[qlo+l+0]&0xF)|(((qh1[qho+l+0]>>0)&3)<<4))-32),(float)((int)((ql1[qlo+l+1]&0xF)|(((qh1[qho+l+1]>>0)&3)<<4))-32),(float)((int)((ql1[qlo+l+2]&0xF)|(((qh1[qho+l+2]>>0)&3)<<4))-32),(float)((int)((ql1[qlo+l+3]&0xF)|(((qh1[qho+l+3]>>0)&3)<<4))-32));
                    float4 v2=float4((float)((int)((ql1[qlo+l+32]&0xF)|(((qh1[qho+l+0]>>2)&3)<<4))-32),(float)((int)((ql1[qlo+l+33]&0xF)|(((qh1[qho+l+1]>>2)&3)<<4))-32),(float)((int)((ql1[qlo+l+34]&0xF)|(((qh1[qho+l+2]>>2)&3)<<4))-32),(float)((int)((ql1[qlo+l+35]&0xF)|(((qh1[qho+l+3]>>2)&3)<<4))-32));
                    float4 v3=float4((float)((int)((ql1[qlo+l+0]>>4)|(((qh1[qho+l+0]>>4)&3)<<4))-32),(float)((int)((ql1[qlo+l+1]>>4)|(((qh1[qho+l+1]>>4)&3)<<4))-32),(float)((int)((ql1[qlo+l+2]>>4)|(((qh1[qho+l+2]>>4)&3)<<4))-32),(float)((int)((ql1[qlo+l+3]>>4)|(((qh1[qho+l+3]>>4)&3)<<4))-32));
                    float4 v4=float4((float)((int)((ql1[qlo+l+32]>>4)|(((qh1[qho+l+0]>>6)&3)<<4))-32),(float)((int)((ql1[qlo+l+33]>>4)|(((qh1[qho+l+1]>>6)&3)<<4))-32),(float)((int)((ql1[qlo+l+34]>>4)|(((qh1[qho+l+2]>>6)&3)<<4))-32),(float)((int)((ql1[qlo+l+35]>>4)|(((qh1[qho+l+3]>>6)&3)<<4))-32));
                    s0 += ds0*dot(w1,xv1)+ds1*dot(w2,xv2)+ds2*dot(w3,xv3)+ds3*dot(w4,xv4);
                    s1 += es0*dot(v1,xv1)+es1*dot(v2,xv2)+es2*dot(v3,xv3)+es3*dot(v4,xv4);
                }
            }
        }
    }
    s0 += simd_shuffle_xor(s0, 4); s0 += simd_shuffle_xor(s0, 2); s0 += simd_shuffle_xor(s0, 1);
    s1 += simd_shuffle_xor(s1, 4); s1 += simd_shuffle_xor(s1, 2); s1 += simd_shuffle_xor(s1, 1);
    if (lane == 0 && row1 < d) { out[row0] = s0; out[row1] = s1; }
}

// ---- Q6_K: 16 lanes per row, whole superblock per lane (nsb <= 16), 16 rows / TG ----
kernel void q6k_ks16(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant uint &nsb_in [[buffer(3)]], constant uint &din [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    uint nsb = nsb_in; uint d = din;
    uint row = ((tg * 8u + sgitg) * 2u) + (tiisg >> 4);
    uint lane = tiisg & 15u;
    float sum = 0.0;
    if (row < d && lane < nsb) {
        device const uchar *blk = wb + (ulong)row * nsb * 210 + (ulong)lane * 210;
        float2 acc = dot_q6_superblock2(blk, x, lane * 256);
        sum = acc.x + acc.y;
    }
    sum += simd_shuffle_xor(sum, 8);
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if ((tiisg & 15u) == 0 && row < d) out[row] = sum;
}

// ---- Q4_K v4: spec structure, uchar4 loads + vector nibble math. Values
// (and float math order) identical to the spec kernel -> bit-identical. ----
kernel void q4k_v4(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant uint &nsb_in [[buffer(3)]], constant uint &din [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    uint nsb = nsb_in; uint d = din;
    uint row = (tg * 8u + sgitg) * 4u + (tiisg >> 3);
    uint lane = tiisg & 7u;
    float sum = 0.0;
    if (row < d) {
        device const uchar *wr = wb + (ulong)row * nsb * 144;
        float slo = 0.0, shi = 0.0;
        for (uint sb = lane; sb < nsb; sb += 8) {
            device const uchar *blk = wr + sb * 144;
            float dall = (float)(*(device const half*)blk); float dmin = (float)(*(device const half*)(blk + 2));
            device const uchar *scales = blk + 4; device const uchar *qs = blk + 16; uint ybase = sb * 256; int is = 0; uint qoff = 0;
            for (uint j = 0; j < 256; j += 64) { uchar sc, m; get_scale_min_k4(is + 0, scales, sc, m); float d1 = dall * sc, m1 = dmin * m;
                get_scale_min_k4(is + 1, scales, sc, m); float d2 = dall * sc, m2 = dmin * m;
                for (uint li = 0; li < 8; li += 1) {
                    uint4 v = uint4(*(device const uchar4*)(qs + qoff + 4 * li));
                    float4 xl = float4(x[ybase+j+4*li+0],x[ybase+j+4*li+1],x[ybase+j+4*li+2],x[ybase+j+4*li+3]);
                    float4 xh = float4(x[ybase+j+32+4*li+0],x[ybase+j+32+4*li+1],x[ybase+j+32+4*li+2],x[ybase+j+32+4*li+3]);
                    slo += dot(float4(v & 15u) * d1 - m1, xl);
                    shi += dot(float4((v >> 4u) & 15u) * d2 - m2, xh);
                }
                qoff += 32; is += 2; }
        }
        sum = slo + shi;
    }
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane == 0 && row < d) out[row] = sum;
}

// ---- Q6_K v4: spec structure, uchar4 loads + vector dequant. Bit-identical
// values; float math order unchanged. ----
kernel void q6k_v4(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant uint &nsb_in [[buffer(3)]], constant uint &din [[buffer(4)]],
    uint tg [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    uint nsb = nsb_in; uint d = din;
    uint row = (tg * 8u + sgitg) * 4u + (tiisg >> 3);
    uint lane = tiisg & 7u;
    float sum = 0.0;
    if (row < d) {
        device const uchar *wr = wb + (ulong)row * nsb * 210;
        float s0 = 0.0, s1 = 0.0;
        for (uint sb = lane; sb < nsb; sb += 8) {
            device const uchar *blk = wr + sb * 210;
            float dall = (float)(*(device const half*)(blk + 208));
            device const uchar *ql = blk; device const uchar *qh = blk + 128; device const char *sc = (device const char*)(blk + 192);
            for (int hf = 0; hf < 2; ++hf) { uint qlo = hf * 64, qho = hf * 32, sco = hf * 8, yo = sb * 256 + hf * 128;
                float ds0=dall*(float)sc[sco+0],ds1=dall*(float)sc[sco+1],ds2=dall*(float)sc[sco+2],ds3=dall*(float)sc[sco+3];
                float ds4=dall*(float)sc[sco+4],ds5=dall*(float)sc[sco+5],ds6=dall*(float)sc[sco+6],ds7=dall*(float)sc[sco+7];
                for (int l = 0; l < 32; l += 4) {
                    uint4 qa = uint4(*(device const uchar4*)(ql + qlo + l));
                    uint4 qb = uint4(*(device const uchar4*)(ql + qlo + l + 32));
                    uint4 qh4 = uint4(*(device const uchar4*)(qh + qho + l));
                    float4 xv1=float4(x[yo+l],x[yo+l+1],x[yo+l+2],x[yo+l+3]); float4 xv2=float4(x[yo+l+32],x[yo+l+33],x[yo+l+34],x[yo+l+35]);
                    float4 xv3=float4(x[yo+l+64],x[yo+l+65],x[yo+l+66],x[yo+l+67]); float4 xv4=float4(x[yo+l+96],x[yo+l+97],x[yo+l+98],x[yo+l+99]);
                    float4 w1=float4(int4((qa & 15u) | ((qh4 & 3u) << 4)) - 32);
                    float4 w2=float4(int4((qb & 15u) | (((qh4 >> 2u) & 3u) << 4)) - 32);
                    float4 w3=float4(int4(((qa >> 4u) & 15u) | (((qh4 >> 4u) & 3u) << 4)) - 32);
                    float4 w4=float4(int4(((qb >> 4u) & 15u) | (((qh4 >> 6u) & 3u) << 4)) - 32);
                    if (l < 16) {
                        s0 += ds0*dot(w1,xv1)+ds2*dot(w2,xv2)+ds4*dot(w3,xv3)+ds6*dot(w4,xv4);
                    } else {
                        s1 += ds1*dot(w1,xv1)+ds3*dot(w2,xv2)+ds5*dot(w3,xv3)+ds7*dot(w4,xv4);
                    }
                }
            }
            sum += s0 + s1; s0 = 0.0; s1 = 0.0;
        }
    }
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane == 0 && row < d) out[row] = sum;
}

`

// Tier 3.5 spec twins: the generated constexpr kernels the engine actually
// dispatches for these shapes (production body verbatim, n/d baked).
// Built with plain concatenation: the bodies contain '%' and '{' that
// fmt.tprintf would treat as verbs.
spec_head :: proc(name: string, n, d: int) -> string {
	// name + signature + baked n/d lines, up to the row/lane line
	return strings.concatenate({
		"\nkernel void ", name, "(device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],\n",
		"    device float *out [[buffer(2)]],\n",
		"    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {\n",
		"    constexpr uint n = ", fmt.tprintf("{}", n), "; constexpr uint d = ", fmt.tprintf("{}", d), ";\n",
	})
}

SPEC_ROW_LINE :: "    uint row = tg * GEMV_ROWS + tid / GEMV_SG; uint lane = tid % GEMV_SG; float sum = 0.0;\n"
SPEC_TAIL :: `    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane % 8 == 0) out[row] = sum;
}
`

spec_q4k :: proc(name: string, n, d: int) -> string {
	return strings.concatenate({
		spec_head(name, n, d),
		SPEC_ROW_LINE,
		"    constexpr uint nsb = ", fmt.tprintf("{}", n / 256),
		"; device const uchar *wr = wb + (ulong)row * nsb * 144;\n",
		`    float slo = 0.0, shi = 0.0;
    for (uint sb = lane; sb < nsb; sb += GEMV_SG) { device const uchar *blk = wr + sb * 144;
        float dall = (float)(*(device const half*)blk); float dmin = (float)(*(device const half*)(blk + 2));
        device const uchar *scales = blk + 4; device const uchar *qs = blk + 16; uint ybase = sb * 256; int is = 0; uint qoff = 0;
        for (uint j = 0; j < 256; j += 64) { uchar sc, m; get_scale_min_k4(is + 0, scales, sc, m); float d1 = dall * sc, m1 = dmin * m;
            get_scale_min_k4(is + 1, scales, sc, m); float d2 = dall * sc, m2 = dmin * m;
            for (uint l = 0; l < 32; l += 4) {
                float4 wl = float4((float)(qs[qoff+l]&0xF),(float)(qs[qoff+l+1]&0xF),(float)(qs[qoff+l+2]&0xF),(float)(qs[qoff+l+3]&0xF));
                float4 wh = float4((float)(qs[qoff+l]>>4),(float)(qs[qoff+l+1]>>4),(float)(qs[qoff+l+2]>>4),(float)(qs[qoff+l+3]>>4));
                float4 xl = float4(x[ybase+j+l],x[ybase+j+l+1],x[ybase+j+l+2],x[ybase+j+l+3]);
                float4 xh = float4(x[ybase+j+32+l],x[ybase+j+32+l+1],x[ybase+j+32+l+2],x[ybase+j+32+l+3]);
                slo += dot(wl * d1 - m1, xl);
                shi += dot(wh * d2 - m2, xh);
            }
            qoff += 32; is += 2; } }
    sum = slo + shi;
`,
		SPEC_TAIL,
	})
}

spec_q6k :: proc(name: string, n, d: int) -> string {
	return strings.concatenate({
		spec_head(name, n, d),
		SPEC_ROW_LINE,
		"    constexpr uint nsb = ", fmt.tprintf("{}", n / 256),
		"; device const uchar *wr = wb + (ulong)row * nsb * 210;\n",
		`    for (uint sb = lane; sb < nsb; sb += GEMV_SG) {
        float2 acc = dot_q6_superblock2(wr + sb * 210, x, sb * 256);
        sum += acc.x + acc.y; }
`,
		SPEC_TAIL,
	})
}

Job :: struct {
	kernel: string,
	kind:   string, // "q4k" | "q6k" | "q8_0"
	n, d:   int,
	count:  int,
	tg_div: int, // rows per threadgroup
	dims:   bool, // kernel takes Dims at buffer 3 (generic) instead of nsb/d uniforms
	label:  string,
}

JOBS := []Job{
	{"q4k_spec12288",  "q4k", 4096, 12288, 66, 32, false, "up/gate 12288x4096 (x66)"},
	{"gemv_q4_k",      "q4k", 4096, 12288, 66, 32, true, "up/gate 12288x4096 (x66)"},
	{"q6k_spec12288",  "q6k", 12288, 4096, 33, 32, false, "down 4096x12288 (x33)"},
	{"gemv_q6_k",      "q6k", 12288, 4096, 33, 32, true, "down 4096x12288 (x33)"},
	{"q4k_v4",         "q4k", 4096, 12288, 66, 32, false, "up/gate 12288x4096 (x66)"},
	{"q6k_v4",         "q6k", 12288, 4096, 33, 32, false, "down 4096x12288 (x33)"},
	{"gemv_q8_0",      "q8_0", 4096, 12288, 1, 32, true, "CEILING q8_0 12288x4096"},
	// IQ ks re-verification (27B shapes): production generic vs the
	// uncommitted K-split kernels, interleaved.
	{"gemv_iq2_xxs",   "iq2xxs", 17408, 5120, 1, 32, true, "iq2xxs 5120x17408 generic"},
	{"gemv_iq2_xxs_ks_l", "iq2xxs", 17408, 5120, 1, 16, false, "iq2xxs 5120x17408 ks"},
	{"gemv_iq1_s",     "iq1s", 17408, 5120, 1, 32, true, "iq1s 5120x17408 generic"},
	{"gemv_iq1_s_ks_l", "iq1s", 17408, 5120, 1, 16, false, "iq1s 5120x17408 ks"},
	{"gemv_q8_0",      "q8_0", 17408, 5120, 1, 32, true, "CEILING q8_0 5120x17408"},
}

block_bytes :: proc(kind: string, n: int) -> int {
	switch kind {
	case "q4k": return (n / 256) * 144
	case "q6k": return (n / 256) * 210
	case "q8_0": return (n / 32) * 34
	case "iq2xxs": return (n / 256) * 66
	case "iq1s": return (n / 256) * 50
	}
	return 0
}

main :: proc() {
	dev := MTL.CreateSystemDefaultDevice()
	if dev == nil {
		fmt.eprintln("no metal device")
		os.exit(1)
	}
	defer dev->release()
	fmt.eprintfln("metal: {}", dev->name()->odinString())
	queue := dev->newCommandQueue()
	defer queue->release()

	specs := strings.concatenate({
		spec_q4k("q4k_spec12288", 4096, 12288),
		spec_q4k("q4k_spec4096", 4096, 4096),
		spec_q6k("q6k_spec12288", 12288, 4096),
		spec_q6k("q6k_spec8192", 4096, 8192),
		spec_q6k("q6k_spec248320", 4096, 248320),
	})
	src, _ := strings.concatenate({q35.MSL_SRC, EXTRA, specs})
	src_ns := NS.String.alloc()->initWithOdinString(src)
	defer src_ns->release()
	opts := MTL.CompileOptions.alloc()->init()
	defer opts->release()
	opts->setFastMathEnabled(true)
	fmt.eprintln("compiling...")
	lib, err := dev->newLibraryWithSource(src_ns, opts)
	if err != nil {
		fmt.eprintfln("compile: {}", err->localizedDescription()->odinString())
		os.exit(1)
	}
	defer lib->release()
	fmt.eprintln("compiled")

	if os.get_env("CHECK", context.temp_allocator) == "1" {
		check_pair(dev, queue, lib, "gemv_q4_k", "q4k_ks16", "q4k", 4096, 4096, 16)
		check_pair(dev, queue, lib, "gemv_q4_k", "q4k_ks8r2", "q4k", 4096, 4096, 64)
		check_pair(dev, queue, lib, "gemv_q4_k", "q4k_ks8r2", "q4k", 4096, 8192, 64)
		check_pair(dev, queue, lib, "gemv_q4_k", "q4k_ks8r4", "q4k", 4096, 4096, 128)
		check_pair(dev, queue, lib, "gemv_q6_k", "q6k_ks32", "q6k", 12288, 4096, 16)
		check_pair(dev, queue, lib, "gemv_q6_k", "q6k_ks8r2", "q6k", 12288, 4096, 64)
		check_pair(dev, queue, lib, "gemv_q6_k", "q6k_ks8r2", "q6k", 4096, 8192, 64)
		check_pair(dev, queue, lib, "gemv_q6_k", "q6k_ks16", "q6k", 4096, 8192, 16)
		check_pair(dev, queue, lib, "gemv_q4_k", "q4k_v4", "q4k", 4096, 4096, 32)
		check_pair(dev, queue, lib, "gemv_q6_k", "q6k_v4", "q6k", 12288, 4096, 32)
		check_pair(dev, queue, lib, "gemv_iq2_xxs", "gemv_iq2_xxs_ks_l", "iq2xxs", 17408, 5120, 16)
		check_pair(dev, queue, lib, "gemv_iq1_s", "gemv_iq1_s_ks_l", "iq1s", 17408, 5120, 16)
		return
	}

	// Interleaved timing: one command buffer per job per round, jobs rotated
	// so every variant samples the same thermal/clock window. Min and median
	// over rounds. (Sequential per-job sweeps on this passive-cooled M3 Air
	// differ run-to-run by 2x with GPU clock drift; interleaving removes the
	// bias.)
	nj := len(JOBS)
	psos := make([]^MTL.ComputePipelineState, nj)
	wbs := make([]^MTL.Buffer, nj)
	xbs := make([]^MTL.Buffer, nj)
	obs := make([]^MTL.Buffer, nj)
	tgs := make([]int, nj)
	for j, i in JOBS {
		psos[i] = make_pso(dev, lib, j.kernel)
		wbytes := j.d * block_bytes(j.kind, j.n)
		wbs[i] = dev->newBufferWithLength(NS.UInteger(wbytes), MTL.ResourceStorageModeShared)
		xbs[i] = dev->newBufferWithLength(NS.UInteger(j.n * 4), MTL.ResourceStorageModeShared)
		obs[i] = dev->newBufferWithLength(NS.UInteger(j.d * 4), MTL.ResourceStorageModeShared)
		ws := wbs[i]->contentsAsSlice([]u8)
		for k := 0; k < len(ws); k += 4096 {
			ws[k] = 1
		}
		xs := xbs[i]->contentsAsSlice([]f32)
		for k := 0; k < len(xs); k += 1 {
			xs[k] = 0.01
		}
	}
	defer for i := 0; i < nj; i += 1 {
		psos[i]->release()
		wbs[i]->release(); xbs[i]->release(); obs[i]->release()
	}
	ROUNDS := 12
	REPS := 5
	samples := make([][dynamic]f64, nj)
	for i := 0; i < nj; i += 1 {
		samples[i] = make([dynamic]f64, 0, ROUNDS)
	}
	// one warm round
	for j, i in JOBS {
		run_once(queue, psos[i], wbs[i], xbs[i], obs[i], j.kind, j.n, j.d, j.tg_div, 1, j.dims)
	}
	for r := 0; r < ROUNDS; r += 1 {
		for j, i in JOBS {
			t0 := time.tick_now()
			run_once(queue, psos[i], wbs[i], xbs[i], obs[i], j.kind, j.n, j.d, j.tg_div, REPS, j.dims)
			ms := f64(time.tick_since(t0)) / 1e6
			append(&samples[i], ms)
		}
	}
	for j, i in JOBS {
		best := math.INF_F64
		sum := 0.0
		for ms in samples[i] {
			if ms < best { best = ms }
			sum += ms
		}
		med := sum / f64(len(samples[i]))
		ms := best / f64(REPS)
		bytes := f64(j.d) * f64(block_bytes(j.kind, j.n))
		gbps := bytes / (ms * 1e-3) / 1e9
		med_gbps := bytes / (med / f64(REPS) * 1e-3) / 1e9
		fmt.eprintfln("{:7.3f} ms ({:5.1f} GB/s)  med {:6.1f} GB/s   {:16s} {}", ms, gbps, med_gbps, j.kernel, j.label)
	}
}

make_pso :: proc(dev: ^MTL.Device, lib: ^MTL.Library, name: string) -> ^MTL.ComputePipelineState {
	ns_name := NS.String.alloc()->initWithOdinString(name)
	defer ns_name->release()
	fn := lib->newFunctionWithName(ns_name)
	if fn == nil {
		fmt.eprintfln("missing {}", name)
		os.exit(1)
	}
	defer fn->release()
	pso, perr := dev->newComputePipelineStateWithFunction(fn)
	if perr != nil {
		fmt.eprintfln("pso {}: {}", name, perr->localizedDescription()->odinString())
		os.exit(1)
	}
	return pso
}

// Fix the per-superblock scale halves to bounded values so synthetic noise
// cannot produce inf/nan; payload bytes stay pseudo-random.
init_weights :: proc(ws: []u8, kind: string, n: int) {
	for i := 0; i < len(ws); i += 1 {
		ws[i] = u8((i * 17 + 3) & 0xff)
	}
	bb := block_bytes(kind, n)
	for off := 0; off + bb <= len(ws); off += bb {
		switch kind {
		case "q4k": // d = 1.0 (0x3C00), dmin = 0.25 (0x3400)
			ws[off] = 0x00; ws[off + 1] = 0x3C
			ws[off + 2] = 0x00; ws[off + 3] = 0x34
		case "q6k": // d = 1.0 at block offset 208
			ws[off + 208] = 0x00; ws[off + 209] = 0x3C
		case "q8_0": // d = 1.0
			ws[off] = 0x00; ws[off + 1] = 0x3C
		case "iq2xxs", "iq1s": // d = 1.0 (IQ blocks start with a half d)
			ws[off] = 0x00; ws[off + 1] = 0x3C
		}
	}
}

run_once :: proc(queue: ^MTL.CommandQueue, pso: ^MTL.ComputePipelineState, wb, xb, ob: ^MTL.Buffer,
	kind: string, n, d, tg_div: int, reps: int, use_dims: bool) {
	cmd := queue->commandBuffer()
	enc := cmd->computeCommandEncoder()
	for _ in 0 ..< reps {
		enc->setComputePipelineState(pso)
		enc->setBuffer(wb, 0, 0)
		enc->setBuffer(xb, 0, 1)
		enc->setBuffer(ob, 0, 2)
		if use_dims {
			dims := Dims{u32(n), u32(d)}
			enc->setBytes(([^]u8)(&dims)[:size_of(Dims)], 3)
		} else {
			nu := u32(n / 256)
			du := u32(d)
			enc->setBytes(([^]u8)(&nu)[:4], 3)
			enc->setBytes(([^]u8)(&du)[:4], 4)
		}
		enc->dispatchThreadgroups(MTL.Size{NS.Integer(d / tg_div), 1, 1}, MTL.Size{256, 1, 1})
	}
	enc->endEncoding()
	cmd->commit()
	cmd->waitUntilCompleted()
}

time_job :: proc(dev: ^MTL.Device, queue: ^MTL.CommandQueue, lib: ^MTL.Library, j: Job) -> f64 {
	pso := make_pso(dev, lib, j.kernel)
	defer pso->release()
	wbytes := j.d * block_bytes(j.kind, j.n)
	wb := dev->newBufferWithLength(NS.UInteger(wbytes), MTL.ResourceStorageModeShared)
	xb := dev->newBufferWithLength(NS.UInteger(j.n * 4), MTL.ResourceStorageModeShared)
	ob := dev->newBufferWithLength(NS.UInteger(j.d * 4), MTL.ResourceStorageModeShared)
	defer wb->release(); defer xb->release(); defer ob->release()
	ws := wb->contentsAsSlice([]u8)
	for i := 0; i < len(ws); i += 4096 {
		ws[i] = 1
	}
	xs := xb->contentsAsSlice([]f32)
	for i := 0; i < len(xs); i += 1 {
		xs[i] = 0.01
	}
	run_once(queue, pso, wb, xb, ob, j.kind, j.n, j.d, j.tg_div, 1, j.dims)
	run_once(queue, pso, wb, xb, ob, j.kind, j.n, j.d, j.tg_div, 1, j.dims)
	best: f64 = 1e18
	for _ in 0 ..< 4 {
		t0 := time.tick_now()
		run_once(queue, pso, wb, xb, ob, j.kind, j.n, j.d, j.tg_div, 10, j.dims)
		ms := f64(time.tick_since(t0)) / 1e6
		if ms < best { best = ms }
	}
	return best / 10.0
}

check_pair :: proc(dev: ^MTL.Device, queue: ^MTL.CommandQueue, lib: ^MTL.Library, a, b, kind: string, n, d, b_tg_div: int) {
	pa := make_pso(dev, lib, a)
	pb := make_pso(dev, lib, b)
	defer pa->release(); defer pb->release()
	wbytes := d * block_bytes(kind, n)
	wb := dev->newBufferWithLength(NS.UInteger(wbytes), MTL.ResourceStorageModeShared)
	xb := dev->newBufferWithLength(NS.UInteger(n * 4), MTL.ResourceStorageModeShared)
	oa := dev->newBufferWithLength(NS.UInteger(d * 4), MTL.ResourceStorageModeShared)
	ob := dev->newBufferWithLength(NS.UInteger(d * 4), MTL.ResourceStorageModeShared)
	defer wb->release(); defer xb->release(); defer oa->release(); defer ob->release()
	ws := wb->contentsAsSlice([]u8)
	init_weights(ws, kind, n)
	xs := xb->contentsAsSlice([]f32)
	for i := 0; i < len(xs); i += 1 {
		xs[i] = f32(i % 13) * 0.01 - 0.05
	}
	run_once(queue, pa, wb, xb, oa, kind, n, d, 32, 1, true)
	run_once(queue, pb, wb, xb, ob, kind, n, d, b_tg_div, 1, false)
	aa := oa->contentsAsSlice([]f32)[:d]
	bb := ob->contentsAsSlice([]f32)[:d]
	max_abs: f32 = 0
	max_rel: f32 = 0
	for i := 0; i < d; i += 1 {
		av := aa[i]; bv := bb[i]
		du := abs(av - bv)
		rel := du / max(abs(av), 1e-3)
		if du > max_abs { max_abs = du }
		if rel > max_rel { max_rel = rel }
	}
	fmt.eprintfln("check {:10s} vs {:10s} n={:5d} d={:5d}: max_abs {:.3e} max_rel {:.3e}", a, b, n, d, max_abs, max_rel)
	if max_rel > 1e-3 {
		fmt.eprintfln("  MISMATCH beyond 1e-3")
	}
}
