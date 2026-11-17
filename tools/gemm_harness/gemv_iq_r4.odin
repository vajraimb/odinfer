// Four-row vs two-row short-K GEMV (n=5120, d=17408). Ten dispatches per
// command buffer. iq2_r2 is the kernel shipping in enc_gemv. iq1_sc is the
// production spec scalar.
package gemv_iq_r4

import q35 "qwen3_5:qwen3_5"

import NS "core:sys/darwin/Foundation"
import "core:fmt"
import "core:os"
import "core:strings"
import "core:time"
import MTL "vendor:darwin/Metal"

N :: 5120
D :: 17408
REPS :: 10

Try :: struct {
	name:   string,
	tg_div: int,
	block:  int,
	group:  int,
}

TRIES := []Try{
	{"iq2_r2", 64, 66, 0},
	{"iq2_r4", 128, 66, 0},
	{"iq1_sc", 32, 50, 1},
	{"iq1_r2", 64, 50, 1},
	{"iq1_r4", 128, 50, 1},
}

main :: proc() {
	dev := MTL.CreateSystemDefaultDevice()
	if dev == nil { fmt.eprintln("no metal"); os.exit(1) }
	defer dev->release()
	queue := dev->newCommandQueue()
	defer queue->release()
	src, _ := strings.concatenate({
		"#include <metal_stdlib>\nusing namespace metal;\n",
		q35.MSL_TABLES,
		EXTRA,
	})
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

	wbytes := D * (N / 256) * 66
	wb := dev->newBufferWithLength(NS.UInteger(wbytes), MTL.ResourceStorageModeShared)
	xb := dev->newBufferWithLength(NS.UInteger(N * 4), MTL.ResourceStorageModeShared)
	ob := dev->newBufferWithLength(NS.UInteger(D * 4), MTL.ResourceStorageModeShared)
	defer wb->release(); defer xb->release(); defer ob->release()
	ws := wb->contentsAsSlice([]u8)
	for i in 0 ..< len(ws) { ws[i] = u8((i * 17 + 3) & 0xff) }
	xs := xb->contentsAsSlice([]f32)
	for i in 0 ..< len(xs) { xs[i] = f32(i % 13) * 0.01 - 0.05 }

	psos := make([]^MTL.ComputePipelineState, len(TRIES))
	for t, i in TRIES { psos[i] = make_pso(dev, lib, t.name) }

	for g in 0 ..< 2 {
		ri := 0
		for t, i in TRIES {
			if t.group == g { ri = i; break }
		}
		rt := TRIES[ri]
		dispatch(queue, psos[ri], wb, xb, ob, D / rt.tg_div)
		ref := make([]f32, D)
		copy(ref, ob->contentsAsSlice([]f32)[:D])
		for t, i in TRIES {
			if t.group != g || i == ri { continue }
			dispatch(queue, psos[i], wb, xb, ob, D / t.tg_div)
			got := ob->contentsAsSlice([]f32)[:D]
			max_rel: f32 = 0
			for k in 0 ..< D {
				df := got[k] - ref[k]
				if df < 0 { df = -df }
				den := ref[k]
				if den < 0 { den = -den }
				if den < 1e-6 { den = 1e-6 }
				rel := df / den
				if rel > max_rel { max_rel = rel }
			}
			fmt.eprintfln("check {} max_rel {}", t.name, max_rel)
		}
		delete(ref)
	}

	for t, i in TRIES {
		best: f64 = 1e18
		tg := D / t.tg_div
		for _ in 0 ..< 4 {
			t0 := time.tick_now()
			cmd := queue->commandBuffer()
			enc := cmd->computeCommandEncoder()
			enc->setComputePipelineState(psos[i])
			enc->setBuffer(wb, 0, 0)
			enc->setBuffer(xb, 0, 1)
			enc->setBuffer(ob, 0, 2)
			for _ in 0 ..< REPS {
				enc->dispatchThreadgroups(MTL.Size{NS.Integer(tg), 1, 1}, MTL.Size{256, 1, 1})
			}
			enc->endEncoding()
			cmd->commit()
			cmd->waitUntilCompleted()
			ms := f64(time.tick_since(t0)) / 1e6 / f64(REPS)
			if ms < best { best = ms }
		}
		bytes := f64(D) * f64(N / 256) * f64(t.block)
		fmt.eprintfln("{:6.3f} ms   {:5.1f} GB/s   {}", best, bytes / (best * 1e-3) / 1e9, t.name)
	}
}

dispatch :: proc(queue: ^MTL.CommandQueue, pso: ^MTL.ComputePipelineState, wb, xb, ob: ^MTL.Buffer, tg: int) {
	cmd := queue->commandBuffer()
	enc := cmd->computeCommandEncoder()
	enc->setComputePipelineState(pso)
	enc->setBuffer(wb, 0, 0)
	enc->setBuffer(xb, 0, 1)
	enc->setBuffer(ob, 0, 2)
	enc->dispatchThreadgroups(MTL.Size{NS.Integer(tg), 1, 1}, MTL.Size{256, 1, 1})
	enc->endEncoding()
	cmd->commit()
	cmd->waitUntilCompleted()
}

make_pso :: proc(dev: ^MTL.Device, lib: ^MTL.Library, name: string) -> ^MTL.ComputePipelineState {
	ns_name := NS.String.alloc()->initWithOdinString(name)
	defer ns_name->release()
	fn := lib->newFunctionWithName(ns_name)
	if fn == nil { fmt.eprintfln("missing {}", name); os.exit(1) }
	defer fn->release()
	pso, perr := dev->newComputePipelineStateWithFunction(fn)
	if perr != nil {
		fmt.eprintfln("pso {}: {}", name, perr->localizedDescription()->odinString())
		os.exit(1)
	}
	return pso
}

EXTRA :: `
inline void iq2_row(device const uchar *blk, device const float *x, thread float &s) {
    float d = (float)(*(device const half*)blk);
    uint ybase = 0;
    #pragma clang loop unroll(disable)
    for (uint ib32 = 0; ib32 < 8; ib32++) {
        uint qb = 2 + 8 * ib32;
        uint aux = (uint)blk[qb + 4] | ((uint)blk[qb + 5] << 8) | ((uint)blk[qb + 6] << 16) | ((uint)blk[qb + 7] << 24);
        float db = d * (0.5f + (float)(aux >> 28)) * 0.25f;
        for (uint l = 0; l < 4; l++) {
            ulong g = iq2xxs_grid[blk[qb + l]];
            uchar sg = ksigns_iq2xs[(aux >> (7 * l)) & 127];
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8; j++) {
                float v = (float)((g >> (8 * j)) & 0xff);
                float z = (sg & (1u << j)) == 0 ? 1.0f : -1.0f;
                s += db * v * z * x[ybase + j];
            }
            ybase += 8;
        }
    }
}

kernel void iq2_r2(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 20;
    uint lane = tid % 8; uint slot = tid / 8;
    uint row0 = tg * 64 + slot; uint row1 = row0 + 32;
    float s0 = 0.0f, s1 = 0.0f;
    device const uchar *wr0 = wb + (ulong)row0 * nsb * 66;
    device const uchar *wr1 = wb + (ulong)row1 * nsb * 66;
    #pragma clang loop unroll(disable)
    for (uint sb = lane; sb < nsb; sb += 8) {
        iq2_row(wr0 + sb * 66, x + sb * 256, s0);
        iq2_row(wr1 + sb * 66, x + sb * 256, s1);
    }
    s0 += simd_shuffle_xor(s0, 4); s0 += simd_shuffle_xor(s0, 2); s0 += simd_shuffle_xor(s0, 1);
    s1 += simd_shuffle_xor(s1, 4); s1 += simd_shuffle_xor(s1, 2); s1 += simd_shuffle_xor(s1, 1);
    if (lane == 0) { out[row0] = s0; out[row1] = s1; }
}

kernel void iq2_r4(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 20;
    uint lane = tid % 8; uint slot = tid / 8;
    uint row0 = tg * 128 + slot;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    device const uchar *a0 = wb + (ulong)(row0 + 0) * nsb * 66;
    device const uchar *a1 = wb + (ulong)(row0 + 32) * nsb * 66;
    device const uchar *a2 = wb + (ulong)(row0 + 64) * nsb * 66;
    device const uchar *a3 = wb + (ulong)(row0 + 96) * nsb * 66;
    #pragma clang loop unroll(disable)
    for (uint sb = lane; sb < nsb; sb += 8) {
        device const float *xv = x + sb * 256;
        iq2_row(a0 + sb * 66, xv, s0);
        iq2_row(a1 + sb * 66, xv, s1);
        iq2_row(a2 + sb * 66, xv, s2);
        iq2_row(a3 + sb * 66, xv, s3);
    }
    s0 += simd_shuffle_xor(s0, 4); s0 += simd_shuffle_xor(s0, 2); s0 += simd_shuffle_xor(s0, 1);
    s1 += simd_shuffle_xor(s1, 4); s1 += simd_shuffle_xor(s1, 2); s1 += simd_shuffle_xor(s1, 1);
    s2 += simd_shuffle_xor(s2, 4); s2 += simd_shuffle_xor(s2, 2); s2 += simd_shuffle_xor(s2, 1);
    s3 += simd_shuffle_xor(s3, 4); s3 += simd_shuffle_xor(s3, 2); s3 += simd_shuffle_xor(s3, 1);
    if (lane == 0) { out[row0] = s0; out[row0 + 32] = s1; out[row0 + 64] = s2; out[row0 + 96] = s3; }
}

inline void iq1_row(device const uchar *blk, device const float *x, thread float &s) {
    float d = (float)(*(device const half*)blk);
    device const uchar *qs = blk + 2; device const uchar *qh = blk + 34;
    uint ybase = 0; uint qoff = 0;
    #pragma clang loop unroll(disable)
    for (uint ib = 0; ib < 8; ib++) {
        uint qhv = (uint)qh[2 * ib] | ((uint)qh[2 * ib + 1] << 8);
        float dl = d * (float)(2 * (int)((qhv >> 12) & 7) + 1);
        float delta = (qhv & 0x8000) == 0 ? 0.125f : -0.125f;
        for (uint l = 0; l < 4; l++) {
            uint idx = (uint)qs[qoff + l] | (((qhv >> (3 * l)) & 7) << 8);
            ulong grid = iq1s_grid[idx];
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8; j++) {
                float gv = (float)((char)((grid >> (8 * j)) & 0xff));
                s += dl * (gv + delta) * x[ybase + j];
            }
            ybase += 8;
        }
        qoff += 4;
    }
}

kernel void iq1_sc(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 20;
    uint row = tg * 32 + tid / 8; uint lane = tid % 8; float sum = 0.0f;
    device const uchar *wr = wb + (ulong)row * nsb * 50;
    #pragma clang loop unroll(disable)
    for (uint sb = lane; sb < nsb; sb += 8) iq1_row(wr + sb * 50, x + sb * 256, sum);
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane == 0) out[row] = sum;
}

kernel void iq1_r2(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 20;
    uint lane = tid % 8; uint slot = tid / 8;
    uint row0 = tg * 64 + slot; uint row1 = row0 + 32;
    float s0 = 0.0f, s1 = 0.0f;
    device const uchar *wr0 = wb + (ulong)row0 * nsb * 50;
    device const uchar *wr1 = wb + (ulong)row1 * nsb * 50;
    #pragma clang loop unroll(disable)
    for (uint sb = lane; sb < nsb; sb += 8) {
        device const float *xv = x + sb * 256;
        iq1_row(wr0 + sb * 50, xv, s0);
        iq1_row(wr1 + sb * 50, xv, s1);
    }
    s0 += simd_shuffle_xor(s0, 4); s0 += simd_shuffle_xor(s0, 2); s0 += simd_shuffle_xor(s0, 1);
    s1 += simd_shuffle_xor(s1, 4); s1 += simd_shuffle_xor(s1, 2); s1 += simd_shuffle_xor(s1, 1);
    if (lane == 0) { out[row0] = s0; out[row1] = s1; }
}

kernel void iq1_r4(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 20;
    uint lane = tid % 8; uint slot = tid / 8;
    uint row0 = tg * 128 + slot;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    device const uchar *a0 = wb + (ulong)(row0 + 0) * nsb * 50;
    device const uchar *a1 = wb + (ulong)(row0 + 32) * nsb * 50;
    device const uchar *a2 = wb + (ulong)(row0 + 64) * nsb * 50;
    device const uchar *a3 = wb + (ulong)(row0 + 96) * nsb * 50;
    #pragma clang loop unroll(disable)
    for (uint sb = lane; sb < nsb; sb += 8) {
        device const float *xv = x + sb * 256;
        iq1_row(a0 + sb * 50, xv, s0);
        iq1_row(a1 + sb * 50, xv, s1);
        iq1_row(a2 + sb * 50, xv, s2);
        iq1_row(a3 + sb * 50, xv, s3);
    }
    s0 += simd_shuffle_xor(s0, 4); s0 += simd_shuffle_xor(s0, 2); s0 += simd_shuffle_xor(s0, 1);
    s1 += simd_shuffle_xor(s1, 4); s1 += simd_shuffle_xor(s1, 2); s1 += simd_shuffle_xor(s1, 1);
    s2 += simd_shuffle_xor(s2, 4); s2 += simd_shuffle_xor(s2, 2); s2 += simd_shuffle_xor(s2, 1);
    s3 += simd_shuffle_xor(s3, 4); s3 += simd_shuffle_xor(s3, 2); s3 += simd_shuffle_xor(s3, 1);
    if (lane == 0) { out[row0] = s0; out[row0 + 32] = s1; out[row0 + 64] = s2; out[row0 + 96] = s3; }
}
`
