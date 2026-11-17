// Inlined IQ1_S short-K GEMV (n=5120, d=17408). iq1_sc matches the
// production spec scalar. Ten dispatches per command buffer, min of 4.
package gemv_iq1_try

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
}

TRIES := []Try{
	{"iq1_sc", 32},
	{"iq1_bit", 32},
	{"iq1_split", 32},
	{"iq1_r2", 64},
	{"iq1_r2bit", 64},
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

	wbytes := D * (N / 256) * 50
	wb := dev->newBufferWithLength(NS.UInteger(wbytes), MTL.ResourceStorageModeShared)
	xb := dev->newBufferWithLength(NS.UInteger(N * 4), MTL.ResourceStorageModeShared)
	ob := dev->newBufferWithLength(NS.UInteger(D * 4), MTL.ResourceStorageModeShared)
	defer wb->release(); defer xb->release(); defer ob->release()
	ws := wb->contentsAsSlice([]u8)
	for i in 0 ..< len(ws) { ws[i] = u8((i * 17 + 3) & 0xff) }
	for b in 0 ..< D * (N / 256) {
		ws[b * 50 + 0] = 0x00
		ws[b * 50 + 1] = 0x3C
	}
	xs := xb->contentsAsSlice([]f32)
	for i in 0 ..< len(xs) { xs[i] = f32(i % 13) * 0.01 - 0.05 }

	psos := make([]^MTL.ComputePipelineState, len(TRIES))
	for t, i in TRIES { psos[i] = make_pso(dev, lib, t.name) }

	dispatch(queue, psos[0], wb, xb, ob, D / TRIES[0].tg_div)
	ref := make([]f32, D)
	copy(ref, ob->contentsAsSlice([]f32)[:D])
	for t, i in TRIES[1:] {
		dispatch(queue, psos[i + 1], wb, xb, ob, D / t.tg_div)
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

	bytes := f64(D) * f64(N / 256) * 50
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
		fmt.eprintfln("{:7.3f} ms   {:5.1f} GB/s   {}", best, bytes / (best * 1e-3) / 1e9, t.name)
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
#define IQ1_DOT(grid, yb, dl, delta, s) \
    do { \
        uchar4 _lo = as_type<uchar4>((uint)(grid)); \
        uchar4 _hi = as_type<uchar4>((uint)((grid) >> 32)); \
        float4 _g0 = float4((float)(char)_lo.x, (float)(char)_lo.y, (float)(char)_lo.z, (float)(char)_lo.w); \
        float4 _g1 = float4((float)(char)_hi.x, (float)(char)_hi.y, (float)(char)_hi.z, (float)(char)_hi.w); \
        float4 _x0 = *(device const float4*)(x + (yb)); \
        float4 _x1 = *(device const float4*)(x + (yb) + 4); \
        s += (dl) * (dot(_g0 + (delta), _x0) + dot(_g1 + (delta), _x1)); \
    } while (0)

kernel void iq1_sc(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 20;
    uint row = tg * 32 + tid / 8; uint lane = tid % 8; float sum = 0.0f;
    device const uchar *wr = wb + (ulong)row * nsb * 50;
    for (uint sb = lane; sb < nsb; sb += 8) {
        device const uchar *blk = wr + sb * 50;
        float d = (float)(*(device const half*)blk);
        device const uchar *qs = blk + 2; device const uchar *qh = blk + 34;
        uint ybase = sb * 256; uint qoff = 0;
        for (uint ib = 0; ib < 8; ib++) {
            uint qhv = (uint)qh[2 * ib] | ((uint)qh[2 * ib + 1] << 8);
            float dl = d * (float)(2 * (int)((qhv >> 12) & 7) + 1);
            float delta = (qhv & 0x8000) == 0 ? 0.125f : -0.125f;
            for (uint l = 0; l < 4; l++) {
                uint idx = (uint)qs[qoff + l] | (((qhv >> (3 * l)) & 7) << 8);
                ulong grid = iq1s_grid[idx];
                for (uint j = 0; j < 8; j++) {
                    float gv = (float)((char)((grid >> (8 * j)) & 0xff));
                    sum += dl * (gv + delta) * x[ybase + j];
                }
                ybase += 8;
            }
            qoff += 4;
        }
    }
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane == 0) out[row] = sum;
}

kernel void iq1_bit(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 20;
    uint row = tg * 32 + tid / 8; uint lane = tid % 8; float sum = 0.0f;
    device const uchar *wr = wb + (ulong)row * nsb * 50;
    #pragma clang loop unroll(disable)
    for (uint sb = lane; sb < nsb; sb += 8) {
        device const uchar *blk = wr + sb * 50;
        float d = (float)(*(device const half*)blk);
        device const uchar *qs = blk + 2; device const uchar *qh = blk + 34;
        uint ybase = sb * 256; uint qoff = 0;
        #pragma clang loop unroll(disable)
        for (uint ib = 0; ib < 8; ib++) {
            uint qhv = (uint)qh[2 * ib] | ((uint)qh[2 * ib + 1] << 8);
            float dl = d * (float)(2 * (int)((qhv >> 12) & 7) + 1);
            float delta = (qhv & 0x8000) == 0 ? 0.125f : -0.125f;
            for (uint l = 0; l < 4; l++) {
                uint idx = (uint)qs[qoff + l] | (((qhv >> (3 * l)) & 7) << 8);
                IQ1_DOT(iq1s_grid[idx], ybase, dl, delta, sum);
                ybase += 8;
            }
            qoff += 4;
        }
    }
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane == 0) out[row] = sum;
}

kernel void iq1_split(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 20;
    uint row = tg * 32 + tid / 8; uint lane = tid % 8; float sum = 0.0f;
    device const uchar *wr = wb + (ulong)row * nsb * 50;
    #pragma clang loop unroll(disable)
    for (uint sb = lane; sb < nsb; sb += 8) {
        device const uchar *blk = wr + sb * 50;
        float d = (float)(*(device const half*)blk);
        device const uchar *qs = blk + 2; device const uchar *qh = blk + 34;
        uint ybase = sb * 256; uint qoff = 0;
        #pragma clang loop unroll(disable)
        for (uint ib = 0; ib < 8; ib++) {
            uint qhv = (uint)qh[2 * ib] | ((uint)qh[2 * ib + 1] << 8);
            float dl = d * (float)(2 * (int)((qhv >> 12) & 7) + 1);
            float delta = (qhv & 0x8000) == 0 ? 0.125f : -0.125f;
            float ag = 0.0f, ax = 0.0f;
            for (uint l = 0; l < 4; l++) {
                uint idx = (uint)qs[qoff + l] | (((qhv >> (3 * l)) & 7) << 8);
                ulong grid = iq1s_grid[idx];
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 8; j++) {
                    float xj = x[ybase + j];
                    float gv = (float)((char)((grid >> (8 * j)) & 0xff));
                    ag += gv * xj;
                    ax += xj;
                }
                ybase += 8;
            }
            sum += dl * ag + (dl * delta) * ax;
            qoff += 4;
        }
    }
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
        device const uchar *b0 = wr0 + sb * 50;
        device const uchar *b1 = wr1 + sb * 50;
        float d0 = (float)(*(device const half*)b0);
        float d1 = (float)(*(device const half*)b1);
        device const uchar *qs0 = b0 + 2; device const uchar *qh0 = b0 + 34;
        device const uchar *qs1 = b1 + 2; device const uchar *qh1 = b1 + 34;
        uint ybase = sb * 256; uint qoff = 0;
        #pragma clang loop unroll(disable)
        for (uint ib = 0; ib < 8; ib++) {
            uint q0 = (uint)qh0[2 * ib] | ((uint)qh0[2 * ib + 1] << 8);
            uint q1 = (uint)qh1[2 * ib] | ((uint)qh1[2 * ib + 1] << 8);
            float dl0 = d0 * (float)(2 * (int)((q0 >> 12) & 7) + 1);
            float dl1 = d1 * (float)(2 * (int)((q1 >> 12) & 7) + 1);
            float de0 = (q0 & 0x8000) == 0 ? 0.125f : -0.125f;
            float de1 = (q1 & 0x8000) == 0 ? 0.125f : -0.125f;
            for (uint l = 0; l < 4; l++) {
                uint i0 = (uint)qs0[qoff + l] | (((q0 >> (3 * l)) & 7) << 8);
                uint i1 = (uint)qs1[qoff + l] | (((q1 >> (3 * l)) & 7) << 8);
                ulong g0 = iq1s_grid[i0];
                ulong g1 = iq1s_grid[i1];
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 8; j++) {
                    float xj = x[ybase + j];
                    float v0 = (float)((char)((g0 >> (8 * j)) & 0xff));
                    float v1 = (float)((char)((g1 >> (8 * j)) & 0xff));
                    s0 += dl0 * (v0 + de0) * xj;
                    s1 += dl1 * (v1 + de1) * xj;
                }
                ybase += 8;
            }
            qoff += 4;
        }
    }
    s0 += simd_shuffle_xor(s0, 4); s0 += simd_shuffle_xor(s0, 2); s0 += simd_shuffle_xor(s0, 1);
    s1 += simd_shuffle_xor(s1, 4); s1 += simd_shuffle_xor(s1, 2); s1 += simd_shuffle_xor(s1, 1);
    if (lane == 0) { out[row0] = s0; out[row1] = s1; }
}

kernel void iq1_r2bit(
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
        device const uchar *b0 = wr0 + sb * 50;
        device const uchar *b1 = wr1 + sb * 50;
        float d0 = (float)(*(device const half*)b0);
        float d1 = (float)(*(device const half*)b1);
        device const uchar *qs0 = b0 + 2; device const uchar *qh0 = b0 + 34;
        device const uchar *qs1 = b1 + 2; device const uchar *qh1 = b1 + 34;
        uint ybase = sb * 256; uint qoff = 0;
        #pragma clang loop unroll(disable)
        for (uint ib = 0; ib < 8; ib++) {
            uint q0 = (uint)qh0[2 * ib] | ((uint)qh0[2 * ib + 1] << 8);
            uint q1 = (uint)qh1[2 * ib] | ((uint)qh1[2 * ib + 1] << 8);
            float dl0 = d0 * (float)(2 * (int)((q0 >> 12) & 7) + 1);
            float dl1 = d1 * (float)(2 * (int)((q1 >> 12) & 7) + 1);
            float de0 = (q0 & 0x8000) == 0 ? 0.125f : -0.125f;
            float de1 = (q1 & 0x8000) == 0 ? 0.125f : -0.125f;
            for (uint l = 0; l < 4; l++) {
                uint i0 = (uint)qs0[qoff + l] | (((q0 >> (3 * l)) & 7) << 8);
                uint i1 = (uint)qs1[qoff + l] | (((q1 >> (3 * l)) & 7) << 8);
                IQ1_DOT(iq1s_grid[i0], ybase, dl0, de0, s0);
                IQ1_DOT(iq1s_grid[i1], ybase, dl1, de1, s1);
                ybase += 8;
            }
            qoff += 4;
        }
    }
    s0 += simd_shuffle_xor(s0, 4); s0 += simd_shuffle_xor(s0, 2); s0 += simd_shuffle_xor(s0, 1);
    s1 += simd_shuffle_xor(s1, 4); s1 += simd_shuffle_xor(s1, 2); s1 += simd_shuffle_xor(s1, 1);
    if (lane == 0) { out[row0] = s0; out[row1] = s1; }
}
`
