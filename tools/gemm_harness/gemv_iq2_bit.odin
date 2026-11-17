// IQ2_XXS: shipping two-row scalar vs float4 bitcast, plus the long-K
// down shape (n=17408, d=5120) scalar vs float4. Ten dispatches, min of 4.
package gemv_iq2_bit

import q35 "qwen3_5:qwen3_5"

import NS "core:sys/darwin/Foundation"
import "core:fmt"
import "core:os"
import "core:strings"
import "core:time"
import MTL "vendor:darwin/Metal"

REPS :: 10

Try :: struct {
	name:   string,
	n, d:   int,
	tg_div: int,
	group:  int,
}

TRIES := []Try{
	{"iq2_r2", 5120, 17408, 64, 0},
	{"iq2_r2bit", 5120, 17408, 64, 0},
	{"iq2_down", 17408, 5120, 32, 1},
	{"iq2_downbit", 17408, 5120, 32, 1},
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

	// Both shapes move the same number of weight bytes.
	wbytes := 17408 * 20 * 66
	wb := dev->newBufferWithLength(NS.UInteger(wbytes), MTL.ResourceStorageModeShared)
	xb := dev->newBufferWithLength(NS.UInteger(17408 * 4), MTL.ResourceStorageModeShared)
	ob := dev->newBufferWithLength(NS.UInteger(17408 * 4), MTL.ResourceStorageModeShared)
	defer wb->release(); defer xb->release(); defer ob->release()
	ws := wb->contentsAsSlice([]u8)
	for i in 0 ..< len(ws) { ws[i] = u8((i * 17 + 3) & 0xff) }
	for b in 0 ..< 17408 * 20 {
		ws[b * 66 + 0] = 0x00
		ws[b * 66 + 1] = 0x3C
	}
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
		dispatch(queue, psos[ri], wb, xb, ob, rt.d / rt.tg_div)
		ref := make([]f32, rt.d)
		copy(ref, ob->contentsAsSlice([]f32)[:rt.d])
		for t, i in TRIES {
			if t.group != g || i == ri { continue }
			dispatch(queue, psos[i], wb, xb, ob, t.d / t.tg_div)
			got := ob->contentsAsSlice([]f32)[:t.d]
			max_rel: f32 = 0
			for k in 0 ..< t.d {
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
		tg := t.d / t.tg_div
		bytes := f64(t.d) * f64(t.n / 256) * 66
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
#define IQ2_ACC(grid, sg, db, x0, x1, s) \
    do { \
        uchar4 _lo = as_type<uchar4>((uint)(grid)); \
        uchar4 _hi = as_type<uchar4>((uint)((grid) >> 32)); \
        float4 _z0 = float4( \
            1.0f - 2.0f * (float)(((sg) >> 0) & 1), \
            1.0f - 2.0f * (float)(((sg) >> 1) & 1), \
            1.0f - 2.0f * (float)(((sg) >> 2) & 1), \
            1.0f - 2.0f * (float)(((sg) >> 3) & 1)); \
        float4 _z1 = float4( \
            1.0f - 2.0f * (float)(((sg) >> 4) & 1), \
            1.0f - 2.0f * (float)(((sg) >> 5) & 1), \
            1.0f - 2.0f * (float)(((sg) >> 6) & 1), \
            1.0f - 2.0f * (float)(((sg) >> 7) & 1)); \
        float4 _v0 = float4(_lo.x, _lo.y, _lo.z, _lo.w) * _z0; \
        float4 _v1 = float4(_hi.x, _hi.y, _hi.z, _hi.w) * _z1; \
        s += (db) * (dot(_v0, x0) + dot(_v1, x1)); \
    } while (0)

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
        device const uchar *blk0 = wr0 + sb * 66;
        device const uchar *blk1 = wr1 + sb * 66;
        float d0 = (float)(*(device const half*)blk0);
        float d1 = (float)(*(device const half*)blk1);
        uint ybase = sb * 256;
        #pragma clang loop unroll(disable)
        for (uint ib32 = 0; ib32 < 8; ib32++) {
            uint qb = 2 + 8 * ib32;
            uint aux0 = (uint)blk0[qb + 4] | ((uint)blk0[qb + 5] << 8) | ((uint)blk0[qb + 6] << 16) | ((uint)blk0[qb + 7] << 24);
            uint aux1 = (uint)blk1[qb + 4] | ((uint)blk1[qb + 5] << 8) | ((uint)blk1[qb + 6] << 16) | ((uint)blk1[qb + 7] << 24);
            float db0 = d0 * (0.5f + (float)(aux0 >> 28)) * 0.25f;
            float db1 = d1 * (0.5f + (float)(aux1 >> 28)) * 0.25f;
            for (uint l = 0; l < 4; l++) {
                ulong g0 = iq2xxs_grid[blk0[qb + l]];
                ulong g1 = iq2xxs_grid[blk1[qb + l]];
                uchar sg0 = ksigns_iq2xs[(aux0 >> (7 * l)) & 127];
                uchar sg1 = ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 8; j++) {
                    float xj = x[ybase + j];
                    float v0 = (float)((g0 >> (8 * j)) & 0xff);
                    float v1 = (float)((g1 >> (8 * j)) & 0xff);
                    float z0 = (sg0 & (1u << j)) == 0 ? 1.0f : -1.0f;
                    float z1 = (sg1 & (1u << j)) == 0 ? 1.0f : -1.0f;
                    s0 += db0 * v0 * z0 * xj;
                    s1 += db1 * v1 * z1 * xj;
                }
                ybase += 8;
            }
        }
    }
    s0 += simd_shuffle_xor(s0, 4); s0 += simd_shuffle_xor(s0, 2); s0 += simd_shuffle_xor(s0, 1);
    s1 += simd_shuffle_xor(s1, 4); s1 += simd_shuffle_xor(s1, 2); s1 += simd_shuffle_xor(s1, 1);
    if (lane == 0) { out[row0] = s0; out[row1] = s1; }
}

kernel void iq2_r2bit(
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
        device const uchar *blk0 = wr0 + sb * 66;
        device const uchar *blk1 = wr1 + sb * 66;
        float d0 = (float)(*(device const half*)blk0);
        float d1 = (float)(*(device const half*)blk1);
        uint ybase = sb * 256;
        #pragma clang loop unroll(disable)
        for (uint ib32 = 0; ib32 < 8; ib32++) {
            uint qb = 2 + 8 * ib32;
            uint aux0 = (uint)blk0[qb + 4] | ((uint)blk0[qb + 5] << 8) | ((uint)blk0[qb + 6] << 16) | ((uint)blk0[qb + 7] << 24);
            uint aux1 = (uint)blk1[qb + 4] | ((uint)blk1[qb + 5] << 8) | ((uint)blk1[qb + 6] << 16) | ((uint)blk1[qb + 7] << 24);
            float db0 = d0 * (0.5f + (float)(aux0 >> 28)) * 0.25f;
            float db1 = d1 * (0.5f + (float)(aux1 >> 28)) * 0.25f;
            for (uint l = 0; l < 4; l++) {
                uchar sg0 = ksigns_iq2xs[(aux0 >> (7 * l)) & 127];
                uchar sg1 = ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
                float4 x0 = *(device const float4*)(x + ybase);
                float4 x1 = *(device const float4*)(x + ybase + 4);
                IQ2_ACC(iq2xxs_grid[blk0[qb + l]], sg0, db0, x0, x1, s0);
                IQ2_ACC(iq2xxs_grid[blk1[qb + l]], sg1, db1, x0, x1, s1);
                ybase += 8;
            }
        }
    }
    s0 += simd_shuffle_xor(s0, 4); s0 += simd_shuffle_xor(s0, 2); s0 += simd_shuffle_xor(s0, 1);
    s1 += simd_shuffle_xor(s1, 4); s1 += simd_shuffle_xor(s1, 2); s1 += simd_shuffle_xor(s1, 1);
    if (lane == 0) { out[row0] = s0; out[row1] = s1; }
}

kernel void iq2_down(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 68;
    uint row = tg * 32 + tid / 8; uint lane = tid % 8; float sum = 0.0f;
    device const uchar *wr = wb + (ulong)row * nsb * 66;
    #pragma clang loop unroll(disable)
    for (uint sb = lane; sb < nsb; sb += 8) {
        device const uchar *blk = wr + sb * 66;
        float d = (float)(*(device const half*)blk);
        uint ybase = sb * 256;
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
                    sum += db * v * z * x[ybase + j];
                }
                ybase += 8;
            }
        }
    }
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane == 0) out[row] = sum;
}

kernel void iq2_downbit(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 68;
    uint row = tg * 32 + tid / 8; uint lane = tid % 8; float sum = 0.0f;
    device const uchar *wr = wb + (ulong)row * nsb * 66;
    #pragma clang loop unroll(disable)
    for (uint sb = lane; sb < nsb; sb += 8) {
        device const uchar *blk = wr + sb * 66;
        float d = (float)(*(device const half*)blk);
        uint ybase = sb * 256;
        #pragma clang loop unroll(disable)
        for (uint ib32 = 0; ib32 < 8; ib32++) {
            uint qb = 2 + 8 * ib32;
            uint aux = (uint)blk[qb + 4] | ((uint)blk[qb + 5] << 8) | ((uint)blk[qb + 6] << 16) | ((uint)blk[qb + 7] << 24);
            float db = d * (0.5f + (float)(aux >> 28)) * 0.25f;
            for (uint l = 0; l < 4; l++) {
                uchar sg = ksigns_iq2xs[(aux >> (7 * l)) & 127];
                float4 x0 = *(device const float4*)(x + ybase);
                float4 x1 = *(device const float4*)(x + ybase + 4);
                IQ2_ACC(iq2xxs_grid[blk[qb + l]], sg, db, x0, x1, sum);
                ybase += 8;
            }
        }
    }
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane == 0) out[row] = sum;
}
`
