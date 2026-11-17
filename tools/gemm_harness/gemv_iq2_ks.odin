// IQ2_XXS short-K: shipping 8-lane r2 vs scale factored out of the
// inner product vs a 32-wide K split. n=5120, d=17408.
package gemv_iq2_ks

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
	for b in 0 ..< D * (N / 256) {
		ws[b * 66 + 0] = 0x00
		ws[b * 66 + 1] = 0x3C
	}
	xs := xb->contentsAsSlice([]f32)
	for i in 0 ..< len(xs) { xs[i] = f32(i % 13) * 0.01 - 0.05 }

	names := []string{"iq2_r2", "iq2_fac", "iq2_ks2"}
	tgs := []int{D / 64, D / 64, D / 16}
	psos := make([]^MTL.ComputePipelineState, len(names))
	for name, i in names { psos[i] = make_pso(dev, lib, name) }
	run(queue, psos[0], wb, xb, ob, tgs[0])
	ref := make([]f32, D)
	copy(ref, ob->contentsAsSlice([]f32)[:D])
	for name, i in names[1:] {
		out_s := ob->contentsAsSlice([]f32)[:D]
		for k in 0 ..< D { out_s[k] = 1e9 }
		run(queue, psos[i + 1], wb, xb, ob, tgs[i + 1])
		got := ob->contentsAsSlice([]f32)[:D]
		max_rel: f32 = 0
		miss := 0
		for k in 0 ..< D {
			if got[k] == 1e9 { miss += 1; continue }
			df := got[k] - ref[k]
			if df < 0 { df = -df }
			den := ref[k]
			if den < 0 { den = -den }
			if den < 1e-6 { den = 1e-6 }
			rel := df / den
			if rel > max_rel { max_rel = rel }
		}
		fmt.eprintfln("check {} max_rel {} unwritten {}", name, max_rel, miss)
	}
	delete(ref)
	bytes := f64(wbytes)
	for name, i in names {
		best: f64 = 1e18
		for _ in 0 ..< 4 {
			t0 := time.tick_now()
			cmd := queue->commandBuffer()
			enc := cmd->computeCommandEncoder()
			enc->setComputePipelineState(psos[i])
			enc->setBuffer(wb, 0, 0)
			enc->setBuffer(xb, 0, 1)
			enc->setBuffer(ob, 0, 2)
			for _ in 0 ..< REPS {
				enc->dispatchThreadgroups(MTL.Size{NS.Integer(tgs[i]), 1, 1}, MTL.Size{256, 1, 1})
			}
			enc->endEncoding()
			cmd->commit()
			cmd->waitUntilCompleted()
			ms := f64(time.tick_since(t0)) / 1e6 / f64(REPS)
			if ms < best { best = ms }
		}
		fmt.eprintfln("{:7.3f} ms   {:5.1f} GB/s   {}", best, bytes / (best * 1e-3) / 1e9, name)
	}
}

run :: proc(queue: ^MTL.CommandQueue, pso: ^MTL.ComputePipelineState, wb, xb, ob: ^MTL.Buffer, groups: int) {
	cmd := queue->commandBuffer()
	enc := cmd->computeCommandEncoder()
	enc->setComputePipelineState(pso)
	enc->setBuffer(wb, 0, 0)
	enc->setBuffer(xb, 0, 1)
	enc->setBuffer(ob, 0, 2)
	enc->dispatchThreadgroups(MTL.Size{NS.Integer(groups), 1, 1}, MTL.Size{256, 1, 1})
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
kernel void iq2_r2(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 20;
    uint lane = tid % 8; uint slot = tid / 8;
    uint row0 = tg * 64 + slot;
    uint row1 = row0 + 32;
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
    if (lane % 8 == 0) { out[row0] = s0; out[row1] = s1; }
}

kernel void iq2_fac(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 20;
    uint lane = tid % 8; uint slot = tid / 8;
    uint row0 = tg * 64 + slot;
    uint row1 = row0 + 32;
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
            float a0 = 0.0f, a1 = 0.0f;
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
                    a0 += v0 * z0 * xj;
                    a1 += v1 * z1 * xj;
                }
                ybase += 8;
            }
            s0 += db0 * a0;
            s1 += db1 * a1;
        }
    }
    s0 += simd_shuffle_xor(s0, 4); s0 += simd_shuffle_xor(s0, 2); s0 += simd_shuffle_xor(s0, 1);
    s1 += simd_shuffle_xor(s1, 4); s1 += simd_shuffle_xor(s1, 2); s1 += simd_shuffle_xor(s1, 1);
    if (lane % 8 == 0) { out[row0] = s0; out[row1] = s1; }
}

kernel void iq2_ks2(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint nb32 = 160;
    uint row0 = (tg * 8u + sgitg) * 2u;
    uint row1 = row0 + 1u;
    float s0 = 0.0f, s1 = 0.0f;
    for (uint ib32 = tiisg; ib32 < nb32; ib32 += 32) {
        uint yb = ib32 * 32;
        float yl[32];
        for (uint i = 0; i < 32; ++i) yl[i] = x[yb + i];
        uint ibl = ib32 >> 3;
        uint ib = ib32 & 7;
        device const uchar *b0 = wb + ((ulong)row0 * 20u + ibl) * 66;
        device const uchar *b1 = wb + ((ulong)row1 * 20u + ibl) * 66;
        float d0 = (float)(*(device const half *)b0);
        float d1 = (float)(*(device const half *)b1);
        uint qb = 2 + 8 * ib;
        uint aux0 = (uint)b0[qb + 4] | ((uint)b0[qb + 5] << 8) | ((uint)b0[qb + 6] << 16) | ((uint)b0[qb + 7] << 24);
        uint aux1 = (uint)b1[qb + 4] | ((uint)b1[qb + 5] << 8) | ((uint)b1[qb + 6] << 16) | ((uint)b1[qb + 7] << 24);
        float db0 = d0 * (0.5f + (float)(aux0 >> 28)) * 0.25f;
        float db1 = d1 * (0.5f + (float)(aux1 >> 28)) * 0.25f;
        float a0 = 0.0f, a1 = 0.0f;
        for (uint l = 0; l < 4; ++l) {
            ulong g0 = iq2xxs_grid[b0[qb + l]];
            ulong g1 = iq2xxs_grid[b1[qb + l]];
            uchar sg0 = ksigns_iq2xs[(aux0 >> (7 * l)) & 127];
            uchar sg1 = ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
            for (uint j = 0; j < 8; ++j) {
                float xj = yl[8 * l + j];
                float v0 = (float)((g0 >> (8 * j)) & 0xff);
                float v1 = (float)((g1 >> (8 * j)) & 0xff);
                float z0 = (sg0 & (1u << j)) == 0 ? 1.0f : -1.0f;
                float z1 = (sg1 & (1u << j)) == 0 ? 1.0f : -1.0f;
                a0 += v0 * z0 * xj;
                a1 += v1 * z1 * xj;
            }
        }
        s0 += db0 * a0;
        s1 += db1 * a1;
    }
    s0 = simd_sum(s0);
    s1 = simd_sum(s1);
    if (tiisg == 0) { out[row0] = s0; out[row1] = s1; }
}
`
