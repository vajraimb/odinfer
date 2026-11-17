// Long-K down GEMV: spec-style 8-lane scalar vs simdgroup K-split.
// n=17408, d=5120. Ten dispatches, min of 4.
package gemv_iq_longk

import q35 "qwen3_5:qwen3_5"

import NS "core:sys/darwin/Foundation"
import "core:fmt"
import "core:os"
import "core:strings"
import "core:time"
import MTL "vendor:darwin/Metal"

N :: 17408
D :: 5120
NSB :: 68
REPS :: 10

main :: proc() {
	dev := MTL.CreateSystemDefaultDevice()
	if dev == nil { fmt.eprintln("no metal"); os.exit(1) }
	defer dev->release()
	queue := dev->newCommandQueue()
	defer queue->release()
	// Same translation unit as the production library. Snippet compiles have
	// moved this kernel by ~30% when a sibling changed.
	src, _ := strings.concatenate({q35.MSL_SRC, EXTRA})
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

	w2 := dev->newBufferWithLength(NS.UInteger(D * NSB * 66), MTL.ResourceStorageModeShared)
	w1 := dev->newBufferWithLength(NS.UInteger(D * NSB * 50), MTL.ResourceStorageModeShared)
	xb := dev->newBufferWithLength(NS.UInteger(N * 4), MTL.ResourceStorageModeShared)
	ob := dev->newBufferWithLength(NS.UInteger(D * 4), MTL.ResourceStorageModeShared)
	defer w2->release(); defer w1->release(); defer xb->release(); defer ob->release()
	fill(w2->contentsAsSlice([]u8), 66)
	fill(w1->contentsAsSlice([]u8), 50)
	xs := xb->contentsAsSlice([]f32)
	for i in 0 ..< len(xs) { xs[i] = f32(i % 13) * 0.01 - 0.05 }

	names := []string{"iq2_sc", "iq2_ks", "iq1_sc", "iq1_ks"}
	tgs := []int{D / 32, D / 16, D / 32, D / 16}
	wbs := []^MTL.Buffer{w2, w2, w1, w1}
	bytes := []f64{f64(D * NSB * 66), f64(D * NSB * 66), f64(D * NSB * 50), f64(D * NSB * 50)}
	psos := make([]^MTL.ComputePipelineState, len(names))
	for name, i in names { psos[i] = make_pso(dev, lib, name) }

	run(queue, psos[0], w2, xb, ob, tgs[0])
	ref2 := make([]f32, D)
	copy(ref2, ob->contentsAsSlice([]f32)[:D])
	check(queue, psos[1], w2, xb, ob, tgs[1], ref2, names[1])
	delete(ref2)

	run(queue, psos[2], w1, xb, ob, tgs[2])
	ref1 := make([]f32, D)
	copy(ref1, ob->contentsAsSlice([]f32)[:D])
	check(queue, psos[3], w1, xb, ob, tgs[3], ref1, names[3])
	delete(ref1)

	for name, i in names {
		best: f64 = 1e18
		for _ in 0 ..< 4 {
			t0 := time.tick_now()
			cmd := queue->commandBuffer()
			enc := cmd->computeCommandEncoder()
			enc->setComputePipelineState(psos[i])
			enc->setBuffer(wbs[i], 0, 0)
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
		fmt.eprintfln("{:7.3f} ms   {:5.1f} GB/s   {}", best, bytes[i] / (best * 1e-3) / 1e9, name)
	}
}

fill :: proc(ws: []u8, blk: int) {
	for i in 0 ..< len(ws) { ws[i] = u8((i * 17 + 3) & 0xff) }
	for b in 0 ..< len(ws) / blk {
		ws[b * blk + 0] = 0x00
		ws[b * blk + 1] = 0x3C
	}
}

check :: proc(queue: ^MTL.CommandQueue, pso: ^MTL.ComputePipelineState, wb, xb, ob: ^MTL.Buffer, groups: int, ref: []f32, name: string) {
	out_s := ob->contentsAsSlice([]f32)[:D]
	for k in 0 ..< D { out_s[k] = 1e9 }
	run(queue, pso, wb, xb, ob, groups)
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
kernel void iq2_sc(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 68;
    uint row = tg * 32 + tid / 8; uint lane = tid % 8; float sum = 0.0f;
    device const uchar *wr = wb + (ulong)row * nsb * 66;
    for (uint sb = lane; sb < nsb; sb += 8) {
        device const uchar *blk = wr + sb * 66;
        float d = (float)(*(device const half*)blk);
        uint ybase = sb * 256;
        for (uint ib32 = 0; ib32 < 8; ib32++) {
            uint qb = 2 + 8 * ib32;
            uint aux = (uint)blk[qb + 4] | ((uint)blk[qb + 5] << 8) | ((uint)blk[qb + 6] << 16) | ((uint)blk[qb + 7] << 24);
            float db = d * (0.5f + (float)(aux >> 28)) * 0.25f;
            for (uint l = 0; l < 4; l++) {
                ulong grid = iq2xxs_grid[blk[qb + l]];
                uchar signs = ksigns_iq2xs[(aux >> (7 * l)) & 127];
                for (uint j = 0; j < 8; j++) {
                    float v = (float)((grid >> (8 * j)) & 0xff);
                    float s = (signs & (1u << j)) == 0 ? 1.0f : -1.0f;
                    sum += db * v * s * x[ybase + j];
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

kernel void iq2_ks(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint nsb = 68;
    constexpr uint nb32 = 544;
    uint row0 = (tg * 8u + sgitg) * 2u;
    uint row1 = row0 + 1u;
    float s0 = 0.0f, s1 = 0.0f;
    for (uint ib32 = tiisg; ib32 < nb32; ib32 += 32) {
        uint yb = ib32 * 32;
        float yl[32];
        for (uint i = 0; i < 32; ++i) yl[i] = x[yb + i];
        uint ibl = ib32 >> 3;
        uint ib = ib32 & 7;
        device const uchar *b0 = wb + ((ulong)row0 * nsb + ibl) * 66;
        device const uchar *b1 = wb + ((ulong)row1 * nsb + ibl) * 66;
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

kernel void iq1_sc(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 68;
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

kernel void iq1_ks(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint nsb = 68;
    constexpr uint nb32 = 544;
    uint rowbase = (tg * 8u + sgitg) * 2u;
    float sumf[2] = {0.0f, 0.0f};
    for (uint ib32 = tiisg; ib32 < nb32; ib32 += 32) {
        uint yb = ib32 * 32;
        float yl[32];
        float sumy = 0.0f;
        for (uint i = 0; i < 32; ++i) { yl[i] = x[yb + i]; sumy += yl[i]; }
        uint ibl = ib32 >> 3;
        uint ib = ib32 & 7;
        for (uint row = 0; row < 2; ++row) {
            device const uchar *blk = wb + ((ulong)(rowbase + row) * nsb + ibl) * 50;
            float d = (float)(*(device const half*)blk);
            device const uchar *qs = blk + 2 + 4 * ib;
            uint qh = (uint)blk[34 + 2 * ib] | ((uint)blk[35 + 2 * ib] << 8);
            float scale = (float)(2 * (int)((qh >> 12) & 7) + 1);
            float delta = (qh & 0x8000) == 0 ? 0.125f : -0.125f;
            float sum = 0.0f;
            for (uint l = 0; l < 4; ++l) {
                uint idx = (uint)qs[l] | (((qh >> (3 * l)) & 7) << 8);
                ulong grid = iq1s_grid[idx];
                for (uint j = 0; j < 8; ++j) {
                    float gv = (float)((char)((grid >> (8 * j)) & 0xff));
                    sum += gv * yl[8 * l + j];
                }
            }
            sumf[row] += d * (sum + delta * sumy) * scale;
        }
    }
    for (uint row = 0; row < 2; ++row) {
        float all = simd_sum(sumf[row]);
        if (tiisg == 0) out[rowbase + row] = all;
    }
}
`
