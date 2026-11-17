// Divergent codebook lookups: constant memory vs one threadgroup copy.
// IQ2_XXS grid is 2 KB, IQ1_S grid is 16 KB. n=5120, d=17408.
// Compiled next to the production library.
package gemv_iq_tg

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

	w2 := dev->newBufferWithLength(NS.UInteger(D * (N / 256) * 66), MTL.ResourceStorageModeShared)
	w1 := dev->newBufferWithLength(NS.UInteger(D * (N / 256) * 50), MTL.ResourceStorageModeShared)
	xb := dev->newBufferWithLength(NS.UInteger(N * 4), MTL.ResourceStorageModeShared)
	ob := dev->newBufferWithLength(NS.UInteger(D * 4), MTL.ResourceStorageModeShared)
	defer w2->release(); defer w1->release(); defer xb->release(); defer ob->release()
	fill(w2->contentsAsSlice([]u8), 66)
	fill(w1->contentsAsSlice([]u8), 50)
	xs := xb->contentsAsSlice([]f32)
	for i in 0 ..< len(xs) { xs[i] = f32(i % 13) * 0.01 - 0.05 }

	p2 := make_pso(dev, lib, "gemv_iq2_xxs_ks")
	p2t := make_pso(dev, lib, "iq2_tg")
	p1 := make_pso(dev, lib, "gemv_iq1_s_ks")
	p1t := make_pso(dev, lib, "iq1_tg")

	run(queue, p2, w2, xb, ob, D / 16, 0, 0)
	ref := make([]f32, D)
	copy(ref, ob->contentsAsSlice([]f32)[:D])
	check(queue, p2t, w2, xb, ob, D / 16, 256 * 8, 128, ref, "iq2_tg")
	delete(ref)

	run(queue, p1, w1, xb, ob, D / 16, 0, 0)
	ref = make([]f32, D)
	copy(ref, ob->contentsAsSlice([]f32)[:D])
	check(queue, p1t, w1, xb, ob, D / 16, 2048 * 8, 0, ref, "iq1_tg")
	delete(ref)

	b2 := f64(D * (N / 256) * 66)
	b1 := f64(D * (N / 256) * 50)
	time_k(queue, p2, w2, xb, ob, D / 16, 0, 0, b2, "iq2_ks")
	time_k(queue, p2t, w2, xb, ob, D / 16, 256 * 8, 128, b2, "iq2_tg")
	time_k(queue, p1, w1, xb, ob, D / 16, 0, 0, b1, "iq1_ks")
	time_k(queue, p1t, w1, xb, ob, D / 16, 2048 * 8, 0, b1, "iq1_tg")
}

fill :: proc(ws: []u8, blk: int) {
	for i in 0 ..< len(ws) { ws[i] = u8((i * 17 + 3) & 0xff) }
	for b in 0 ..< len(ws) / blk {
		ws[b * blk + 0] = 0x00
		ws[b * blk + 1] = 0x3C
	}
}

bind :: proc(enc: ^MTL.ComputeCommandEncoder, pso: ^MTL.ComputePipelineState, wb, xb, ob: ^MTL.Buffer, tg0, tg1: int) {
	enc->setComputePipelineState(pso)
	enc->setBuffer(wb, 0, 0)
	enc->setBuffer(xb, 0, 1)
	enc->setBuffer(ob, 0, 2)
	if tg0 > 0 { enc->setThreadgroupMemoryLength(NS.UInteger(tg0), 0) }
	if tg1 > 0 { enc->setThreadgroupMemoryLength(NS.UInteger(tg1), 1) }
}

check :: proc(queue: ^MTL.CommandQueue, pso: ^MTL.ComputePipelineState, wb, xb, ob: ^MTL.Buffer, groups, tg0, tg1: int, ref: []f32, name: string) {
	out_s := ob->contentsAsSlice([]f32)[:D]
	for k in 0 ..< D { out_s[k] = 1e9 }
	run(queue, pso, wb, xb, ob, groups, tg0, tg1)
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

run :: proc(queue: ^MTL.CommandQueue, pso: ^MTL.ComputePipelineState, wb, xb, ob: ^MTL.Buffer, groups, tg0, tg1: int) {
	cmd := queue->commandBuffer()
	enc := cmd->computeCommandEncoder()
	bind(enc, pso, wb, xb, ob, tg0, tg1)
	enc->dispatchThreadgroups(MTL.Size{NS.Integer(groups), 1, 1}, MTL.Size{256, 1, 1})
	enc->endEncoding()
	cmd->commit()
	cmd->waitUntilCompleted()
}

time_k :: proc(queue: ^MTL.CommandQueue, pso: ^MTL.ComputePipelineState, wb, xb, ob: ^MTL.Buffer, groups, tg0, tg1: int, bytes: f64, name: string) {
	best: f64 = 1e18
	for _ in 0 ..< 4 {
		t0 := time.tick_now()
		cmd := queue->commandBuffer()
		enc := cmd->computeCommandEncoder()
		bind(enc, pso, wb, xb, ob, tg0, tg1)
		for _ in 0 ..< REPS {
			enc->dispatchThreadgroups(MTL.Size{NS.Integer(groups), 1, 1}, MTL.Size{256, 1, 1})
		}
		enc->endEncoding()
		cmd->commit()
		cmd->waitUntilCompleted()
		ms := f64(time.tick_since(t0)) / 1e6 / f64(REPS)
		if ms < best { best = ms }
	}
	fmt.eprintfln("{:7.3f} ms   {:5.1f} GB/s   {}", best, bytes / (best * 1e-3) / 1e9, name)
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
kernel void iq2_tg(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    threadgroup ulong *grid_tg [[threadgroup(0)]],
    threadgroup uchar *signs_tg [[threadgroup(1)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    grid_tg[tid] = iq2xxs_grid[tid];
    if (tid < 128u) signs_tg[tid] = ksigns_iq2xs[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
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
            ulong g0 = grid_tg[b0[qb + l]];
            ulong g1 = grid_tg[b1[qb + l]];
            uchar sg0 = signs_tg[(aux0 >> (7 * l)) & 127];
            uchar sg1 = signs_tg[(aux1 >> (7 * l)) & 127];
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

kernel void iq1_tg(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    threadgroup ulong *grid_tg [[threadgroup(0)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    for (uint i = 0; i < 8; ++i) grid_tg[tid + i * 256u] = iq1s_grid[tid + i * 256u];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr uint nb32 = 160;
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
            device const uchar *blk = wb + ((ulong)(rowbase + row) * 20u + ibl) * 50;
            float d = (float)(*(device const half*)blk);
            device const uchar *qs = blk + 2 + 4 * ib;
            uint qh = (uint)blk[34 + 2 * ib] | ((uint)blk[35 + 2 * ib] << 8);
            float scale = (float)(2 * (int)((qh >> 12) & 7) + 1);
            float delta = (qh & 0x8000) == 0 ? 0.125f : -0.125f;
            float sum = 0.0f;
            for (uint l = 0; l < 4; ++l) {
                uint idx = (uint)qs[l] | (((qh >> (3 * l)) & 7) << 8);
                ulong grid = grid_tg[idx];
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
