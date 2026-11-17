// IQ MMQ at the short-prefill widths. v3 BN=8 covers N=8 and N=16.
// The wide tile is BN=32. n=5120 is the up-projection K, d=17408 the rows.
package gemv_iq_n16

import q35 "qwen3_5:qwen3_5"

import NS "core:sys/darwin/Foundation"
import "core:fmt"
import "core:os"
import "core:strings"
import "core:time"
import MTL "vendor:darwin/Metal"

M :: 17408
K :: 5120
REPS :: 4

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

	nsb := K / 256
	a2 := dev->newBufferWithLength(NS.UInteger(M * nsb * 66), MTL.ResourceStorageModeShared)
	a1 := dev->newBufferWithLength(NS.UInteger(M * nsb * 50), MTL.ResourceStorageModeShared)
	defer a2->release(); defer a1->release()
	fill(a2->contentsAsSlice([]u8), 66)
	fill(a1->contentsAsSlice([]u8), 50)

	p2v := make_pso(dev, lib, "gemm_iq2xxs_f32")
	p2w := make_pso(dev, lib, "gemm_iq2xxs_32_32_64")
	p1v := make_pso(dev, lib, "gemm_iq1s_f32")
	p1w := make_pso(dev, lib, "gemm_iq1s_32_32_64")

	time_n(dev, queue, p2v, a2, 8, 128, 1, M / 32, f64(M * nsb * 66), "iq2 v3 N=8")
	time_n(dev, queue, p2v, a2, 16, 128, 2, M / 32, f64(M * nsb * 66) * 2, "iq2 v3 N=16")
	time_n(dev, queue, p2w, a2, 32, 512, 1, M / 32, f64(M * nsb * 66), "iq2 wide N=32")
	time_n(dev, queue, p1v, a1, 8, 128, 1, M / 32, f64(M * nsb * 50), "iq1 v3 N=8")
	time_n(dev, queue, p1v, a1, 16, 128, 2, M / 32, f64(M * nsb * 50) * 2, "iq1 v3 N=16")
	time_n(dev, queue, p1w, a1, 32, 512, 1, M / 32, f64(M * nsb * 50), "iq1 wide N=32")

	p16 := make_pso(dev, lib, "gemm_iq2xxs_n16tg")
	check_n16(dev, queue, p2v, p16, a2)
	time_n(dev, queue, p16, a2, 16, 256, 1, M / 32, f64(M * nsb * 66), "iq2 bn16 N=16")
	p16b := make_pso(dev, lib, "gemm_iq2xxs_n16b")
	check_n16b(dev, queue, p2v, p16b, a2)
	time_n(dev, queue, p16b, a2, 16, 128, 1, M / 32, f64(M * nsb * 66), "iq2 bn16b N=16")
	p1b := make_pso(dev, lib, "gemm_iq1s_n16b")
	check_n16b(dev, queue, p1v, p1b, a1)
	time_n(dev, queue, p1b, a1, 16, 128, 1, M / 32, f64(M * nsb * 50), "iq1 bn16b N=16")

	a2s := dev->newBufferWithLength(NS.UInteger(M * nsb * 82), MTL.ResourceStorageModeShared)
	a3 := dev->newBufferWithLength(NS.UInteger(M * nsb * 98), MTL.ResourceStorageModeShared)
	defer a2s->release(); defer a3->release()
	fill(a2s->contentsAsSlice([]u8), 82)
	fill(a3->contentsAsSlice([]u8), 98)
	p2sv := make_pso(dev, lib, "gemm_iq2s_f32")
	p3v := make_pso(dev, lib, "gemm_iq3xxs_f32")
	time_n(dev, queue, p2sv, a2s, 16, 128, 2, M / 32, f64(M * nsb * 82) * 2, "iq2s v3 N=16")
	time_n(dev, queue, p3v, a3, 16, 128, 2, M / 32, f64(M * nsb * 98) * 2, "iq3 v3 N=16")
	p2sb := make_pso(dev, lib, "gemm_iq2s_n16b")
	p3b := make_pso(dev, lib, "gemm_iq3xxs_n16b")
	check_n16b(dev, queue, p2sv, p2sb, a2s)
	time_n(dev, queue, p2sb, a2s, 16, 128, 1, M / 32, f64(M * nsb * 82), "iq2s bn16b N=16")
	check_n16b(dev, queue, p3v, p3b, a3)
	time_n(dev, queue, p3b, a3, 16, 128, 1, M / 32, f64(M * nsb * 98), "iq3 bn16b N=16")
}

check_n16b :: proc(dev: ^MTL.Device, queue: ^MTL.CommandQueue, v3, bn: ^MTL.ComputePipelineState, a: ^MTL.Buffer) {
	n := 16
	b := dev->newBufferWithLength(NS.UInteger(K * n * 2), MTL.ResourceStorageModeShared)
	c := dev->newBufferWithLength(NS.UInteger(M * n * 4), MTL.ResourceStorageModeShared)
	defer b->release(); defer c->release()
	hs := b->contentsAsSlice([]u16)
	for i in 0 ..< len(hs) { hs[i] = 0x3C00 }
	dims := [3]u32{u32(M), u32(n), u32(K)}
	run_one(queue, v3, a, b, c, &dims, n / 8, 128)
	ref := make([]f32, M * n)
	copy(ref, c->contentsAsSlice([]f32)[:M * n])
	cs := c->contentsAsSlice([]f32)
	for i in 0 ..< M * n { cs[i] = 1e9 }
	run_one(queue, bn, a, b, c, &dims, 1, 128)
	got := c->contentsAsSlice([]f32)[:M * n]
	max_rel: f32 = 0
	miss := 0
	for i in 0 ..< M * n {
		if got[i] == 1e9 { miss += 1; continue }
		df := got[i] - ref[i]
		if df < 0 { df = -df }
		den := ref[i]
		if den < 0 { den = -den }
		if den < 1e-6 { den = 1e-6 }
		rel := df / den
		if rel > max_rel { max_rel = rel }
	}
	delete(ref)
	fmt.eprintfln("check bn16b max_rel {} unwritten {}", max_rel, miss)
}

check_n16 :: proc(dev: ^MTL.Device, queue: ^MTL.CommandQueue, v3, bn: ^MTL.ComputePipelineState, a: ^MTL.Buffer) {
	n := 16
	b := dev->newBufferWithLength(NS.UInteger(K * n * 2), MTL.ResourceStorageModeShared)
	c := dev->newBufferWithLength(NS.UInteger(M * n * 4), MTL.ResourceStorageModeShared)
	defer b->release(); defer c->release()
	hs := b->contentsAsSlice([]u16)
	for i in 0 ..< len(hs) { hs[i] = 0x3C00 }
	dims := [3]u32{u32(M), u32(n), u32(K)}
	run_one(queue, v3, a, b, c, &dims, n / 8, 128)
	ref := make([]f32, M * n)
	copy(ref, c->contentsAsSlice([]f32)[:M * n])
	cs := c->contentsAsSlice([]f32)
	for i in 0 ..< M * n { cs[i] = 1e9 }
	run_one(queue, bn, a, b, c, &dims, n / 16, 256)
	got := c->contentsAsSlice([]f32)[:M * n]
	max_rel: f32 = 0
	miss := 0
	for i in 0 ..< M * n {
		if got[i] == 1e9 { miss += 1; continue }
		df := got[i] - ref[i]
		if df < 0 { df = -df }
		den := ref[i]
		if den < 0 { den = -den }
		if den < 1e-6 { den = 1e-6 }
		rel := df / den
		if rel > max_rel { max_rel = rel }
	}
	delete(ref)
	fmt.eprintfln("check bn16 max_rel {} unwritten {}", max_rel, miss)
}

run_one :: proc(queue: ^MTL.CommandQueue, pso: ^MTL.ComputePipelineState, a, b, c: ^MTL.Buffer, dims: ^[3]u32, grid_n, threads: int) {
	cmd := queue->commandBuffer()
	enc := cmd->computeCommandEncoder()
	enc->setComputePipelineState(pso)
	enc->setBuffer(a, 0, 0)
	enc->setBuffer(b, 0, 1)
	enc->setBuffer(c, 0, 2)
	enc->setBytes(([^]u8)(dims)[:size_of(dims^)], 3)
	enc->dispatchThreadgroups(MTL.Size{NS.Integer(grid_n), NS.Integer(M / 32), 1}, MTL.Size{NS.Integer(threads), 1, 1})
	enc->endEncoding()
	cmd->commit()
	cmd->waitUntilCompleted()
}

fill :: proc(ws: []u8, blk: int) {
	for i in 0 ..< len(ws) { ws[i] = u8((i * 17 + 3) & 0xff) }
	for b in 0 ..< len(ws) / blk {
		ws[b * blk + 0] = 0x00
		ws[b * blk + 1] = 0x3C
	}
}

time_n :: proc(dev: ^MTL.Device, queue: ^MTL.CommandQueue, pso: ^MTL.ComputePipelineState, a: ^MTL.Buffer, n, threads, grid_n, grid_m: int, bytes: f64, name: string) {
	b := dev->newBufferWithLength(NS.UInteger(K * n * 2), MTL.ResourceStorageModeShared)
	c := dev->newBufferWithLength(NS.UInteger(M * n * 4), MTL.ResourceStorageModeShared)
	defer b->release(); defer c->release()
	hs := b->contentsAsSlice([]u16)
	for i in 0 ..< len(hs) { hs[i] = 0x3C00 }
	dims := [3]u32{u32(M), u32(n), u32(K)}
	best: f64 = 1e18
	for _ in 0 ..< 3 {
		t0 := time.tick_now()
		cmd := queue->commandBuffer()
		enc := cmd->computeCommandEncoder()
		enc->setComputePipelineState(pso)
		enc->setBuffer(a, 0, 0)
		enc->setBuffer(b, 0, 1)
		enc->setBuffer(c, 0, 2)
		enc->setBytes(([^]u8)(&dims)[:size_of(dims)], 3)
		for _ in 0 ..< REPS {
			enc->dispatchThreadgroups(MTL.Size{NS.Integer(grid_n), NS.Integer(grid_m), 1}, MTL.Size{NS.Integer(threads), 1, 1})
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
kernel void gemm_iq2xxs_n16tg(
    device const uchar * A [[buffer(0)]], device const half * B [[buffer(1)]],
    device float * C [[buffer(2)]], constant GemmDims & dims [[buffer(3)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiitg [[thread_index_in_threadgroup]]) {
    const uint M = dims.M, N = dims.N, K = dims.K;
    const uint r0 = tgpig.y * 32, c0 = tgpig.x * 16;
    const ulong row_bytes = (ulong)(K / 256) * 66;
    threadgroup half sa[32 * 32];
    threadgroup half sb[32 * 16];
    simdgroup_half8x8 a_tile, b_tile;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.0f);
    const uint sg_row = (sgitg / 2) * 8;
    const uint sg_col = (sgitg % 2) * 8;
    for (uint k = 0; k < K; k += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiitg < 128u) {
            uint row = tiitg >> 2, col_base = (tiitg & 3) * 8;
            if (r0 + row < M) {
                uint e_base = k + col_base;
                device const uchar * blk = A + (ulong)(r0 + row) * row_bytes + (uint)(e_base / 256) * 66;
                int ew = int(e_base & 255);
                float d = (float)(*(device const half *)blk);
                uint ib32 = (uint)ew >> 5;
                uint qb = 2 + 8 * ib32;
                uint aux1 = (uint)blk[qb + 4] | ((uint)blk[qb + 5] << 8) | ((uint)blk[qb + 6] << 16) | ((uint)blk[qb + 7] << 24);
                float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
                uint l = ((uint)ew >> 3) & 3;
                ulong grid = iq2xxs_grid[blk[qb + l]];
                uchar signs = ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
                for (uint j = 0; j < 8; j++) {
                    float v = (float)((grid >> (8 * j)) & 0xff);
                    float s = (signs & kmask_iq2xs[j]) == 0 ? 1.0f : -1.0f;
                    sa[row * 32 + col_base + j] = (half)(db * v * s);
                }
            }
        }
        for (uint t = tiitg; t < 32u * 16u; t += 256u) {
            uint kr = t >> 4, nc = t & 15;
            sb[t] = B[(ulong)(k + kr) * N + c0 + nc];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ik = 0; ik < 4; ik++) {
            simdgroup_load(a_tile, sa + sg_row * 32 + ik * 8, 32);
            simdgroup_load(b_tile, sb + ik * 128 + sg_col, 16);
            simdgroup_multiply_accumulate(acc, a_tile, b_tile, acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (r0 + sg_row < M) simdgroup_store(acc, C + (ulong)(r0 + sg_row) * N + c0 + sg_col, N);
}

kernel void gemm_iq2xxs_n16b(
    device const uchar * A [[buffer(0)]], device const half * B [[buffer(1)]],
    device float * C [[buffer(2)]], constant GemmDims & dims [[buffer(3)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiitg [[thread_index_in_threadgroup]]) {
    const uint M = dims.M, N = dims.N, K = dims.K;
    const uint r0 = tgpig.y * 32, c0 = tgpig.x * 16;
    const ulong row_bytes = (ulong)(K / 256) * 66;
    threadgroup half sa[32 * 32];
    threadgroup half sb[32 * 16];
    simdgroup_half8x8 a_tile, b0, b1;
    simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    const uint sg_row = sgitg * 8;
    for (uint k = 0; k < K; k += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            uint row = tiitg >> 2, col_base = (tiitg & 3) * 8;
            if (r0 + row < M) {
                uint e_base = k + col_base;
                device const uchar * blk = A + (ulong)(r0 + row) * row_bytes + (uint)(e_base / 256) * 66;
                int ew = int(e_base & 255);
                float d = (float)(*(device const half *)blk);
                uint ib32 = (uint)ew >> 5;
                uint qb = 2 + 8 * ib32;
                uint aux1 = (uint)blk[qb + 4] | ((uint)blk[qb + 5] << 8) | ((uint)blk[qb + 6] << 16) | ((uint)blk[qb + 7] << 24);
                float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
                uint l = ((uint)ew >> 3) & 3;
                ulong grid = iq2xxs_grid[blk[qb + l]];
                uchar signs = ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
                for (uint j = 0; j < 8; j++) {
                    float v = (float)((grid >> (8 * j)) & 0xff);
                    float s = (signs & kmask_iq2xs[j]) == 0 ? 1.0f : -1.0f;
                    sa[row * 32 + col_base + j] = (half)(db * v * s);
                }
            }
        }
        for (uint t = tiitg; t < 32u * 16u; t += 128u) {
            uint kr = t >> 4, nc = t & 15;
            sb[t] = B[(ulong)(k + kr) * N + c0 + nc];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ik = 0; ik < 4; ik++) {
            simdgroup_load(a_tile, sa + sg_row * 32 + ik * 8, 32);
            simdgroup_load(b0, sb + ik * 128, 16);
            simdgroup_load(b1, sb + ik * 128 + 8, 16);
            simdgroup_multiply_accumulate(acc0, a_tile, b0, acc0);
            simdgroup_multiply_accumulate(acc1, a_tile, b1, acc1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (r0 + sg_row < M) {
        simdgroup_store(acc0, C + (ulong)(r0 + sg_row) * N + c0, N);
        simdgroup_store(acc1, C + (ulong)(r0 + sg_row) * N + c0 + 8, N);
    }
}

kernel void gemm_iq1s_n16b(
    device const uchar * A [[buffer(0)]], device const half * B [[buffer(1)]],
    device float * C [[buffer(2)]], constant GemmDims & dims [[buffer(3)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiitg [[thread_index_in_threadgroup]]) {
    const uint M = dims.M, N = dims.N, K = dims.K;
    const uint r0 = tgpig.y * 32, c0 = tgpig.x * 16;
    const ulong row_bytes = (ulong)(K / 256) * 50;
    threadgroup half sa[32 * 32];
    threadgroup half sb[32 * 16];
    simdgroup_half8x8 a_tile, b0, b1;
    simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    const uint sg_row = sgitg * 8;
    for (uint k = 0; k < K; k += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            uint row = tiitg >> 2, col_base = (tiitg & 3) * 8;
            if (r0 + row < M) {
                uint e_base = k + col_base;
                device const uchar * blk = A + (ulong)(r0 + row) * row_bytes + (uint)(e_base / 256) * 50;
                int ew = int(e_base & 255);
                float d = (float)(*(device const half *)blk);
                device const uchar * qs = blk + 2; device const uchar * qh = blk + 34;
                uint ib = (uint)ew >> 5;
                uint qhv = (uint)qh[2 * ib] | ((uint)qh[2 * ib + 1] << 8);
                float dl = d * (float)(2 * (int)((qhv >> 12) & 7) + 1);
                float delta = (qhv & 0x8000) == 0 ? 0.125f : -0.125f;
                uint l = ((uint)ew >> 3) & 3;
                uint idx = (uint)qs[4 * ib + l] | (((qhv >> (3 * l)) & 7) << 8);
                ulong grid = iq1s_grid[idx];
                for (uint j = 0; j < 8; j++) {
                    float gv = (float)((char)((grid >> (8 * j)) & 0xff));
                    sa[row * 32 + col_base + j] = (half)(dl * (gv + delta));
                }
            }
        }
        for (uint t = tiitg; t < 32u * 16u; t += 128u) {
            uint kr = t >> 4, nc = t & 15;
            sb[t] = B[(ulong)(k + kr) * N + c0 + nc];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ik = 0; ik < 4; ik++) {
            simdgroup_load(a_tile, sa + sg_row * 32 + ik * 8, 32);
            simdgroup_load(b0, sb + ik * 128, 16);
            simdgroup_load(b1, sb + ik * 128 + 8, 16);
            simdgroup_multiply_accumulate(acc0, a_tile, b0, acc0);
            simdgroup_multiply_accumulate(acc1, a_tile, b1, acc1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (r0 + sg_row < M) {
        simdgroup_store(acc0, C + (ulong)(r0 + sg_row) * N + c0, N);
        simdgroup_store(acc1, C + (ulong)(r0 + sg_row) * N + c0 + 8, N);
    }
}

kernel void gemm_iq2s_n16b(
    device const uchar * A [[buffer(0)]], device const half * B [[buffer(1)]],
    device float * C [[buffer(2)]], constant GemmDims & dims [[buffer(3)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiitg [[thread_index_in_threadgroup]]) {
    const uint M = dims.M, N = dims.N, K = dims.K;
    const uint r0 = tgpig.y * 32, c0 = tgpig.x * 16;
    const ulong row_bytes = (ulong)(K / 256) * 82;
    threadgroup half sa[32 * 32];
    threadgroup half sb[32 * 16];
    simdgroup_half8x8 a_tile, b0, b1;
    simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    const uint sg_row = sgitg * 8;
    for (uint k = 0; k < K; k += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            uint row = tiitg >> 2, col_base = (tiitg & 3) * 8;
            if (r0 + row < M) {
                uint e_base = k + col_base;
                device const uchar * blk = A + (ulong)(r0 + row) * row_bytes + (uint)(e_base / 256) * 82;
                int ew = int(e_base & 255);
                float d = (float)(*(device const half *)blk);
                device const uchar * qs = blk + 2; device const uchar * signs = blk + 34;
                device const uchar * qh = blk + 66; device const uchar * scales = blk + 74;
                uint ib32 = (uint)ew >> 5;
                uchar sc = scales[ib32];
                float db0 = d * (0.5f + (float)(sc & 0xF)) * 0.25f;
                float db1 = d * (0.5f + (float)(sc >> 4)) * 0.25f;
                uint l = ((uint)ew >> 3) & 3;
                uint idx = (uint)qs[4 * ib32 + l] | (((uint)qh[ib32] << (8 - 2 * l)) & 0x300);
                ulong grid = iq2s_grid[idx];
                uchar sn = signs[4 * ib32 + l];
                float dl = (l < 2) ? db0 : db1;
                for (uint j = 0; j < 8; j++) {
                    float v = (float)((grid >> (8 * j)) & 0xff);
                    float s = (sn & kmask_iq2xs[j]) == 0 ? 1.0f : -1.0f;
                    sa[row * 32 + col_base + j] = (half)(dl * v * s);
                }
            }
        }
        for (uint t = tiitg; t < 32u * 16u; t += 128u) {
            uint kr = t >> 4, nc = t & 15;
            sb[t] = B[(ulong)(k + kr) * N + c0 + nc];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ik = 0; ik < 4; ik++) {
            simdgroup_load(a_tile, sa + sg_row * 32 + ik * 8, 32);
            simdgroup_load(b0, sb + ik * 128, 16);
            simdgroup_load(b1, sb + ik * 128 + 8, 16);
            simdgroup_multiply_accumulate(acc0, a_tile, b0, acc0);
            simdgroup_multiply_accumulate(acc1, a_tile, b1, acc1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (r0 + sg_row < M) {
        simdgroup_store(acc0, C + (ulong)(r0 + sg_row) * N + c0, N);
        simdgroup_store(acc1, C + (ulong)(r0 + sg_row) * N + c0 + 8, N);
    }
}

kernel void gemm_iq3xxs_n16b(
    device const uchar * A [[buffer(0)]], device const half * B [[buffer(1)]],
    device float * C [[buffer(2)]], constant GemmDims & dims [[buffer(3)]],
    uint2 tgpig [[threadgroup_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiitg [[thread_index_in_threadgroup]]) {
    const uint M = dims.M, N = dims.N, K = dims.K;
    const uint r0 = tgpig.y * 32, c0 = tgpig.x * 16;
    const ulong row_bytes = (ulong)(K / 256) * 98;
    threadgroup half sa[32 * 32];
    threadgroup half sb[32 * 16];
    simdgroup_half8x8 a_tile, b0, b1;
    simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    const uint sg_row = sgitg * 8;
    for (uint k = 0; k < K; k += 32) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            uint row = tiitg >> 2, col_base = (tiitg & 3) * 8;
            if (r0 + row < M) {
                uint e_base = k + col_base;
                device const uchar * blk = A + (ulong)(r0 + row) * row_bytes + (uint)(e_base / 256) * 98;
                int ew = int(e_base & 255);
                float d = (float)(*(device const half *)blk);
                device const uchar * qs = blk + 2; device const uchar * sas = blk + 66;
                uint ib32 = (uint)ew >> 5;
                uint sbase = 4 * ib32;
                uint aux = (uint)sas[sbase] | ((uint)sas[sbase + 1] << 8) | ((uint)sas[sbase + 2] << 16) | ((uint)sas[sbase + 3] << 24);
                float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
                uint l = ((uint)ew >> 3) & 3;
                uchar signs = ksigns_iq2xs[(aux >> (7 * l)) & 127];
                uint g1 = iq3xxs_grid[qs[8 * ib32 + 2 * l + 0]];
                uint g2 = iq3xxs_grid[qs[8 * ib32 + 2 * l + 1]];
                for (uint j = 0; j < 4; j++) {
                    float v1 = (float)((g1 >> (8 * j)) & 0xff);
                    float v2 = (float)((g2 >> (8 * j)) & 0xff);
                    float s1 = (signs & kmask_iq2xs[j + 0]) == 0 ? 1.0f : -1.0f;
                    float s2 = (signs & kmask_iq2xs[j + 4]) == 0 ? 1.0f : -1.0f;
                    sa[row * 32 + col_base + j + 0] = (half)(db * v1 * s1);
                    sa[row * 32 + col_base + j + 4] = (half)(db * v2 * s2);
                }
            }
        }
        for (uint t = tiitg; t < 32u * 16u; t += 128u) {
            uint kr = t >> 4, nc = t & 15;
            sb[t] = B[(ulong)(k + kr) * N + c0 + nc];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint ik = 0; ik < 4; ik++) {
            simdgroup_load(a_tile, sa + sg_row * 32 + ik * 8, 32);
            simdgroup_load(b0, sb + ik * 128, 16);
            simdgroup_load(b1, sb + ik * 128 + 8, 16);
            simdgroup_multiply_accumulate(acc0, a_tile, b0, acc0);
            simdgroup_multiply_accumulate(acc1, a_tile, b1, acc1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (r0 + sg_row < M) {
        simdgroup_store(acc0, C + (ulong)(r0 + sg_row) * N + c0, N);
        simdgroup_store(acc1, C + (ulong)(r0 + sg_row) * N + c0 + 8, N);
    }
}
`
