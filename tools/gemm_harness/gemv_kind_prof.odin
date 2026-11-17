// Time the production GEMV kernels on the 27B decode shapes.
// One call is wall-clock around command-buffer wait. Totals multiply by how
// many times that shape runs in one token (ffn up+gate share a shape).
// Generic kernels, not the constexpr spec variants.
package gemv_kind_prof

import q35 "qwen3_5:qwen3_5"

import NS "core:sys/darwin/Foundation"
import "core:fmt"
import "core:os"
import "core:strings"
import "core:time"
import MTL "vendor:darwin/Metal"

Dims :: struct { n, d: u32 }

Job :: struct {
	kernel: string,
	n, d:   int,
	count:  int,
	block:  int, // bytes per 256 elements (IQ4_NL is unused here)
	label:  string,
}

JOBS := []Job{
	{"gemv_iq2_xxs", 5120, 17408, 45, 66, "iq2xxs up/gate 5120x17408"},
	{"gemv_iq2_xxs", 17408, 5120, 26, 66, "iq2xxs down 17408x5120"},
	{"gemv_iq1_s", 5120, 17408, 53, 50, "iq1s up/gate 5120x17408"},
	{"gemv_iq1_s", 17408, 5120, 20, 50, "iq1s down 17408x5120"},
	{"gemv_iq2_xxs", 5120, 10240, 21, 66, "iq2xxs qkv 5120x10240"},
	{"gemv_iq2_xxs", 5120, 6144, 29, 66, "iq2xxs gate 5120x6144"},
	{"gemv_iq2_xxs", 5120, 12288, 11, 66, "iq2xxs q 5120x12288"},
	{"gemv_iq2_s", 17408, 5120, 7, 82, "iq2s down 17408x5120"},
	{"gemv_iq2_s", 5120, 17408, 9, 82, "iq2s up/gate 5120x17408"},
	{"gemv_iq3_xxs", 17408, 5120, 5, 98, "iq3xxs down 17408x5120"},
	{"gemv_iq3_xxs", 5120, 17408, 8, 98, "iq3xxs up/gate 5120x17408"},
	{"gemv_iq2_xs", 5120, 17408, 5, 74, "iq2xs up 5120x17408"},
	{"gemv_iq3_xxs", 6144, 5120, 18, 98, "iq3xxs out 6144x5120"},
	{"gemv_iq3_s", 17408, 5120, 3, 110, "iq3s down 17408x5120"},
	{"gemv_iq1_s", 5120, 10240, 11, 50, "iq1s qkv 5120x10240"},
	{"gemv_iq2_xs", 5120, 10240, 7, 74, "iq2xs qkv 5120x10240"},
	{"gemv_iq2_s", 6144, 5120, 10, 82, "iq2s out 6144x5120"},
	{"gemv_iq2_xs", 6144, 5120, 11, 74, "iq2xs out 6144x5120"},
	{"gemv_q3_k", 5120, 248320, 1, 110, "q3k output 5120x248320"},
	{"gemv_iq2_xxs", 6144, 5120, 10, 66, "iq2xxs out 6144x5120"},
	{"gemv_iq1_m", 5120, 10240, 8, 56, "iq1m qkv 5120x10240"},
	{"gemv_iq1_m", 5120, 17408, 6, 56, "iq1m up/gate 5120x17408"},
	{"gemv_iq2_xs", 17408, 5120, 2, 74, "iq2xs down 17408x5120"},
	{"gemv_iq4_xs", 5120, 17408, 1, 136, "iq4xs up 5120x17408"},
	{"gemv_q3_k", 5120, 17408, 1, 110, "q3k gate 5120x17408"},
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

	src, _ := strings.concatenate({q35.MSL_SRC, V2})
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
		check_pair(dev, queue, lib, "gemv_iq2_xxs", "gemv_iq2_xxs_v2", 2048, 64, 66)
		check_pair(dev, queue, lib, "gemv_iq1_s", "gemv_iq1_s_v2", 2048, 64, 50)
		return
	}
	if os.get_env("SPEC", context.temp_allocator) == "1" {
		time_spec(dev, queue, lib, "gemv_iq2xxs_spec_f4", 5120, 17408, 66)
		time_spec(dev, queue, lib, "gemv_iq2xxs_spec_sc", 5120, 17408, 66)
		time_spec(dev, queue, lib, "gemv_iq1s_spec_f4", 5120, 17408, 50)
		time_spec(dev, queue, lib, "gemv_iq1s_spec_sc", 5120, 17408, 50)
		return
	}

	total: f64 = 0
	for j in JOBS {
		ms := time_job(dev, queue, lib, j)
		sum := ms * f64(j.count)
		total += sum
		bytes := f64(j.d) * f64(j.n / 256) * f64(j.block)
		gbps := bytes / (ms * 1e-3) / 1e9
		fmt.eprintfln("{:6.2f} ms x {:2d} = {:7.1f} ms/tok   {:5.1f} GB/s   {}", ms, j.count, sum, gbps, j.label)
	}
	fmt.eprintfln("sum of these GEMVs: {:.1f} ms/tok", total)
	fmt.eprintln("--- float4 variants ---")
	check_pair(dev, queue, lib, "gemv_iq2_xxs", "gemv_iq2_xxs_v2", 2048, 64, 66)
	check_pair(dev, queue, lib, "gemv_iq1_s", "gemv_iq1_s_v2", 2048, 64, 50)
	v2s := []Job{
		{"gemv_iq2_xxs_v2", 5120, 17408, 45, 66, "v2 iq2xxs up/gate"},
		{"gemv_iq2_xxs_v2", 17408, 5120, 26, 66, "v2 iq2xxs down"},
		{"gemv_iq1_s_v2", 5120, 17408, 53, 50, "v2 iq1s up/gate"},
		{"gemv_iq1_s_v2", 17408, 5120, 20, 50, "v2 iq1s down"},
	}
	for j in v2s {
		ms := time_job(dev, queue, lib, j)
		bytes := f64(j.d) * f64(j.n / 256) * f64(j.block)
		gbps := bytes / (ms * 1e-3) / 1e9
		fmt.eprintfln("{:6.2f} ms x {:2d} = {:7.1f} ms/tok   {:5.1f} GB/s   {}", ms, j.count, ms * f64(j.count), gbps, j.label)
	}
}

time_job :: proc(dev: ^MTL.Device, queue: ^MTL.CommandQueue, lib: ^MTL.Library, j: Job) -> f64 {
	ns_name := NS.String.alloc()->initWithOdinString(j.kernel)
	defer ns_name->release()
	fn := lib->newFunctionWithName(ns_name)
	if fn == nil {
		fmt.eprintfln("missing {}", j.kernel)
		os.exit(1)
	}
	defer fn->release()
	pso, perr := dev->newComputePipelineStateWithFunction(fn)
	if perr != nil {
		fmt.eprintfln("pso {}: {}", j.kernel, perr->localizedDescription()->odinString())
		os.exit(1)
	}
	defer pso->release()

	wbytes := j.d * (j.n / 256) * j.block
	wb := dev->newBufferWithLength(NS.UInteger(wbytes), MTL.ResourceStorageModeShared)
	xb := dev->newBufferWithLength(NS.UInteger(j.n * 4), MTL.ResourceStorageModeShared)
	ob := dev->newBufferWithLength(NS.UInteger(j.d * 4), MTL.ResourceStorageModeShared)
	defer wb->release()
	defer xb->release()
	defer ob->release()
	// Touch pages so the timed runs are not first-fault.
	ws := wb->contentsAsSlice([]u8)
	for i := 0; i < len(ws); i += 4096 {
		ws[i] = 1
	}
	xs := xb->contentsAsSlice([]f32)
	for i in 0 ..< len(xs) {
		xs[i] = 0.01
	}

	dims := Dims{u32(j.n), u32(j.d)}
	once :: proc(queue: ^MTL.CommandQueue, pso: ^MTL.ComputePipelineState, wb, xb, ob: ^MTL.Buffer, dims: Dims, d: int) {
		cmd := queue->commandBuffer()
		enc := cmd->computeCommandEncoder()
		enc->setComputePipelineState(pso)
		enc->setBuffer(wb, 0, 0)
		enc->setBuffer(xb, 0, 1)
		enc->setBuffer(ob, 0, 2)
		dims_local := dims
		enc->setBytes(([^]u8)(&dims_local)[:size_of(Dims)], 3)
		enc->dispatchThreadgroups(MTL.Size{NS.Integer(d / 32), 1, 1}, MTL.Size{256, 1, 1})
		enc->endEncoding()
		cmd->commit()
		cmd->waitUntilCompleted()
	}
	once(queue, pso, wb, xb, ob, dims, j.d)
	once(queue, pso, wb, xb, ob, dims, j.d)
	best: f64 = 1e18
	for _ in 0 ..< 4 {
		t0 := time.tick_now()
		once(queue, pso, wb, xb, ob, dims, j.d)
		ms := f64(time.tick_since(t0)) / 1e6
		if ms < best { best = ms }
	}
	return best
}

check_pair :: proc(dev: ^MTL.Device, queue: ^MTL.CommandQueue, lib: ^MTL.Library, a, b: string, n, d, block: int) {
	pa := make_pso(dev, lib, a)
	pb := make_pso(dev, lib, b)
	defer pa->release()
	defer pb->release()
	wbytes := d * (n / 256) * block
	wb := dev->newBufferWithLength(NS.UInteger(wbytes), MTL.ResourceStorageModeShared)
	xb := dev->newBufferWithLength(NS.UInteger(n * 4), MTL.ResourceStorageModeShared)
	oa := dev->newBufferWithLength(NS.UInteger(d * 4), MTL.ResourceStorageModeShared)
	ob := dev->newBufferWithLength(NS.UInteger(d * 4), MTL.ResourceStorageModeShared)
	defer wb->release(); defer xb->release(); defer oa->release(); defer ob->release()
	ws := wb->contentsAsSlice([]u8)
	for i in 0 ..< len(ws) { ws[i] = u8((i * 17 + 3) & 0xff) }
	// Modest f16 scale at the start of every superblock so a mismatch is not a huge-half artifact.
	nb := n / 256
	for row in 0 ..< d {
		for sb in 0 ..< nb {
			off := (row * nb + sb) * block
			ws[off] = 0x00
			ws[off + 1] = 0x3C
		}
	}
	xs := xb->contentsAsSlice([]f32)
	for i in 0 ..< len(xs) { xs[i] = f32(i % 13) * 0.01 - 0.05 }
	dims := Dims{u32(n), u32(d)}
	run :: proc(queue: ^MTL.CommandQueue, pso: ^MTL.ComputePipelineState, wb, xb, ob: ^MTL.Buffer, dims: Dims, d: int) {
		cmd := queue->commandBuffer()
		enc := cmd->computeCommandEncoder()
		enc->setComputePipelineState(pso)
		enc->setBuffer(wb, 0, 0); enc->setBuffer(xb, 0, 1); enc->setBuffer(ob, 0, 2)
		dims_local := dims
		enc->setBytes(([^]u8)(&dims_local)[:size_of(Dims)], 3)
		enc->dispatchThreadgroups(MTL.Size{NS.Integer(d / 32), 1, 1}, MTL.Size{256, 1, 1})
		enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted()
	}
	run(queue, pa, wb, xb, oa, dims, d)
	run(queue, pb, wb, xb, ob, dims, d)
	aa := oa->contentsAsSlice([]f32)[:d]
	bb := ob->contentsAsSlice([]f32)[:d]
	max_abs: f32 = 0
	max_rel: f32 = 0
	for i in 0 ..< d {
		df := aa[i] - bb[i]
		if df < 0 { df = -df }
		if df > max_abs { max_abs = df }
		den := aa[i]
		if den < 0 { den = -den }
		if den < 1e-6 { den = 1e-6 }
		rel := df / den
		if rel > max_rel { max_rel = rel }
	}
	fmt.eprintfln("check {} vs {} max_abs {} max_rel {} a0 {} b0 {}", a, b, max_abs, max_rel, aa[0], bb[0])
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

time_spec :: proc(dev: ^MTL.Device, queue: ^MTL.CommandQueue, lib: ^MTL.Library, name: string, n, d, block: int) {
	pso := make_pso(dev, lib, name)
	defer pso->release()
	wbytes := d * (n / 256) * block
	wb := dev->newBufferWithLength(NS.UInteger(wbytes), MTL.ResourceStorageModeShared)
	xb := dev->newBufferWithLength(NS.UInteger(n * 4), MTL.ResourceStorageModeShared)
	ob := dev->newBufferWithLength(NS.UInteger(d * 4), MTL.ResourceStorageModeShared)
	defer wb->release(); defer xb->release(); defer ob->release()
	ws := wb->contentsAsSlice([]u8)
	for i := 0; i < len(ws); i += 4096 { ws[i] = 1 }
	xs := xb->contentsAsSlice([]f32)
	for i in 0 ..< len(xs) { xs[i] = 0.01 }
	once :: proc(queue: ^MTL.CommandQueue, pso: ^MTL.ComputePipelineState, wb, xb, ob: ^MTL.Buffer, d: int) {
		cmd := queue->commandBuffer()
		enc := cmd->computeCommandEncoder()
		enc->setComputePipelineState(pso)
		enc->setBuffer(wb, 0, 0); enc->setBuffer(xb, 0, 1); enc->setBuffer(ob, 0, 2)
		enc->dispatchThreadgroups(MTL.Size{NS.Integer(d / 32), 1, 1}, MTL.Size{256, 1, 1})
		enc->endEncoding(); cmd->commit(); cmd->waitUntilCompleted()
	}
	once(queue, pso, wb, xb, ob, d)
	once(queue, pso, wb, xb, ob, d)
	best: f64 = 1e18
	for _ in 0 ..< 4 {
		t0 := time.tick_now()
		once(queue, pso, wb, xb, ob, d)
		ms := f64(time.tick_since(t0)) / 1e6
		if ms < best { best = ms }
	}
	bytes := f64(d) * f64(n / 256) * f64(block)
	fmt.eprintfln("{:6.2f} ms   {:5.1f} GB/s   {}", best, bytes / (best * 1e-3) / 1e9, name)
}

V2 :: `
kernel void gemv_iq2_xxs_v2(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant Dims &dim [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    uint row = tg * GEMV_ROWS + tid / GEMV_SG; uint lane = tid % GEMV_SG;
    float acc0 = 0.0f, acc1 = 0.0f;
    if (row < dim.d) { uint nsb = dim.n / 256; device const uchar *wr = wb + (ulong)row * nsb * 66;
        for (uint sb = lane; sb < nsb; sb += GEMV_SG) { device const uchar *blk = wr + sb * 66;
            float dscale = (float)(*(device const half*)blk);
            uint ybase = sb * 256;
            for (uint ib32 = 0; ib32 < 8; ib32++) {
                uint qb = 2 + 8 * ib32;
                uint aux1 = (uint)blk[qb + 4] | ((uint)blk[qb + 5] << 8) | ((uint)blk[qb + 6] << 16) | ((uint)blk[qb + 7] << 24);
                float db = dscale * (0.5f + (float)(aux1 >> 28)) * 0.25f;
                for (uint l = 0; l < 4; l++) {
                    ulong grid = iq2xxs_grid[blk[qb + l]];
                    uchar signs = ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
                    float4 sg0 = float4(
                        1.0f - 2.0f * float((signs & kmask_iq2xs[0]) != 0),
                        1.0f - 2.0f * float((signs & kmask_iq2xs[1]) != 0),
                        1.0f - 2.0f * float((signs & kmask_iq2xs[2]) != 0),
                        1.0f - 2.0f * float((signs & kmask_iq2xs[3]) != 0));
                    float4 sg1 = float4(
                        1.0f - 2.0f * float((signs & kmask_iq2xs[4]) != 0),
                        1.0f - 2.0f * float((signs & kmask_iq2xs[5]) != 0),
                        1.0f - 2.0f * float((signs & kmask_iq2xs[6]) != 0),
                        1.0f - 2.0f * float((signs & kmask_iq2xs[7]) != 0));
                    float4 gv0 = float4((float)((grid >> 0) & 0xff), (float)((grid >> 8) & 0xff), (float)((grid >> 16) & 0xff), (float)((grid >> 24) & 0xff));
                    float4 gv1 = float4((float)((grid >> 32) & 0xff), (float)((grid >> 40) & 0xff), (float)((grid >> 48) & 0xff), (float)((grid >> 56) & 0xff));
                    float4 x0 = float4(x[ybase], x[ybase+1], x[ybase+2], x[ybase+3]);
                    float4 x1 = float4(x[ybase+4], x[ybase+5], x[ybase+6], x[ybase+7]);
                    acc0 += db * dot(gv0 * sg0, x0);
                    acc1 += db * dot(gv1 * sg1, x1);
                    ybase += 8;
                }
            } } }
    float sum = acc0 + acc1;
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane % 8 == 0 && row < dim.d) out[row] = sum;
}

kernel void gemv_iq1_s_v2(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant Dims &dim [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    uint row = tg * GEMV_ROWS + tid / GEMV_SG; uint lane = tid % GEMV_SG;
    float acc0 = 0.0f, acc1 = 0.0f;
    if (row < dim.d) { uint nsb = dim.n / 256; device const uchar *wr = wb + (ulong)row * nsb * 50;
        for (uint sb = lane; sb < nsb; sb += GEMV_SG) { device const uchar *blk = wr + sb * 50;
            float dscale = (float)(*(device const half*)blk);
            device const uchar *qs = blk + 2; device const uchar *qh = blk + 34;
            uint ybase = sb * 256; uint qoff = 0;
            for (uint ib = 0; ib < 8; ib++) {
                uint qhv = (uint)qh[2 * ib] | ((uint)qh[2 * ib + 1] << 8);
                float dl = dscale * (float)(2 * (int)((qhv >> 12) & 7) + 1);
                float delta = (qhv & 0x8000) == 0 ? 0.125f : -0.125f;
                for (uint l = 0; l < 4; l++) {
                    uint idx = (uint)qs[qoff + l] | (((qhv >> (3 * l)) & 7) << 8);
                    ulong grid = iq1s_grid[idx];
                    float4 gv0 = float4(
                        (float)(char)((grid >> 0) & 0xff), (float)(char)((grid >> 8) & 0xff),
                        (float)(char)((grid >> 16) & 0xff), (float)(char)((grid >> 24) & 0xff));
                    float4 gv1 = float4(
                        (float)(char)((grid >> 32) & 0xff), (float)(char)((grid >> 40) & 0xff),
                        (float)(char)((grid >> 48) & 0xff), (float)(char)((grid >> 56) & 0xff));
                    float4 x0 = float4(x[ybase], x[ybase+1], x[ybase+2], x[ybase+3]);
                    float4 x1 = float4(x[ybase+4], x[ybase+5], x[ybase+6], x[ybase+7]);
                    acc0 += dl * dot(gv0 + delta, x0);
                    acc1 += dl * dot(gv1 + delta, x1);
                    ybase += 8;
                }
                qoff += 4;
            } } }
    float sum = acc0 + acc1;
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane % 8 == 0 && row < dim.d) out[row] = sum;
}

kernel void gemv_iq2xxs_spec_f4(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint n = 5120; constexpr uint d = 17408; constexpr uint nsb = n / 256;
    uint row = tg * GEMV_ROWS + tid / GEMV_SG; uint lane = tid % GEMV_SG;
    float acc0 = 0.0f, acc1 = 0.0f;
    device const uchar *wr = wb + (ulong)row * nsb * 66;
    for (uint sb = lane; sb < nsb; sb += GEMV_SG) { device const uchar *blk = wr + sb * 66;
        float dscale = (float)(*(device const half*)blk);
        uint ybase = sb * 256;
        for (uint ib32 = 0; ib32 < 8; ib32++) {
            uint qb = 2 + 8 * ib32;
            uint aux1 = (uint)blk[qb + 4] | ((uint)blk[qb + 5] << 8) | ((uint)blk[qb + 6] << 16) | ((uint)blk[qb + 7] << 24);
            float db = dscale * (0.5f + (float)(aux1 >> 28)) * 0.25f;
            for (uint l = 0; l < 4; l++) {
                ulong grid = iq2xxs_grid[blk[qb + l]];
                uchar signs = ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
                float4 sg0 = float4(
                    1.0f - 2.0f * float((signs & kmask_iq2xs[0]) != 0),
                    1.0f - 2.0f * float((signs & kmask_iq2xs[1]) != 0),
                    1.0f - 2.0f * float((signs & kmask_iq2xs[2]) != 0),
                    1.0f - 2.0f * float((signs & kmask_iq2xs[3]) != 0));
                float4 sg1 = float4(
                    1.0f - 2.0f * float((signs & kmask_iq2xs[4]) != 0),
                    1.0f - 2.0f * float((signs & kmask_iq2xs[5]) != 0),
                    1.0f - 2.0f * float((signs & kmask_iq2xs[6]) != 0),
                    1.0f - 2.0f * float((signs & kmask_iq2xs[7]) != 0));
                float4 gv0 = float4((float)((grid >> 0) & 0xff), (float)((grid >> 8) & 0xff), (float)((grid >> 16) & 0xff), (float)((grid >> 24) & 0xff));
                float4 gv1 = float4((float)((grid >> 32) & 0xff), (float)((grid >> 40) & 0xff), (float)((grid >> 48) & 0xff), (float)((grid >> 56) & 0xff));
                float4 x0 = float4(x[ybase], x[ybase+1], x[ybase+2], x[ybase+3]);
                float4 x1 = float4(x[ybase+4], x[ybase+5], x[ybase+6], x[ybase+7]);
                acc0 += db * dot(gv0 * sg0, x0);
                acc1 += db * dot(gv1 * sg1, x1);
                ybase += 8;
            }
        }
    }
    float sum = acc0 + acc1;
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane % 8 == 0) out[row] = sum;
}

kernel void gemv_iq2xxs_spec_sc(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint n = 5120; constexpr uint d = 17408; constexpr uint nsb = n / 256;
    uint row = tg * GEMV_ROWS + tid / GEMV_SG; uint lane = tid % GEMV_SG; float sum = 0.0;
    device const uchar *wr = wb + (ulong)row * nsb * 66;
    for (uint sb = lane; sb < nsb; sb += GEMV_SG) { device const uchar *blk = wr + sb * 66;
        float dscale = (float)(*(device const half*)blk);
        uint ybase = sb * 256;
        for (uint ib32 = 0; ib32 < 8; ib32++) {
            uint qb = 2 + 8 * ib32;
            uint aux1 = (uint)blk[qb + 4] | ((uint)blk[qb + 5] << 8) | ((uint)blk[qb + 6] << 16) | ((uint)blk[qb + 7] << 24);
            float db = dscale * (0.5f + (float)(aux1 >> 28)) * 0.25f;
            for (uint l = 0; l < 4; l++) {
                ulong grid = iq2xxs_grid[blk[qb + l]];
                uchar signs = ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
                for (uint j = 0; j < 8; j++) {
                    float v = (float)((grid >> (8 * j)) & 0xff);
                    float s = (signs & kmask_iq2xs[j]) == 0 ? 1.0f : -1.0f;
                    sum += db * v * s * x[ybase + j];
                }
                ybase += 8;
            }
        }
    }
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane % 8 == 0) out[row] = sum;
}

kernel void gemv_iq1s_spec_f4(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint n = 5120; constexpr uint d = 17408; constexpr uint nsb = n / 256;
    uint row = tg * GEMV_ROWS + tid / GEMV_SG; uint lane = tid % GEMV_SG;
    float acc0 = 0.0f, acc1 = 0.0f;
    device const uchar *wr = wb + (ulong)row * nsb * 50;
    for (uint sb = lane; sb < nsb; sb += GEMV_SG) { device const uchar *blk = wr + sb * 50;
        float dscale = (float)(*(device const half*)blk);
        device const uchar *qs = blk + 2; device const uchar *qh = blk + 34;
        uint ybase = sb * 256; uint qoff = 0;
        for (uint ib = 0; ib < 8; ib++) {
            uint qhv = (uint)qh[2 * ib] | ((uint)qh[2 * ib + 1] << 8);
            float dl = dscale * (float)(2 * (int)((qhv >> 12) & 7) + 1);
            float delta = (qhv & 0x8000) == 0 ? 0.125f : -0.125f;
            for (uint l = 0; l < 4; l++) {
                uint idx = (uint)qs[qoff + l] | (((qhv >> (3 * l)) & 7) << 8);
                ulong grid = iq1s_grid[idx];
                float4 gv0 = float4(
                    (float)(char)((grid >> 0) & 0xff), (float)(char)((grid >> 8) & 0xff),
                    (float)(char)((grid >> 16) & 0xff), (float)(char)((grid >> 24) & 0xff));
                float4 gv1 = float4(
                    (float)(char)((grid >> 32) & 0xff), (float)(char)((grid >> 40) & 0xff),
                    (float)(char)((grid >> 48) & 0xff), (float)(char)((grid >> 56) & 0xff));
                float4 x0 = float4(x[ybase], x[ybase+1], x[ybase+2], x[ybase+3]);
                float4 x1 = float4(x[ybase+4], x[ybase+5], x[ybase+6], x[ybase+7]);
                acc0 += dl * dot(gv0 + delta, x0);
                acc1 += dl * dot(gv1 + delta, x1);
                ybase += 8;
            }
            qoff += 4;
        }
    }
    float sum = acc0 + acc1;
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane % 8 == 0) out[row] = sum;
}

kernel void gemv_iq1s_spec_sc(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint n = 5120; constexpr uint d = 17408; constexpr uint nsb = n / 256;
    uint row = tg * GEMV_ROWS + tid / GEMV_SG; uint lane = tid % GEMV_SG; float sum = 0.0;
    device const uchar *wr = wb + (ulong)row * nsb * 50;
    for (uint sb = lane; sb < nsb; sb += GEMV_SG) { device const uchar *blk = wr + sb * 50;
        float dscale = (float)(*(device const half*)blk);
        device const uchar *qs = blk + 2; device const uchar *qh = blk + 34;
        uint ybase = sb * 256; uint qoff = 0;
        for (uint ib = 0; ib < 8; ib++) {
            uint qhv = (uint)qh[2 * ib] | ((uint)qh[2 * ib + 1] << 8);
            float dl = dscale * (float)(2 * (int)((qhv >> 12) & 7) + 1);
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
    if (lane % 8 == 0) out[row] = sum;
}
`
