// IQ1_S short-K. Production 2-row K-split holds yl[32] in registers.
// These variants drop that array: float4 x for 2 rows, and 4/8 rows that
// reload x so the accumulators fit. Compiled next to the production library.
// n=5120, d=17408.
package gemv_iq1_rows

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

	wb := dev->newBufferWithLength(NS.UInteger(D * (N / 256) * 50), MTL.ResourceStorageModeShared)
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

	names := []string{"gemv_iq1_s_ks", "iq1_ilp", "iq1_r4", "iq1_r8"}
	tgs := []int{D / 16, D / 16, D / 32, D / 64}
	bytes := f64(D * (N / 256) * 50)
	psos := make([]^MTL.ComputePipelineState, len(names))
	for name, i in names { psos[i] = make_pso(dev, lib, name) }

	run(queue, psos[0], wb, xb, ob, tgs[0])
	ref := make([]f32, D)
	copy(ref, ob->contentsAsSlice([]f32)[:D])
	for name, i in names[1:] {
		check(queue, psos[i + 1], wb, xb, ob, tgs[i + 1], ref, name)
	}
	delete(ref)
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

check :: proc(queue: ^MTL.CommandQueue, pso: ^MTL.ComputePipelineState, wb, xb, ob: ^MTL.Buffer, groups: int, ref: []f32, name: string) {
	out_s := ob->contentsAsSlice([]f32)[:D]
	for k in 0 ..< D { out_s[k] = 1e9 }
	run(queue, pso, wb, xb, ob, groups)
	got := ob->contentsAsSlice([]f32)[:D]
	max_rel: f32 = 0
	max_abs: f32 = 0
	miss := 0
	for k in 0 ..< D {
		if got[k] == 1e9 { miss += 1; continue }
		df := got[k] - ref[k]
		if df < 0 { df = -df }
		if df > max_abs { max_abs = df }
		den := ref[k]
		if den < 0 { den = -den }
		if den < 1e-6 { den = 1e-6 }
		rel := df / den
		if rel > max_rel { max_rel = rel }
	}
	fmt.eprintfln("check {} max_rel {} max_abs {} unwritten {}", name, max_rel, max_abs, miss)
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
__attribute__((always_inline)) float iq1_dot8(ulong grid, thread const float *yl) {
    uint lo = (uint)grid;
    uint hi = (uint)(grid >> 32);
    uchar4 a = as_type<uchar4>(lo);
    uchar4 b = as_type<uchar4>(hi);
    return (float)((char)a[0]) * yl[0] + (float)((char)a[1]) * yl[1]
         + (float)((char)a[2]) * yl[2] + (float)((char)a[3]) * yl[3]
         + (float)((char)b[0]) * yl[4] + (float)((char)b[1]) * yl[5]
         + (float)((char)b[2]) * yl[6] + (float)((char)b[3]) * yl[7];
}

kernel void iq1_ilp(
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
        float y0 = x[yb+0],  y1 = x[yb+1],  y2 = x[yb+2],  y3 = x[yb+3];
        float y4 = x[yb+4],  y5 = x[yb+5],  y6 = x[yb+6],  y7 = x[yb+7];
        float y8 = x[yb+8],  y9 = x[yb+9],  y10= x[yb+10], y11= x[yb+11];
        float y12= x[yb+12], y13= x[yb+13], y14= x[yb+14], y15= x[yb+15];
        float y16= x[yb+16], y17= x[yb+17], y18= x[yb+18], y19= x[yb+19];
        float y20= x[yb+20], y21= x[yb+21], y22= x[yb+22], y23= x[yb+23];
        float y24= x[yb+24], y25= x[yb+25], y26= x[yb+26], y27= x[yb+27];
        float y28= x[yb+28], y29= x[yb+29], y30= x[yb+30], y31= x[yb+31];
        float sumy = y0+y1+y2+y3+y4+y5+y6+y7+y8+y9+y10+y11+y12+y13+y14+y15
                   + y16+y17+y18+y19+y20+y21+y22+y23+y24+y25+y26+y27+y28+y29+y30+y31;
        float yl0[8] = {y0,y1,y2,y3,y4,y5,y6,y7};
        float yl1[8] = {y8,y9,y10,y11,y12,y13,y14,y15};
        float yl2[8] = {y16,y17,y18,y19,y20,y21,y22,y23};
        float yl3[8] = {y24,y25,y26,y27,y28,y29,y30,y31};
        uint ibl = ib32 >> 3;
        uint ib = ib32 & 7;
        device const uchar *b0 = wb + ((ulong)row0 * 20u + ibl) * 50;
        device const uchar *b1 = wb + ((ulong)row1 * 20u + ibl) * 50;
        float d0 = (float)(*(device const half *)b0);
        float d1 = (float)(*(device const half *)b1);
        device const uchar *q0 = b0 + 2 + 4 * ib;
        device const uchar *q1 = b1 + 2 + 4 * ib;
        uint h0 = (uint)b0[34 + 2 * ib] | ((uint)b0[35 + 2 * ib] << 8);
        uint h1 = (uint)b1[34 + 2 * ib] | ((uint)b1[35 + 2 * ib] << 8);
        float sc0 = (float)(2 * (int)((h0 >> 12) & 7) + 1);
        float sc1 = (float)(2 * (int)((h1 >> 12) & 7) + 1);
        float del0 = (h0 & 0x8000) == 0 ? 0.125f : -0.125f;
        float del1 = (h1 & 0x8000) == 0 ? 0.125f : -0.125f;
        ulong g00 = iq1s_grid[(uint)q0[0] | (((h0 >> 0) & 7) << 8)];
        ulong g01 = iq1s_grid[(uint)q0[1] | (((h0 >> 3) & 7) << 8)];
        ulong g02 = iq1s_grid[(uint)q0[2] | (((h0 >> 6) & 7) << 8)];
        ulong g03 = iq1s_grid[(uint)q0[3] | (((h0 >> 9) & 7) << 8)];
        ulong g10 = iq1s_grid[(uint)q1[0] | (((h1 >> 0) & 7) << 8)];
        ulong g11 = iq1s_grid[(uint)q1[1] | (((h1 >> 3) & 7) << 8)];
        ulong g12 = iq1s_grid[(uint)q1[2] | (((h1 >> 6) & 7) << 8)];
        ulong g13 = iq1s_grid[(uint)q1[3] | (((h1 >> 9) & 7) << 8)];
        float a0 = iq1_dot8(g00, yl0) + iq1_dot8(g01, yl1) + iq1_dot8(g02, yl2) + iq1_dot8(g03, yl3);
        float a1 = iq1_dot8(g10, yl0) + iq1_dot8(g11, yl1) + iq1_dot8(g12, yl2) + iq1_dot8(g13, yl3);
        s0 += d0 * (a0 + del0 * sumy) * sc0;
        s1 += d1 * (a1 + del1 * sumy) * sc1;
    }
    s0 = simd_sum(s0);
    s1 = simd_sum(s1);
    if (tiisg == 0) { out[row0] = s0; out[row1] = s1; }
}

template<int NR>
__attribute__((always_inline)) void iq1_rn(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint nb32 = 160;
    uint rowbase = (tg * 8u + sgitg) * NR;
    float sumf[NR] = {0.0f};
    for (uint ib32 = tiisg; ib32 < nb32; ib32 += 32) {
        uint yb = ib32 * 32;
        volatile device const float *xv = x + yb;
        float sumy = 0.0f;
        for (uint i = 0; i < 32; ++i) sumy += xv[i];
        uint ibl = ib32 >> 3;
        uint ib = ib32 & 7;
        for (uint row = 0; row < NR; ++row) {
            device const uchar *blk = wb + ((ulong)(rowbase + row) * 20u + ibl) * 50;
            float d = (float)(*(device const half*)blk);
            device const uchar *qs = blk + 2 + 4 * ib;
            uint qh = (uint)blk[34 + 2 * ib] | ((uint)blk[35 + 2 * ib] << 8);
            float scale = (float)(2 * (int)((qh >> 12) & 7) + 1);
            float delta = (qh & 0x8000) == 0 ? 0.125f : -0.125f;
            float sum = 0.0f;
            for (uint l = 0; l < 4; ++l) {
                uint idx = (uint)qs[l] | (((qh >> (3 * l)) & 7) << 8);
                ulong grid = iq1s_grid[idx];
                uint off = 8 * l;
                for (uint j = 0; j < 8; ++j) {
                    float gv = (float)((char)((grid >> (8 * j)) & 0xff));
                    sum += gv * xv[off + j];
                }
            }
            sumf[row] += d * (sum + delta * sumy) * scale;
        }
    }
    for (uint row = 0; row < NR; ++row) {
        float all = simd_sum(sumf[row]);
        if (tiisg == 0) out[rowbase + row] = all;
    }
}

kernel void iq1_r4(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    iq1_rn<4>(wb, x, out, tg, tiisg, sgitg);
}

kernel void iq1_r8(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    iq1_rn<8>(wb, x, out, tg, tiisg, sgitg);
}
`
