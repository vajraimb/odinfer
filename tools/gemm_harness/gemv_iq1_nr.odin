// IQ1_S short-K: llama.cpp-style 32-wide K split, several rows sharing one
// x tile, with (gv+delta) factored into gv*x + delta*sum(x).
// n=5120, d=17408. Ten dispatches, min of 4.
package gemv_iq1_nr

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
	nib, nerr := os.read_entire_file_from_path("/tmp/iq1s_nib.metal", context.allocator)
	if nerr != nil { fmt.eprintln("nib table"); os.exit(1) }
	defer delete(nib)
	src, _ := strings.concatenate({
		"#include <metal_stdlib>\nusing namespace metal;\n",
		q35.MSL_TABLES,
		"\nconstant uint iq1s_nib[2048] = {\n",
		string(nib),
		"\n};\n",
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

	// rows covered per threadgroup: sc 32, nr2 16, nr4 32
	names := []string{"iq1_sc", "iq1_nr2", "iq1_nib2"}
	tgs := []int{D / 32, D / 16, D / 16}
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
	bytes := f64(D) * f64(N / 256) * 50
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
kernel void iq1_sc(
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

template<int NR>
void iq1_nr(
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
        float yl[32];
        float sumy = 0.0f;
        for (uint i = 0; i < 32; ++i) { yl[i] = x[yb + i]; sumy += yl[i]; }
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
            uint qv = qh;
            for (uint l = 0; l < 4; ++l) {
                uint idx = (uint)qs[l] | (((qv >> (3 * l)) & 7) << 8);
                ulong grid = iq1s_grid[idx];
                for (uint j = 0; j < 8; ++j) {
                    float gv = (float)((char)((grid >> (8 * j)) & 0xff));
                    sum += gv * yl[8 * l + j];
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

kernel void iq1_nr2(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    iq1_nr<2>(wb, x, out, tg, tiisg, sgitg);
}

kernel void iq1_nib2(
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
        float4 y0 = *(device const float4 *)(x + yb);
        float4 y1 = *(device const float4 *)(x + yb + 4);
        float4 y2 = *(device const float4 *)(x + yb + 8);
        float4 y3 = *(device const float4 *)(x + yb + 12);
        float4 y4 = *(device const float4 *)(x + yb + 16);
        float4 y5 = *(device const float4 *)(x + yb + 20);
        float4 y6 = *(device const float4 *)(x + yb + 24);
        float4 y7 = *(device const float4 *)(x + yb + 28);
        float sumy = y0[0]+y0[1]+y0[2]+y0[3]+y1[0]+y1[1]+y1[2]+y1[3]
                   + y2[0]+y2[1]+y2[2]+y2[3]+y3[0]+y3[1]+y3[2]+y3[3]
                   + y4[0]+y4[1]+y4[2]+y4[3]+y5[0]+y5[1]+y5[2]+y5[3]
                   + y6[0]+y6[1]+y6[2]+y6[3]+y7[0]+y7[1]+y7[2]+y7[3];
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
        float c0 = (h0 & 0x8000) ? -1.125f : -0.875f;
        float c1 = (h1 & 0x8000) ? -1.125f : -0.875f;
        float a0 = 0.0f, a1 = 0.0f;
        uint g;
        g = iq1s_nib[(uint)q0[0] | ((h0 & 7) << 8)];
        a0 += y0[0]*(g & 15) + y1[0]*((g >> 4) & 15);
        a0 += y0[1]*((g >> 8) & 15) + y1[1]*((g >> 12) & 15);
        a0 += y0[2]*((g >> 16) & 15) + y1[2]*((g >> 20) & 15);
        a0 += y0[3]*((g >> 24) & 15) + y1[3]*((g >> 28) & 15);
        g = iq1s_nib[(uint)q0[1] | (((h0 >> 3) & 7) << 8)];
        a0 += y2[0]*(g & 15) + y3[0]*((g >> 4) & 15);
        a0 += y2[1]*((g >> 8) & 15) + y3[1]*((g >> 12) & 15);
        a0 += y2[2]*((g >> 16) & 15) + y3[2]*((g >> 20) & 15);
        a0 += y2[3]*((g >> 24) & 15) + y3[3]*((g >> 28) & 15);
        g = iq1s_nib[(uint)q0[2] | (((h0 >> 6) & 7) << 8)];
        a0 += y4[0]*(g & 15) + y5[0]*((g >> 4) & 15);
        a0 += y4[1]*((g >> 8) & 15) + y5[1]*((g >> 12) & 15);
        a0 += y4[2]*((g >> 16) & 15) + y5[2]*((g >> 20) & 15);
        a0 += y4[3]*((g >> 24) & 15) + y5[3]*((g >> 28) & 15);
        g = iq1s_nib[(uint)q0[3] | (((h0 >> 9) & 7) << 8)];
        a0 += y6[0]*(g & 15) + y7[0]*((g >> 4) & 15);
        a0 += y6[1]*((g >> 8) & 15) + y7[1]*((g >> 12) & 15);
        a0 += y6[2]*((g >> 16) & 15) + y7[2]*((g >> 20) & 15);
        a0 += y6[3]*((g >> 24) & 15) + y7[3]*((g >> 28) & 15);
        g = iq1s_nib[(uint)q1[0] | ((h1 & 7) << 8)];
        a1 += y0[0]*(g & 15) + y1[0]*((g >> 4) & 15);
        a1 += y0[1]*((g >> 8) & 15) + y1[1]*((g >> 12) & 15);
        a1 += y0[2]*((g >> 16) & 15) + y1[2]*((g >> 20) & 15);
        a1 += y0[3]*((g >> 24) & 15) + y1[3]*((g >> 28) & 15);
        g = iq1s_nib[(uint)q1[1] | (((h1 >> 3) & 7) << 8)];
        a1 += y2[0]*(g & 15) + y3[0]*((g >> 4) & 15);
        a1 += y2[1]*((g >> 8) & 15) + y3[1]*((g >> 12) & 15);
        a1 += y2[2]*((g >> 16) & 15) + y3[2]*((g >> 20) & 15);
        a1 += y2[3]*((g >> 24) & 15) + y3[3]*((g >> 28) & 15);
        g = iq1s_nib[(uint)q1[2] | (((h1 >> 6) & 7) << 8)];
        a1 += y4[0]*(g & 15) + y5[0]*((g >> 4) & 15);
        a1 += y4[1]*((g >> 8) & 15) + y5[1]*((g >> 12) & 15);
        a1 += y4[2]*((g >> 16) & 15) + y5[2]*((g >> 20) & 15);
        a1 += y4[3]*((g >> 24) & 15) + y5[3]*((g >> 28) & 15);
        g = iq1s_nib[(uint)q1[3] | (((h1 >> 9) & 7) << 8)];
        a1 += y6[0]*(g & 15) + y7[0]*((g >> 4) & 15);
        a1 += y6[1]*((g >> 8) & 15) + y7[1]*((g >> 12) & 15);
        a1 += y6[2]*((g >> 16) & 15) + y7[2]*((g >> 20) & 15);
        a1 += y6[3]*((g >> 24) & 15) + y7[3]*((g >> 28) & 15);
        s0 += d0 * (a0 + sumy * c0) * sc0;
        s1 += d1 * (a1 + sumy * c1) * sc1;
    }
    s0 = simd_sum(s0);
    s1 = simd_sum(s1);
    if (tiisg == 0) { out[row0] = s0; out[row1] = s1; }
}

`
