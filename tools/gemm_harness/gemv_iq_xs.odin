// IQ2_XS and IQ1_M, n=5120 d=17408. Spec-style scalar vs simdgroup K-split,
// compiled next to the production Metal library.
package gemv_iq_xs

import q35 "qwen3_5:qwen3_5"

import NS "core:sys/darwin/Foundation"
import "core:fmt"
import "core:os"
import "core:strings"
import "core:time"
import MTL "vendor:darwin/Metal"

N :: 5120
D :: 17408
NSB :: 20
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

	wxs := dev->newBufferWithLength(NS.UInteger(D * NSB * 74), MTL.ResourceStorageModeShared)
	wm := dev->newBufferWithLength(NS.UInteger(D * NSB * 56), MTL.ResourceStorageModeShared)
	xb := dev->newBufferWithLength(NS.UInteger(N * 4), MTL.ResourceStorageModeShared)
	ob := dev->newBufferWithLength(NS.UInteger(D * 4), MTL.ResourceStorageModeShared)
	defer wxs->release(); defer wm->release(); defer xb->release(); defer ob->release()
	fill_half(wxs->contentsAsSlice([]u8), 74)
	fill_raw(wm->contentsAsSlice([]u8))
	xs := xb->contentsAsSlice([]f32)
	for i in 0 ..< len(xs) { xs[i] = f32(i % 13) * 0.01 - 0.05 }

	names := []string{"iq2xs_sc", "iq2xs_ks", "iq1m_sc", "iq1m_ks"}
	tgs := []int{D / 32, D / 16, D / 32, D / 16}
	wbs := []^MTL.Buffer{wxs, wxs, wm, wm}
	bytes := []f64{f64(D * NSB * 74), f64(D * NSB * 74), f64(D * NSB * 56), f64(D * NSB * 56)}
	psos := make([]^MTL.ComputePipelineState, len(names))
	for name, i in names { psos[i] = make_pso(dev, lib, name) }

	run(queue, psos[0], wxs, xb, ob, tgs[0])
	ref2 := make([]f32, D)
	copy(ref2, ob->contentsAsSlice([]f32)[:D])
	check(queue, psos[1], wxs, xb, ob, tgs[1], ref2, names[1])
	delete(ref2)

	run(queue, psos[2], wm, xb, ob, tgs[2])
	ref1 := make([]f32, D)
	copy(ref1, ob->contentsAsSlice([]f32)[:D])
	check(queue, psos[3], wm, xb, ob, tgs[3], ref1, names[3])
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

fill_half :: proc(ws: []u8, blk: int) {
	for i in 0 ..< len(ws) { ws[i] = u8((i * 17 + 3) & 0xff) }
	for b in 0 ..< len(ws) / blk {
		ws[b * blk + 0] = 0x00
		ws[b * blk + 1] = 0x3C
	}
}

fill_raw :: proc(ws: []u8) {
	for i in 0 ..< len(ws) { ws[i] = u8((i * 17 + 3) & 0xff) }
}

check :: proc(queue: ^MTL.CommandQueue, pso: ^MTL.ComputePipelineState, wb, xb, ob: ^MTL.Buffer, groups: int, ref: []f32, name: string) {
	out_s := ob->contentsAsSlice([]f32)[:D]
	for k in 0 ..< D { out_s[k] = 1e9 }
	run(queue, pso, wb, xb, ob, groups)
	got := ob->contentsAsSlice([]f32)[:D]
	max_rel: f32 = 0
	max_abs: f32 = 0
	worst: f32 = 0
	gotw: f32 = 0
	miss := 0
	for k in 0 ..< D {
		if got[k] == 1e9 { miss += 1; continue }
		df := got[k] - ref[k]
		if df < 0 { df = -df }
		if df > max_abs {
			max_abs = df
			worst = ref[k]
			gotw = got[k]
		}
		den := ref[k]
		if den < 0 { den = -den }
		if den < 1e-6 { den = 1e-6 }
		rel := df / den
		if rel > max_rel { max_rel = rel }
	}
	fmt.eprintfln("check {} max_rel {} max_abs {} ref {} got {} unwritten {}", name, max_rel, max_abs, worst, gotw, miss)
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
kernel void iq2xs_sc(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 20;
    uint row = tg * 32 + tid / 8; uint lane = tid % 8; float sum = 0.0f;
    device const uchar *wr = wb + (ulong)row * nsb * 74;
    for (uint sb = lane; sb < nsb; sb += 8) {
        device const uchar *blk = wr + sb * 74;
        float d = (float)(*(device const half*)blk);
        uint ybase = sb * 256;
        for (uint ib32 = 0; ib32 < 8; ib32++) {
            uchar sc = blk[66 + ib32];
            float db0 = d * (0.5f + (float)(sc & 0xF)) * 0.25f;
            float db1 = d * (0.5f + (float)(sc >> 4)) * 0.25f;
            for (uint l = 0; l < 4; l++) {
                uint qoff = 2 + 2 * (4 * ib32 + l);
                uint q = (uint)blk[qoff] | ((uint)blk[qoff + 1] << 8);
                ulong grid = iq2xs_grid[q & 511];
                uchar signs = ksigns_iq2xs[q >> 9];
                float dl = (l < 2) ? db0 : db1;
                for (uint j = 0; j < 8; j++) {
                    float v = (float)((grid >> (8 * j)) & 0xff);
                    float s = (signs & kmask_iq2xs[j]) == 0 ? 1.0f : -1.0f;
                    sum += dl * v * s * x[ybase + j];
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

kernel void iq2xs_ks(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint nsb = 20;
    constexpr uint nb32 = 160;
    uint rowbase = (tg * 8u + sgitg) * 2u;
    float sumf[2] = {0.0f, 0.0f};
    for (uint ib32 = tiisg; ib32 < nb32; ib32 += 32) {
        uint yb = ib32 * 32;
        float yl[32];
        for (uint i = 0; i < 32; ++i) yl[i] = x[yb + i];
        uint ibl = ib32 >> 3;
        uint ib = ib32 & 7;
        for (uint row = 0; row < 2; ++row) {
            device const uchar *blk = wb + ((ulong)(rowbase + row) * nsb + ibl) * 74;
            float d = (float)(*(device const half*)blk);
            uchar scv = blk[66 + ib];
            float db0 = d * (0.5f + (float)(scv & 0xF)) * 0.25f;
            float db1 = d * (0.5f + (float)(scv >> 4)) * 0.25f;
            float lo = 0.0f, hi = 0.0f;
            for (uint l = 0; l < 4; ++l) {
                uint qoff = 2 + 2 * (4 * ib + l);
                uint q = (uint)blk[qoff] | ((uint)blk[qoff + 1] << 8);
                ulong grid = iq2xs_grid[q & 511];
                uchar signs = ksigns_iq2xs[q >> 9];
                float acc = 0.0f;
                for (uint j = 0; j < 8; ++j) {
                    float v = (float)((grid >> (8 * j)) & 0xff);
                    float s = (signs & kmask_iq2xs[j]) == 0 ? 1.0f : -1.0f;
                    acc += v * s * yl[8 * l + j];
                }
                if (l < 2) lo += acc; else hi += acc;
            }
            sumf[row] += db0 * lo + db1 * hi;
        }
    }
    for (uint row = 0; row < 2; ++row) {
        float all = simd_sum(sumf[row]);
        if (tiisg == 0) out[rowbase + row] = all;
    }
}

kernel void iq1m_sc(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 20;
    uint row = tg * 32 + tid / 8; uint lane = tid % 8; float sum = 0.0f;
    device const uchar *wr = wb + (ulong)row * nsb * 56;
    for (uint sb = lane; sb < nsb; sb += 8) {
        device const uchar *blk = wr + sb * 56;
        device const uchar *qs = blk;
        device const uchar *qh = blk + 32;
        device const uchar *scales = blk + 48;
        uint sc0 = (uint)scales[0] | ((uint)scales[1] << 8);
        uint sc1 = (uint)scales[2] | ((uint)scales[3] << 8);
        uint sc2 = (uint)scales[4] | ((uint)scales[5] << 8);
        uint sc3 = (uint)scales[6] | ((uint)scales[7] << 8);
        uint scale = (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000);
        float d = half_to_float((ushort)scale);
        uint ybase = sb * 256; uint qoff = 0; uint qhoff = 0;
        uint sc[4] = {sc0, sc1, sc2, sc3};
        for (uint ib = 0; ib < 8; ib++) {
            uint scv = (ib & 1) == 0 ? sc[ib / 2] : (sc[ib / 2] >> 6);
            float dl1 = d * (float)(2 * (int)((scv >> 0) & 0x7) + 1);
            float dl2 = d * (float)(2 * (int)((scv >> 3) & 0x7) + 1);
            uint idx0 = (uint)qs[qoff + 0] | (((uint)qh[qhoff + 0] << 8) & 0x700);
            uint idx1 = (uint)qs[qoff + 1] | (((uint)qh[qhoff + 0] << 4) & 0x700);
            uint idx2 = (uint)qs[qoff + 2] | (((uint)qh[qhoff + 1] << 8) & 0x700);
            uint idx3 = (uint)qs[qoff + 3] | (((uint)qh[qhoff + 1] << 4) & 0x700);
            float del0 = (qh[qhoff + 0] & 0x08) == 0 ? 0.125f : -0.125f;
            float del1 = (qh[qhoff + 0] & 0x80) == 0 ? 0.125f : -0.125f;
            float del2 = (qh[qhoff + 1] & 0x08) == 0 ? 0.125f : -0.125f;
            float del3 = (qh[qhoff + 1] & 0x80) == 0 ? 0.125f : -0.125f;
            for (uint j = 0; j < 8; j++) {
                ulong g0 = iq1s_grid[idx0]; float gv = (float)((char)((g0 >> (8 * j)) & 0xff));
                sum += dl1 * (gv + del0) * x[ybase + j];
                ulong g1 = iq1s_grid[idx1]; gv = (float)((char)((g1 >> (8 * j)) & 0xff));
                sum += dl1 * (gv + del1) * x[ybase + 8 + j];
                ulong g2 = iq1s_grid[idx2]; gv = (float)((char)((g2 >> (8 * j)) & 0xff));
                sum += dl2 * (gv + del2) * x[ybase + 16 + j];
                ulong g3 = iq1s_grid[idx3]; gv = (float)((char)((g3 >> (8 * j)) & 0xff));
                sum += dl2 * (gv + del3) * x[ybase + 24 + j];
            }
            ybase += 32; qoff += 4; qhoff += 2;
        }
    }
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane == 0) out[row] = sum;
}

kernel void iq1m_ks(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint nsb = 20;
    constexpr uint nb32 = 160;
    uint rowbase = (tg * 8u + sgitg) * 2u;
    float sumf[2] = {0.0f, 0.0f};
    for (uint ib32 = tiisg; ib32 < nb32; ib32 += 32) {
        uint yb = ib32 * 32;
        float yl[32];
        for (uint i = 0; i < 32; ++i) yl[i] = x[yb + i];
        uint ibl = ib32 >> 3;
        uint ib = ib32 & 7;
        for (uint row = 0; row < 2; ++row) {
            device const uchar *blk = wb + ((ulong)(rowbase + row) * nsb + ibl) * 56;
            device const uchar *qs = blk + 4 * ib;
            device const uchar *qh = blk + 32 + 2 * ib;
            device const uchar *scales = blk + 48;
            uint sc0 = (uint)scales[0] | ((uint)scales[1] << 8);
            uint sc1 = (uint)scales[2] | ((uint)scales[3] << 8);
            uint sc2 = (uint)scales[4] | ((uint)scales[5] << 8);
            uint sc3 = (uint)scales[6] | ((uint)scales[7] << 8);
            uint scale = (sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000);
            float d = half_to_float((ushort)scale);
            uint scv = (ib & 1) == 0 ? ((ib < 2) ? sc0 : (ib < 4) ? sc1 : (ib < 6) ? sc2 : sc3)
                                     : (((ib < 2) ? sc0 : (ib < 4) ? sc1 : (ib < 6) ? sc2 : sc3) >> 6);
            float dl1 = d * (float)(2 * (int)((scv >> 0) & 0x7) + 1);
            float dl2 = d * (float)(2 * (int)((scv >> 3) & 0x7) + 1);
            uint idx0 = (uint)qs[0] | (((uint)qh[0] << 8) & 0x700);
            uint idx1 = (uint)qs[1] | (((uint)qh[0] << 4) & 0x700);
            uint idx2 = (uint)qs[2] | (((uint)qh[1] << 8) & 0x700);
            uint idx3 = (uint)qs[3] | (((uint)qh[1] << 4) & 0x700);
            float del0 = (qh[0] & 0x08) == 0 ? 0.125f : -0.125f;
            float del1 = (qh[0] & 0x80) == 0 ? 0.125f : -0.125f;
            float del2 = (qh[1] & 0x08) == 0 ? 0.125f : -0.125f;
            float del3 = (qh[1] & 0x80) == 0 ? 0.125f : -0.125f;
            ulong g0 = iq1s_grid[idx0];
            ulong g1 = iq1s_grid[idx1];
            ulong g2 = iq1s_grid[idx2];
            ulong g3 = iq1s_grid[idx3];
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            float y0 = 0.0f, y1 = 0.0f, y2 = 0.0f, y3 = 0.0f;
            for (uint j = 0; j < 8; ++j) {
                a0 += (float)((char)((g0 >> (8 * j)) & 0xff)) * yl[j];
                y0 += yl[j];
                a1 += (float)((char)((g1 >> (8 * j)) & 0xff)) * yl[8 + j];
                y1 += yl[8 + j];
                a2 += (float)((char)((g2 >> (8 * j)) & 0xff)) * yl[16 + j];
                y2 += yl[16 + j];
                a3 += (float)((char)((g3 >> (8 * j)) & 0xff)) * yl[24 + j];
                y3 += yl[24 + j];
            }
            sumf[row] += dl1 * (a0 + del0 * y0) + dl1 * (a1 + del1 * y1)
                       + dl2 * (a2 + del2 * y2) + dl2 * (a3 + del3 * y3);
        }
    }
    for (uint row = 0; row < 2; ++row) {
        float all = simd_sum(sumf[row]);
        if (tiisg == 0) out[rowbase + row] = all;
    }
}
`
