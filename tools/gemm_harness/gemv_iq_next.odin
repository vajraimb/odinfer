// IQ2_S and IQ3_XXS short-K: spec-style scalar vs simdgroup K-split.
// n=5120, d=17408. Compiled next to the production Metal library.
package gemv_iq_next

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

	w2 := dev->newBufferWithLength(NS.UInteger(D * NSB * 82), MTL.ResourceStorageModeShared)
	w3 := dev->newBufferWithLength(NS.UInteger(D * NSB * 98), MTL.ResourceStorageModeShared)
	xb := dev->newBufferWithLength(NS.UInteger(N * 4), MTL.ResourceStorageModeShared)
	ob := dev->newBufferWithLength(NS.UInteger(D * 4), MTL.ResourceStorageModeShared)
	defer w2->release(); defer w3->release(); defer xb->release(); defer ob->release()
	fill(w2->contentsAsSlice([]u8), 82)
	fill(w3->contentsAsSlice([]u8), 98)
	xs := xb->contentsAsSlice([]f32)
	for i in 0 ..< len(xs) { xs[i] = f32(i % 13) * 0.01 - 0.05 }

	names := []string{"iq2s_sc", "iq2s_ks", "iq3_sc", "iq3_ks"}
	tgs := []int{D / 32, D / 16, D / 32, D / 16}
	wbs := []^MTL.Buffer{w2, w2, w3, w3}
	bytes := []f64{f64(D * NSB * 82), f64(D * NSB * 82), f64(D * NSB * 98), f64(D * NSB * 98)}
	psos := make([]^MTL.ComputePipelineState, len(names))
	for name, i in names { psos[i] = make_pso(dev, lib, name) }

	run(queue, psos[0], w2, xb, ob, tgs[0])
	ref2 := make([]f32, D)
	copy(ref2, ob->contentsAsSlice([]f32)[:D])
	check(queue, psos[1], w2, xb, ob, tgs[1], ref2, names[1])
	delete(ref2)

	run(queue, psos[2], w3, xb, ob, tgs[2])
	ref3 := make([]f32, D)
	copy(ref3, ob->contentsAsSlice([]f32)[:D])
	check(queue, psos[3], w3, xb, ob, tgs[3], ref3, names[3])
	delete(ref3)

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
	nsb: u32 = NSB
	p2 := make_pso(dev, lib, "iq2s_rt")
	p3 := make_pso(dev, lib, "iq3_rt")
	run(queue, psos[0], w2, xb, ob, tgs[0])
	ref2b := make([]f32, D)
	copy(ref2b, ob->contentsAsSlice([]f32)[:D])
	check_rt(queue, p2, w2, xb, ob, D / 16, &nsb, ref2b, "iq2s_rt")
	delete(ref2b)
	run(queue, psos[2], w3, xb, ob, tgs[2])
	ref3b := make([]f32, D)
	copy(ref3b, ob->contentsAsSlice([]f32)[:D])
	check_rt(queue, p3, w3, xb, ob, D / 16, &nsb, ref3b, "iq3_rt")
	delete(ref3b)
	time_rt(queue, p2, w2, xb, ob, D / 16, &nsb, f64(D * NSB * 82), "iq2s_rt")
	time_rt(queue, p3, w3, xb, ob, D / 16, &nsb, f64(D * NSB * 98), "iq3_rt")
}

check_rt :: proc(queue: ^MTL.CommandQueue, pso: ^MTL.ComputePipelineState, wb, xb, ob: ^MTL.Buffer, groups: int, nsb: ^u32, ref: []f32, name: string) {
	out_s := ob->contentsAsSlice([]f32)[:D]
	for k in 0 ..< D { out_s[k] = 1e9 }
	cmd := queue->commandBuffer()
	enc := cmd->computeCommandEncoder()
	enc->setComputePipelineState(pso)
	enc->setBuffer(wb, 0, 0)
	enc->setBuffer(xb, 0, 1)
	enc->setBuffer(ob, 0, 2)
	enc->setBytes(([^]u8)(nsb)[:size_of(u32)], 3)
	enc->dispatchThreadgroups(MTL.Size{NS.Integer(groups), 1, 1}, MTL.Size{256, 1, 1})
	enc->endEncoding()
	cmd->commit()
	cmd->waitUntilCompleted()
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

time_rt :: proc(queue: ^MTL.CommandQueue, pso: ^MTL.ComputePipelineState, wb, xb, ob: ^MTL.Buffer, groups: int, nsb: ^u32, bytes: f64, name: string) {
	best: f64 = 1e18
	for _ in 0 ..< 4 {
		t0 := time.tick_now()
		cmd := queue->commandBuffer()
		enc := cmd->computeCommandEncoder()
		enc->setComputePipelineState(pso)
		enc->setBuffer(wb, 0, 0)
		enc->setBuffer(xb, 0, 1)
		enc->setBuffer(ob, 0, 2)
		enc->setBytes(([^]u8)(nsb)[:size_of(u32)], 3)
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
kernel void iq2s_sc(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 20;
    uint row = tg * 32 + tid / 8; uint lane = tid % 8; float sum = 0.0f;
    device const uchar *wr = wb + (ulong)row * nsb * 82;
    for (uint sb = lane; sb < nsb; sb += 8) {
        device const uchar *blk = wr + sb * 82;
        float d = (float)(*(device const half*)blk);
        device const uchar *qs = blk + 2; device const uchar *signs = blk + 34;
        device const uchar *qh = blk + 66; device const uchar *scales = blk + 74;
        uint ybase = sb * 256; uint qoff = 0; uint signoff = 0;
        for (uint ib32 = 0; ib32 < 8; ib32++) {
            uchar sc = scales[ib32];
            float db0 = d * (0.5f + (float)(sc & 0xF)) * 0.25f;
            float db1 = d * (0.5f + (float)(sc >> 4)) * 0.25f;
            for (uint l = 0; l < 4; l++) {
                uint idx = (uint)qs[qoff + l] | (((uint)qh[ib32] << (8 - 2 * l)) & 0x300);
                ulong grid = iq2s_grid[idx];
                uchar sn = signs[signoff + l];
                float dl = (l < 2) ? db0 : db1;
                for (uint j = 0; j < 8; j++) {
                    float v = (float)((grid >> (8 * j)) & 0xff);
                    float s = (sn & kmask_iq2xs[j]) == 0 ? 1.0f : -1.0f;
                    sum += dl * v * s * x[ybase + j];
                }
                ybase += 8;
            }
            qoff += 4; signoff += 4;
        }
    }
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane == 0) out[row] = sum;
}

kernel void iq2s_ks(
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
            device const uchar *blk = wb + ((ulong)(rowbase + row) * nsb + ibl) * 82;
            float d = (float)(*(device const half*)blk);
            device const uchar *qs = blk + 2 + 4 * ib;
            device const uchar *sg = blk + 34 + 4 * ib;
            uint qh = blk[66 + ib];
            uchar scv = blk[74 + ib];
            float db0 = d * (0.5f + (float)(scv & 0xF)) * 0.25f;
            float db1 = d * (0.5f + (float)(scv >> 4)) * 0.25f;
            float lo = 0.0f, hi = 0.0f;
            for (uint l = 0; l < 4; ++l) {
                uint idx = (uint)qs[l] | ((qh << (8 - 2 * l)) & 0x300);
                ulong grid = iq2s_grid[idx];
                uchar sn = sg[l];
                float acc = 0.0f;
                for (uint j = 0; j < 8; ++j) {
                    float v = (float)((grid >> (8 * j)) & 0xff);
                    float s = (sn & kmask_iq2xs[j]) == 0 ? 1.0f : -1.0f;
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

kernel void iq3_sc(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 20;
    uint row = tg * 32 + tid / 8; uint lane = tid % 8; float sum = 0.0f;
    device const uchar *wr = wb + (ulong)row * nsb * 98;
    for (uint sb = lane; sb < nsb; sb += 8) {
        device const uchar *blk = wr + sb * 98;
        float d = (float)(*(device const half*)blk);
        device const uchar *qs = blk + 2; device const uchar *sas = blk + 66;
        uint ybase = sb * 256; uint qoff = 0;
        for (uint ib32 = 0; ib32 < 8; ib32++) {
            uint sb_ = 4 * ib32;
            uint aux = (uint)sas[sb_] | ((uint)sas[sb_ + 1] << 8) | ((uint)sas[sb_ + 2] << 16) | ((uint)sas[sb_ + 3] << 24);
            float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
            for (uint l = 0; l < 4; l++) {
                uchar signs = ksigns_iq2xs[(aux >> (7 * l)) & 127];
                uint g1 = iq3xxs_grid[qs[qoff + 2 * l + 0]];
                uint g2 = iq3xxs_grid[qs[qoff + 2 * l + 1]];
                for (uint j = 0; j < 4; j++) {
                    float v1 = (float)((g1 >> (8 * j)) & 0xff);
                    float v2 = (float)((g2 >> (8 * j)) & 0xff);
                    float s1 = (signs & kmask_iq2xs[j + 0]) == 0 ? 1.0f : -1.0f;
                    float s2 = (signs & kmask_iq2xs[j + 4]) == 0 ? 1.0f : -1.0f;
                    sum += db * v1 * s1 * x[ybase + j + 0];
                    sum += db * v2 * s2 * x[ybase + j + 4];
                }
                ybase += 8;
            }
            qoff += 8;
        }
    }
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane == 0) out[row] = sum;
}

kernel void iq3_ks(
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
            device const uchar *blk = wb + ((ulong)(rowbase + row) * nsb + ibl) * 98;
            float d = (float)(*(device const half*)blk);
            device const uchar *qs = blk + 2 + 8 * ib;
            device const uchar *sas = blk + 66 + 4 * ib;
            uint aux = (uint)sas[0] | ((uint)sas[1] << 8) | ((uint)sas[2] << 16) | ((uint)sas[3] << 24);
            float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
            float acc = 0.0f;
            for (uint l = 0; l < 4; ++l) {
                uchar signs = ksigns_iq2xs[(aux >> (7 * l)) & 127];
                uint g1 = iq3xxs_grid[qs[2 * l + 0]];
                uint g2 = iq3xxs_grid[qs[2 * l + 1]];
                for (uint j = 0; j < 4; ++j) {
                    float v1 = (float)((g1 >> (8 * j)) & 0xff);
                    float v2 = (float)((g2 >> (8 * j)) & 0xff);
                    float s1 = (signs & kmask_iq2xs[j + 0]) == 0 ? 1.0f : -1.0f;
                    float s2 = (signs & kmask_iq2xs[j + 4]) == 0 ? 1.0f : -1.0f;
                    acc += v1 * s1 * yl[8 * l + j];
                    acc += v2 * s2 * yl[8 * l + j + 4];
                }
            }
            sumf[row] += db * acc;
        }
    }
    for (uint row = 0; row < 2; ++row) {
        float all = simd_sum(sumf[row]);
        if (tiisg == 0) out[rowbase + row] = all;
    }
}

kernel void iq2s_rt(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant uint &nsb_in [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    uint nsb = nsb_in;
    uint nb32 = nsb * 8u;
    uint rowbase = (tg * 8u + sgitg) * 2u;
    float sumf[2] = {0.0f, 0.0f};
    for (uint ib32 = tiisg; ib32 < nb32; ib32 += 32) {
        uint yb = ib32 * 32;
        float yl[32];
        for (uint i = 0; i < 32; ++i) yl[i] = x[yb + i];
        uint ibl = ib32 >> 3;
        uint ib = ib32 & 7;
        for (uint row = 0; row < 2; ++row) {
            device const uchar *blk = wb + ((ulong)(rowbase + row) * nsb + ibl) * 82;
            float d = (float)(*(device const half*)blk);
            device const uchar *qs = blk + 2 + 4 * ib;
            device const uchar *sg = blk + 34 + 4 * ib;
            uint qh = blk[66 + ib];
            uchar scv = blk[74 + ib];
            float db0 = d * (0.5f + (float)(scv & 0xF)) * 0.25f;
            float db1 = d * (0.5f + (float)(scv >> 4)) * 0.25f;
            float lo = 0.0f, hi = 0.0f;
            for (uint l = 0; l < 4; ++l) {
                uint idx = (uint)qs[l] | ((qh << (8 - 2 * l)) & 0x300);
                ulong grid = iq2s_grid[idx];
                uchar sn = sg[l];
                float acc = 0.0f;
                for (uint j = 0; j < 8; ++j) {
                    float v = (float)((grid >> (8 * j)) & 0xff);
                    float s = (sn & kmask_iq2xs[j]) == 0 ? 1.0f : -1.0f;
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

kernel void iq3_rt(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant uint &nsb_in [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    uint nsb = nsb_in;
    uint nb32 = nsb * 8u;
    uint rowbase = (tg * 8u + sgitg) * 2u;
    float sumf[2] = {0.0f, 0.0f};
    for (uint ib32 = tiisg; ib32 < nb32; ib32 += 32) {
        uint yb = ib32 * 32;
        float yl[32];
        for (uint i = 0; i < 32; ++i) yl[i] = x[yb + i];
        uint ibl = ib32 >> 3;
        uint ib = ib32 & 7;
        for (uint row = 0; row < 2; ++row) {
            device const uchar *blk = wb + ((ulong)(rowbase + row) * nsb + ibl) * 98;
            float d = (float)(*(device const half*)blk);
            device const uchar *qs = blk + 2 + 8 * ib;
            device const uchar *sas = blk + 66 + 4 * ib;
            uint aux = (uint)sas[0] | ((uint)sas[1] << 8) | ((uint)sas[2] << 16) | ((uint)sas[3] << 24);
            float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
            float acc = 0.0f;
            for (uint l = 0; l < 4; ++l) {
                uchar signs = ksigns_iq2xs[(aux >> (7 * l)) & 127];
                uint g1 = iq3xxs_grid[qs[2 * l + 0]];
                uint g2 = iq3xxs_grid[qs[2 * l + 1]];
                for (uint j = 0; j < 4; ++j) {
                    float v1 = (float)((g1 >> (8 * j)) & 0xff);
                    float v2 = (float)((g2 >> (8 * j)) & 0xff);
                    float s1 = (signs & kmask_iq2xs[j + 0]) == 0 ? 1.0f : -1.0f;
                    float s2 = (signs & kmask_iq2xs[j + 4]) == 0 ? 1.0f : -1.0f;
                    acc += v1 * s1 * yl[8 * l + j];
                    acc += v2 * s2 * yl[8 * l + j + 4];
                }
            }
            sumf[row] += db * acc;
        }
    }
    for (uint row = 0; row < 2; ++row) {
        float all = simd_sum(sumf[row]);
        if (tiisg == 0) out[rowbase + row] = all;
    }
}
`
