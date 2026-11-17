// Two-row vs spec scalar for the short-K kinds that are not IQ2_XXS.
// n=5120, d=17408. Ten dispatches per command buffer, min of 4.
package gemv_iq_r2more

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

Try :: struct {
	name:   string,
	tg_div: int,
	group:  int,
}

TRIES := []Try{
	{"iq2s_sc", 32, 0},
	{"iq2s_r2", 64, 0},
	{"iq3_sc", 32, 1},
	{"iq3_r2", 64, 1},
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

	wbytes := D * (N / 256) * 98
	wb := dev->newBufferWithLength(NS.UInteger(wbytes), MTL.ResourceStorageModeShared)
	xb := dev->newBufferWithLength(NS.UInteger(N * 4), MTL.ResourceStorageModeShared)
	ob := dev->newBufferWithLength(NS.UInteger(D * 4), MTL.ResourceStorageModeShared)
	defer wb->release(); defer xb->release(); defer ob->release()
	ws := wb->contentsAsSlice([]u8)
	for i in 0 ..< len(ws) { ws[i] = u8((i * 17 + 3) & 0xff) }
	nblk82 := D * (N / 256)
	for b in 0 ..< nblk82 {
		if b * 82 + 1 < len(ws) {
			ws[b * 82 + 0] = 0x00
			ws[b * 82 + 1] = 0x3C
		}
	}
	for b in 0 ..< nblk82 {
		ws[b * 98 + 0] = 0x00
		ws[b * 98 + 1] = 0x3C
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
		dispatch(queue, psos[ri], wb, xb, ob, D / rt.tg_div)
		ref := make([]f32, D)
		copy(ref, ob->contentsAsSlice([]f32)[:D])
		for t, i in TRIES {
			if t.group != g || i == ri { continue }
			dispatch(queue, psos[i], wb, xb, ob, D / t.tg_div)
			got := ob->contentsAsSlice([]f32)[:D]
			max_rel: f32 = 0
			for k in 0 ..< D {
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
		tg := D / t.tg_div
		block: f64 = 82
		if t.group == 1 { block = 98 }
		bytes := f64(D) * f64(N / 256) * block
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
kernel void iq2s_sc(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 20;
    uint row = tg * 32 + tid / 8; uint lane = tid % 8; float sum = 0.0f;
    device const uchar *wr = wb + (ulong)row * nsb * 82;
    #pragma clang loop unroll(disable)
    for (uint sb = lane; sb < nsb; sb += 8) {
        device const uchar *blk = wr + sb * 82;
        float d = (float)(*(device const half*)blk);
        device const uchar *qs = blk + 2; device const uchar *signs = blk + 34;
        device const uchar *qh = blk + 66; device const uchar *scales = blk + 74;
        uint ybase = sb * 256; uint qoff = 0; uint signoff = 0;
        #pragma clang loop unroll(disable)
        for (uint ib32 = 0; ib32 < 8; ib32++) {
            uchar sc = scales[ib32];
            float db0 = d * (0.5f + (float)(sc & 0xF)) * 0.25f;
            float db1 = d * (0.5f + (float)(sc >> 4)) * 0.25f;
            for (uint l = 0; l < 4; l++) {
                uint idx = (uint)qs[qoff + l] | (((uint)qh[ib32] << (8 - 2 * l)) & 0x300);
                ulong grid = iq2s_grid[idx];
                uchar sn = signs[signoff + l];
                float dl = (l < 2) ? db0 : db1;
                #pragma clang loop unroll(full)
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

kernel void iq2s_r2(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 20;
    uint lane = tid % 8; uint slot = tid / 8;
    uint row0 = tg * 64 + slot; uint row1 = row0 + 32;
    float s0 = 0.0f, s1 = 0.0f;
    device const uchar *wr0 = wb + (ulong)row0 * nsb * 82;
    device const uchar *wr1 = wb + (ulong)row1 * nsb * 82;
    #pragma clang loop unroll(disable)
    for (uint sb = lane; sb < nsb; sb += 8) {
        device const uchar *b0 = wr0 + sb * 82;
        device const uchar *b1 = wr1 + sb * 82;
        float d0 = (float)(*(device const half*)b0);
        float d1 = (float)(*(device const half*)b1);
        device const uchar *qs0 = b0 + 2; device const uchar *sg0 = b0 + 34;
        device const uchar *qh0 = b0 + 66; device const uchar *sc0 = b0 + 74;
        device const uchar *qs1 = b1 + 2; device const uchar *sg1 = b1 + 34;
        device const uchar *qh1 = b1 + 66; device const uchar *sc1 = b1 + 74;
        uint ybase = sb * 256; uint qoff = 0; uint signoff = 0;
        #pragma clang loop unroll(disable)
        for (uint ib32 = 0; ib32 < 8; ib32++) {
            uchar c0 = sc0[ib32]; uchar c1 = sc1[ib32];
            float db00 = d0 * (0.5f + (float)(c0 & 0xF)) * 0.25f;
            float db01 = d0 * (0.5f + (float)(c0 >> 4)) * 0.25f;
            float db10 = d1 * (0.5f + (float)(c1 & 0xF)) * 0.25f;
            float db11 = d1 * (0.5f + (float)(c1 >> 4)) * 0.25f;
            for (uint l = 0; l < 4; l++) {
                uint i0 = (uint)qs0[qoff + l] | (((uint)qh0[ib32] << (8 - 2 * l)) & 0x300);
                uint i1 = (uint)qs1[qoff + l] | (((uint)qh1[ib32] << (8 - 2 * l)) & 0x300);
                ulong g0 = iq2s_grid[i0]; ulong g1 = iq2s_grid[i1];
                uchar n0 = sg0[signoff + l]; uchar n1 = sg1[signoff + l];
                float dl0 = (l < 2) ? db00 : db01;
                float dl1 = (l < 2) ? db10 : db11;
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 8; j++) {
                    float xj = x[ybase + j];
                    float v0 = (float)((g0 >> (8 * j)) & 0xff);
                    float v1 = (float)((g1 >> (8 * j)) & 0xff);
                    float z0 = (n0 & kmask_iq2xs[j]) == 0 ? 1.0f : -1.0f;
                    float z1 = (n1 & kmask_iq2xs[j]) == 0 ? 1.0f : -1.0f;
                    s0 += dl0 * v0 * z0 * xj;
                    s1 += dl1 * v1 * z1 * xj;
                }
                ybase += 8;
            }
            qoff += 4; signoff += 4;
        }
    }
    s0 += simd_shuffle_xor(s0, 4); s0 += simd_shuffle_xor(s0, 2); s0 += simd_shuffle_xor(s0, 1);
    s1 += simd_shuffle_xor(s1, 4); s1 += simd_shuffle_xor(s1, 2); s1 += simd_shuffle_xor(s1, 1);
    if (lane == 0) { out[row0] = s0; out[row1] = s1; }
}

kernel void iq3_sc(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 20;
    uint row = tg * 32 + tid / 8; uint lane = tid % 8; float sum = 0.0f;
    device const uchar *wr = wb + (ulong)row * nsb * 98;
    #pragma clang loop unroll(disable)
    for (uint sb = lane; sb < nsb; sb += 8) {
        device const uchar *blk = wr + sb * 98;
        float d = (float)(*(device const half*)blk);
        device const uchar *qs = blk + 2; device const uchar *sas = blk + 66;
        uint ybase = sb * 256; uint qoff = 0;
        #pragma clang loop unroll(disable)
        for (uint ib32 = 0; ib32 < 8; ib32++) {
            uint sb_ = 4 * ib32;
            uint aux = (uint)sas[sb_] | ((uint)sas[sb_ + 1] << 8) | ((uint)sas[sb_ + 2] << 16) | ((uint)sas[sb_ + 3] << 24);
            float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
            for (uint l = 0; l < 4; l++) {
                uchar signs = ksigns_iq2xs[(aux >> (7 * l)) & 127];
                uint g1 = iq3xxs_grid[qs[qoff + 2 * l + 0]];
                uint g2 = iq3xxs_grid[qs[qoff + 2 * l + 1]];
                #pragma clang loop unroll(full)
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

kernel void iq3_r2(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 20;
    uint lane = tid % 8; uint slot = tid / 8;
    uint row0 = tg * 64 + slot; uint row1 = row0 + 32;
    float s0 = 0.0f, s1 = 0.0f;
    device const uchar *wr0 = wb + (ulong)row0 * nsb * 98;
    device const uchar *wr1 = wb + (ulong)row1 * nsb * 98;
    #pragma clang loop unroll(disable)
    for (uint sb = lane; sb < nsb; sb += 8) {
        device const uchar *b0 = wr0 + sb * 98;
        device const uchar *b1 = wr1 + sb * 98;
        float d0 = (float)(*(device const half*)b0);
        float d1 = (float)(*(device const half*)b1);
        device const uchar *qs0 = b0 + 2; device const uchar *sas0 = b0 + 66;
        device const uchar *qs1 = b1 + 2; device const uchar *sas1 = b1 + 66;
        uint ybase = sb * 256; uint qoff = 0;
        #pragma clang loop unroll(disable)
        for (uint ib32 = 0; ib32 < 8; ib32++) {
            uint sb_ = 4 * ib32;
            uint aux0 = (uint)sas0[sb_] | ((uint)sas0[sb_ + 1] << 8) | ((uint)sas0[sb_ + 2] << 16) | ((uint)sas0[sb_ + 3] << 24);
            uint aux1 = (uint)sas1[sb_] | ((uint)sas1[sb_ + 1] << 8) | ((uint)sas1[sb_ + 2] << 16) | ((uint)sas1[sb_ + 3] << 24);
            float db0 = d0 * (0.5f + (float)(aux0 >> 28)) * 0.5f;
            float db1 = d1 * (0.5f + (float)(aux1 >> 28)) * 0.5f;
            for (uint l = 0; l < 4; l++) {
                uchar n0 = ksigns_iq2xs[(aux0 >> (7 * l)) & 127];
                uchar n1 = ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
                uint g10 = iq3xxs_grid[qs0[qoff + 2 * l + 0]];
                uint g20 = iq3xxs_grid[qs0[qoff + 2 * l + 1]];
                uint g11 = iq3xxs_grid[qs1[qoff + 2 * l + 0]];
                uint g21 = iq3xxs_grid[qs1[qoff + 2 * l + 1]];
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 4; j++) {
                    float x0 = x[ybase + j];
                    float x1 = x[ybase + j + 4];
                    float v10 = (float)((g10 >> (8 * j)) & 0xff);
                    float v20 = (float)((g20 >> (8 * j)) & 0xff);
                    float v11 = (float)((g11 >> (8 * j)) & 0xff);
                    float v21 = (float)((g21 >> (8 * j)) & 0xff);
                    float z10 = (n0 & kmask_iq2xs[j + 0]) == 0 ? 1.0f : -1.0f;
                    float z20 = (n0 & kmask_iq2xs[j + 4]) == 0 ? 1.0f : -1.0f;
                    float z11 = (n1 & kmask_iq2xs[j + 0]) == 0 ? 1.0f : -1.0f;
                    float z21 = (n1 & kmask_iq2xs[j + 4]) == 0 ? 1.0f : -1.0f;
                    s0 += db0 * v10 * z10 * x0;
                    s0 += db0 * v20 * z20 * x1;
                    s1 += db1 * v11 * z11 * x0;
                    s1 += db1 * v21 * z21 * x1;
                }
                ybase += 8;
            }
            qoff += 8;
        }
    }
    s0 += simd_shuffle_xor(s0, 4); s0 += simd_shuffle_xor(s0, 2); s0 += simd_shuffle_xor(s0, 1);
    s1 += simd_shuffle_xor(s1, 4); s1 += simd_shuffle_xor(s1, 2); s1 += simd_shuffle_xor(s1, 1);
    if (lane == 0) { out[row0] = s0; out[row1] = s1; }
}
`
