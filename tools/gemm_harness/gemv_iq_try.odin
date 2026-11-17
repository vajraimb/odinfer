// Same-run wall clock for IQ2_XXS GEMV variants on the 27B up shape
// (n=5120, d=17408). Ten dispatches per command buffer, so the number is
// kernel time rather than one-launch overhead. spec_sc matches production
// (constexpr nsb, scalar).
package gemv_iq_try

import q35 "qwen3_5:qwen3_5"

import NS "core:sys/darwin/Foundation"
import "core:fmt"
import "core:os"
import "core:strings"
import "core:time"
import MTL "vendor:darwin/Metal"

Try :: struct {
	name:   string,
	n, d:   int,
	tg_div: int,
	block:  int,
	group:  int,
}

TRIES := []Try{
	{"spec_sc", 5120, 17408, 32, 66, 0},
	{"spec_nounroll", 5120, 17408, 32, 66, 0},
	{"row2", 5120, 17408, 64, 66, 0},
	{"down_sc", 17408, 5120, 32, 66, 1},
	{"down_nounroll", 17408, 5120, 32, 66, 1},
	{"down_row2", 17408, 5120, 64, 66, 1},
	{"iq1_sc", 5120, 17408, 32, 50, 2},
	{"iq1_nounroll", 5120, 17408, 32, 50, 2},
	{"iq1_row2", 5120, 17408, 64, 50, 2},
}

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

	wbytes := 17408 * (17408 / 256) * 66
	wb := dev->newBufferWithLength(NS.UInteger(wbytes), MTL.ResourceStorageModeShared)
	xb := dev->newBufferWithLength(NS.UInteger(17408 * 4), MTL.ResourceStorageModeShared)
	ob := dev->newBufferWithLength(NS.UInteger(17408 * 4), MTL.ResourceStorageModeShared)
	defer wb->release(); defer xb->release(); defer ob->release()
	ws := wb->contentsAsSlice([]u8)
	for i in 0 ..< len(ws) { ws[i] = u8((i * 17 + 3) & 0xff) }
	xs := xb->contentsAsSlice([]f32)
	for i in 0 ..< len(xs) { xs[i] = f32(i % 13) * 0.01 - 0.05 }

	psos := make([]^MTL.ComputePipelineState, len(TRIES))
	for t, i in TRIES {
		psos[i] = make_pso(dev, lib, t.name)
	}
	ref_of := [3]int{0, 0, 0}
	for t, i in TRIES {
		if i == 0 || t.group != TRIES[i - 1].group { ref_of[t.group] = i }
	}
	for g in 0 ..< 3 {
		ri := ref_of[g]
		rt := TRIES[ri]
		dispatch(queue, psos[ri], wb, xb, ob, rt.d / rt.tg_div)
		ref := make([]f32, rt.d)
		copy(ref, ob->contentsAsSlice([]f32)[:rt.d])
		for t, i in TRIES {
			if t.group != g || i == ri { continue }
			dispatch(queue, psos[i], wb, xb, ob, t.d / t.tg_div)
			got := ob->contentsAsSlice([]f32)[:t.d]
			max_rel: f32 = 0
			for k in 0 ..< t.d {
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
		tg := t.d / t.tg_div
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
		bytes := f64(t.d) * f64(t.n / 256) * f64(t.block)
		fmt.eprintfln("{:6.3f} ms   {:5.1f} GB/s   {}", best, bytes / (best * 1e-3) / 1e9, t.name)
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
kernel void spec_sc(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 5120 / 256;
    uint row = tg * 32 + tid / 8; uint lane = tid % 8; float sum = 0.0;
    device const uchar *wr = wb + (ulong)row * nsb * 66;
    for (uint sb = lane; sb < nsb; sb += 8) { device const uchar *blk = wr + sb * 66;
        float d = (float)(*(device const half*)blk);
        uint ybase = sb * 256;
        for (uint ib32 = 0; ib32 < 8; ib32++) {
            uint qb = 2 + 8 * ib32;
            uint aux1 = (uint)blk[qb + 4] | ((uint)blk[qb + 5] << 8) | ((uint)blk[qb + 6] << 16) | ((uint)blk[qb + 7] << 24);
            float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
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

kernel void spec_nounroll(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 5120 / 256;
    uint row = tg * 32 + tid / 8; uint lane = tid % 8; float sum = 0.0;
    device const uchar *wr = wb + (ulong)row * nsb * 66;
    #pragma clang loop unroll(disable)
    for (uint sb = lane; sb < nsb; sb += 8) { device const uchar *blk = wr + sb * 66;
        float d = (float)(*(device const half*)blk);
        uint ybase = sb * 256;
        #pragma clang loop unroll(disable)
        for (uint ib32 = 0; ib32 < 8; ib32++) {
            uint qb = 2 + 8 * ib32;
            uint aux1 = (uint)blk[qb + 4] | ((uint)blk[qb + 5] << 8) | ((uint)blk[qb + 6] << 16) | ((uint)blk[qb + 7] << 24);
            float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
            for (uint l = 0; l < 4; l++) {
                ulong grid = iq2xxs_grid[blk[qb + l]];
                uchar signs = ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
                #pragma clang loop unroll(full)
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
    if (lane % 8 == 0) out[row] = sum;
}

kernel void spec_2acc(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 5120 / 256;
    uint row = tg * 32 + tid / 8; uint lane = tid % 8;
    float a0 = 0.0f, a1 = 0.0f;
    device const uchar *wr = wb + (ulong)row * nsb * 66;
    #pragma clang loop unroll(disable)
    for (uint sb = lane; sb < nsb; sb += 8) { device const uchar *blk = wr + sb * 66;
        float d = (float)(*(device const half*)blk);
        uint ybase = sb * 256;
        #pragma clang loop unroll(disable)
        for (uint ib32 = 0; ib32 < 8; ib32++) {
            uint qb = 2 + 8 * ib32;
            uint aux1 = (uint)blk[qb + 4] | ((uint)blk[qb + 5] << 8) | ((uint)blk[qb + 6] << 16) | ((uint)blk[qb + 7] << 24);
            float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
            for (uint l = 0; l < 4; l++) {
                ulong grid = iq2xxs_grid[blk[qb + l]];
                uchar signs = ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
                float b0 = 0.0f, b1 = 0.0f;
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 8; j++) {
                    float v = (float)((grid >> (8 * j)) & 0xff);
                    float s = 1.0f - 2.0f * float((signs & (1u << j)) != 0);
                    float term = db * v * s * x[ybase + j];
                    if ((j & 1) == 0) b0 += term; else b1 += term;
                }
                a0 += b0; a1 += b1;
                ybase += 8;
            }
        }
    }
    float sum = a0 + a1;
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane % 8 == 0) out[row] = sum;
}

kernel void spec_f4_nounroll(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 5120 / 256;
    uint row = tg * 32 + tid / 8; uint lane = tid % 8;
    float a0 = 0.0f, a1 = 0.0f;
    device const uchar *wr = wb + (ulong)row * nsb * 66;
    #pragma clang loop unroll(disable)
    for (uint sb = lane; sb < nsb; sb += 8) { device const uchar *blk = wr + sb * 66;
        float d = (float)(*(device const half*)blk);
        uint ybase = sb * 256;
        #pragma clang loop unroll(disable)
        for (uint ib32 = 0; ib32 < 8; ib32++) {
            uint qb = 2 + 8 * ib32;
            uint aux1 = (uint)blk[qb + 4] | ((uint)blk[qb + 5] << 8) | ((uint)blk[qb + 6] << 16) | ((uint)blk[qb + 7] << 24);
            float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
            #pragma clang loop unroll(full)
            for (uint l = 0; l < 4; l++) {
                ulong grid = iq2xxs_grid[blk[qb + l]];
                uchar signs = ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
                float4 sg0 = float4(
                    1.0f - 2.0f * float((signs & 1) != 0),
                    1.0f - 2.0f * float((signs & 2) != 0),
                    1.0f - 2.0f * float((signs & 4) != 0),
                    1.0f - 2.0f * float((signs & 8) != 0));
                float4 sg1 = float4(
                    1.0f - 2.0f * float((signs & 16) != 0),
                    1.0f - 2.0f * float((signs & 32) != 0),
                    1.0f - 2.0f * float((signs & 64) != 0),
                    1.0f - 2.0f * float((signs & 128) != 0));
                float4 gv0 = float4((float)((grid >> 0) & 0xff), (float)((grid >> 8) & 0xff), (float)((grid >> 16) & 0xff), (float)((grid >> 24) & 0xff));
                float4 gv1 = float4((float)((grid >> 32) & 0xff), (float)((grid >> 40) & 0xff), (float)((grid >> 48) & 0xff), (float)((grid >> 56) & 0xff));
                float4 x0 = float4(x[ybase], x[ybase+1], x[ybase+2], x[ybase+3]);
                float4 x1 = float4(x[ybase+4], x[ybase+5], x[ybase+6], x[ybase+7]);
                a0 += db * dot(gv0 * sg0, x0);
                a1 += db * dot(gv1 * sg1, x1);
                ybase += 8;
            }
        }
    }
    float sum = a0 + a1;
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane % 8 == 0) out[row] = sum;
}

kernel void row2(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 5120 / 256;
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

kernel void row2_f4(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 5120 / 256;
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
                uchar zn0 = ksigns_iq2xs[(aux0 >> (7 * l)) & 127];
                uchar zn1 = ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
                float4 x0 = float4(x[ybase], x[ybase+1], x[ybase+2], x[ybase+3]);
                float4 x1 = float4(x[ybase+4], x[ybase+5], x[ybase+6], x[ybase+7]);
                float4 v0 = float4((float)(g0 & 0xff), (float)((g0 >> 8) & 0xff), (float)((g0 >> 16) & 0xff), (float)((g0 >> 24) & 0xff));
                float4 v1 = float4((float)((g0 >> 32) & 0xff), (float)((g0 >> 40) & 0xff), (float)((g0 >> 48) & 0xff), (float)((g0 >> 56) & 0xff));
                float4 w0 = float4((float)(g1 & 0xff), (float)((g1 >> 8) & 0xff), (float)((g1 >> 16) & 0xff), (float)((g1 >> 24) & 0xff));
                float4 w1 = float4((float)((g1 >> 32) & 0xff), (float)((g1 >> 40) & 0xff), (float)((g1 >> 48) & 0xff), (float)((g1 >> 56) & 0xff));
                float4 z0a = float4(1.f - 2.f * float((zn0 & 1) != 0), 1.f - 2.f * float((zn0 & 2) != 0), 1.f - 2.f * float((zn0 & 4) != 0), 1.f - 2.f * float((zn0 & 8) != 0));
                float4 z0b = float4(1.f - 2.f * float((zn0 & 16) != 0), 1.f - 2.f * float((zn0 & 32) != 0), 1.f - 2.f * float((zn0 & 64) != 0), 1.f - 2.f * float((zn0 & 128) != 0));
                float4 z1a = float4(1.f - 2.f * float((zn1 & 1) != 0), 1.f - 2.f * float((zn1 & 2) != 0), 1.f - 2.f * float((zn1 & 4) != 0), 1.f - 2.f * float((zn1 & 8) != 0));
                float4 z1b = float4(1.f - 2.f * float((zn1 & 16) != 0), 1.f - 2.f * float((zn1 & 32) != 0), 1.f - 2.f * float((zn1 & 64) != 0), 1.f - 2.f * float((zn1 & 128) != 0));
                s0 += db0 * (dot(v0 * z0a, x0) + dot(v1 * z0b, x1));
                s1 += db1 * (dot(w0 * z1a, x0) + dot(w1 * z1b, x1));
                ybase += 8;
            }
        }
    }
    s0 += simd_shuffle_xor(s0, 4); s0 += simd_shuffle_xor(s0, 2); s0 += simd_shuffle_xor(s0, 1);
    s1 += simd_shuffle_xor(s1, 4); s1 += simd_shuffle_xor(s1, 2); s1 += simd_shuffle_xor(s1, 1);
    if (lane % 8 == 0) { out[row0] = s0; out[row1] = s1; }
}

kernel void down_sc(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 17408 / 256;
    uint row = tg * 32 + tid / 8; uint lane = tid % 8; float sum = 0.0;
    device const uchar *wr = wb + (ulong)row * nsb * 66;
    for (uint sb = lane; sb < nsb; sb += 8) { device const uchar *blk = wr + sb * 66;
        float d = (float)(*(device const half*)blk);
        uint ybase = sb * 256;
        for (uint ib32 = 0; ib32 < 8; ib32++) {
            uint qb = 2 + 8 * ib32;
            uint aux1 = (uint)blk[qb + 4] | ((uint)blk[qb + 5] << 8) | ((uint)blk[qb + 6] << 16) | ((uint)blk[qb + 7] << 24);
            float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
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

kernel void down_nounroll(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 17408 / 256;
    uint row = tg * 32 + tid / 8; uint lane = tid % 8; float sum = 0.0;
    device const uchar *wr = wb + (ulong)row * nsb * 66;
    #pragma clang loop unroll(disable)
    for (uint sb = lane; sb < nsb; sb += 8) { device const uchar *blk = wr + sb * 66;
        float d = (float)(*(device const half*)blk);
        uint ybase = sb * 256;
        #pragma clang loop unroll(disable)
        for (uint ib32 = 0; ib32 < 8; ib32++) {
            uint qb = 2 + 8 * ib32;
            uint aux1 = (uint)blk[qb + 4] | ((uint)blk[qb + 5] << 8) | ((uint)blk[qb + 6] << 16) | ((uint)blk[qb + 7] << 24);
            float db = d * (0.5f + (float)(aux1 >> 28)) * 0.25f;
            for (uint l = 0; l < 4; l++) {
                ulong grid = iq2xxs_grid[blk[qb + l]];
                uchar signs = ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
                #pragma clang loop unroll(full)
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
    if (lane % 8 == 0) out[row] = sum;
}

kernel void down_row2(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 17408 / 256;
    uint lane = tid % 8; uint slot = tid / 8;
    uint row0 = tg * 64 + slot; uint row1 = row0 + 32;
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

kernel void iq1_sc(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 5120 / 256;
    uint row = tg * 32 + tid / 8; uint lane = tid % 8; float sum = 0.0;
    device const uchar *wr = wb + (ulong)row * nsb * 50;
    for (uint sb = lane; sb < nsb; sb += 8) { device const uchar *blk = wr + sb * 50;
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
    if (lane % 8 == 0) out[row] = sum;
}

kernel void iq1_nounroll(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 5120 / 256;
    uint row = tg * 32 + tid / 8; uint lane = tid % 8; float sum = 0.0;
    device const uchar *wr = wb + (ulong)row * nsb * 50;
    #pragma clang loop unroll(disable)
    for (uint sb = lane; sb < nsb; sb += 8) { device const uchar *blk = wr + sb * 50;
        float d = (float)(*(device const half*)blk);
        device const uchar *qs = blk + 2; device const uchar *qh = blk + 34;
        uint ybase = sb * 256; uint qoff = 0;
        #pragma clang loop unroll(disable)
        for (uint ib = 0; ib < 8; ib++) {
            uint qhv = (uint)qh[2 * ib] | ((uint)qh[2 * ib + 1] << 8);
            float dl = d * (float)(2 * (int)((qhv >> 12) & 7) + 1);
            float delta = (qhv & 0x8000) == 0 ? 0.125f : -0.125f;
            for (uint l = 0; l < 4; l++) {
                uint idx = (uint)qs[qoff + l] | (((qhv >> (3 * l)) & 7) << 8);
                ulong grid = iq1s_grid[idx];
                #pragma clang loop unroll(full)
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

kernel void iq1_row2(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 5120 / 256;
    uint lane = tid % 8; uint slot = tid / 8;
    uint row0 = tg * 64 + slot; uint row1 = row0 + 32;
    float s0 = 0.0f, s1 = 0.0f;
    device const uchar *wr0 = wb + (ulong)row0 * nsb * 50;
    device const uchar *wr1 = wb + (ulong)row1 * nsb * 50;
    #pragma clang loop unroll(disable)
    for (uint sb = lane; sb < nsb; sb += 8) {
        device const uchar *b0 = wr0 + sb * 50;
        device const uchar *b1 = wr1 + sb * 50;
        float d0 = (float)(*(device const half*)b0);
        float d1 = (float)(*(device const half*)b1);
        device const uchar *qs0 = b0 + 2; device const uchar *qh0 = b0 + 34;
        device const uchar *qs1 = b1 + 2; device const uchar *qh1 = b1 + 34;
        uint ybase = sb * 256; uint qoff = 0;
        #pragma clang loop unroll(disable)
        for (uint ib = 0; ib < 8; ib++) {
            uint q0 = (uint)qh0[2 * ib] | ((uint)qh0[2 * ib + 1] << 8);
            uint q1 = (uint)qh1[2 * ib] | ((uint)qh1[2 * ib + 1] << 8);
            float dl0 = d0 * (float)(2 * (int)((q0 >> 12) & 7) + 1);
            float dl1 = d1 * (float)(2 * (int)((q1 >> 12) & 7) + 1);
            float delta0 = (q0 & 0x8000) == 0 ? 0.125f : -0.125f;
            float delta1 = (q1 & 0x8000) == 0 ? 0.125f : -0.125f;
            for (uint l = 0; l < 4; l++) {
                uint i0 = (uint)qs0[qoff + l] | (((q0 >> (3 * l)) & 7) << 8);
                uint i1 = (uint)qs1[qoff + l] | (((q1 >> (3 * l)) & 7) << 8);
                ulong g0 = iq1s_grid[i0];
                ulong g1 = iq1s_grid[i1];
                #pragma clang loop unroll(full)
                for (uint j = 0; j < 8; j++) {
                    float xj = x[ybase + j];
                    float v0 = (float)((char)((g0 >> (8 * j)) & 0xff));
                    float v1 = (float)((char)((g1 >> (8 * j)) & 0xff));
                    s0 += dl0 * (v0 + delta0) * xj;
                    s1 += dl1 * (v1 + delta1) * xj;
                }
                ybase += 8;
            }
            qoff += 4;
        }
    }
    s0 += simd_shuffle_xor(s0, 4); s0 += simd_shuffle_xor(s0, 2); s0 += simd_shuffle_xor(s0, 1);
    s1 += simd_shuffle_xor(s1, 4); s1 += simd_shuffle_xor(s1, 2); s1 += simd_shuffle_xor(s1, 1);
    if (lane % 8 == 0) { out[row0] = s0; out[row1] = s1; }
}
`
