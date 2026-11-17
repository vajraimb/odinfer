// Q3_K output-head GEMV (n=5120, d=248320). Production runs the generic
// kernel. q3_spec bakes nsb; q3_r2 is the short-K two-row form.
// Ten dispatches per command buffer, min of 4.
package gemv_q3_try

import NS "core:sys/darwin/Foundation"
import "core:fmt"
import "core:os"
import "core:time"
import MTL "vendor:darwin/Metal"

N :: 5120
D :: 248320
REPS :: 10

Try :: struct {
	name:   string,
	tg_div: int,
	dims:   bool,
}

TRIES := []Try{
	{"q3_gen", 32, true},
	{"q3_f4", 32, false},
	{"q3_r2", 64, false},
}

main :: proc() {
	dev := MTL.CreateSystemDefaultDevice()
	if dev == nil { fmt.eprintln("no metal"); os.exit(1) }
	defer dev->release()
	queue := dev->newCommandQueue()
	defer queue->release()
	src_ns := NS.String.alloc()->initWithOdinString(EXTRA)
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

	wbytes := D * (N / 256) * 110
	wb := dev->newBufferWithLength(NS.UInteger(wbytes), MTL.ResourceStorageModeShared)
	xb := dev->newBufferWithLength(NS.UInteger(N * 4), MTL.ResourceStorageModeShared)
	ob := dev->newBufferWithLength(NS.UInteger(D * 4), MTL.ResourceStorageModeShared)
	if wb == nil || xb == nil || ob == nil {
		fmt.eprintln("buffer alloc failed")
		os.exit(1)
	}
	defer wb->release(); defer xb->release(); defer ob->release()
	ws := wb->contentsAsSlice([]u8)
	for i in 0 ..< len(ws) { ws[i] = u8((i * 17 + 3) & 0xff) }
	// Modest block scales. Random f16 d is often huge and makes the
	// dot-vs-scalar check meaningless.
	for b in 0 ..< D * (N / 256) {
		ws[b * 110 + 108] = 0x00
		ws[b * 110 + 109] = 0x3C
	}
	xs := xb->contentsAsSlice([]f32)
	for i in 0 ..< len(xs) { xs[i] = f32(i % 13) * 0.01 - 0.05 }

	psos := make([]^MTL.ComputePipelineState, len(TRIES))
	for t, i in TRIES { psos[i] = make_pso(dev, lib, t.name) }

	dispatch(queue, psos[0], wb, xb, ob, D / TRIES[0].tg_div, true)
	ref := make([]f32, D)
	copy(ref, ob->contentsAsSlice([]f32)[:D])
	for t, i in TRIES[1:] {
		dispatch(queue, psos[i + 1], wb, xb, ob, D / t.tg_div, t.dims)
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

	bytes := f64(D) * f64(N / 256) * 110
	for t, i in TRIES {
		best: f64 = 1e18
		tg := D / t.tg_div
		for _ in 0 ..< 4 {
			t0 := time.tick_now()
			cmd := queue->commandBuffer()
			enc := cmd->computeCommandEncoder()
			enc->setComputePipelineState(psos[i])
			enc->setBuffer(wb, 0, 0)
			enc->setBuffer(xb, 0, 1)
			enc->setBuffer(ob, 0, 2)
			if t.dims {
				dims := [2]u32{N, D}
				enc->setBytes(([^]u8)(rawptr(&dims))[:size_of(dims)], 3)
			}
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

dispatch :: proc(queue: ^MTL.CommandQueue, pso: ^MTL.ComputePipelineState, wb, xb, ob: ^MTL.Buffer, tg: int, dims_on: bool) {
	cmd := queue->commandBuffer()
	enc := cmd->computeCommandEncoder()
	enc->setComputePipelineState(pso)
	enc->setBuffer(wb, 0, 0)
	enc->setBuffer(xb, 0, 1)
	enc->setBuffer(ob, 0, 2)
	if dims_on {
		dims := [2]u32{N, D}
		enc->setBytes(([^]u8)(rawptr(&dims))[:size_of(dims)], 3)
	}
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
#include <metal_stdlib>
using namespace metal;

struct Dims { uint n; uint d; };

inline uchar q3k_scale_byte(uint is, uint aux0, uint aux1, uint aux2, uint aux3) {
    uint av = is < 4 ? aux0 : (is < 8 ? aux1 : (is < 12 ? aux2 : aux3));
    return (uchar)((av >> (8 * (is & 3))) & 0xff);
}

#define Q3_BLOCK(blk, s) \
    do { \
        device const uchar *_hm = (blk); \
        device const uchar *_qs = (blk) + 32; \
        device const uchar *_scales = (blk) + 96; \
        float _d = (float)(*(device const half*)((blk) + 108)); \
        uint _aux0 = (uint)_scales[0] | ((uint)_scales[1] << 8) | ((uint)_scales[2] << 16) | ((uint)_scales[3] << 24); \
        uint _aux1 = (uint)_scales[4] | ((uint)_scales[5] << 8) | ((uint)_scales[6] << 16) | ((uint)_scales[7] << 24); \
        uint _tmp  = (uint)_scales[8] | ((uint)_scales[9] << 8) | ((uint)_scales[10] << 16) | ((uint)_scales[11] << 24); \
        const uint _KMASK1 = 0x03030303u; \
        const uint _KMASK2 = 0x0f0f0f0fu; \
        uint _aux2 = ((_aux0 >> 4) & _KMASK2) | (((_tmp >> 4) & _KMASK1) << 4); \
        uint _aux3 = ((_aux1 >> 4) & _KMASK2) | (((_tmp >> 6) & _KMASK1) << 4); \
        _aux0 = (_aux0 & _KMASK2) | (((_tmp >> 0) & _KMASK1) << 4); \
        _aux1 = (_aux1 & _KMASK2) | (((_tmp >> 2) & _KMASK1) << 4); \
        uint _ybase = sb * 256; \
        uint _qoff = 0; \
        uchar _m = 1; \
        uint _is = 0; \
        for (uint _h = 0; _h < 2; _h++) { \
            uint _shift = 0; \
            for (uint _ss = 0; _ss < 4; _ss++) { \
                float _dl = _d * (float)((int)(char)q3k_scale_byte(_is, _aux0, _aux1, _aux2, _aux3) - 32); \
                _is += 1; \
                for (uint _l = 0; _l < 16; _l++) { \
                    int _q = (int)((_qs[_qoff + _l] >> _shift) & 3) - (int)((_hm[_l] & _m) == 0 ? 4 : 0); \
                    s += _dl * (float)_q * x[_ybase + _l]; \
                } \
                _dl = _d * (float)((int)(char)q3k_scale_byte(_is, _aux0, _aux1, _aux2, _aux3) - 32); \
                _is += 1; \
                for (uint _l = 0; _l < 16; _l++) { \
                    int _q = (int)((_qs[_qoff + 16 + _l] >> _shift) & 3) - (int)((_hm[16 + _l] & _m) == 0 ? 4 : 0); \
                    s += _dl * (float)_q * x[_ybase + 16 + _l]; \
                } \
                _shift += 2; _m <<= 1; _ybase += 32; \
            } \
            _qoff += 32; \
        } \
    } while (0)

kernel void q3_gen(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]], constant Dims &dim [[buffer(3)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    uint row = tg * 32 + tid / 8; uint lane = tid % 8; float sum = 0.0f;
    if (row < dim.d) {
        uint nsb = dim.n / 256;
        device const uchar *wr = wb + (ulong)row * nsb * 110;
        for (uint sb = lane; sb < nsb; sb += 8) { Q3_BLOCK(wr + sb * 110, sum); }
    }
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane == 0 && row < dim.d) out[row] = sum;
}

kernel void q3_f4(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 20;
    uint row = tg * 32 + tid / 8; uint lane = tid % 8; float sum = 0.0f;
    device const uchar *wr = wb + (ulong)row * nsb * 110;
    #pragma clang loop unroll(disable)
    for (uint sb = lane; sb < nsb; sb += 8) {
        device const uchar *blk = wr + sb * 110;
        device const uchar *hm = blk;
        device const uchar *qs = blk + 32;
        device const uchar *scales = blk + 96;
        float d = (float)(*(device const half*)(blk + 108));
        uint aux0 = (uint)scales[0] | ((uint)scales[1] << 8) | ((uint)scales[2] << 16) | ((uint)scales[3] << 24);
        uint aux1 = (uint)scales[4] | ((uint)scales[5] << 8) | ((uint)scales[6] << 16) | ((uint)scales[7] << 24);
        uint tmp  = (uint)scales[8] | ((uint)scales[9] << 8) | ((uint)scales[10] << 16) | ((uint)scales[11] << 24);
        const uint KMASK1 = 0x03030303u; const uint KMASK2 = 0x0f0f0f0fu;
        uint aux2 = ((aux0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        uint aux3 = ((aux1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        aux0 = (aux0 & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
        aux1 = (aux1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
        uint ybase = sb * 256; uint qoff = 0; uchar m = 1; uint is = 0;
        for (uint h = 0; h < 2; h++) {
            uint shift = 0;
            for (uint ss = 0; ss < 4; ss++) {
                float dl = d * (float)((int)(char)q3k_scale_byte(is, aux0, aux1, aux2, aux3) - 32); is += 1;
                for (uint l = 0; l < 16; l += 4) {
                    uchar4 qv = uchar4(qs[qoff + l], qs[qoff + l + 1], qs[qoff + l + 2], qs[qoff + l + 3]);
                    uchar4 hv = uchar4(hm[l], hm[l + 1], hm[l + 2], hm[l + 3]);
                    float4 qq = float4(
                        (float)((int)((qv.x >> shift) & 3) - (int)((hv.x & m) == 0 ? 4 : 0)),
                        (float)((int)((qv.y >> shift) & 3) - (int)((hv.y & m) == 0 ? 4 : 0)),
                        (float)((int)((qv.z >> shift) & 3) - (int)((hv.z & m) == 0 ? 4 : 0)),
                        (float)((int)((qv.w >> shift) & 3) - (int)((hv.w & m) == 0 ? 4 : 0)));
                    float4 xv = *(device const float4*)(x + ybase + l);
                    sum += dl * dot(qq, xv);
                }
                dl = d * (float)((int)(char)q3k_scale_byte(is, aux0, aux1, aux2, aux3) - 32); is += 1;
                for (uint l = 0; l < 16; l += 4) {
                    uchar4 qv = uchar4(qs[qoff + 16 + l], qs[qoff + 16 + l + 1], qs[qoff + 16 + l + 2], qs[qoff + 16 + l + 3]);
                    uchar4 hv = uchar4(hm[16 + l], hm[16 + l + 1], hm[16 + l + 2], hm[16 + l + 3]);
                    float4 qq = float4(
                        (float)((int)((qv.x >> shift) & 3) - (int)((hv.x & m) == 0 ? 4 : 0)),
                        (float)((int)((qv.y >> shift) & 3) - (int)((hv.y & m) == 0 ? 4 : 0)),
                        (float)((int)((qv.z >> shift) & 3) - (int)((hv.z & m) == 0 ? 4 : 0)),
                        (float)((int)((qv.w >> shift) & 3) - (int)((hv.w & m) == 0 ? 4 : 0)));
                    float4 xv = *(device const float4*)(x + ybase + 16 + l);
                    sum += dl * dot(qq, xv);
                }
                shift += 2; m <<= 1; ybase += 32;
            }
            qoff += 32;
        }
    }
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 1);
    if (lane == 0) out[row] = sum;
}

kernel void q3_r2(
    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],
    device float *out [[buffer(2)]],
    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {
    constexpr uint nsb = 20;
    uint lane = tid % 8; uint slot = tid / 8;
    uint row0 = tg * 64 + slot; uint row1 = row0 + 32;
    float s0 = 0.0f, s1 = 0.0f;
    device const uchar *wr0 = wb + (ulong)row0 * nsb * 110;
    device const uchar *wr1 = wb + (ulong)row1 * nsb * 110;
    #pragma clang loop unroll(disable)
    for (uint sb = lane; sb < nsb; sb += 8) {
        Q3_BLOCK(wr0 + sb * 110, s0);
        Q3_BLOCK(wr1 + sb * 110, s1);
    }
    s0 += simd_shuffle_xor(s0, 4); s0 += simd_shuffle_xor(s0, 2); s0 += simd_shuffle_xor(s0, 1);
    s1 += simd_shuffle_xor(s1, 4); s1 += simd_shuffle_xor(s1, 2); s1 += simd_shuffle_xor(s1, 1);
    if (lane == 0) { out[row0] = s0; out[row1] = s1; }
}
`
