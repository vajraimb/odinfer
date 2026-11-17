// Lane x row sweep for short-K IQ GEMV (n=5120, d=17408).
// Helpers are shared so the ranking is within one compile. A winner still
// has to be re-timed inlined before it can ship.
package gemv_lane_sweep

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
THREADS :: 256

Try :: struct {
	name:   string,
	lanes:  int,
	rows:   int,
	block:  int,
	group:  int,
}

TRIES := []Try{
	{"iq1_l4", 4, 1, 50, 0},
	{"iq1_l4r2", 4, 2, 50, 0},
	{"iq1_l8", 8, 1, 50, 0},
	{"iq1_l8r2", 8, 2, 50, 0},
	{"iq1_l16", 16, 1, 50, 0},
	{"iq1_l16r2", 16, 2, 50, 0},
	{"iq2_l4", 4, 1, 66, 1},
	{"iq2_l4r2", 4, 2, 66, 1},
	{"iq2_l8", 8, 1, 66, 1},
	{"iq2_l8r2", 8, 2, 66, 1},
	{"iq2_l16", 16, 1, 66, 1},
	{"iq2_l16r2", 16, 2, 66, 1},
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
		HELPERS,
		emit_kernels(),
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

	wbytes := D * (N / 256) * 66
	wb := dev->newBufferWithLength(NS.UInteger(wbytes), MTL.ResourceStorageModeShared)
	xb := dev->newBufferWithLength(NS.UInteger(N * 4), MTL.ResourceStorageModeShared)
	ob := dev->newBufferWithLength(NS.UInteger(D * 4), MTL.ResourceStorageModeShared)
	defer wb->release(); defer xb->release(); defer ob->release()
	ws := wb->contentsAsSlice([]u8)
	for i in 0 ..< len(ws) { ws[i] = u8((i * 17 + 3) & 0xff) }
	for b in 0 ..< D * (N / 256) {
		ws[b * 66 + 0] = 0x00
		ws[b * 66 + 1] = 0x3C
		if b * 50 + 1 < len(ws) {
			ws[b * 50 + 0] = 0x00
			ws[b * 50 + 1] = 0x3C
		}
	}
	xs := xb->contentsAsSlice([]f32)
	for i in 0 ..< len(xs) { xs[i] = f32(i % 13) * 0.01 - 0.05 }

	psos := make([]^MTL.ComputePipelineState, len(TRIES))
	for t, i in TRIES { psos[i] = make_pso(dev, lib, t.name) }

	for g in 0 ..< 2 {
		ri := -1
		for t, i in TRIES {
			if t.group == g && t.lanes == 8 && t.rows == 1 { ri = i; break }
		}
		rt := TRIES[ri]
		dispatch(queue, psos[ri], wb, xb, ob, rows_per_tg(rt))
		ref := make([]f32, D)
		copy(ref, ob->contentsAsSlice([]f32)[:D])
		for t, i in TRIES {
			if t.group != g || i == ri { continue }
			dispatch(queue, psos[i], wb, xb, ob, rows_per_tg(t))
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
		tg := D / rows_per_tg(t)
		bytes := f64(D) * f64(N / 256) * f64(t.block)
		for _ in 0 ..< 4 {
			t0 := time.tick_now()
			cmd := queue->commandBuffer()
			enc := cmd->computeCommandEncoder()
			enc->setComputePipelineState(psos[i])
			enc->setBuffer(wb, 0, 0)
			enc->setBuffer(xb, 0, 1)
			enc->setBuffer(ob, 0, 2)
			for _ in 0 ..< REPS {
				enc->dispatchThreadgroups(MTL.Size{NS.Integer(tg), 1, 1}, MTL.Size{THREADS, 1, 1})
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

rows_per_tg :: proc(t: Try) -> int {
	return (THREADS / t.lanes) * t.rows
}

dispatch :: proc(queue: ^MTL.CommandQueue, pso: ^MTL.ComputePipelineState, wb, xb, ob: ^MTL.Buffer, rows: int) {
	cmd := queue->commandBuffer()
	enc := cmd->computeCommandEncoder()
	enc->setComputePipelineState(pso)
	enc->setBuffer(wb, 0, 0)
	enc->setBuffer(xb, 0, 1)
	enc->setBuffer(ob, 0, 2)
	enc->dispatchThreadgroups(MTL.Size{NS.Integer(D / rows), 1, 1}, MTL.Size{THREADS, 1, 1})
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

emit_kernels :: proc() -> string {
	b := strings.builder_make()
	for t in TRIES {
		slots := THREADS / t.lanes
		fmt.sbprintf(&b, "kernel void {}(\n", t.name)
		fmt.sbprintf(&b, "    device const uchar *wb [[buffer(0)]], device const float *x [[buffer(1)]],\n")
		fmt.sbprintf(&b, "    device float *out [[buffer(2)]],\n")
		fmt.sbprintf(&b, "    uint tg [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {{\n")
		fmt.sbprintf(&b, "    constexpr uint nsb = 20;\n")
		fmt.sbprintf(&b, "    uint lane = tid %% {}; uint slot = tid / {};\n", t.lanes, t.lanes)
		fmt.sbprintf(&b, "    uint row0 = tg * {} + slot;\n", slots * t.rows)
		fn := "iq1_acc"
		if t.group == 1 { fn = "iq2_acc" }
		if t.rows == 1 {
			fmt.sbprintf(&b, "    float s0 = 0.0f;\n")
			fmt.sbprintf(&b, "    device const uchar *a0 = wb + (ulong)row0 * nsb * {};\n", t.block)
			fmt.sbprintf(&b, "    #pragma clang loop unroll(disable)\n")
			fmt.sbprintf(&b, "    for (uint sb = lane; sb < nsb; sb += {}) {}(a0 + sb * {}, x + sb * 256, s0);\n", t.lanes, fn, t.block)
			write_reduce(&b, t.lanes, "s0")
			fmt.sbprintf(&b, "    if (lane == 0) out[row0] = s0;\n}}\n")
		} else {
			fmt.sbprintf(&b, "    uint row1 = row0 + {};\n", slots)
			fmt.sbprintf(&b, "    float s0 = 0.0f, s1 = 0.0f;\n")
			fmt.sbprintf(&b, "    device const uchar *a0 = wb + (ulong)row0 * nsb * {};\n", t.block)
			fmt.sbprintf(&b, "    device const uchar *a1 = wb + (ulong)row1 * nsb * {};\n", t.block)
			fmt.sbprintf(&b, "    #pragma clang loop unroll(disable)\n")
			fmt.sbprintf(&b, "    for (uint sb = lane; sb < nsb; sb += {}) {{\n", t.lanes)
			fmt.sbprintf(&b, "        device const float *xv = x + sb * 256;\n")
			fmt.sbprintf(&b, "        {}(a0 + sb * {}, xv, s0);\n", fn, t.block)
			fmt.sbprintf(&b, "        {}(a1 + sb * {}, xv, s1);\n", fn, t.block)
			fmt.sbprintf(&b, "    }}\n")
			write_reduce(&b, t.lanes, "s0")
			write_reduce(&b, t.lanes, "s1")
			fmt.sbprintf(&b, "    if (lane == 0) {{ out[row0] = s0; out[row1] = s1; }}\n}}\n")
		}
	}
	return strings.to_string(b)
}

write_reduce :: proc(b: ^strings.Builder, lanes: int, acc: string) {
	m := lanes / 2
	for m >= 1 {
		fmt.sbprintf(b, "    {} += simd_shuffle_xor({}, {});\n", acc, acc, m)
		m /= 2
	}
}

HELPERS :: `
inline void iq1_acc(device const uchar *blk, device const float *x, thread float &s) {
    float d = (float)(*(device const half*)blk);
    device const uchar *qs = blk + 2; device const uchar *qh = blk + 34;
    uint ybase = 0; uint qoff = 0;
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
                s += dl * (gv + delta) * x[ybase + j];
            }
            ybase += 8;
        }
        qoff += 4;
    }
}

inline void iq2_acc(device const uchar *blk, device const float *x, thread float &s) {
    float d = (float)(*(device const half*)blk);
    uint ybase = 0;
    #pragma clang loop unroll(disable)
    for (uint ib32 = 0; ib32 < 8; ib32++) {
        uint qb = 2 + 8 * ib32;
        uint aux = (uint)blk[qb + 4] | ((uint)blk[qb + 5] << 8) | ((uint)blk[qb + 6] << 16) | ((uint)blk[qb + 7] << 24);
        float db = d * (0.5f + (float)(aux >> 28)) * 0.25f;
        for (uint l = 0; l < 4; l++) {
            ulong g = iq2xxs_grid[blk[qb + l]];
            uchar sg = ksigns_iq2xs[(aux >> (7 * l)) & 127];
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 8; j++) {
                float v = (float)((g >> (8 * j)) & 0xff);
                float z = (sg & (1u << j)) == 0 ? 1.0f : -1.0f;
                s += db * v * z * x[ybase + j];
            }
            ybase += 8;
        }
    }
}
`
