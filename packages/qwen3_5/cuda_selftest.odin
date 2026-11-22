/* CUDA correctness checks against the CPU reference in packages/ggml and ops.odin.
   cuda_nvrtc_check compiles PTX and does not need a GPU.
   cuda_selftest needs a driver and a device (sm_70+). */

package qwen3_5

import cu "cuda:cuda"
import ggml "ggml:ggml"

import "core:c"
import "core:fmt"
import "core:math"
import "core:os"
import "core:strconv"
import "core:time"

@(private = "file")
_cuda_self_import_keep :: proc() {
	_ = cu.BACKEND_TAG
	_ = ggml.QK_K
	_ = c.int(0)
	_ = fmt.tprintf("")
	_ = math.abs(f32(0))
	_ = os.O_RDONLY
	_, _ = strconv.parse_int("0")
	_ = time.now()
}

when GPU_CUDA {

@(private)
s_dev: cu.Device
@(private)
s_ctx: bool
@(private)
s_mod: cu.Module

cuda_nvrtc_check :: proc(arch: string) -> bool {
	opt := fmt.tprintf("--gpu-architecture=%s", arch)
	ptx, log, err := cu.compile_ptx(CUDA_KERNEL_SRC, "odinfer_cuda.cu", []string{opt})
	if len(log) > 0 {
		fmt.eprintfln("nvrtc: %s", log)
	}
	if err != cu.OK {
		fmt.eprintf("cuda: NVRTC compile %s failed: %s\n", arch, cu.error_string(err))
		return false
	}
	fmt.eprintfln("cuda: NVRTC compiled %s (%d bytes PTX) via %s", arch, len(ptx), cu.nv_loaded_from)
	return true
}

@(private)
s_fn :: proc(name: string) -> cu.Function {
	fn, err := cu.module_function(s_mod, name)
	if err != cu.OK || fn == nil {
		fmt.eprintf("cuda_selftest: missing %s (%s)\n", name, cu.error_string(err))
		return nil
	}
	return fn
}

@(private)
s_close :: proc() {
	if s_mod != nil {
		cu.drv.ModuleUnload(s_mod)
		s_mod = nil
	}
	if s_ctx {
		cu.drv.DevicePrimaryCtxRelease_v2(s_dev)
		s_ctx = false
	}
}

@(private)
s_open :: proc() -> bool {
	if cu.load_driver() != cu.OK {
		fmt.eprintln("cuda_selftest: libcuda not found")
		return false
	}
	if cu.drv.Init(0) != cu.OK {
		fmt.eprintln("cuda_selftest: cuInit failed")
		return false
	}
	n: c.int
	if cu.drv.DeviceGetCount(&n) != cu.OK || n < 1 {
		fmt.eprintln("cuda_selftest: no CUDA device")
		return false
	}
	if cu.drv.DeviceGet(&s_dev, 0) != cu.OK {
		return false
	}
	major, minor: c.int
	cu.drv.DeviceGetAttribute(&major, cu.ATTR_COMPUTE_CAPABILITY_MAJOR, s_dev)
	cu.drv.DeviceGetAttribute(&minor, cu.ATTR_COMPUTE_CAPABILITY_MINOR, s_dev)
	ctx: cu.Context
	if cu.drv.DevicePrimaryCtxRetain(&ctx, s_dev) != cu.OK {
		return false
	}
	s_ctx = true
	if cu.drv.CtxSetCurrent(ctx) != cu.OK {
		s_close()
		return false
	}
	arch := fmt.tprintf("compute_%d%d", int(major), int(minor))
	opt := fmt.tprintf("--gpu-architecture=%s", arch)
	ptx, log, err := cu.compile_ptx(CUDA_KERNEL_SRC, "odinfer_cuda.cu", []string{opt})
	if len(log) > 0 {
		fmt.eprintfln("nvrtc: %s", log)
	}
	if err != cu.OK {
		fmt.eprintf("cuda_selftest: compile %s failed: %s\n", arch, cu.error_string(err))
		s_close()
		return false
	}
	mod, jit, merr := cu.module_load_ptx(ptx)
	if len(jit) > 0 {
		fmt.eprintfln("jit: %s", jit)
	}
	if merr != cu.OK {
		fmt.eprintf("cuda_selftest: module load failed: %s (%d)\n", cu.error_string(merr), int(merr))
		s_close()
		return false
	}
	s_mod = mod
	fmt.eprintfln("cuda_selftest: device sm_%d%d arch %s", int(major), int(minor), arch)
	return true
}

Case_Err :: struct {
	max_abs: f32,
	tol:     f32,
	bad:     bool,
	printed: int,
}

@(private)
s_note :: proc(st: ^Case_Err, name: string, got, want, atol, rtol: f32) {
	d := math.abs(got - want)
	tol := atol + rtol * math.abs(want)
	if math.is_nan(d) || d > st.max_abs {
		st.max_abs = d
		st.tol = tol
	}
	if math.is_nan(got) || math.is_inf(got) || d > tol {
		st.bad = true
		if st.printed < 4 {
			fmt.eprintf("cuda_selftest: %s got %.6g want %.6g diff %.3g tol %.3g\n", name, got, want, d, tol)
			st.printed += 1
		}
	}
}

// Printed for every case, pass or fail. tol is the allowance at the element
// with the largest absolute error (atol + rtol*|ref|).
@(private)
s_report :: proc(label: string, st: Case_Err, atol, rtol: f32) -> bool {
	status := "PASS"
	if st.bad {
		status = "FAIL"
	}
	fmt.eprintfln("cuda_selftest: %s %s  max_abs=%.6g  tol=%.6g  (atol=%.3g rtol=%.3g)", status, label, st.max_abs, st.tol, atol, rtol)
	return !st.bad
}

@(private)
fill_weight :: proc(kind: ggml.GGML_Type, rows, cols: int) -> []u8 {
	elems, bsz := ggml.ggml_block_info(kind)
	nb := cols / elems
	raw := make([]u8, rows * nb * bsz)
	if kind == .F32 {
		f := (cast([^]f32)raw_data(raw))[:rows * cols]
		for i in 0 ..< rows * cols {
			f[i] = (f32(i % 7) - 3) * 0.02
		}
		return raw
	}
	if kind == .F16 {
		for i in 0 ..< rows * cols {
			write_f16_le(raw[i * 2:], (f32(i % 7) - 3) * 0.02)
		}
		return raw
	}
	for i in 0 ..< len(raw) {
		raw[i] = u8((i * 17 + 9) & 0xFF)
	}
	for b in 0 ..< rows * nb {
		base := b * bsz
		#partial switch kind {
		case .Q8_K:
			bits := transmute(u32)(f32(0.02))
			raw[base + 0] = u8(bits)
			raw[base + 1] = u8(bits >> 8)
			raw[base + 2] = u8(bits >> 16)
			raw[base + 3] = u8(bits >> 24)
		case .Q2_K:
			write_f16_le(raw[base + 80:], 0.05)
			write_f16_le(raw[base + 82:], 0.01)
		case .Q3_K:
			write_f16_le(raw[base + 108:], 0.05)
		case .Q6_K:
			write_f16_le(raw[base + 208:], 0.02)
		case .Q4_1, .Q5_1:
			write_f16_le(raw[base:], 0.05)
			write_f16_le(raw[base + 2:], 0.01)
		case .Q4_K, .Q5_K:
			write_f16_le(raw[base:], 0.05)
			write_f16_le(raw[base + 2:], 0.01)
		case:
			write_f16_le(raw[base:], 0.05)
		}
	}
	return raw
}

@(private)
cpu_gemv :: proc(kind: ggml.GGML_Type, raw: []u8, x: []f32, rows, cols: int) -> []f32 {
	elems, bsz := ggml.ggml_block_info(kind)
	rb := (cols / elems) * bsz
	out := make([]f32, rows)
	scratch := make([]f32, cols)
	defer delete(scratch)
	for r in 0 ..< rows {
		ggml.dequant_row(kind, raw[r * rb:], cols, scratch)
		sum: f32 = 0
		for j in 0 ..< cols {
			sum += scratch[j] * x[j]
		}
		out[r] = sum
	}
	return out
}

@(private)
run_gemv :: proc(kind: ggml.GGML_Type, rows, cols: int, atol, rtol: f32) -> bool {
	tag := cuda_kind_tag(kind)
	fn := s_fn(fmt.tprintf("gemv_%s", tag))
	if fn == nil {
		return false
	}
	raw := fill_weight(kind, rows, cols)
	defer delete(raw)
	x := make([]f32, cols)
	defer delete(x)
	for i in 0 ..< cols {
		x[i] = (f32(i % 5) - 2) * 0.05
	}
	ref := cpu_gemv(kind, raw, x, rows, cols)
	defer delete(ref)
	dw, e1 := cu.alloc(len(raw))
	dx, e2 := cu.alloc(cols * 4)
	dy, e3 := cu.alloc(rows * 4)
	defer cu.free(dw)
	defer cu.free(dx)
	defer cu.free(dy)
	if e1 != cu.OK || e2 != cu.OK || e3 != cu.OK {
		fmt.eprintln("cuda_selftest: alloc failed")
		return false
	}
	cu.upload(dw, raw_data(raw), len(raw))
	cu.upload(dx, raw_data(x), cols * 4)
	wb, xx, yy := dw, dx, dy
	nn, dd := i32(cols), i32(rows)
	args := [?]rawptr{&wb, &xx, &yy, &nn, &dd}
	if cu.launch(fn, u32(rows), 1, 1, 32, 1, 1, 0, args[:]) != cu.OK {
		fmt.eprintf("cuda_selftest: gemv %s launch failed\n", tag)
		return false
	}
	got := make([]f32, rows)
	defer delete(got)
	cu.download(raw_data(got), dy, rows * 4)
	st: Case_Err
	for r in 0 ..< rows {
		s_note(&st, fmt.tprintf("gemv %s row %d", tag, r), got[r], ref[r], atol, rtol)
	}
	return s_report(fmt.tprintf("gemv %s", tag), st, atol, rtol)
}

@(private)
run_gemv2 :: proc(kind: ggml.GGML_Type, rows, cols: int, atol, rtol: f32) -> bool {
	tag := cuda_kind_tag(kind)
	fn := s_fn(fmt.tprintf("gemv2_%s", tag))
	qfn := s_fn("quant_q8")
	if fn == nil || qfn == nil {
		fmt.eprintfln("cuda_selftest: gemv2 %s missing", tag)
		return false
	}
	raw := fill_weight(kind, rows, cols)
	defer delete(raw)
	x := make([]f32, cols)
	defer delete(x)
	for i in 0 ..< cols {
		x[i] = (f32(i % 5) - 2) * 0.05
	}
	ref := cpu_gemv(kind, raw, x, rows, cols)
	defer delete(ref)
	dw, e1 := cu.alloc(len(raw) + 64)
	dx, e2 := cu.alloc(cols * 4)
	dy, e3 := cu.alloc(rows * 4)
	ng := cols / 32
	dqs, e4 := cu.alloc(cols)
	dsc, e5 := cu.alloc(ng * 4)
	dsm, e6 := cu.alloc(ng * 8)
	defer cu.free(dw)
	defer cu.free(dx)
	defer cu.free(dy)
	defer cu.free(dqs)
	defer cu.free(dsc)
	defer cu.free(dsm)
	if e1 != cu.OK || e2 != cu.OK || e3 != cu.OK || e4 != cu.OK || e5 != cu.OK || e6 != cu.OK {
		fmt.eprintln("cuda_selftest: gemv2 alloc failed")
		return false
	}
	cu.upload(dw, raw_data(raw), len(raw))
	cu.upload(dx, raw_data(x), cols * 4)
	xx, qs, sc, sm := dx, dqs, dsc, dsm
	nn := i32(cols)
	qargs := [?]rawptr{&xx, &qs, &sc, &sm, &nn}
	if cu.launch(qfn, u32((ng + 127) / 128), 1, 1, 128, 1, 1, 0, qargs[:]) != cu.OK {
		fmt.eprintfln("cuda_selftest: quant_q8 launch failed (%s)", tag)
		return false
	}
	wb, yy := dw, dy
	dd := i32(rows)
	acc := i32(0)
	warps := 4
	gx := u32((rows + warps * 4 - 1) / (warps * 4))
	bx := u32(warps * 32)
	launched: bool
	if kind == .Q4_K || kind == .Q5_K {
		args := [?]rawptr{&wb, &qs, &sc, &sm, &yy, &nn, &dd, &acc}
		launched = cu.launch(fn, gx, 1, 1, bx, 1, 1, 0, args[:]) == cu.OK
	} else {
		args := [?]rawptr{&wb, &qs, &sc, &yy, &nn, &dd, &acc}
		launched = cu.launch(fn, gx, 1, 1, bx, 1, 1, 0, args[:]) == cu.OK
	}
	if !launched || cu.sync() != cu.OK {
		fmt.eprintfln("cuda_selftest: gemv2 launch failed (%s)", tag)
		return false
	}
	got := make([]f32, rows)
	defer delete(got)
	cu.download(raw_data(got), dy, rows * 4)
	st: Case_Err
	for r in 0 ..< rows {
		s_note(&st, fmt.tprintf("gemv2 %s row %d", tag, r), got[r], ref[r], atol, rtol)
	}
	ok := s_report(fmt.tprintf("gemv2 %s %dx%d", tag, rows, cols), st, atol, rtol)
	if !ok {
		return false
	}
	if kind == .Q4_K {
		nfn := s_fn("gemv2_q4nc_k")
		xfn := s_fn("gemv2_q4x_k")
		if nfn == nil || xfn == nil {
			fmt.eprintln("cuda_selftest: gemv2_q4nc_k or gemv2_q4x_k missing")
			return false
		}
		acc = 0
		nargs := [?]rawptr{&wb, &qs, &sc, &sm, &yy, &nn, &dd, &acc}
		if cu.launch(nfn, gx, 1, 1, bx, 1, 1, 0, nargs[:]) != cu.OK || cu.sync() != cu.OK {
			fmt.eprintln("cuda_selftest: gemv2_q4nc_k launch failed")
			return false
		}
		cu.download(raw_data(got), dy, rows * 4)
		st = {}
		for r in 0 ..< rows {
			s_note(&st, fmt.tprintf("gemv2_q4nc row %d", r), got[r], ref[r], atol, rtol)
		}
		if !s_report(fmt.tprintf("gemv2_q4nc %dx%d", rows, cols), st, atol, rtol) {
			return false
		}
		gx8 := u32((rows + warps * 8 - 1) / (warps * 8))
		acc = 0
		xargs := [?]rawptr{&wb, &qs, &sc, &sm, &yy, &nn, &dd, &acc}
		if cu.launch(xfn, gx8, 1, 1, bx, 1, 1, 0, xargs[:]) != cu.OK || cu.sync() != cu.OK {
			fmt.eprintln("cuda_selftest: gemv2_q4x_k launch failed")
			return false
		}
		cu.download(raw_data(got), dy, rows * 4)
		st = {}
		for r in 0 ..< rows {
			s_note(&st, fmt.tprintf("gemv2_q4x row %d", r), got[r], ref[r], atol, rtol)
		}
		if !s_report(fmt.tprintf("gemv2_q4x %dx%d", rows, cols), st, atol, rtol) {
			return false
		}
		if cols / 256 >= 2 {
			if cu.memset(dy, 0, rows * 4) != cu.OK {
				fmt.eprintln("cuda_selftest: q4x split memset failed")
				return false
			}
			acc = 0
			sargs := [?]rawptr{&wb, &qs, &sc, &sm, &yy, &nn, &dd, &acc}
			if cu.launch(xfn, gx8, 2, 1, bx, 1, 1, 0, sargs[:]) != cu.OK || cu.sync() != cu.OK {
				fmt.eprintln("cuda_selftest: gemv2_q4x split launch failed")
				return false
			}
			cu.download(raw_data(got), dy, rows * 4)
			st = {}
			for r in 0 ..< rows {
				s_note(&st, fmt.tprintf("gemv2_q4x split row %d", r), got[r], ref[r], atol, rtol)
			}
		if !s_report(fmt.tprintf("gemv2_q4x split %dx%d", rows, cols), st, atol, rtol) {
			return false
		}
		}
		wfn := s_fn("gemv2_q4w_k")
		if wfn == nil {
			fmt.eprintln("cuda_selftest: gemv2_q4w_k missing")
			return false
		}
		gxw := u32((rows + warps - 1) / warps)
		acc = 0
		wargs := [?]rawptr{&wb, &qs, &sc, &sm, &yy, &nn, &dd, &acc}
		if cu.launch(wfn, gxw, 1, 1, bx, 1, 1, 0, wargs[:]) != cu.OK || cu.sync() != cu.OK {
			fmt.eprintln("cuda_selftest: gemv2_q4w_k launch failed")
			return false
		}
		cu.download(raw_data(got), dy, rows * 4)
		st = {}
		for r in 0 ..< rows {
			s_note(&st, fmt.tprintf("gemv2_q4w row %d", r), got[r], ref[r], atol, rtol)
		}
		if !s_report(fmt.tprintf("gemv2_q4w %dx%d", rows, cols), st, atol, rtol) {
			return false
		}
		pfn := s_fn("gemv2_q4_pair_k")
		if pfn == nil {
			fmt.eprintln("cuda_selftest: gemv2_q4_pair_k missing")
			return false
		}
		raw2 := fill_weight(kind, rows, cols)
		defer delete(raw2)
		for i in 0 ..< len(raw2) {
			raw2[i] += 3
		}
		dw2, ep2 := cu.alloc(len(raw2) + 64)
		dy0, ey0 := cu.alloc(rows * 4)
		dy1, ey1 := cu.alloc(rows * 4)
		defer cu.free(dw2)
		defer cu.free(dy0)
		defer cu.free(dy1)
		if ep2 != cu.OK || ey0 != cu.OK || ey1 != cu.OK {
			fmt.eprintln("cuda_selftest: gemv2_q4_pair alloc failed")
			return false
		}
		cu.upload(dw2, raw_data(raw2), len(raw2))
		w2p, y0p, y1p := dw2, dy0, dy1
		acc = 0
		one := [?]rawptr{&w2p, &qs, &sc, &sm, &y1p, &nn, &dd, &acc}
		if cu.launch(fn, gx, 1, 1, bx, 1, 1, 0, one[:]) != cu.OK || cu.sync() != cu.OK {
			fmt.eprintln("cuda_selftest: gemv2_q4 pair reference launch failed")
			return false
		}
		ref1 := make([]f32, rows)
		defer delete(ref1)
		cu.download(raw_data(ref1), dy1, rows * 4)
		acc = 0
		back := [?]rawptr{&wb, &qs, &sc, &sm, &yy, &nn, &dd, &acc}
		if cu.launch(fn, gx, 1, 1, bx, 1, 1, 0, back[:]) != cu.OK || cu.sync() != cu.OK {
			fmt.eprintln("cuda_selftest: gemv2_q4 pair reference launch failed")
			return false
		}
		cu.download(raw_data(got), dy, rows * 4)
		pargs := [?]rawptr{&wb, &w2p, &qs, &sc, &sm, &y0p, &y1p, &nn, &dd}
		if cu.launch(pfn, gx, 1, 1, bx, 1, 1, 0, pargs[:]) != cu.OK || cu.sync() != cu.OK {
			fmt.eprintln("cuda_selftest: gemv2_q4_pair_k launch failed")
			return false
		}
		pair0 := make([]f32, rows)
		pair1 := make([]f32, rows)
		defer delete(pair0)
		defer delete(pair1)
		cu.download(raw_data(pair0), dy0, rows * 4)
		cu.download(raw_data(pair1), dy1, rows * 4)
		st = {}
		for r in 0 ..< rows {
			s_note(&st, fmt.tprintf("gemv2_q4_pair y0 row %d", r), pair0[r], got[r], 1e-4, 1e-5)
			s_note(&st, fmt.tprintf("gemv2_q4_pair y1 row %d", r), pair1[r], ref1[r], 1e-4, 1e-5)
		}
		if !s_report(fmt.tprintf("gemv2_q4_pair %dx%d", rows, cols), st, 1e-4, 1e-5) {
			return false
		}
	}
	if kind == .Q6_K {
		sfn := s_fn("gemv2_q6s_k")
		if sfn == nil {
			fmt.eprintln("cuda_selftest: gemv2_q6s_k missing")
			return false
		}
		args := [?]rawptr{&wb, &qs, &sc, &yy, &nn, &dd, &acc}
		if cu.launch(sfn, gx, 1, 1, bx, 1, 1, 0, args[:]) != cu.OK || cu.sync() != cu.OK {
			fmt.eprintln("cuda_selftest: gemv2_q6s_k launch failed")
			return false
		}
		cu.download(raw_data(got), dy, rows * 4)
		st = {}
		for r in 0 ..< rows {
			s_note(&st, fmt.tprintf("gemv2_q6s row %d", r), got[r], ref[r], atol, rtol)
		}
		if !s_report(fmt.tprintf("gemv2_q6s %dx%d", rows, cols), st, atol, rtol) {
			return false
		}
	}
	if kind == .Q4_K || kind == .Q6_K {
		rname := "gemv2r1_q4_k"
		if kind == .Q6_K {
			rname = "gemv2r1_q6_k"
		}
		rfn := s_fn(rname)
		if rfn == nil {
			fmt.eprintfln("cuda_selftest: %s missing", rname)
			return false
		}
		gx1 := u32((rows + warps - 1) / warps)
		acc = 0
		launched = false
		if kind == .Q4_K {
			args := [?]rawptr{&wb, &qs, &sc, &sm, &yy, &nn, &dd, &acc}
			launched = cu.launch(rfn, gx1, 1, 1, bx, 1, 1, 0, args[:]) == cu.OK
		} else {
			args := [?]rawptr{&wb, &qs, &sc, &yy, &nn, &dd, &acc}
			launched = cu.launch(rfn, gx1, 1, 1, bx, 1, 1, 0, args[:]) == cu.OK
		}
		if !launched || cu.sync() != cu.OK {
			fmt.eprintfln("cuda_selftest: %s launch failed", rname)
			return false
		}
		cu.download(raw_data(got), dy, rows * 4)
		st = {}
		for r in 0 ..< rows {
			s_note(&st, fmt.tprintf("%s row %d", rname, r), got[r], ref[r], atol, rtol)
		}
		if !s_report(fmt.tprintf("%s %dx%d", rname, rows, cols), st, atol, rtol) {
			return false
		}
		// accumulate=1 adds onto whatever is already in the output.
		bias := make([]f32, rows)
		defer delete(bias)
		for r in 0 ..< rows {
			bias[r] = 0.25
		}
		cu.upload(dy, raw_data(bias), rows * 4)
		acc = 1
		if kind == .Q4_K {
			args := [?]rawptr{&wb, &qs, &sc, &sm, &yy, &nn, &dd, &acc}
			launched = cu.launch(rfn, gx1, 1, 1, bx, 1, 1, 0, args[:]) == cu.OK
		} else {
			args := [?]rawptr{&wb, &qs, &sc, &yy, &nn, &dd, &acc}
			launched = cu.launch(rfn, gx1, 1, 1, bx, 1, 1, 0, args[:]) == cu.OK
		}
		if !launched || cu.sync() != cu.OK {
			fmt.eprintfln("cuda_selftest: %s accumulate launch failed", rname)
			return false
		}
		cu.download(raw_data(got), dy, rows * 4)
		st = {}
		for r in 0 ..< rows {
			s_note(&st, fmt.tprintf("%s accum row %d", rname, r), got[r], ref[r] + 0.25, atol, rtol)
		}
		if !s_report(fmt.tprintf("%s accumulate %dx%d", rname, rows, cols), st, atol, rtol) {
			return false
		}
		if cols / 256 >= 2 {
			if cu.memset(dy, 0, rows * 4) != cu.OK {
				fmt.eprintln("cuda_selftest: split memset failed")
				return false
			}
			acc = 0
			launched = false
			if kind == .Q4_K {
				args := [?]rawptr{&wb, &qs, &sc, &sm, &yy, &nn, &dd, &acc}
				launched = cu.launch(fn, gx, 2, 1, bx, 1, 1, 0, args[:]) == cu.OK
			} else {
				args := [?]rawptr{&wb, &qs, &sc, &yy, &nn, &dd, &acc}
				launched = cu.launch(fn, gx, 2, 1, bx, 1, 1, 0, args[:]) == cu.OK
			}
			if !launched || cu.sync() != cu.OK {
				fmt.eprintfln("cuda_selftest: gemv2 split launch failed (%s)", tag)
				return false
			}
			cu.download(raw_data(got), dy, rows * 4)
			st = {}
			for r in 0 ..< rows {
				s_note(&st, fmt.tprintf("gemv2 split %s row %d", tag, r), got[r], ref[r], atol, rtol)
			}
			if !s_report(fmt.tprintf("gemv2 split %s %dx%d", tag, rows, cols), st, atol, rtol) {
				return false
			}
		}
	}
	return true
}

@(private)
run_mmq :: proc(kind: ggml.GGML_Type, rows, cols, T: int) -> bool {
	tag := cuda_kind_tag(kind)
	fn := s_fn(fmt.tprintf("mmq_%s", tag))
	if fn == nil {
		return false
	}
	raw := fill_weight(kind, rows, cols)
	defer delete(raw)
	x := make([]f32, T * cols)
	defer delete(x)
	for t in 0 ..< T {
		for j in 0 ..< cols {
			x[t * cols + j] = (f32((t + 1) * (j + 3) % 9) - 4) * 0.03
		}
	}
	ref := make([]f32, T * rows)
	defer delete(ref)
	for t in 0 ..< T {
		row := cpu_gemv(kind, raw, x[t * cols:][:cols], rows, cols)
		copy(ref[t * rows:][:rows], row)
		delete(row)
	}
	dw, _ := cu.alloc(len(raw))
	dx, _ := cu.alloc(T * cols * 4)
	dy, _ := cu.alloc(T * rows * 4)
	defer cu.free(dw)
	defer cu.free(dx)
	defer cu.free(dy)
	cu.upload(dw, raw_data(raw), len(raw))
	cu.upload(dx, raw_data(x), T * cols * 4)
	wb, xx, yy := dw, dx, dy
	nn, dd, tt := i32(cols), i32(rows), i32(T)
	args := [?]rawptr{&wb, &xx, &yy, &nn, &dd, &tt}
	if cu.launch(fn, u32(rows), 1, 1, 32, 1, 1, 0, args[:]) != cu.OK {
		fmt.eprintf("cuda_selftest: mmq %s launch failed\n", tag)
		return false
	}
	got := make([]f32, T * rows)
	defer delete(got)
	cu.download(raw_data(got), dy, T * rows * 4)
	st: Case_Err
	atol, rtol := f32(2e-2), f32(1e-3)
	for i in 0 ..< T * rows {
		s_note(&st, fmt.tprintf("mmq %s [%d]", tag, i), got[i], ref[i], atol, rtol)
	}
	return s_report(fmt.tprintf("mmq %s T=%d", tag, T), st, atol, rtol)
}

@(private)
run_tc :: proc(kind: ggml.GGML_Type, rows, cols, T: int, atol, rtol: f32) -> bool {
	tag := cuda_kind_tag(kind)
	fn := s_fn(fmt.tprintf("tc_%s", tag))
	if fn == nil {
		fmt.eprintf("cuda_selftest: missing tc_%s\n", tag)
		return false
	}
	raw := fill_weight(kind, rows, cols)
	defer delete(raw)
	x := make([]f32, T * cols)
	defer delete(x)
	for t in 0 ..< T {
		for j in 0 ..< cols {
			x[t * cols + j] = (f32((t + 1) * (j + 3) % 9) - 4) * 0.03
		}
	}
	ref := make([]f32, T * rows)
	defer delete(ref)
	for t in 0 ..< T {
		row := cpu_gemv(kind, raw, x[t * cols:][:cols], rows, cols)
		copy(ref[t * rows:][:rows], row)
		delete(row)
	}
	dw, _ := cu.alloc(len(raw))
	dx, _ := cu.alloc(T * cols * 4)
	dy, _ := cu.alloc(T * rows * 4)
	defer cu.free(dw)
	defer cu.free(dx)
	defer cu.free(dy)
	cu.upload(dw, raw_data(raw), len(raw))
	cu.upload(dx, raw_data(x), T * cols * 4)
	wb, xx, yy := dw, dx, dy
	nn, dd, tt := i32(cols), i32(rows), i32(T)
	args := [?]rawptr{&wb, &xx, &yy, &nn, &dd, &tt}
	gx := u32((rows + 63) / 64)
	gy := u32((T + 63) / 64)
	if cu.launch(fn, gx, gy, 1, 512, 1, 1, 0, args[:]) != cu.OK {
		fmt.eprintf("cuda_selftest: tc %s launch failed\n", tag)
		return false
	}
	got := make([]f32, T * rows)
	defer delete(got)
	cu.download(raw_data(got), dy, T * rows * 4)
	st: Case_Err
	for i in 0 ..< T * rows {
		s_note(&st, fmt.tprintf("tc %s [%d]", tag, i), got[i], ref[i], atol, rtol)
	}
	return s_report(fmt.tprintf("tc %s T=%d rows=%d cols=%d", tag, T, rows, cols), st, atol, rtol)
}

@(private)
run_tc2 :: proc(kind: ggml.GGML_Type, rows, cols, T: int, atol, rtol: f32, wide: bool) -> bool {
	name := "tc2s_q4_k"
	if wide {
		name = "tc2_q4_k"
	}
	if kind == .Q6_K {
		name = "tc2s_q6_k"
		if wide {
			name = "tc2_q6_k"
		}
	}
	fn := s_fn(name)
	if fn == nil {
		fmt.eprintf("cuda_selftest: missing %s\n", name)
		return false
	}
	raw := fill_weight(kind, rows, cols)
	defer delete(raw)
	x := make([]f32, T * cols)
	defer delete(x)
	for t in 0 ..< T {
		for j in 0 ..< cols {
			x[t * cols + j] = (f32((t + 1) * (j + 3) % 9) - 4) * 0.03
		}
	}
	ref := make([]f32, T * rows)
	defer delete(ref)
	for t in 0 ..< T {
		row := cpu_gemv(kind, raw, x[t * cols:][:cols], rows, cols)
		copy(ref[t * rows:][:rows], row)
		delete(row)
	}
	dw, _ := cu.alloc(len(raw))
	dx, _ := cu.alloc(T * cols * 4)
	dy, _ := cu.alloc(T * rows * 4)
	defer cu.free(dw)
	defer cu.free(dx)
	defer cu.free(dy)
	cu.upload(dw, raw_data(raw), len(raw))
	cu.upload(dx, raw_data(x), T * cols * 4)
	wb, xx, yy := dw, dx, dy
	nn, dd, tt := i32(cols), i32(rows), i32(T)
	args := [?]rawptr{&wb, &xx, &yy, &nn, &dd, &tt}
	gx := u32((rows + 31) / 32)
	gy := u32((T + 31) / 32)
	if wide {
		gy = u32((T + 127) / 128)
	}
	if cu.launch(fn, gx, gy, 1, 256, 1, 1, 0, args[:]) != cu.OK {
		fmt.eprintf("cuda_selftest: %s launch failed\n", name)
		return false
	}
	got := make([]f32, T * rows)
	defer delete(got)
	cu.download(raw_data(got), dy, T * rows * 4)
	st: Case_Err
	for i in 0 ..< T * rows {
		s_note(&st, fmt.tprintf("%s [%d]", name, i), got[i], ref[i], atol, rtol)
	}
	return s_report(fmt.tprintf("%s T=%d rows=%d cols=%d", name, T, rows, cols), st, atol, rtol)
}

@(private)
run_tc2m :: proc(kind: ggml.GGML_Type, rows, cols, T: int, atol, rtol: f32) -> bool {
	name := "tc2m_q4_k"
	if kind == .Q6_K {
		name = "tc2m_q6_k"
	}
	fn := s_fn(name)
	if fn == nil {
		fmt.eprintf("cuda_selftest: missing %s\n", name)
		return false
	}
	raw := fill_weight(kind, rows, cols)
	defer delete(raw)
	x := make([]f32, T * cols)
	defer delete(x)
	for t in 0 ..< T {
		for j in 0 ..< cols {
			x[t * cols + j] = (f32((t + 1) * (j + 3) % 9) - 4) * 0.03
		}
	}
	ref := make([]f32, T * rows)
	defer delete(ref)
	for t in 0 ..< T {
		row := cpu_gemv(kind, raw, x[t * cols:][:cols], rows, cols)
		copy(ref[t * rows:][:rows], row)
		delete(row)
	}
	dw, _ := cu.alloc(len(raw))
	dx, _ := cu.alloc(T * cols * 4)
	dy, _ := cu.alloc(T * rows * 4)
	defer cu.free(dw)
	defer cu.free(dx)
	defer cu.free(dy)
	cu.upload(dw, raw_data(raw), len(raw))
	cu.upload(dx, raw_data(x), T * cols * 4)
	wb, xx, yy := dw, dx, dy
	nn, dd, tt := i32(cols), i32(rows), i32(T)
	args := [?]rawptr{&wb, &xx, &yy, &nn, &dd, &tt}
	gx := u32((rows + 63) / 64)
	gy := u32((T + 63) / 64)
	if cu.launch(fn, gx, gy, 1, 512, 1, 1, 0, args[:]) != cu.OK {
		fmt.eprintf("cuda_selftest: %s launch failed\n", name)
		return false
	}
	got := make([]f32, T * rows)
	defer delete(got)
	cu.download(raw_data(got), dy, T * rows * 4)
	st: Case_Err
	for i in 0 ..< T * rows {
		s_note(&st, fmt.tprintf("%s [%d]", name, i), got[i], ref[i], atol, rtol)
	}
	return s_report(fmt.tprintf("%s T=%d rows=%d cols=%d", name, T, rows, cols), st, atol, rtol)
}

// pgemm is not launched. On the V100 it was slower than the scalar tile.
// The kernel stays so this compare still runs. fp16 products against the
// fp32 CPU GEMV. A swapped WMMA layout is O(1), not this tolerance.
@(private)
run_pgemm :: proc(kind: ggml.GGML_Type, rows, cols, T: int, atol, rtol: f32) -> bool {
	name := "pgemm_q4_k"
	if kind == .Q6_K {
		name = "pgemm_q6_k"
	}
	fn := s_fn(name)
	if fn == nil {
		fmt.eprintf("cuda_selftest: missing %s\n", name)
		return false
	}
	raw := fill_weight(kind, rows, cols)
	defer delete(raw)
	x := make([]f32, T * cols)
	defer delete(x)
	for t in 0 ..< T {
		for j in 0 ..< cols {
			x[t * cols + j] = (f32((t + 1) * (j + 3) % 9) - 4) * 0.03
		}
	}
	ref := make([]f32, T * rows)
	defer delete(ref)
	for t in 0 ..< T {
		row := cpu_gemv(kind, raw, x[t * cols:][:cols], rows, cols)
		copy(ref[t * rows:][:rows], row)
		delete(row)
	}
	dw, _ := cu.alloc(len(raw))
	dx, _ := cu.alloc(T * cols * 4)
	dy, _ := cu.alloc(T * rows * 4)
	defer cu.free(dw)
	defer cu.free(dx)
	defer cu.free(dy)
	cu.upload(dw, raw_data(raw), len(raw))
	cu.upload(dx, raw_data(x), T * cols * 4)
	wb, xx, yy := dw, dx, dy
	nn, dd, tt := i32(cols), i32(rows), i32(T)
	args := [?]rawptr{&wb, &xx, &yy, &nn, &dd, &tt}
	gx := u32((rows + 63) / 64)
	gy := u32((T + 63) / 64)
	if cu.launch(fn, gx, gy, 1, 512, 1, 1, 0, args[:]) != cu.OK {
		fmt.eprintf("cuda_selftest: %s launch failed\n", name)
		return false
	}
	got := make([]f32, T * rows)
	defer delete(got)
	cu.download(raw_data(got), dy, T * rows * 4)
	st: Case_Err
	for i in 0 ..< T * rows {
		s_note(&st, fmt.tprintf("%s [%d]", name, i), got[i], ref[i], atol, rtol)
	}
	return s_report(fmt.tprintf("%s T=%d rows=%d cols=%d", name, T, rows, cols), st, atol, rtol)
}

// wgemm dequants to fp16 and accumulates with WMMA. Same allowance as pgemm:
// atol 5e-2, rtol 5e-2 against the fp32 CPU GEMV. Not bit-exact.
@(private)
run_wgemm :: proc(kind: ggml.GGML_Type, rows, cols, T: int, atol, rtol: f32) -> bool {
	name := "wgemm_q4_k"
	if kind == .Q6_K {
		name = "wgemm_q6_k"
	} else if kind == .Q5_K {
		name = "wgemm_q5_k"
	}
	fn := s_fn(name)
	if fn == nil {
		fmt.eprintf("cuda_selftest: missing %s\n", name)
		return false
	}
	if cu.func_set_max_dynamic_shared(fn, int(WG_SMEM)) != cu.OK {
		fmt.eprintf("cuda_selftest: %s cuFuncSetAttribute failed\n", name)
		return false
	}
	raw := fill_weight(kind, rows, cols)
	defer delete(raw)
	x := make([]f32, T * cols)
	defer delete(x)
	for t in 0 ..< T {
		for j in 0 ..< cols {
			x[t * cols + j] = (f32((t + 1) * (j + 3) % 9) - 4) * 0.03
		}
	}
	ref := make([]f32, T * rows)
	defer delete(ref)
	for t in 0 ..< T {
		row := cpu_gemv(kind, raw, x[t * cols:][:cols], rows, cols)
		copy(ref[t * rows:][:rows], row)
		delete(row)
	}
	dw, _ := cu.alloc(len(raw))
	dx, _ := cu.alloc(T * cols * 4)
	dy, _ := cu.alloc(T * rows * 4)
	defer cu.free(dw)
	defer cu.free(dx)
	defer cu.free(dy)
	cu.upload(dw, raw_data(raw), len(raw))
	cu.upload(dx, raw_data(x), T * cols * 4)
	wb, xx, yy := dw, dx, dy
	nn, dd, tt := i32(cols), i32(rows), i32(T)
	args := [?]rawptr{&wb, &xx, &yy, &nn, &dd, &tt}
	gx := u32((rows + 127) / 128)
	gy := u32((T + 127) / 128)
	if cu.launch(fn, gx, gy, 1, 256, 1, 1, WG_SMEM, args[:]) != cu.OK || cu.sync() != cu.OK {
		fmt.eprintf("cuda_selftest: %s launch failed\n", name)
		return false
	}
	got := make([]f32, T * rows)
	defer delete(got)
	cu.download(raw_data(got), dy, T * rows * 4)
	st: Case_Err
	for i in 0 ..< T * rows {
		s_note(&st, fmt.tprintf("%s [%d]", name, i), got[i], ref[i], atol, rtol)
	}
	if !s_report(fmt.tprintf("%s T=%d rows=%d cols=%d", name, T, rows, cols), st, atol, rtol) {
		return false
	}
	xname := "wgemm_q4_x16"
	if kind == .Q6_K {
		xname = "wgemm_q6_x16"
	} else if kind == .Q5_K {
		xname = "wgemm_q5_x16"
	}
	xfn := s_fn(xname)
	pfn := s_fn("pack_f16")
	if xfn == nil || pfn == nil {
		fmt.eprintf("cuda_selftest: missing %s or pack_f16\n", xname)
		return false
	}
	if cu.func_set_max_dynamic_shared(xfn, int(WG_SMEM)) != cu.OK {
		fmt.eprintf("cuda_selftest: %s cuFuncSetAttribute failed\n", xname)
		return false
	}
	dxh, _ := cu.alloc(T * cols * 2)
	dy2, _ := cu.alloc(T * rows * 4)
	defer cu.free(dxh)
	defer cu.free(dy2)
	nel := i32(T * cols)
	pargs := [?]rawptr{&xx, &dxh, &nel}
	blocks := u32((T * cols + 255) / 256)
	if cu.launch(pfn, blocks, 1, 1, 256, 1, 1, 0, pargs[:]) != cu.OK {
		fmt.eprintln("cuda_selftest: pack_f16 launch failed")
		return false
	}
	yy2 := dy2
	xargs := [?]rawptr{&wb, &dxh, &yy2, &nn, &dd, &tt}
	if cu.launch(xfn, gx, gy, 1, 256, 1, 1, WG_SMEM, xargs[:]) != cu.OK || cu.sync() != cu.OK {
		fmt.eprintf("cuda_selftest: %s launch failed\n", xname)
		return false
	}
	got2 := make([]f32, T * rows)
	defer delete(got2)
	cu.download(raw_data(got2), dy2, T * rows * 4)
	st2: Case_Err
	for i in 0 ..< T * rows {
		s_note(&st2, fmt.tprintf("%s [%d]", xname, i), got2[i], got[i], 0, 0)
	}
	return s_report(fmt.tprintf("%s matches %s", xname, name), st2, 0, 0)
}

// mmqk is the Q4_K / Q6_K prefill. Same q8 + dp4a math as gemv2, so that
// compare stays tight (2e-3). The fp32 CPU GEMV sees the q8 rounding, and
// uses the same allowance as run_gemv2: atol 1, rtol 5e-2. On a V100,
// mmqk_q6_k matched gemv2 within 7.6e-6, while a few elements missed
// 2e-2 + 2e-2*|ref| by 0.05-0.16.
@(private)
run_mmqk :: proc(kind: ggml.GGML_Type, rows, cols, T: int) -> bool {
	tag := cuda_kind_tag(kind)
	name := "mmqk_q4_k"
	if kind == .Q6_K {
		name = "mmqk_q6_k"
	}
	fn := s_fn(name)
	qfn := s_fn("quant_q8_batch")
	gfn := s_fn(fmt.tprintf("gemv2_%s", tag))
	q1 := s_fn("quant_q8")
	if fn == nil || qfn == nil || gfn == nil || q1 == nil {
		fmt.eprintf("cuda_selftest: missing %s, quant_q8_batch, or gemv2_%s\n", name, tag)
		return false
	}
	raw := fill_weight(kind, rows, cols)
	defer delete(raw)
	x := make([]f32, T * cols)
	defer delete(x)
	for t in 0 ..< T {
		for j in 0 ..< cols {
			x[t * cols + j] = (f32((t + 1) * (j + 3) % 9) - 4) * 0.03
		}
	}
	ref := make([]f32, T * rows)
	defer delete(ref)
	for t in 0 ..< T {
		row := cpu_gemv(kind, raw, x[t * cols:][:cols], rows, cols)
		copy(ref[t * rows:][:rows], row)
		delete(row)
	}
	ng := cols / 32
	dw, e1 := cu.alloc(len(raw) + 64)
	dx, e2 := cu.alloc(T * cols * 4)
	dy, e3 := cu.alloc(T * rows * 4)
	dpack, e4 := cu.alloc(T * cols)
	dsc, e5 := cu.alloc(T * ng * 4)
	dsm, e6 := cu.alloc(T * ng * 8)
	dqs, e7 := cu.alloc(cols)
	dscale, e8 := cu.alloc(ng * 4)
	dsum, e9 := cu.alloc(ng * 8)
	drow, e10 := cu.alloc(rows * 4)
	defer cu.free(dw)
	defer cu.free(dx)
	defer cu.free(dy)
	defer cu.free(dpack)
	defer cu.free(dsc)
	defer cu.free(dsm)
	defer cu.free(dqs)
	defer cu.free(dscale)
	defer cu.free(dsum)
	defer cu.free(drow)
	if e1 != cu.OK || e2 != cu.OK || e3 != cu.OK || e4 != cu.OK || e5 != cu.OK || e6 != cu.OK || e7 != cu.OK || e8 != cu.OK || e9 != cu.OK || e10 != cu.OK {
		fmt.eprintf("cuda_selftest: %s alloc failed\n", name)
		return false
	}
	cu.upload(dw, raw_data(raw), len(raw))
	cu.upload(dx, raw_data(x), T * cols * 4)
	xx, qp, sc, sm := dx, dpack, dsc, dsm
	nn, tt := i32(cols), i32(T)
	qargs := [?]rawptr{&xx, &qp, &sc, &sm, &nn, &tt}
	blocks := u32((ng * T + 255) / 256)
	if cu.launch(qfn, blocks, 1, 1, 256, 1, 1, 0, qargs[:]) != cu.OK {
		fmt.eprintf("cuda_selftest: quant_q8_batch launch failed (%s)\n", name)
		return false
	}
	wb, yy := dw, dy
	dd := i32(rows)
	args := [?]rawptr{&wb, &qp, &sc, &sm, &yy, &nn, &dd, &tt}
	// Q4 is 64 rows per block (8 per warp). Q6 stays at 32. 22528 bytes of q8.
	bm := 32
	if kind == .Q4_K {
		bm = 64
	}
	gx := u32((rows + bm - 1) / bm)
	gy := u32((T + 63) / 64)
	if cu.launch(fn, gx, gy, 1, 256, 1, 1, 22528, args[:]) != cu.OK || cu.sync() != cu.OK {
		fmt.eprintf("cuda_selftest: %s launch failed\n", name)
		return false
	}
	got := make([]f32, T * rows)
	defer delete(got)
	cu.download(raw_data(got), dy, T * rows * 4)
	st_cpu: Case_Err
	// Same q8-vs-fp32 allowance as gemv2 (run_gemv2 passes atol 1, rtol 5e-2).
	cpu_atol, cpu_rtol: f32 = 1, 5e-2
	for i in 0 ..< T * rows {
		s_note(&st_cpu, fmt.tprintf("%s cpu [%d]", name, i), got[i], ref[i], cpu_atol, cpu_rtol)
	}
	warps := 4
	ggx := u32((rows + warps * 4 - 1) / (warps * 4))
	bx := u32(warps * 32)
	acc := i32(0)
	st_g: Case_Err
	grow := make([]f32, rows)
	defer delete(grow)
	for t in 0 ..< T {
		rowx := cu.ptr_add(dx, t * cols * 4)
		qs, gsc, gsm := dqs, dscale, dsum
		rn := i32(cols)
		one := [?]rawptr{&rowx, &qs, &gsc, &gsm, &rn}
		if cu.launch(q1, u32((ng + 127) / 128), 1, 1, 128, 1, 1, 0, one[:]) != cu.OK {
			fmt.eprintf("cuda_selftest: quant_q8 launch failed (%s)\n", name)
			return false
		}
		ry := drow
		rd := i32(rows)
		launched: bool
		if kind == .Q4_K {
			gargs := [?]rawptr{&wb, &qs, &gsc, &gsm, &ry, &rn, &rd, &acc}
			launched = cu.launch(gfn, ggx, 1, 1, bx, 1, 1, 0, gargs[:]) == cu.OK
		} else {
			gargs := [?]rawptr{&wb, &qs, &gsc, &ry, &rn, &rd, &acc}
			launched = cu.launch(gfn, ggx, 1, 1, bx, 1, 1, 0, gargs[:]) == cu.OK
		}
		if !launched || cu.sync() != cu.OK {
			fmt.eprintf("cuda_selftest: gemv2 launch failed (%s)\n", name)
			return false
		}
		cu.download(raw_data(grow), drow, rows * 4)
		for r in 0 ..< rows {
			s_note(&st_g, fmt.tprintf("%s gemv2 t=%d r=%d", name, t, r), got[t * rows + r], grow[r], 2e-3, 2e-3)
		}
	}
	ok_cpu := s_report(fmt.tprintf("%s vs cpu T=%d rows=%d cols=%d", name, T, rows, cols), st_cpu, cpu_atol, cpu_rtol)
	ok_g := s_report(fmt.tprintf("%s vs gemv2 T=%d rows=%d cols=%d", name, T, rows, cols), st_g, 2e-3, 2e-3)
	return ok_cpu && ok_g
}

// Two same-shaped matrices on one q8 tile. The second weight is a different
// byte pattern so a swapped output fails. Same allowances as run_mmqk.
// The pair block is 32 rows even for Q4.
@(private)
run_mmqk_pair :: proc(kind: ggml.GGML_Type, rows, cols, T: int) -> bool {
	tag := cuda_kind_tag(kind)
	name := "mmqk_q4_pair"
	if kind == .Q6_K {
		name = "mmqk_q6_pair"
	}
	fn := s_fn(name)
	qfn := s_fn("quant_q8_batch")
	gfn := s_fn(fmt.tprintf("gemv2_%s", tag))
	q1 := s_fn("quant_q8")
	if fn == nil || qfn == nil || gfn == nil || q1 == nil {
		fmt.eprintf("cuda_selftest: missing %s\n", name)
		return false
	}
	raw0 := fill_weight(kind, rows, cols)
	raw1 := fill_weight(kind, rows, cols)
	defer delete(raw0)
	defer delete(raw1)
	// Q4 can take +19 on every byte and still land inside the fp32 allowance.
	// Q6 cannot: +19 also hits the fp16 superblock scale and the int8 scales,
	// and the shared q8 rounding (the same one gemv2 uses) then exceeds that
	// allowance. The pair matched gemv2 at ~2e-4 on that pattern, so the
	// second matrix was not a different kernel. Perturb ql/qh only and leave
	// the scales and d that fill_weight wrote.
	if kind == .Q6_K {
		_, bsz := ggml.ggml_block_info(kind)
		for i in 0 ..< len(raw1) {
			if i % bsz < 192 {
				raw1[i] ~= 0x3C
			}
		}
	} else {
		for i in 0 ..< len(raw1) {
			raw1[i] = u8((int(raw1[i]) + 19) & 0xFF)
		}
	}
	x := make([]f32, T * cols)
	defer delete(x)
	for t in 0 ..< T {
		for j in 0 ..< cols {
			x[t * cols + j] = (f32((t + 1) * (j + 3) % 9) - 4) * 0.03
		}
	}
	ref0 := make([]f32, T * rows)
	ref1 := make([]f32, T * rows)
	defer delete(ref0)
	defer delete(ref1)
	for t in 0 ..< T {
		row := cpu_gemv(kind, raw0, x[t * cols:][:cols], rows, cols)
		copy(ref0[t * rows:][:rows], row)
		delete(row)
		row = cpu_gemv(kind, raw1, x[t * cols:][:cols], rows, cols)
		copy(ref1[t * rows:][:rows], row)
		delete(row)
	}
	ng := cols / 32
	dw0, e0 := cu.alloc(len(raw0) + 64)
	dw1, e1 := cu.alloc(len(raw1) + 64)
	dx, e2 := cu.alloc(T * cols * 4)
	dy0, e3 := cu.alloc(T * rows * 4)
	dy1, e4 := cu.alloc(T * rows * 4)
	dpack, e5 := cu.alloc(T * cols)
	dsc, e6 := cu.alloc(T * ng * 4)
	dsm, e7 := cu.alloc(T * ng * 8)
	dqs, e8 := cu.alloc(cols)
	dscale, e9 := cu.alloc(ng * 4)
	dsum, e10 := cu.alloc(ng * 8)
	drow, e11 := cu.alloc(rows * 4)
	defer cu.free(dw0)
	defer cu.free(dw1)
	defer cu.free(dx)
	defer cu.free(dy0)
	defer cu.free(dy1)
	defer cu.free(dpack)
	defer cu.free(dsc)
	defer cu.free(dsm)
	defer cu.free(dqs)
	defer cu.free(dscale)
	defer cu.free(dsum)
	defer cu.free(drow)
	if e0 != cu.OK || e1 != cu.OK || e2 != cu.OK || e3 != cu.OK || e4 != cu.OK || e5 != cu.OK || e6 != cu.OK || e7 != cu.OK || e8 != cu.OK || e9 != cu.OK || e10 != cu.OK || e11 != cu.OK {
		fmt.eprintf("cuda_selftest: %s alloc failed\n", name)
		return false
	}
	cu.upload(dw0, raw_data(raw0), len(raw0))
	cu.upload(dw1, raw_data(raw1), len(raw1))
	cu.upload(dx, raw_data(x), T * cols * 4)
	xx, qp, sc, sm := dx, dpack, dsc, dsm
	nn, tt := i32(cols), i32(T)
	qargs := [?]rawptr{&xx, &qp, &sc, &sm, &nn, &tt}
	blocks := u32((ng * T + 255) / 256)
	if cu.launch(qfn, blocks, 1, 1, 256, 1, 1, 0, qargs[:]) != cu.OK {
		fmt.eprintf("cuda_selftest: quant_q8_batch launch failed (%s)\n", name)
		return false
	}
	wb, wb2, yy, yy2 := dw0, dw1, dy0, dy1
	dd := i32(rows)
	args := [?]rawptr{&wb, &wb2, &qp, &sc, &sm, &yy, &yy2, &nn, &dd, &tt}
	gx := u32((rows + 31) / 32)
	gy := u32((T + 63) / 64)
	if cu.launch(fn, gx, gy, 1, 256, 1, 1, 22528, args[:]) != cu.OK || cu.sync() != cu.OK {
		fmt.eprintf("cuda_selftest: %s launch failed\n", name)
		return false
	}
	got0 := make([]f32, T * rows)
	got1 := make([]f32, T * rows)
	defer delete(got0)
	defer delete(got1)
	cu.download(raw_data(got0), dy0, T * rows * 4)
	cu.download(raw_data(got1), dy1, T * rows * 4)
	cpu_atol, cpu_rtol: f32 = 1, 5e-2
	st_cpu: Case_Err
	for i in 0 ..< T * rows {
		s_note(&st_cpu, fmt.tprintf("%s cpu0 [%d]", name, i), got0[i], ref0[i], cpu_atol, cpu_rtol)
		s_note(&st_cpu, fmt.tprintf("%s cpu1 [%d]", name, i), got1[i], ref1[i], cpu_atol, cpu_rtol)
	}
	warps := 4
	ggx := u32((rows + warps * 4 - 1) / (warps * 4))
	bx := u32(warps * 32)
	acc := i32(0)
	st_g: Case_Err
	grow := make([]f32, rows)
	defer delete(grow)
	for t in 0 ..< T {
		rowx := cu.ptr_add(dx, t * cols * 4)
		qs, gsc, gsm := dqs, dscale, dsum
		rn := i32(cols)
		one := [?]rawptr{&rowx, &qs, &gsc, &gsm, &rn}
		if cu.launch(q1, u32((ng + 127) / 128), 1, 1, 128, 1, 1, 0, one[:]) != cu.OK {
			fmt.eprintf("cuda_selftest: quant_q8 launch failed (%s)\n", name)
			return false
		}
		ry := drow
		rd := i32(rows)
		for which in 0 ..< 2 {
			wsel := dw0
			got := got0
			if which == 1 {
				wsel = dw1
				got = got1
			}
			launched: bool
			if kind == .Q4_K {
				gargs := [?]rawptr{&wsel, &qs, &gsc, &gsm, &ry, &rn, &rd, &acc}
				launched = cu.launch(gfn, ggx, 1, 1, bx, 1, 1, 0, gargs[:]) == cu.OK
			} else {
				gargs := [?]rawptr{&wsel, &qs, &gsc, &ry, &rn, &rd, &acc}
				launched = cu.launch(gfn, ggx, 1, 1, bx, 1, 1, 0, gargs[:]) == cu.OK
			}
			if !launched || cu.sync() != cu.OK {
				fmt.eprintf("cuda_selftest: gemv2 launch failed (%s)\n", name)
				return false
			}
			cu.download(raw_data(grow), drow, rows * 4)
			for r in 0 ..< rows {
				s_note(&st_g, fmt.tprintf("%s gemv2 w%d t=%d r=%d", name, which, t, r), got[t * rows + r], grow[r], 2e-3, 2e-3)
			}
		}
	}
	ok_cpu := s_report(fmt.tprintf("%s vs cpu T=%d rows=%d cols=%d", name, T, rows, cols), st_cpu, cpu_atol, cpu_rtol)
	ok_g := s_report(fmt.tprintf("%s vs gemv2 T=%d rows=%d cols=%d", name, T, rows, cols), st_g, 2e-3, 2e-3)
	return ok_cpu && ok_g
}

// tc2r must match the scalar tc kernel bit for bit. A CPU tolerance of 5e-2
// also passes kernels that disagree with each other by ~0.04, which is enough
// to change a sampled token. This compare is the token-relevant check.
// harsh=true uses a wider activation range so a later CPU compare is not only
// the tiny (i%9) pattern.
@(private)
run_tc_pair :: proc(kind: ggml.GGML_Type, rows, cols, T: int, harsh: bool) -> bool {
	tag := cuda_kind_tag(kind)
	name := "tc2r_q4_k"
	if kind == .Q6_K {
		name = "tc2r_q6_k"
	}
	fn_r := s_fn(name)
	fn_s := s_fn(fmt.tprintf("tc_%s", tag))
	if fn_r == nil || fn_s == nil {
		fmt.eprintf("cuda_selftest: missing %s or tc_%s\n", name, tag)
		return false
	}
	raw := fill_weight(kind, rows, cols)
	defer delete(raw)
	x := make([]f32, T * cols)
	defer delete(x)
	for t in 0 ..< T {
		for j in 0 ..< cols {
			if harsh {
				x[t * cols + j] = (f32((t * 13 + j * 7) % 251) - 125) * 0.5
			} else {
				x[t * cols + j] = (f32((t * 13 + j * 7) % 251) - 125) * 0.02
			}
		}
	}
	dw, _ := cu.alloc(len(raw))
	dx, _ := cu.alloc(T * cols * 4)
	dy, _ := cu.alloc(T * rows * 4)
	dz, _ := cu.alloc(T * rows * 4)
	defer cu.free(dw)
	defer cu.free(dx)
	defer cu.free(dy)
	defer cu.free(dz)
	cu.upload(dw, raw_data(raw), len(raw))
	cu.upload(dx, raw_data(x), T * cols * 4)
	wb, xx := dw, dx
	nn, dd, tt := i32(cols), i32(rows), i32(T)
	ys, yr := dy, dz
	args_s := [?]rawptr{&wb, &xx, &ys, &nn, &dd, &tt}
	args_r := [?]rawptr{&wb, &xx, &yr, &nn, &dd, &tt}
	gx := u32((rows + 63) / 64)
	gy_s := u32((T + 63) / 64)
	gy_r := u32((T + 127) / 128)
	if cu.memset(dy, 0, T * rows * 4) != cu.OK || cu.memset(dz, 0, T * rows * 4) != cu.OK {
		fmt.eprintf("cuda_selftest: %s memset failed\n", name)
		return false
	}
	if cu.launch(fn_s, gx, gy_s, 1, 512, 1, 1, 0, args_s[:]) != cu.OK {
		fmt.eprintf("cuda_selftest: tc_%s launch failed\n", tag)
		return false
	}
	if cu.launch(fn_r, gx, gy_r, 1, 512, 1, 1, 0, args_r[:]) != cu.OK {
		fmt.eprintf("cuda_selftest: %s launch failed\n", name)
		return false
	}
	got_s := make([]f32, T * rows)
	got_r := make([]f32, T * rows)
	defer delete(got_s)
	defer delete(got_r)
	cu.download(raw_data(got_s), dy, T * rows * 4)
	cu.download(raw_data(got_r), dz, T * rows * 4)
	st: Case_Err
	for i in 0 ..< T * rows {
		s_note(&st, fmt.tprintf("%s vs tc [%d]", name, i), got_r[i], got_s[i], 0, 0)
	}
	// fp16 versus the fp32 CPU dot. This bound only catches a broken kernel.
	// It is not a token-match guarantee; the bit-exact compare above is.
	ref_atol, ref_rtol: f32 = 5e-2, 5e-2
	if cols >= 1024 {
		ref_atol, ref_rtol = 0.25, 0.05
	}
	ref := cpu_gemv(kind, raw, x[:cols], rows, cols)
	defer delete(ref)
	st_cpu: Case_Err
	for i in 0 ..< rows {
		s_note(&st_cpu, fmt.tprintf("tc %s cpu [%d]", tag, i), got_s[i], ref[i], ref_atol, ref_rtol)
	}
	ok_pair := s_report(fmt.tprintf("%s bit-exact vs tc T=%d rows=%d cols=%d", name, T, rows, cols), st, 0, 0)
	ok_cpu := s_report(fmt.tprintf("tc %s vs cpu row0 T=%d rows=%d cols=%d", tag, T, rows, cols), st_cpu, ref_atol, ref_rtol)
	return ok_pair && ok_cpu
}

@(private)
run_elem :: proc() -> bool {
	// RMSNorm
	n := 64
	x := make([]f32, n)
	w := make([]f32, n)
	ref := make([]f32, n)
	defer delete(x)
	defer delete(w)
	defer delete(ref)
	for i in 0 ..< n {
		x[i] = (f32(i % 9) - 4) * 0.1
		w[i] = 0.8 + f32(i % 5) * 0.05
	}
	rmsnorm(ref, x, w, 1e-6)
	dx, _ := cu.alloc(n * 4)
	dw, _ := cu.alloc(n * 4)
	dy, _ := cu.alloc(n * 4)
	defer cu.free(dx)
	defer cu.free(dw)
	defer cu.free(dy)
	cu.upload(dx, raw_data(x), n * 4)
	cu.upload(dw, raw_data(w), n * 4)
	fn := s_fn("rmsnorm_k")
	if fn == nil {
		return false
	}
	a, b, c0 := dx, dy, dw
	sz := i32(n)
	ep := f32(1e-6)
	args := [?]rawptr{&a, &b, &c0, &sz, &ep}
	if cu.launch(fn, 1, 1, 1, 128, 1, 1, 0, args[:]) != cu.OK {
		fmt.eprintln("cuda_selftest: rmsnorm launch failed")
		return false
	}
	got := make([]f32, n)
	defer delete(got)
	cu.download(raw_data(got), dy, n * 4)
	st_rms: Case_Err
	for i in 0 ..< n {
		s_note(&st_rms, fmt.tprintf("rmsnorm[%d]", i), got[i], ref[i], 1e-4, 1e-4)
	}
	if !s_report("rmsnorm", st_rms, 1e-4, 1e-4) {
		return false
	}

	// residual add then the same RMSNorm, one kernel. n is a multiple of 32.
	yv := make([]f32, n)
	sumv := make([]f32, n)
	defer delete(yv)
	defer delete(sumv)
	for i in 0 ..< n {
		yv[i] = (f32(i % 7) - 3) * 0.05
		sumv[i] = x[i] + yv[i]
	}
	ref2 := make([]f32, n)
	defer delete(ref2)
	rmsnorm(ref2, sumv, w, 1e-6)
	dy2, _ := cu.alloc(n * 4)
	dqs, _ := cu.alloc(n)
	dsc, _ := cu.alloc((n / 32) * 4)
	dsm, _ := cu.alloc((n / 32) * 8)
	defer cu.free(dy2)
	defer cu.free(dqs)
	defer cu.free(dsc)
	defer cu.free(dsm)
	cu.upload(dx, raw_data(x), n * 4)
	cu.upload(dy2, raw_data(yv), n * 4)
	rn := s_fn("residual_rmsnorm_q8")
	if rn == nil {
		fmt.eprintln("cuda_selftest: residual_rmsnorm_q8 missing")
		return false
	}
	rx, ry, ro, rw := dx, dy2, dy, dw
	rqs, rsc, rsm := dqs, dsc, dsm
	rn_args := [?]rawptr{&rx, &ry, &ro, &rw, &rqs, &rsc, &rsm, &sz, &ep}
	if cu.launch(rn, 1, 1, 1, 128, 1, 1, 0, rn_args[:]) != cu.OK {
		fmt.eprintln("cuda_selftest: residual_rmsnorm_q8 launch failed")
		return false
	}
	got2 := make([]f32, n)
	gotx := make([]f32, n)
	defer delete(got2)
	defer delete(gotx)
	cu.download(raw_data(got2), dy, n * 4)
	cu.download(raw_data(gotx), dx, n * 4)
	st_rn: Case_Err
	for i in 0 ..< n {
		s_note(&st_rn, fmt.tprintf("residual_norm.x[%d]", i), gotx[i], sumv[i], 1e-5, 1e-5)
		s_note(&st_rn, fmt.tprintf("residual_norm[%d]", i), got2[i], ref2[i], 1e-4, 1e-4)
	}
	if !s_report("residual_rmsnorm_q8", st_rn, 1e-4, 1e-4) {
		return false
	}

	// RoPE, 2 heads, head_dim 8, rotary 4
	hd, rot, nh, pos := 8, 4, 2, 3
	vec := make([]f32, nh * hd)
	cpuv := make([]f32, nh * hd)
	defer delete(vec)
	defer delete(cpuv)
	for i in 0 ..< len(vec) {
		vec[i] = (f32(i % 5) - 2) * 0.3
		cpuv[i] = vec[i]
	}
	inv := make([]f32, rot / 2)
	defer delete(inv)
	inv_freq_default(10000, rot, inv)
	for h in 0 ..< nh {
		apply_mrope_text_head(cpuv[h * hd:][:hd], pos, inv, rot)
	}
	dv, _ := cu.alloc(len(vec) * 4)
	defer cu.free(dv)
	cu.upload(dv, raw_data(vec), len(vec) * 4)
	rf := s_fn("rope_partial")
	if rf == nil {
		return false
	}
	vv := dv
	ihd, ipos, irot, inh := i32(hd), i32(pos), i32(rot), i32(nh)
	theta := f32(10000)
	slot: cu.Device_Ptr
	rargs := [?]rawptr{&vv, &ihd, &ipos, &theta, &irot, &inh, &slot}
	total := nh * (rot / 2)
	if cu.launch(rf, u32((total + 127) / 128), 1, 1, 128, 1, 1, 0, rargs[:]) != cu.OK {
		fmt.eprintln("cuda_selftest: rope launch failed")
		return false
	}
	gotv := make([]f32, len(vec))
	defer delete(gotv)
	cu.download(raw_data(gotv), dv, len(vec) * 4)
	st_rope: Case_Err
	for i in 0 ..< len(vec) {
		s_note(&st_rope, fmt.tprintf("rope[%d]", i), gotv[i], cpuv[i], 1e-4, 1e-4)
	}
	if !s_report("rope", st_rope, 1e-4, 1e-4) {
		return false
	}

	// qnorm_rope matches qnorm_extract then rope_partial, across two tokens.
	{
		head, nper, nt, rot, pos0 := 8, 2, 2, 4, 3
		heads := nt * nper
		qsrc := make([]f32, heads * head * 2)
		qn := make([]f32, head)
		defer delete(qsrc)
		defer delete(qn)
		for i in 0 ..< len(qsrc) {
			qsrc[i] = (f32(i % 7) - 3) * 0.2
		}
		for i in 0 ..< head {
			qn[i] = 0.5 + f32(i) * 0.1
		}
		da, _ := cu.alloc(len(qsrc) * 4)
		db, _ := cu.alloc(heads * head * 4)
		dc, _ := cu.alloc(heads * head * 4)
		dw, _ := cu.alloc(head * 4)
		defer cu.free(da)
		defer cu.free(db)
		defer cu.free(dc)
		defer cu.free(dw)
		cu.upload(da, raw_data(qsrc), len(qsrc) * 4)
		cu.upload(dw, raw_data(qn), head * 4)
		qf := s_fn("qnorm_extract")
		rf := s_fn("rope_partial")
		ff := s_fn("qnorm_rope")
		if qf == nil || rf == nil || ff == nil {
			return false
		}
		pa, pb, pw := da, db, dw
		ih, ieps := i32(head), f32(1e-6)
		qargs := [?]rawptr{&pa, &pb, &pw, &ih, &ieps}
		if cu.launch(qf, u32(heads), 1, 1, 128, 1, 1, 0, qargs[:]) != cu.OK {
			return false
		}
		theta := f32(10000)
		irot := i32(rot)
		inh := i32(nper)
		slot: cu.Device_Ptr
		for t in 0 ..< nt {
			row := cu.ptr_add(db, t * nper * head * 4)
			ipos := i32(pos0 + t)
			rargs := [?]rawptr{&row, &ih, &ipos, &theta, &irot, &inh, &slot}
			if cu.launch(rf, 1, 1, 1, 128, 1, 1, 0, rargs[:]) != cu.OK {
				return false
			}
		}
		refq := make([]f32, heads * head)
		gotq := make([]f32, heads * head)
		defer delete(refq)
		defer delete(gotq)
		cu.download(raw_data(refq), db, len(refq) * 4)
		pc := dc
		ipos0 := i32(pos0)
		inper := i32(nper)
		fargs := [?]rawptr{&pa, &pc, &pw, &ih, &ieps, &ipos0, &theta, &irot, &inper, &slot}
		if cu.launch(ff, u32(heads), 1, 1, 128, 1, 1, 0, fargs[:]) != cu.OK {
			fmt.eprintln("cuda_selftest: qnorm_rope launch failed")
			return false
		}
		cu.download(raw_data(gotq), dc, len(gotq) * 4)
		st_qr: Case_Err
		for i in 0 ..< len(gotq) {
			s_note(&st_qr, fmt.tprintf("qnorm_rope[%d]", i), gotq[i], refq[i], 1e-4, 1e-4)
		}
		if !s_report("qnorm_rope", st_qr, 1e-4, 1e-4) {
			return false
		}

		kf := s_fn("rmsnorm_k")
		kr := s_fn("rmsnorm_rope")
		if kf == nil || kr == nil {
			return false
		}
		kin := make([]f32, heads * head)
		defer delete(kin)
		for i in 0 ..< len(kin) {
			kin[i] = (f32(i % 5) - 2) * 0.4
		}
		dk, _ := cu.alloc(len(kin) * 4)
		dk2, _ := cu.alloc(len(kin) * 4)
		defer cu.free(dk)
		defer cu.free(dk2)
		cu.upload(dk, raw_data(kin), len(kin) * 4)
		cu.upload(dk2, raw_data(kin), len(kin) * 4)
		pk, po := dk, dk
		kargs := [?]rawptr{&pk, &po, &pw, &ih, &ieps}
		if cu.launch(kf, u32(heads), 1, 1, 128, 1, 1, 0, kargs[:]) != cu.OK {
			return false
		}
		for t in 0 ..< nt {
			row := cu.ptr_add(dk, t * nper * head * 4)
			ipos := i32(pos0 + t)
			rargs := [?]rawptr{&row, &ih, &ipos, &theta, &irot, &inh, &slot}
			if cu.launch(rf, 1, 1, 1, 128, 1, 1, 0, rargs[:]) != cu.OK {
				return false
			}
		}
		refk := make([]f32, len(kin))
		gotk := make([]f32, len(kin))
		defer delete(refk)
		defer delete(gotk)
		cu.download(raw_data(refk), dk, len(refk) * 4)
		pk2 := dk2
		kargs2 := [?]rawptr{&pk2, &pw, &ih, &ieps, &ipos0, &theta, &irot, &inper, &slot}
		if cu.launch(kr, u32(heads), 1, 1, 128, 1, 1, 0, kargs2[:]) != cu.OK {
			fmt.eprintln("cuda_selftest: rmsnorm_rope launch failed")
			return false
		}
		cu.download(raw_data(gotk), dk2, len(gotk) * 4)
		st_kr: Case_Err
		for i in 0 ..< len(gotk) {
			s_note(&st_kr, fmt.tprintf("rmsnorm_rope[%d]", i), gotk[i], refk[i], 1e-4, 1e-4)
		}
		if !s_report("rmsnorm_rope", st_kr, 1e-4, 1e-4) {
			return false
		}
	}

	// conv1d_chunk matches two conv1d_step calls, including the state shift.
	{
		cd, kern, nt := 4, 3, 2
		km1 := kern - 1
		inp := make([]f32, nt * cd)
		w := make([]f32, cd * kern)
		st0 := make([]f32, cd * km1)
		defer delete(inp)
		defer delete(w)
		defer delete(st0)
		for i in 0 ..< len(inp) {
			inp[i] = (f32(i) - 3) * 0.3
		}
		for i in 0 ..< len(w) {
			w[i] = 0.1 * f32(i + 1)
		}
		for i in 0 ..< len(st0) {
			st0[i] = 0.05 * f32(i)
		}
		cf := s_fn("conv1d_step")
		cc := s_fn("conv1d_chunk")
		if cf == nil || cc == nil {
			return false
		}
		di, _ := cu.alloc(len(inp) * 4)
		dout, _ := cu.alloc(len(inp) * 4)
		dwc, _ := cu.alloc(len(w) * 4)
		ds, _ := cu.alloc(len(st0) * 4)
		ds2, _ := cu.alloc(len(st0) * 4)
		dout2, _ := cu.alloc(len(inp) * 4)
		defer cu.free(di)
		defer cu.free(dout)
		defer cu.free(dwc)
		defer cu.free(ds)
		defer cu.free(ds2)
		defer cu.free(dout2)
		cu.upload(di, raw_data(inp), len(inp) * 4)
		cu.upload(dwc, raw_data(w), len(w) * 4)
		cu.upload(ds, raw_data(st0), len(st0) * 4)
		cu.upload(ds2, raw_data(st0), len(st0) * 4)
		icd, ik := i32(cd), i32(kern)
		pwv, pst := dwc, ds
		for t in 0 ..< nt {
			pin := cu.ptr_add(di, t * cd * 4)
			pout := cu.ptr_add(dout, t * cd * 4)
			cargs := [?]rawptr{&pout, &pin, &pst, &pwv, &icd, &ik}
			if cu.launch(cf, 1, 1, 1, 128, 1, 1, 0, cargs[:]) != cu.OK {
				return false
			}
		}
		refc := make([]f32, len(inp))
		gotc := make([]f32, len(inp))
		refs := make([]f32, len(st0))
		gots := make([]f32, len(st0))
		defer delete(refc)
		defer delete(gotc)
		defer delete(refs)
		defer delete(gots)
		cu.download(raw_data(refc), dout, len(refc) * 4)
		cu.download(raw_data(refs), ds, len(refs) * 4)
		pin2, pout2, pst2 := di, dout2, ds2
		itt := i32(nt)
		cargs2 := [?]rawptr{&pout2, &pin2, &pst2, &pwv, &icd, &ik, &itt}
		if cu.launch(cc, 1, 1, 1, 128, 1, 1, 0, cargs2[:]) != cu.OK {
			fmt.eprintln("cuda_selftest: conv1d_chunk launch failed")
			return false
		}
		cu.download(raw_data(gotc), dout2, len(gotc) * 4)
		cu.download(raw_data(gots), ds2, len(gots) * 4)
		st_cv: Case_Err
		for i in 0 ..< len(gotc) {
			s_note(&st_cv, fmt.tprintf("conv[%d]", i), gotc[i], refc[i], 1e-5, 1e-5)
		}
		for i in 0 ..< len(gots) {
			s_note(&st_cv, fmt.tprintf("conv_state[%d]", i), gots[i], refs[i], 1e-5, 1e-5)
		}
		if !s_report("conv_chunk", st_cv, 1e-5, 1e-5) {
			return false
		}
	}

	// SwiGLU
	m := 17
	hb := make([]f32, m)
	hb2 := make([]f32, m)
	defer delete(hb)
	defer delete(hb2)
	for i in 0 ..< m {
		hb[i] = (f32(i) - 8) * 0.4
		hb2[i] = (f32(i % 3) - 1) * 0.7
	}
	want := make([]f32, m)
	defer delete(want)
	for i in 0 ..< m {
		want[i] = silu_f32(hb[i]) * hb2[i]
	}
	dh, _ := cu.alloc(m * 4)
	dh2, _ := cu.alloc(m * 4)
	defer cu.free(dh)
	defer cu.free(dh2)
	cu.upload(dh, raw_data(hb), m * 4)
	cu.upload(dh2, raw_data(hb2), m * 4)
	sf := s_fn("swiglu")
	if sf == nil {
		return false
	}
	ha, hb2p := dh, dh2
	nn := i32(m)
	sargs := [?]rawptr{&ha, &hb2p, &nn}
	if cu.launch(sf, 1, 1, 1, 32, 1, 1, 0, sargs[:]) != cu.OK {
		return false
	}
	gout := make([]f32, m)
	defer delete(gout)
	cu.download(raw_data(gout), dh, m * 4)
	st_swi: Case_Err
	for i in 0 ..< m {
		s_note(&st_swi, fmt.tprintf("swiglu[%d]", i), gout[i], want[i], 1e-5, 1e-5)
	}
	return s_report("swiglu", st_swi, 1e-5, 1e-5)
}

@(private)
run_kv_attn :: proc() -> bool {
	// One head, head_dim 32, two positions. Mixed V store + attention.
	hd := 32
	k := make([]f32, hd)
	v := make([]f32, hd)
	q := make([]f32, hd)
	defer delete(k)
	defer delete(v)
	defer delete(q)
	for i in 0 ..< hd {
		k[i] = (f32(i % 7) - 3) * 0.2
		v[i] = (f32(i % 5) - 1) * 0.5
		q[i] = (f32(i % 4) - 1.5) * 0.3
	}
	// Spike an outlier so the mixed-V index is not trivial.
	v[3] = 4.5
	seq := 4
	k_pos := 34
	v_pos := 40
	kc := make([]u8, seq * k_pos)
	vc := make([]u8, seq * v_pos)
	defer delete(kc)
	defer delete(vc)
	dk, _ := cu.alloc(hd * 4)
	dv, _ := cu.alloc(hd * 4)
	dq, _ := cu.alloc(hd * 4)
	dkc, _ := cu.alloc(len(kc))
	dvc, _ := cu.alloc(len(vc))
	dsc, _ := cu.alloc(seq * 4)
	dxb, _ := cu.alloc(hd * 4)
	defer cu.free(dk)
	defer cu.free(dv)
	defer cu.free(dq)
	defer cu.free(dkc)
	defer cu.free(dvc)
	defer cu.free(dsc)
	defer cu.free(dxb)
	cu.upload(dk, raw_data(k), hd * 4)
	cu.upload(dv, raw_data(v), hd * 4)
	cu.memset(dkc, 0, len(kc))
	cu.memset(dvc, 0, len(vc))
	store := s_fn("store_kv_q8v")
	if store == nil {
		return false
	}
	// pos 0 = k, pos 1 = a second copy scaled, so attention has two keys.
	for pos in 0 ..< 2 {
		scale := f32(1)
		if pos == 1 {
			scale = 0.5
		}
		ks := make([]f32, hd)
		vs := make([]f32, hd)
		for i in 0 ..< hd {
			ks[i] = k[i] * scale
			vs[i] = v[i] * scale
		}
		cu.upload(dk, raw_data(ks), hd * 4)
		cu.upload(dv, raw_data(vs), hd * 4)
		a, b, c0, d := dk, dv, dkc, dvc
		nb, kb, vb, ps := i32(1), i32(k_pos), i32(v_pos), i32(pos)
		slot: cu.Device_Ptr
		args := [?]rawptr{&a, &b, &c0, &d, &nb, &kb, &vb, &ps, &slot}
		if cu.launch(store, 1, 1, 1, 1, 1, 1, 0, args[:]) != cu.OK {
			fmt.eprintln("cuda_selftest: store_kv_q8v failed")
			delete(ks)
			delete(vs)
			return false
		}
		delete(ks)
		delete(vs)
	}
	cu.download(raw_data(kc), dkc, len(kc))
	cu.download(raw_data(vc), dvc, len(vc))
	// CPU quant of pos 0 and compare dequant (f16 scale may differ by an ulp).
	refk := make([]u8, 34)
	refv := make([]u8, 40)
	defer delete(refk)
	defer delete(refv)
	quantize_q8_block(k, refk)
	quantize_q8_v_mix_block(v, refv)
	dkf := make([]f32, 32)
	dvf := make([]f32, 32)
	rkf := make([]f32, 32)
	rvf := make([]f32, 32)
	defer delete(dkf)
	defer delete(dvf)
	defer delete(rkf)
	defer delete(rvf)
	dequantize_q8_block(kc[:34], dkf)
	dequantize_q8_v_mix_block(vc[:40], dvf)
	dequantize_q8_block(refk, rkf)
	dequantize_q8_v_mix_block(refv, rvf)
	st_q8: Case_Err
	for i in 0 ..< 32 {
		s_note(&st_q8, fmt.tprintf("q8k[%d]", i), dkf[i], rkf[i], 2e-2, 1e-3)
		s_note(&st_q8, fmt.tprintf("q8v[%d]", i), dvf[i], rvf[i], 2e-2, 1e-3)
	}
	if !s_report("q8v store", st_q8, 2e-2, 1e-3) {
		return false
	}

	cu.upload(dq, raw_data(q), hd * 4)
	attn := s_fn("attention_q8v")
	if attn == nil {
		return false
	}
	qa, kca, vca, xb, sc := dq, dkc, dvc, dxb, dsc
	ihd, ikb, ivb, mul, ps, sl := i32(hd), i32(k_pos), i32(v_pos), i32(1), i32(1), i32(seq)
	slot: cu.Device_Ptr
	stride := i32(0)
	aargs := [?]rawptr{&qa, &kca, &vca, &xb, &sc, &ihd, &ikb, &ivb, &mul, &ps, &sl, &slot, &stride}
	if cu.launch(attn, 1, 1, 1, 128, 1, 1, 0, aargs[:]) != cu.OK {
		fmt.eprintln("cuda_selftest: attention_q8v failed")
		return false
	}
	got := make([]f32, hd)
	defer delete(got)
	cu.download(raw_data(got), dxb, hd * 4)

	// CPU attention over the dequantized cache the GPU just wrote.
	k0 := make([]f32, hd)
	k1 := make([]f32, hd)
	v0 := make([]f32, hd)
	v1 := make([]f32, hd)
	defer delete(k0)
	defer delete(k1)
	defer delete(v0)
	defer delete(v1)
	dequantize_q8_block(kc[0:34], k0)
	dequantize_q8_block(kc[k_pos:k_pos + 34], k1)
	dequantize_q8_v_mix_block(vc[0:40], v0)
	dequantize_q8_v_mix_block(vc[v_pos:v_pos + 40], v1)
	dot := proc(a, b: []f32) -> f32 {
		s: f32 = 0
		for i in 0 ..< len(a) {
			s += a[i] * b[i]
		}
		return s
	}
	scale := 1 / math.sqrt_f32(f32(hd))
	s0 := dot(q, k0) * scale
	s1 := dot(q, k1) * scale
	m := max(s0, s1)
	e0 := math.exp_f32(s0 - m)
	e1 := math.exp_f32(s1 - m)
	inv := 1 / (e0 + e1)
	st_attn: Case_Err
	for i in 0 ..< hd {
		want := (e0 * v0[i] + e1 * v1[i]) * inv
		s_note(&st_attn, fmt.tprintf("attn[%d]", i), got[i], want, 2e-2, 1e-3)
	}
	return s_report("attention_q8v", st_attn, 2e-2, 1e-3)
}

// Split attention against attention_q8v on one cache. head_dim 256, two KV
// heads, four query heads per KV head, so every warp in the part kernel is
// live. Lengths cover one position, a full part, a partial tail, the
// previous long decode sizes (300, 1800), and the contexts this server
// now runs (8192, 32768). Allowance is the same band as attention_q8v
// vs the CPU reference. The fp32 softmax merge is not bit-exact.
@(private)
run_attn_split :: proc() -> bool {
	hd := 256
	n_kv := 2
	kv_mul := 4
	n_heads := n_kv * kv_mul
	npos_max := 32768
	nb_head := hd / 32
	kv_dim := n_kv * hd
	nb := kv_dim / 32
	k_pos := nb * Q8_0_BLOCK_BYTES
	v_pos := nb * Q8_V_MIX_BLOCK_BYTES
	kc := make([]u8, npos_max * k_pos)
	vc := make([]u8, npos_max * v_pos)
	q := make([]f32, n_heads * hd)
	blk := make([]f32, 32)
	defer delete(kc)
	defer delete(vc)
	defer delete(q)
	defer delete(blk)
	for t in 0 ..< npos_max {
		for kvh in 0 ..< n_kv {
			for b in 0 ..< nb_head {
				gid := kvh * nb_head + b
				for j in 0 ..< 32 {
					i := b * 32 + j
					blk[j] = (f32((i * 3 + t * 5 + kvh * 7) % 11) - 5) * 0.15
				}
				quantize_q8_block(blk, kc[t * k_pos + gid * Q8_0_BLOCK_BYTES:][:Q8_0_BLOCK_BYTES])
				for j in 0 ..< 32 {
					i := b * 32 + j
					blk[j] = (f32((i * 2 + t + kvh) % 9) - 4) * 0.2
					if j == 3 {
						blk[j] = 3.5 + f32(t) * 0.01
					}
				}
				quantize_q8_v_mix_block(blk, vc[t * v_pos + gid * Q8_V_MIX_BLOCK_BYTES:][:Q8_V_MIX_BLOCK_BYTES])
			}
		}
	}
	for h in 0 ..< n_heads {
		for i in 0 ..< hd {
			q[h * hd + i] = (f32((i + h * 3) % 13) - 6) * 0.1
		}
	}
	dq, _ := cu.alloc(len(q) * 4)
	dkc, _ := cu.alloc(len(kc))
	dvc, _ := cu.alloc(len(vc))
	dxb, _ := cu.alloc(len(q) * 4)
	dsc, _ := cu.alloc(n_heads * npos_max * 4)
	nsplit_max := (npos_max + 31) / 32
	dpm, _ := cu.alloc(nsplit_max * n_heads * 4)
	dpl, _ := cu.alloc(nsplit_max * n_heads * 4)
	dpo, _ := cu.alloc(nsplit_max * n_heads * hd * 4)
	defer cu.free(dq)
	defer cu.free(dkc)
	defer cu.free(dvc)
	defer cu.free(dxb)
	defer cu.free(dsc)
	defer cu.free(dpm)
	defer cu.free(dpl)
	defer cu.free(dpo)
	if cu.upload(dq, raw_data(q), len(q) * 4) != cu.OK || cu.upload(dkc, raw_data(kc), len(kc)) != cu.OK || cu.upload(dvc, raw_data(vc), len(vc)) != cu.OK {
		fmt.eprintln("cuda_selftest: attn split upload failed")
		return false
	}
	old_fn := s_fn("attention_q8v")
	part_fn := s_fn("attention_q8v_part")
	red_fn := s_fn("attention_q8v_reduce")
	if old_fn == nil || part_fn == nil || red_fn == nil {
		fmt.eprintln("cuda_selftest: attn split kernels missing")
		return false
	}
	smem := u32(kv_mul * hd * 4 + 32 * nb_head * 34 + 32 * nb_head * 40)
	got_old := make([]f32, n_heads * hd)
	got_new := make([]f32, n_heads * hd)
	defer delete(got_old)
	defer delete(got_new)
	lengths := [?]int{1, 32, 33, 70, 128, 300, 1800, 8192, 32768}
	ok := true
	for npos in lengths {
		qa, kca, vca, xb, sc := dq, dkc, dvc, dxb, dsc
		ihd := i32(hd)
		ikb := i32(k_pos)
		ivb := i32(v_pos)
		mul := i32(kv_mul)
		ps := i32(npos - 1)
		sl := i32(npos_max)
		slot: cu.Device_Ptr
		stride := i32(0)
		oargs := [?]rawptr{&qa, &kca, &vca, &xb, &sc, &ihd, &ikb, &ivb, &mul, &ps, &sl, &slot, &stride}
		if cu.launch(old_fn, u32(n_heads), 1, 1, 128, 1, 1, 0, oargs[:]) != cu.OK || cu.sync() != cu.OK {
			fmt.eprintfln("cuda_selftest: attention_q8v launch failed at npos {}", npos)
			return false
		}
		if cu.download(raw_data(got_old), dxb, len(got_old) * 4) != cu.OK {
			return false
		}
		pm, pl, po := dpm, dpl, dpo
		nh := i32(n_heads)
		part := i32(32)
		pargs := [?]rawptr{&qa, &kca, &vca, &pm, &pl, &po, &ihd, &ikb, &ivb, &mul, &nh, &ps, &part, &slot}
		nsplit := (npos + 31) / 32
		if cu.launch(part_fn, u32(n_kv), u32(nsplit), 1, 128, 1, 1, smem, pargs[:]) != cu.OK || cu.sync() != cu.OK {
			fmt.eprintfln("cuda_selftest: attention_q8v_part launch failed at npos {}", npos)
			return false
		}
		ns := i32(nsplit)
		rargs := [?]rawptr{&xb, &pm, &pl, &po, &ihd, &nh, &ns}
		if cu.launch(red_fn, u32(n_heads), 1, 1, 128, 1, 1, 0, rargs[:]) != cu.OK || cu.sync() != cu.OK {
			fmt.eprintfln("cuda_selftest: attention_q8v_reduce launch failed at npos {}", npos)
			return false
		}
		if cu.download(raw_data(got_new), dxb, len(got_new) * 4) != cu.OK {
			return false
		}
		st: Case_Err
		for i in 0 ..< len(got_new) {
			s_note(&st, fmt.tprintf("split[%d] npos %d", i, npos), got_new[i], got_old[i], 2e-2, 1e-3)
		}
		if !s_report(fmt.tprintf("attn_split npos={}", npos), st, 2e-2, 1e-3) {
			ok = false
		}
		// 128-wide decode parts. kv_mul here is 4, the 9B group. Same
		// reduce and the same allowance as the 32-wide split.
		if npos == 128 || npos == 300 || npos == 1800 {
			long_fn := s_fn("attention_q8v_long4")
			if long_fn == nil {
				fmt.eprintln("cuda_selftest: attention_q8v_long4 missing")
				return false
			}
			long_part := i32(128)
			lsplit := (npos + 127) / 128
			lsmem := u32((kv_mul * hd + kv_mul * 128 + kv_mul * 2 + hd) * 4)
			largs := [?]rawptr{&qa, &kca, &vca, &pm, &pl, &po, &ihd, &ikb, &ivb, &mul, &nh, &ps, &long_part, &slot}
			if cu.launch(long_fn, u32(n_kv), u32(lsplit), 1, 128, 1, 1, lsmem, largs[:]) != cu.OK || cu.sync() != cu.OK {
				fmt.eprintfln("cuda_selftest: attention_q8v_long4 launch failed at npos {}", npos)
				return false
			}
			lns := i32(lsplit)
			lrargs := [?]rawptr{&xb, &pm, &pl, &po, &ihd, &nh, &lns}
			if cu.launch(red_fn, u32(n_heads), 1, 1, 128, 1, 1, 0, lrargs[:]) != cu.OK || cu.sync() != cu.OK {
				fmt.eprintfln("cuda_selftest: attention_q8v_long reduce failed at npos {}", npos)
				return false
			}
			if cu.download(raw_data(got_new), dxb, len(got_new) * 4) != cu.OK {
				return false
			}
			st_long: Case_Err
			for i in 0 ..< len(got_new) {
				s_note(&st_long, fmt.tprintf("long[%d] npos %d", i, npos), got_new[i], got_old[i], 2e-2, 1e-3)
			}
			if !s_report(fmt.tprintf("attn_long npos={}", npos), st_long, 2e-2, 1e-3) {
				ok = false
			}
		}
	}
	return ok
}

// Prefill flash against attention_q8v on one mixed-V cache. kv_mul is 6,
// the 27B group, so K and V for a 32-token tile are loaded once and reused
// by every query head in the group. Each case is a causal chunk: token tok
// attends through pos0+tok. T=128 is a full production chunk; T=7 is a
// partial last query tile (br is 2 for this shape). Online softmax is the
// same fp32 merge as the split reduce, applied as tiles arrive, so it is
// not bit-exact. Allowance is 2e-2 absolute plus 1e-3 times |reference|.
@(private)
run_attn_flash :: proc() -> bool {
	hd := 256
	n_kv := 2
	kv_mul := 6
	n_heads := n_kv * kv_mul
	npos_max := 32768
	t_max := 128
	nb_head := hd / 32
	kv_dim := n_kv * hd
	nb := kv_dim / 32
	k_pos := nb * Q8_0_BLOCK_BYTES
	v_pos := nb * Q8_V_MIX_BLOCK_BYTES
	br := attn_flash_pick(kv_mul, hd)
	if br < 1 {
		fmt.eprintln("cuda_selftest: attn flash does not fit this shape")
		return false
	}
	kc := make([]u8, npos_max * k_pos)
	vc := make([]u8, npos_max * v_pos)
	q := make([]f32, t_max * n_heads * hd)
	blk := make([]f32, 32)
	defer delete(kc)
	defer delete(vc)
	defer delete(q)
	defer delete(blk)
	for t in 0 ..< npos_max {
		for kvh in 0 ..< n_kv {
			for b in 0 ..< nb_head {
				gid := kvh * nb_head + b
				for j in 0 ..< 32 {
					i := b * 32 + j
					blk[j] = (f32((i * 3 + t * 5 + kvh * 7) % 11) - 5) * 0.15
				}
				quantize_q8_block(blk, kc[t * k_pos + gid * Q8_0_BLOCK_BYTES:][:Q8_0_BLOCK_BYTES])
				for j in 0 ..< 32 {
					i := b * 32 + j
					blk[j] = (f32((i * 2 + t + kvh) % 9) - 4) * 0.2
					if j == 3 {
						blk[j] = 3.5 + f32(t) * 0.01
					}
				}
				quantize_q8_v_mix_block(blk, vc[t * v_pos + gid * Q8_V_MIX_BLOCK_BYTES:][:Q8_V_MIX_BLOCK_BYTES])
			}
		}
	}
	for tok in 0 ..< t_max {
		for h in 0 ..< n_heads {
			for i in 0 ..< hd {
				q[(tok * n_heads + h) * hd + i] = (f32((i + h * 3 + tok * 5) % 13) - 6) * 0.1
			}
		}
	}
	dq, _ := cu.alloc(len(q) * 4)
	dkc, _ := cu.alloc(len(kc))
	dvc, _ := cu.alloc(len(vc))
	dxb, _ := cu.alloc(t_max * n_heads * hd * 4)
	dref, _ := cu.alloc(n_heads * hd * 4)
	dsc, _ := cu.alloc(n_heads * npos_max * 4)
	defer cu.free(dq)
	defer cu.free(dkc)
	defer cu.free(dvc)
	defer cu.free(dxb)
	defer cu.free(dref)
	defer cu.free(dsc)
	if dq == 0 || dkc == 0 || dvc == 0 || dxb == 0 || dref == 0 || dsc == 0 {
		fmt.eprintln("cuda_selftest: attn flash alloc failed")
		return false
	}
	if cu.upload(dq, raw_data(q), len(q) * 4) != cu.OK || cu.upload(dkc, raw_data(kc), len(kc)) != cu.OK || cu.upload(dvc, raw_data(vc), len(vc)) != cu.OK {
		fmt.eprintln("cuda_selftest: attn flash upload failed")
		return false
	}
	old_fn := s_fn("attention_q8v")
	flash_fn := s_fn("attention_q8v_flash")
	if old_fn == nil || flash_fn == nil {
		fmt.eprintln("cuda_selftest: attn flash kernels missing")
		return false
	}
	smem := u32(attn_flash_smem(br, kv_mul, hd))
	got_old := make([]f32, t_max * n_heads * hd)
	got_new := make([]f32, t_max * n_heads * hd)
	one := make([]f32, n_heads * hd)
	defer delete(got_old)
	defer delete(got_new)
	defer delete(one)
	// npos is the context the last token of the chunk can see. pos0 = npos-T.
	cases := [?][2]int{{128, 128}, {128, 7}, {128, 4}, {300, 4}, {1800, 4}, {8192, 4}, {32768, 4}}
	ok := true
	for cs in cases {
		npos := cs[0]
		T := cs[1]
		pos0 := npos - T
		kca, vca, sc := dkc, dvc, dsc
		ihd := i32(hd)
		ikb := i32(k_pos)
		ivb := i32(v_pos)
		mul := i32(kv_mul)
		sl := i32(npos_max)
		slot: cu.Device_Ptr
		stride0 := i32(0)
		for tok in 0 ..< T {
			qa := cu.ptr_add(dq, tok * n_heads * hd * 4)
			xb := dref
			ps := i32(pos0 + tok)
			oargs := [?]rawptr{&qa, &kca, &vca, &xb, &sc, &ihd, &ikb, &ivb, &mul, &ps, &sl, &slot, &stride0}
			if cu.launch(old_fn, u32(n_heads), 1, 1, 128, 1, 1, 0, oargs[:]) != cu.OK || cu.sync() != cu.OK {
				fmt.eprintfln("cuda_selftest: attention_q8v launch failed at npos {} T {}", npos, T)
				return false
			}
			if cu.download(raw_data(one), dref, len(one) * 4) != cu.OK {
				return false
			}
			copy(got_old[tok * n_heads * hd:], one)
		}
		qa := dq
		xb := dxb
		nh := i32(n_heads)
		ps := i32(pos0)
		nt := i32(T)
		stride := i32(n_heads * hd)
		ibr := i32(br)
		fargs := [?]rawptr{&qa, &kca, &vca, &xb, &ihd, &ikb, &ivb, &mul, &nh, &ps, &nt, &stride, &ibr}
		gy := u32((T + br - 1) / br)
		if cu.launch(flash_fn, u32(n_kv), gy, 1, 128, 1, 1, smem, fargs[:]) != cu.OK || cu.sync() != cu.OK {
			fmt.eprintfln("cuda_selftest: attention_q8v_flash launch failed at npos {} T {}", npos, T)
			return false
		}
		nfloat := T * n_heads * hd
		if cu.download(raw_data(got_new), dxb, nfloat * 4) != cu.OK {
			return false
		}
		st: Case_Err
		for i in 0 ..< nfloat {
			s_note(&st, fmt.tprintf("flash[%d] npos %d T %d", i, npos, T), got_new[i], got_old[i], 2e-2, 1e-3)
		}
		if !s_report(fmt.tprintf("attn_flash npos={} T={}", npos, T), st, 2e-2, 1e-3) {
			ok = false
		}
	}
	return ok
}

// WMMA prefill against attention_q8v. fp16 Q/K/V products, fp32 accumulate,
// online softmax. kv_mul 4 is the 9B group (16 tokens x 4 heads = 64 queries).
// kv_mul 6 is the 27B group. Allowance is 1e-1 absolute plus 5e-2 times
// |reference|, wider than the fp32 flash band because the products are fp16.
@(private)
run_attn_wmma :: proc() -> bool {
	ok := true
	wide := [?][2]int{{128, 16}, {128, 7}, {128, 64}, {300, 16}, {1800, 16}, {8192, 16}, {32768, 16}}
	short := [?][2]int{{128, 16}, {1800, 16}}
	if !run_attn_wmma_shape(4, wide[:]) {
		ok = false
	}
	if !run_attn_wmma_shape(6, short[:]) {
		ok = false
	}
	return ok
}

@(private)
run_attn_wmma_shape :: proc(kv_mul: int, cases: [][2]int) -> bool {
	hd := 256
	n_kv := 2
	n_heads := n_kv * kv_mul
	npos_max := 1
	t_max := 1
	for cs in cases {
		if cs[0] > npos_max {
			npos_max = cs[0]
		}
		if cs[1] > t_max {
			t_max = cs[1]
		}
	}
	nb_head := hd / 32
	kv_dim := n_kv * hd
	nb := kv_dim / 32
	k_pos := nb * Q8_0_BLOCK_BYTES
	v_pos := nb * Q8_V_MIX_BLOCK_BYTES
	kc := make([]u8, npos_max * k_pos)
	vc := make([]u8, npos_max * v_pos)
	q := make([]f32, t_max * n_heads * hd)
	blk := make([]f32, 32)
	defer delete(kc)
	defer delete(vc)
	defer delete(q)
	defer delete(blk)
	for t in 0 ..< npos_max {
		for kvh in 0 ..< n_kv {
			for b in 0 ..< nb_head {
				gid := kvh * nb_head + b
				for j in 0 ..< 32 {
					i := b * 32 + j
					blk[j] = (f32((i * 3 + t * 5 + kvh * 7) % 11) - 5) * 0.15
				}
				quantize_q8_block(blk, kc[t * k_pos + gid * Q8_0_BLOCK_BYTES:][:Q8_0_BLOCK_BYTES])
				for j in 0 ..< 32 {
					i := b * 32 + j
					blk[j] = (f32((i * 2 + t + kvh) % 9) - 4) * 0.2
					if j == 3 {
						blk[j] = 3.5 + f32(t) * 0.01
					}
				}
				quantize_q8_v_mix_block(blk, vc[t * v_pos + gid * Q8_V_MIX_BLOCK_BYTES:][:Q8_V_MIX_BLOCK_BYTES])
			}
		}
	}
	for tok in 0 ..< t_max {
		for h in 0 ..< n_heads {
			for i in 0 ..< hd {
				q[(tok * n_heads + h) * hd + i] = (f32((i + h * 3 + tok * 5) % 13) - 6) * 0.1
			}
		}
	}
	q_cap := (t_max + ATTN_WMMA_BQ - 1) / ATTN_WMMA_BQ
	nslot := ATTN_WMMA_PARTS * q_cap * n_heads * ATTN_WMMA_BQ
	dq, _ := cu.alloc(len(q) * 4)
	dkc, _ := cu.alloc(len(kc))
	dvc, _ := cu.alloc(len(vc))
	dxb, _ := cu.alloc(t_max * n_heads * hd * 4)
	dref, _ := cu.alloc(n_heads * hd * 4)
	dsc, _ := cu.alloc(n_heads * npos_max * 4)
	dpo, _ := cu.alloc(nslot * hd * 4)
	dpm, _ := cu.alloc(nslot * 4)
	dpl, _ := cu.alloc(nslot * 4)
	defer cu.free(dq)
	defer cu.free(dkc)
	defer cu.free(dvc)
	defer cu.free(dxb)
	defer cu.free(dref)
	defer cu.free(dsc)
	defer cu.free(dpo)
	defer cu.free(dpm)
	defer cu.free(dpl)
	if dq == 0 || dkc == 0 || dvc == 0 || dxb == 0 || dref == 0 || dsc == 0 || dpo == 0 || dpm == 0 || dpl == 0 {
		fmt.eprintln("cuda_selftest: attn wmma alloc failed")
		return false
	}
	if cu.upload(dq, raw_data(q), len(q) * 4) != cu.OK || cu.upload(dkc, raw_data(kc), len(kc)) != cu.OK || cu.upload(dvc, raw_data(vc), len(vc)) != cu.OK {
		fmt.eprintln("cuda_selftest: attn wmma upload failed")
		return false
	}
	old_fn := s_fn("attention_q8v")
	wmma_fn := s_fn("attention_q8v_wmma")
	red_fn := s_fn("attention_q8v_wmma_reduce")
	if old_fn == nil || wmma_fn == nil || red_fn == nil {
		fmt.eprintln("cuda_selftest: attn wmma kernels missing")
		return false
	}
	if cu.func_set_max_dynamic_shared(wmma_fn, ATTN_WMMA_SMEM) != cu.OK {
		fmt.eprintln("cuda_selftest: attn wmma could not raise dynamic smem")
		return false
	}
	got_old := make([]f32, t_max * n_heads * hd)
	got_new := make([]f32, t_max * n_heads * hd)
	one := make([]f32, n_heads * hd)
	defer delete(got_old)
	defer delete(got_new)
	defer delete(one)
	ok := true
	for cs in cases {
		npos := cs[0]
		T := cs[1]
		if T > npos {
			fmt.eprintfln("cuda_selftest: attn wmma T {} > npos {}", T, npos)
			return false
		}
		pos0 := npos - T
		plan := attn_wmma_plan(T, pos0, n_kv)
		kca, vca, sc := dkc, dvc, dsc
		ihd := i32(hd)
		ikb := i32(k_pos)
		ivb := i32(v_pos)
		mul := i32(kv_mul)
		sl := i32(npos_max)
		slot: cu.Device_Ptr
		stride0 := i32(0)
		for tok in 0 ..< T {
			qa := cu.ptr_add(dq, tok * n_heads * hd * 4)
			xb := dref
			ps := i32(pos0 + tok)
			oargs := [?]rawptr{&qa, &kca, &vca, &xb, &sc, &ihd, &ikb, &ivb, &mul, &ps, &sl, &slot, &stride0}
			if cu.launch(old_fn, u32(n_heads), 1, 1, 128, 1, 1, 0, oargs[:]) != cu.OK || cu.sync() != cu.OK {
				fmt.eprintfln("cuda_selftest: attention_q8v launch failed at wmma npos {} T {}", npos, T)
				return false
			}
			if cu.download(raw_data(one), dref, len(one) * 4) != cu.OK {
				return false
			}
			copy(got_old[tok * n_heads * hd:], one)
		}
		qa := dq
		po, pm, pl := dpo, dpm, dpl
		nh := i32(n_heads)
		ps := i32(pos0)
		nt := i32(T)
		stride := i32(n_heads * hd)
		qt := i32(plan.q_tiles)
		plen := i32(plan.part_len)
		fargs := [?]rawptr{&qa, &kca, &vca, &po, &pm, &pl, &ihd, &ikb, &ivb, &mul, &nh, &ps, &nt, &stride, &qt, &plen}
		if cu.launch(wmma_fn, u32(plan.q_tiles), u32(n_kv), u32(plan.nparts), 512, 1, 1, u32(ATTN_WMMA_SMEM), fargs[:]) != cu.OK || cu.sync() != cu.OK {
			fmt.eprintfln("cuda_selftest: attention_q8v_wmma launch failed at npos {} T {}", npos, T)
			return false
		}
		out := dxb
		np := i32(plan.nparts)
		rargs := [?]rawptr{&out, &po, &pm, &pl, &nh, &qt, &np, &ps, &nt, &stride}
		if cu.launch(red_fn, u32(n_heads), u32(plan.q_tiles), 1, 128, 1, 1, 0, rargs[:]) != cu.OK || cu.sync() != cu.OK {
			fmt.eprintfln("cuda_selftest: attention_q8v_wmma_reduce launch failed at npos {} T {}", npos, T)
			return false
		}
		nfloat := T * n_heads * hd
		if cu.download(raw_data(got_new), dxb, nfloat * 4) != cu.OK {
			return false
		}
		st: Case_Err
		for i in 0 ..< nfloat {
			s_note(&st, fmt.tprintf("wmma[%d] npos %d T %d kv_mul %d", i, npos, T, kv_mul), got_new[i], got_old[i], 1e-1, 5e-2)
		}
		if !s_report(fmt.tprintf("attn_wmma kv_mul={} npos={} T={}", kv_mul, npos, T), st, 1e-1, 5e-2) {
			ok = false
		}
	}
	return ok
}

@(private)
run_delta :: proc() -> bool {
	hkd, hvd := 4, 4
	state := make([]f32, hkd * hvd)
	q := make([]f32, hkd)
	k := make([]f32, hkd)
	v := make([]f32, hvd)
	defer delete(state)
	defer delete(q)
	defer delete(k)
	defer delete(v)
	for i in 0 ..< len(state) {
		state[i] = (f32(i % 5) - 2) * 0.1
	}
	for i in 0 ..< hkd {
		q[i] = 0.2
		k[i] = (f32(i) - 1) * 0.3
	}
	for i in 0 ..< hvd {
		v[i] = (f32(i) - 0.5) * 0.4
	}
	bv := f32(0.3)
	av := f32(0.2)
	dec := f32(-0.7)
	dt := f32(0.1)
	beta := sigmoid_f32(bv)
	g := math.exp_f32(dec * softplus_f32(av + dt))
	cpu_state := make([]f32, len(state))
	copy(cpu_state, state)
	defer delete(cpu_state)
	out := make([]f32, hvd)
	scratch := make([]f32, hvd)
	defer delete(out)
	defer delete(scratch)
	delta_recurrent_step(out, cpu_state, q, k, v, scratch, beta, g, hkd, hvd)

	ds, _ := cu.alloc(len(state) * 4)
	dq, _ := cu.alloc(hkd * 4)
	dk, _ := cu.alloc(hkd * 4)
	dv, _ := cu.alloc(hvd * 4)
	db, _ := cu.alloc(4)
	da, _ := cu.alloc(4)
	dd, _ := cu.alloc(4)
	dtb, _ := cu.alloc(4)
	dout, _ := cu.alloc(hvd * 4)
	defer cu.free(ds)
	defer cu.free(dq)
	defer cu.free(dk)
	defer cu.free(dv)
	defer cu.free(db)
	defer cu.free(da)
	defer cu.free(dd)
	defer cu.free(dtb)
	defer cu.free(dout)
	cu.upload(ds, raw_data(state), len(state) * 4)
	cu.upload(dq, raw_data(q), hkd * 4)
	cu.upload(dk, raw_data(k), hkd * 4)
	cu.upload(dv, raw_data(v), hvd * 4)
	b1 := [1]f32{bv}
	a1 := [1]f32{av}
	d1 := [1]f32{dec}
	t1 := [1]f32{dt}
	cu.upload(db, raw_data(b1[:]), 4)
	cu.upload(da, raw_data(a1[:]), 4)
	cu.upload(dd, raw_data(d1[:]), 4)
	cu.upload(dtb, raw_data(t1[:]), 4)
	fn := s_fn("delta_recurrent")
	if fn == nil {
		return false
	}
	o, st, qq, kk, vv, bb, aa, decp, dtp := dout, ds, dq, dk, dv, db, da, dd, dtb
	ik, iv, nk := i32(hkd), i32(hvd), i32(1)
	args := [?]rawptr{&o, &st, &qq, &kk, &vv, &bb, &aa, &decp, &dtp, &ik, &iv, &nk}
	if cu.launch(fn, 1, 1, 1, 128, 1, 1, 0, args[:]) != cu.OK {
		fmt.eprintln("cuda_selftest: delta launch failed")
		return false
	}
	gout := make([]f32, hvd)
	gstate := make([]f32, len(state))
	defer delete(gout)
	defer delete(gstate)
	cu.download(raw_data(gout), dout, hvd * 4)
	cu.download(raw_data(gstate), ds, len(state) * 4)
	st_out: Case_Err
	for i in 0 ..< hvd {
		s_note(&st_out, fmt.tprintf("delta out[%d]", i), gout[i], out[i], 1e-4, 1e-4)
	}
	if !s_report("delta out", st_out, 1e-4, 1e-4) {
		return false
	}
	st_state: Case_Err
	for i in 0 ..< len(state) {
		s_note(&st_state, fmt.tprintf("delta state[%d]", i), gstate[i], cpu_state[i], 1e-4, 1e-4)
	}
	return s_report("delta state", st_state, 1e-4, 1e-4)
}

// Two tokens, two value heads, one key head. The chunk kernel must match
// stepping delta_recurrent_step in token order.
@(private)
run_delta_chunk :: proc() -> bool {
	return run_delta_case(4, 4, 1, 2, 2, 0)
}

// which: 0 global recurrence, 1 smem recurrence, 2 chunked WMMA.
@(private)
run_delta_case :: proc(hkd, hvd, nkh, nvh, T, which: int) -> bool {
	name := "delta_recurrent_chunk"
	shared := u32(0)
	block := u32(128)
	atol: f32 = 1e-4
	if which == 1 {
		name = "delta_recurrent_chunk_sm"
		shared = u32(hkd * hvd * 4)
	} else if which == 2 {
		name = "delta_chunk_wmma"
		shared = 49152
		block = 256
		atol = 2e-3
	}
	fn := s_fn(name)
	if fn == nil {
		fmt.eprintf("cuda_selftest: %s missing\n", name)
		return false
	}
	if which == 1 && shared > 48 * 1024 {
		if cu.func_set_max_dynamic_shared(fn, int(shared)) != cu.OK {
			fmt.eprintf("cuda_selftest: %s could not raise dynamic smem to %d\n", name, shared)
			return false
		}
	}
	state := make([]f32, nvh * hkd * hvd)
	q := make([]f32, T * nkh * hkd)
	k := make([]f32, T * nkh * hkd)
	v := make([]f32, T * nvh * hvd)
	b := make([]f32, T * nvh)
	a := make([]f32, T * nvh)
	dec := make([]f32, nvh)
	dt := make([]f32, nvh)
	defer delete(state)
	defer delete(q)
	defer delete(k)
	defer delete(v)
	defer delete(b)
	defer delete(a)
	defer delete(dec)
	defer delete(dt)
	for i in 0 ..< len(state) {
		state[i] = (f32(i % 5) - 2) * 0.05
	}
	for t in 0 ..< T {
		for i in 0 ..< nkh * hkd {
			q[t * nkh * hkd + i] = 0.1 * f32(t + 1) + 0.01 * f32(i)
			k[t * nkh * hkd + i] = 0.05 * f32(i + 1) - 0.02 * f32(t)
		}
		for h in 0 ..< nvh {
			b[t * nvh + h] = 0.2 * f32(h + 1) + 0.1 * f32(t)
			a[t * nvh + h] = -0.3 + 0.1 * f32(t + h)
			for j in 0 ..< hvd {
				v[(t * nvh + h) * hvd + j] = 0.15 * f32(j + 1) - 0.05 * f32(t)
			}
		}
	}
	for h in 0 ..< nvh {
		dec[h] = -0.4 - 0.1 * f32(h)
		dt[h] = 0.05 * f32(h + 1)
	}
	if which == 2 {
		// Production l2-normalizes k and scales q by 1/sqrt(head_k). The raw
		// pattern has ||k|| around 40, and the fp32 WY substitution overflows.
		inv_sqrt := 1 / math.sqrt(f32(hkd))
		for t in 0 ..< T {
			for h in 0 ..< nkh {
				base := t * nkh * hkd + h * hkd
				ss: f32
				for i in 0 ..< hkd {
					ss += k[base + i] * k[base + i]
				}
				inv := 1 / math.sqrt(ss)
				for i in 0 ..< hkd {
					k[base + i] *= inv
				}
				ss = 0
				for i in 0 ..< hkd {
					ss += q[base + i] * q[base + i]
				}
				inv = inv_sqrt / math.sqrt(ss)
				for i in 0 ..< hkd {
					q[base + i] *= inv
				}
			}
		}
	}
	cpu_state := make([]f32, len(state))
	copy(cpu_state, state)
	defer delete(cpu_state)
	cpu_out := make([]f32, T * nvh * hvd)
	defer delete(cpu_out)
	scratch := make([]f32, hvd)
	defer delete(scratch)
	for t in 0 ..< T {
		for h in 0 ..< nvh {
			beta := sigmoid_f32(b[t * nvh + h])
			g := math.exp_f32(dec[h] * softplus_f32(a[t * nvh + h] + dt[h]))
			kh := h % nkh
			delta_recurrent_step(
				cpu_out[(t * nvh + h) * hvd:][:hvd],
				cpu_state[h * hkd * hvd:][:hkd * hvd],
				q[t * nkh * hkd + kh * hkd:][:hkd],
				k[t * nkh * hkd + kh * hkd:][:hkd],
				v[(t * nvh + h) * hvd:][:hvd],
				scratch, beta, g, hkd, hvd,
			)
		}
	}
	ds, _ := cu.alloc(len(state) * 4)
	dq, _ := cu.alloc(len(q) * 4)
	dk, _ := cu.alloc(len(k) * 4)
	dv, _ := cu.alloc(len(v) * 4)
	db, _ := cu.alloc(len(b) * 4)
	da, _ := cu.alloc(len(a) * 4)
	dd, _ := cu.alloc(len(dec) * 4)
	dtb, _ := cu.alloc(len(dt) * 4)
	dout, _ := cu.alloc(len(cpu_out) * 4)
	defer cu.free(ds)
	defer cu.free(dq)
	defer cu.free(dk)
	defer cu.free(dv)
	defer cu.free(db)
	defer cu.free(da)
	defer cu.free(dd)
	defer cu.free(dtb)
	defer cu.free(dout)
	cu.upload(ds, raw_data(state), len(state) * 4)
	cu.upload(dq, raw_data(q), len(q) * 4)
	cu.upload(dk, raw_data(k), len(k) * 4)
	cu.upload(dv, raw_data(v), len(v) * 4)
	cu.upload(db, raw_data(b), len(b) * 4)
	cu.upload(da, raw_data(a), len(a) * 4)
	cu.upload(dd, raw_data(dec), len(dec) * 4)
	cu.upload(dtb, raw_data(dt), len(dt) * 4)
	o, stp, qq, kk, vv, bb, aa, decp, dtp := dout, ds, dq, dk, dv, db, da, dd, dtb
	ik, iv, nk, nv := i32(hkd), i32(hvd), i32(nkh), i32(nvh)
	tt, qs, vs := i32(T), i32(nkh * hkd), i32(nvh * hvd)
	args := [?]rawptr{&o, &stp, &qq, &kk, &vv, &bb, &aa, &decp, &dtp, &ik, &iv, &nk, &nv, &tt, &qs, &vs}
	if cu.launch(fn, u32(nvh), 1, 1, block, 1, 1, shared, args[:]) != cu.OK || cu.sync() != cu.OK {
		fmt.eprintf("cuda_selftest: %s launch failed\n", name)
		return false
	}
	gout := make([]f32, len(cpu_out))
	gstate := make([]f32, len(state))
	defer delete(gout)
	defer delete(gstate)
	cu.download(raw_data(gout), dout, len(cpu_out) * 4)
	cu.download(raw_data(gstate), ds, len(state) * 4)
	st_out: Case_Err
	for i in 0 ..< len(cpu_out) {
		s_note(&st_out, fmt.tprintf("%s out[%d]", name, i), gout[i], cpu_out[i], atol, atol)
	}
	if !s_report(fmt.tprintf("%s out", name), st_out, atol, atol) {
		return false
	}
	st_state: Case_Err
	for i in 0 ..< len(state) {
		s_note(&st_state, fmt.tprintf("%s state[%d]", name, i), gstate[i], cpu_state[i], atol, atol)
	}
	return s_report(fmt.tprintf("%s state", name), st_state, atol, atol)
}

cuda_selftest :: proc() -> bool {
	if !s_open() {
		return false
	}
	defer s_close()
	ok := true
	kinds := [?]ggml.GGML_Type{
		.F32, .F16, .Q8_0, .Q4_0, .Q4_1, .Q5_0, .Q5_1,
		.Q4_K, .Q6_K, .Q2_K, .Q3_K, .Q5_K, .Q8_K,
	}
	for kind in kinds {
		atol, rtol: f32 = 2e-2, 1e-3
		if kind == .F32 || kind == .F16 {
			atol, rtol = 1e-4, 1e-5
		}
		if !run_gemv(kind, 3, 256, atol, rtol) {
			ok = false
		}
	}
	// dp4a GEMV vs the same CPU reference. q8 of x is approximate, so the
	// tolerance is wider than the scalar kernel; a swapped scale is still O(1).
	gemv2 := [?]ggml.GGML_Type{.Q4_K, .Q6_K, .Q8_0, .Q5_0, .Q5_K}
	for kind in gemv2 {
		// q8 of x is the only approximation. A swapped nibble or scale is O(|ref|).
		if !run_gemv2(kind, 64, 512, 1, 5e-2) {
			ok = false
		}
		// A row count that is not a multiple of 4 exercises the tail.
		if !run_gemv2(kind, 6, 256, 1, 5e-2) {
			ok = false
		}
	}
	mmq_kinds := [?]ggml.GGML_Type{.F32, .Q8_0, .Q4_K, .Q6_K, .Q5_0}
	for kind in mmq_kinds {
		if !run_mmq(kind, 3, 256, 5) {
			ok = false
		}
	}
	// fp16 tiles vs the fp32 CPU dot. K=256, values O(0.1). A wrong WMMA
	// layout is off by O(1), not by this tolerance.
	tc_kinds := [?]ggml.GGML_Type{
		.F32, .F16, .Q8_0, .Q4_0, .Q5_0, .Q4_K, .Q6_K, .Q5_K, .Q8_K,
	}
	for kind in tc_kinds {
		atol, rtol: f32 = 5e-2, 5e-2
		if kind == .F32 || kind == .F16 {
			atol, rtol = 2e-2, 1e-2
		}
		if !run_tc(kind, 64, 256, 32, atol, rtol) {
			ok = false
		}
	}
	// Partial tile: T and rows are not multiples of 16.
	if !run_tc(.Q4_K, 40, 256, 20, 5e-2, 5e-2) {
		ok = false
	}
	if !run_tc(.Q8_0, 40, 64, 20, 5e-2, 5e-2) {
		ok = false
	}
	if !run_tc(.Q5_0, 40, 64, 20, 5e-2, 5e-2) {
		ok = false
	}
	if !run_tc2(.Q4_K, 64, 256, 32, 5e-2, 5e-2, false) || !run_tc2(.Q6_K, 64, 256, 32, 5e-2, 5e-2, false) {
		ok = false
	}
	if !run_tc2(.Q4_K, 40, 256, 20, 5e-2, 5e-2, false) || !run_tc2(.Q6_K, 40, 512, 20, 5e-2, 5e-2, false) {
		ok = false
	}
	// 80 tokens forces the second 64-token chunk that reuses the fp16 weight tile.
	if !run_tc2(.Q4_K, 64, 256, 80, 5e-2, 5e-2, true) || !run_tc2(.Q6_K, 48, 512, 160, 5e-2, 5e-2, true) {
		ok = false
	}
	if !run_tc2(.Q4_K, 64, 256, 32, 5e-2, 5e-2, true) || !run_tc2(.Q6_K, 40, 256, 20, 5e-2, 5e-2, true) {
		ok = false
	}
	// tc2m is the opt-in 64x64 block-scale tile. T=80 is a partial second
	// token tile; 40 rows is a partial row tile; K=512 crosses two superblocks.
	if !run_tc2m(.Q4_K, 64, 256, 64, 5e-2, 5e-2) || !run_tc2m(.Q4_K, 40, 256, 80, 5e-2, 5e-2) {
		ok = false
	}
	if !run_tc2m(.Q6_K, 64, 512, 64, 5e-2, 5e-2) || !run_tc2m(.Q6_K, 48, 256, 96, 5e-2, 5e-2) {
		ok = false
	}
	if !run_tc2m(.Q6_K, 40, 512, 128, 5e-2, 5e-2) {
		ok = false
	}
	// pgemm: exact 64x64 tile, a partial tile, K across two superblocks, and
	// a token count that is not a multiple of 64.
	pgemm_shapes := [?][3]int{{64, 256, 8}, {64, 256, 32}, {64, 256, 128}, {40, 256, 20}, {64, 512, 64}, {48, 512, 96}, {96, 1024, 127}}
	for shape in pgemm_shapes {
		if !run_pgemm(.Q4_K, shape[0], shape[1], shape[2], 5e-2, 5e-2) {
			ok = false
		}
		if !run_pgemm(.Q6_K, shape[0], shape[1], shape[2], 5e-2, 5e-2) {
			ok = false
		}
		// Same shapes. mmqk must match gemv2 within 2e-3. The CPU compare uses gemv2's q8 allowance.
		if !run_mmqk(.Q4_K, shape[0], shape[1], shape[2]) || !run_mmqk(.Q6_K, shape[0], shape[1], shape[2]) {
			ok = false
		}
		if !run_mmqk_pair(.Q4_K, shape[0], shape[1], shape[2]) || !run_mmqk_pair(.Q6_K, shape[0], shape[1], shape[2]) {
			ok = false
		}
	}
	// 128x128 tile, a partial row and token tile, K across two superblocks,
	// and a token count that is not a multiple of 128. fp16 WMMA vs fp32 CPU.
	wgemm_shapes := [?][3]int{{128, 256, 128}, {96, 256, 20}, {130, 512, 160}, {40, 256, 7}, {256, 1024, 200}, {48, 768, 96}}
	for shape in wgemm_shapes {
		if !run_wgemm(.Q4_K, shape[0], shape[1], shape[2], 5e-2, 5e-2) {
			ok = false
		}
		if !run_wgemm(.Q6_K, shape[0], shape[1], shape[2], 5e-2, 5e-2) {
			ok = false
		}
		if !run_wgemm(.Q5_K, shape[0], shape[1], shape[2], 5e-2, 5e-2) {
			ok = false
		}
	}
	// Partial rows, partial token tiles, K past one superblock, and a wide
	// activation range. tc2r has to match scalar tc exactly on each of these.
	pair_q4 := [?][3]int{{64, 256, 128}, {65, 512, 65}, {96, 1024, 127}, {130, 256, 129}, {40, 2048, 160}, {1, 256, 128}, {63, 512, 192}}
	for shape in pair_q4 {
		if !run_tc_pair(.Q4_K, shape[0], shape[1], shape[2], false) {
			ok = false
		}
	}
	if !run_tc_pair(.Q4_K, 80, 4096, 128, true) {
		ok = false
	}
	pair_q6 := [?][3]int{{64, 256, 128}, {48, 512, 96}, {33, 768, 70}, {96, 1024, 200}}
	for shape in pair_q6 {
		if !run_tc_pair(.Q6_K, shape[0], shape[1], shape[2], false) {
			ok = false
		}
	}
	if !run_tc_pair(.Q6_K, 70, 2048, 129, true) {
		ok = false
	}
	if !run_delta_chunk() {
		ok = false
	}
	// hvd is not a multiple of the block size, and the 128x128 production
	// state (64 KiB) which needs the dynamic-smem cap raised.
	if !run_delta_case(16, 20, 2, 2, 3, 1) || !run_delta_case(128, 128, 1, 1, 2, 1) {
		ok = false
	}
	// Chunked WMMA against the recurrent step. T=32 is two full chunks.
	// T=20 leaves a partial chunk. Allowance is the fp16-operand drift.
	if !run_delta_case(128, 128, 1, 1, 32, 2) || !run_delta_case(128, 128, 1, 2, 20, 2) {
		ok = false
	}
	if !run_elem() {
		ok = false
	}
	if !run_kv_attn() {
		ok = false
	}
	if !run_attn_split() {
		ok = false
	}
	if !run_attn_flash() {
		ok = false
	}
	if !run_attn_wmma() {
		ok = false
	}
	if !run_delta() {
		ok = false
	}
	if ok {
		fmt.eprintln("cuda_selftest: PASS")
	} else {
		fmt.eprintln("cuda_selftest: FAIL")
	}
	return ok
}

// Representative 9B prefill GEMMs (T=64) plus a small Q8_0 / Q5_0 pair for
// the 0.8B MoE quants. Times tensor-core WMMA against the MMQ fallback.
// FLOPs count a multiply and an add: 2*T*rows*cols.
cuda_tc_bench :: proc() -> bool {
	if !s_open() {
		return false
	}
	defer s_close()
	Shape :: struct {
		kind:       ggml.GGML_Type,
		rows, cols: int,
		T:          int,
		name:       string,
	}
	shapes := [?]Shape{
		{.Q4_K, 12288, 4096, 64, "9B w1/w3"},
		{.Q4_K, 8192, 4096, 64, "9B wq"},
		{.Q4_K, 4096, 4096, 64, "9B wo"},
		{.Q6_K, 4096, 12288, 64, "9B w2"},
		{.Q8_0, 1024, 1024, 32, "0.8B Q8_0"},
		{.Q5_0, 1024, 1024, 32, "0.8B Q5_0"},
	}
	fmt.eprintln("cuda_bench: TC is WMMA 16x16x16 fp16, fp32 acc. MMQ is the warp fallback.")
	for s in shapes {
		tag := cuda_kind_tag(s.kind)
		tc := s_fn(fmt.tprintf("tc_%s", tag))
		mmq := s_fn(fmt.tprintf("mmq_%s", tag))
		if tc == nil || mmq == nil {
			fmt.eprintf("cuda_bench: missing kernel for %s\n", tag)
			return false
		}
		raw := fill_weight(s.kind, s.rows, s.cols)
		x := make([]f32, s.T * s.cols)
		for i in 0 ..< len(x) {
			x[i] = (f32(i % 11) - 5) * 0.02
		}
		dw, e1 := cu.alloc(len(raw))
		dx, e2 := cu.alloc(len(x) * 4)
		dy, e3 := cu.alloc(s.T * s.rows * 4)
		if e1 != cu.OK || e2 != cu.OK || e3 != cu.OK {
			fmt.eprintln("cuda_bench: alloc failed")
			delete(raw)
			delete(x)
			return false
		}
		cu.upload(dw, raw_data(raw), len(raw))
		cu.upload(dx, raw_data(x), len(x) * 4)
		delete(raw)
		delete(x)
		wb, xx, yy := dw, dx, dy
		nn, dd, tt := i32(s.cols), i32(s.rows), i32(s.T)
		args := [?]rawptr{&wb, &xx, &yy, &nn, &dd, &tt}
		gx := u32((s.rows + 63) / 64)
		gy := u32((s.T + 63) / 64)
		tc_ms := bench_ms(tc, args[:], gx, gy, 512)
		mmq_ms := bench_ms(mmq, args[:], u32(s.rows), 1, 32)
		cu.free(dw)
		cu.free(dx)
		cu.free(dy)
		flops := 2 * f64(s.T) * f64(s.rows) * f64(s.cols)
		tc_tflops := flops / (tc_ms * 1e-3) / 1e12
		mmq_tflops := flops / (mmq_ms * 1e-3) / 1e12
		fmt.eprintfln(
			"cuda_bench: %s %s  %dx%dx%d  TC %.2f ms  %.2f TFLOPS   MMQ %.2f ms  %.2f TFLOPS   TC/MMQ %.2fx",
			s.name, tag, s.rows, s.T, s.cols, tc_ms, tc_tflops, mmq_ms, mmq_tflops, mmq_ms / tc_ms,
		)
	}
	return true
}

@(private)
bench_ms :: proc(fn: cu.Function, args: []rawptr, gx, gy, bx: u32) -> f64 {
	for _ in 0 ..< 2 {
		cu.launch(fn, gx, gy, 1, bx, 1, 1, 0, args)
	}
	cu.sync()
	iters := 5
	t0 := time.now()
	for _ in 0 ..< iters {
		cu.launch(fn, gx, gy, 1, bx, 1, 1, 0, args)
		cu.sync()
	}
	return f64(time.duration_milliseconds(time.since(t0))) / f64(iters)
}

Gemv_Shape :: struct {
	name: string,
	kind: ggml.GGML_Type,
	rows: int,
	cols: int,
}

@(private)
gemv_gbs :: proc(kind: ggml.GGML_Type, rows, cols: int, ms: f64) -> (bytes: f64, gbs: f64) {
	elems, bsz := ggml.ggml_block_info(kind)
	bytes = f64(rows) * f64(cols / elems) * f64(bsz)
	if ms > 0 {
		gbs = bytes / ms / 1e6
	}
	return
}

// cuda_gemv_bench times the scalar GEMV and the bandwidth GEMV on the
// T=1 shapes that dominate 9B and 27B decode. The q8 of x is done once,
// outside the timed loop, matching a cached activation.
cuda_gemv_bench :: proc() {
	if !s_open() {
		return
	}
	defer s_close()
	peak := 1134.0
	if s := os.get_env("ODINFER_CUDA_PEAK_GBS", context.temp_allocator); s != "" {
		if v, ok := strconv.parse_f64(s); ok && v > 0 {
			peak = v
		}
	}
	fmt.eprintfln("cuda_gemv: T=1  peak %.0f GB/s (ODINFER_CUDA_PEAK_GBS). GB/s counts the weight bytes only.", peak)
	fmt.eprintln("cuda_gemv: r4 is gy=1 of the unstaged kernel. dec is the production launch (Q6 staged when K>=16384 or rows>8192, split from the 80-SM rule). pair is one K sweep of two Q4 matrices versus two r4 launches.")
	shapes := [?]Gemv_Shape{
		{"9B w1", .Q4_K, 12288, 4096},
		{"9B wq", .Q4_K, 8192, 4096},
		{"9B wo", .Q4_K, 4096, 4096},
		{"9B w2", .Q6_K, 4096, 12288},
		{"9B out", .Q6_K, 248320, 4096},
		{"27B w1", .Q4_K, 17408, 5120},
		{"27B wq", .Q4_K, 12288, 5120},
		{"27B wo", .Q4_K, 5120, 6144},
		{"27B w2", .Q6_K, 5120, 17408},
		{"27B out", .Q6_K, 248320, 5120},
		{"q8_0", .Q8_0, 2048, 2048},
		{"q5_0", .Q5_0, 2048, 2048},
		{"q5_k", .Q5_K, 5120, 5120},
	}
	for sh in shapes {
		gemv_bench_one(sh, peak)
	}
}

@(private)
gemv_bench_one :: proc(sh: Gemv_Shape, peak: f64) {
	tag := cuda_kind_tag(sh.kind)
	old := s_fn(fmt.tprintf("gemv_%s", tag))
	nw := s_fn(fmt.tprintf("gemv2_%s", tag))
	qfn := s_fn("quant_q8")
	if old == nil || nw == nil || qfn == nil {
		fmt.eprintfln("  %s %s  kernel missing", sh.name, tag)
		return
	}
	elems, bsz := ggml.ggml_block_info(sh.kind)
	nbytes := sh.rows * (sh.cols / elems) * bsz
	raw := fill_weight(sh.kind, sh.rows, sh.cols)
	defer delete(raw)
	x := make([]f32, sh.cols)
	defer delete(x)
	for i in 0 ..< sh.cols {
		x[i] = (f32(i % 7) - 3) * 0.02
	}
	dw, e1 := cu.alloc(nbytes + 64)
	dx, e2 := cu.alloc(sh.cols * 4)
	dy, e3 := cu.alloc(sh.rows * 4)
	ng := sh.cols / 32
	dqs, e4 := cu.alloc(sh.cols)
	dsc, e5 := cu.alloc(ng * 4)
	dsm, e6 := cu.alloc(ng * 8)
	defer cu.free(dw)
	defer cu.free(dx)
	defer cu.free(dy)
	defer cu.free(dqs)
	defer cu.free(dsc)
	defer cu.free(dsm)
	if e1 != cu.OK || e2 != cu.OK || e3 != cu.OK || e4 != cu.OK || e5 != cu.OK || e6 != cu.OK {
		fmt.eprintfln("  %s %s  %dx%d  alloc failed (%.1f MiB weights)", sh.name, tag, sh.rows, sh.cols, f64(nbytes) / (1024.0 * 1024.0))
		return
	}
	cu.upload(dw, raw_data(raw), nbytes)
	cu.upload(dx, raw_data(x), sh.cols * 4)
	xx, qs, sc, sm := dx, dqs, dsc, dsm
	nn := i32(sh.cols)
	qargs := [?]rawptr{&xx, &qs, &sc, &sm, &nn}
	cu.launch(qfn, u32((ng + 127) / 128), 1, 1, 128, 1, 1, 0, qargs[:])
	cu.sync()
	wb, yy := dw, dy
	dd := i32(sh.rows)
	old_args := [?]rawptr{&wb, &xx, &yy, &nn, &dd}
	warm, iters := 2, 5
	if sh.rows > 100000 {
		warm, iters = 1, 3
	}
	old_ms := gemv_time(old, old_args[:], u32(sh.rows), 1, 32, warm, iters)
	warps := 8
	gx := u32((sh.rows + warps * 4 - 1) / (warps * 4))
	bx := u32(warps * 32)
	acc := i32(0)
	new_ms: f64
	if sh.kind == .Q4_K || sh.kind == .Q5_K {
		args := [?]rawptr{&wb, &qs, &sc, &sm, &yy, &nn, &dd, &acc}
		new_ms = gemv_time(nw, args[:], gx, 1, bx, warm, iters)
	} else {
		args := [?]rawptr{&wb, &qs, &sc, &yy, &nn, &dd, &acc}
		new_ms = gemv_time(nw, args[:], gx, 1, bx, warm, iters)
	}
	r1_ms := f64(-1)
	if sh.kind == .Q4_K || sh.kind == .Q6_K {
		rname := "gemv2r1_q4_k"
		if sh.kind == .Q6_K {
			rname = "gemv2r1_q6_k"
		}
		rfn := s_fn(rname)
		if rfn != nil {
			gx1 := u32((sh.rows + warps - 1) / warps)
			if sh.kind == .Q4_K {
				args := [?]rawptr{&wb, &qs, &sc, &sm, &yy, &nn, &dd, &acc}
				r1_ms = gemv_time(rfn, args[:], gx1, 1, bx, warm, iters)
			} else {
				args := [?]rawptr{&wb, &qs, &sc, &yy, &nn, &dd, &acc}
				r1_ms = gemv_time(rfn, args[:], gx1, 1, bx, warm, iters)
			}
		}
	}
	_, old_gbs := gemv_gbs(sh.kind, sh.rows, sh.cols, old_ms)
	bytes, new_gbs := gemv_gbs(sh.kind, sh.rows, sh.cols, new_ms)
	fmt.eprintfln("  %s %s  %dx%d  %.1f MiB", sh.name, tag, sh.rows, sh.cols, bytes / (1024.0 * 1024.0))
	fmt.eprintfln("    r4   %.3f ms  %.1f GB/s  %.1f%% of peak", new_ms, new_gbs, new_gbs / peak * 100)
	// gy=1 is the line above. Rows <= 8192 are the shapes whose production
	// launch splits K. s2/s4 zero the output and atomicAdd, which is what
	// decode runs. 27B wo was 35% and 27B w2 30% on the unsplit line.
	if (sh.kind == .Q4_K || sh.kind == .Q6_K) && sh.rows <= 8192 {
		splits := [?]u32{2, 4}
		for sp in splits {
			ms: f64
			if sh.kind == .Q4_K {
				args := [?]rawptr{&wb, &qs, &sc, &sm, &yy, &nn, &dd, &acc}
				ms = gemv_time_split(nw, args[:], gx, sp, bx, dy, sh.rows * 4, warm, iters)
			} else {
				args := [?]rawptr{&wb, &qs, &sc, &yy, &nn, &dd, &acc}
				ms = gemv_time_split(nw, args[:], gx, sp, bx, dy, sh.rows * 4, warm, iters)
			}
			if ms > 0 {
				_, gbs := gemv_gbs(sh.kind, sh.rows, sh.cols, ms)
				fmt.eprintfln("    s{}   %.3f ms  %.1f GB/s  %.1f%% of peak", sp, ms, gbs, gbs / peak * 100)
			}
		}
	}
	if sh.kind == .Q6_K {
		sfn := s_fn("gemv2_q6s_k")
		if sfn != nil {
			args := [?]rawptr{&wb, &qs, &sc, &yy, &nn, &dd, &acc}
			st_ms := gemv_time(sfn, args[:], gx, 1, bx, warm, iters)
			if st_ms > 0 {
				_, st_gbs := gemv_gbs(sh.kind, sh.rows, sh.cols, st_ms)
				fmt.eprintfln("    q6s  %.3f ms  %.1f GB/s  %.1f%% of peak", st_ms, st_gbs, st_gbs / peak * 100)
			}
			if sh.rows <= 8192 {
				st2 := gemv_time_split(sfn, args[:], gx, 2, bx, dy, sh.rows * 4, warm, iters)
				if st2 > 0 {
					_, st2_gbs := gemv_gbs(sh.kind, sh.rows, sh.cols, st2)
					fmt.eprintfln("    q6s2 %.3f ms  %.1f GB/s  %.1f%% of peak", st2, st2_gbs, st2_gbs / peak * 100)
				}
			}
		}
	}
	if sh.kind == .Q4_K {
		nfn := s_fn("gemv2_q4nc_k")
		xfn := s_fn("gemv2_q4x_k")
		if nfn != nil {
			args := [?]rawptr{&wb, &qs, &sc, &sm, &yy, &nn, &dd, &acc}
			nc_ms := gemv_time(nfn, args[:], gx, 1, bx, warm, iters)
			if nc_ms > 0 {
				_, nc_gbs := gemv_gbs(sh.kind, sh.rows, sh.cols, nc_ms)
				fmt.eprintfln("    nc   %.3f ms  %.1f GB/s  %.1f%% of peak", nc_ms, nc_gbs, nc_gbs / peak * 100)
			}
		}
		if xfn != nil {
			gx8 := u32((sh.rows + warps * 8 - 1) / (warps * 8))
			args := [?]rawptr{&wb, &qs, &sc, &sm, &yy, &nn, &dd, &acc}
			x_ms := gemv_time(xfn, args[:], gx8, 1, bx, warm, iters)
			if x_ms > 0 {
				_, x_gbs := gemv_gbs(sh.kind, sh.rows, sh.cols, x_ms)
				fmt.eprintfln("    r8   %.3f ms  %.1f GB/s  %.1f%% of peak", x_ms, x_gbs, x_gbs / peak * 100)
			}
		}
	}
	if r1_ms > 0 {
		_, r1_gbs := gemv_gbs(sh.kind, sh.rows, sh.cols, r1_ms)
		fmt.eprintfln("    r1   %.3f ms  %.1f GB/s  %.1f%% of peak", r1_ms, r1_gbs, r1_gbs / peak * 100)
	}
	dec_fn := nw
	dec_name := "r4"
	if sh.kind == .Q6_K && (sh.cols >= 16384 || sh.rows > 8192) {
		if sfn := s_fn("gemv2_q6s_k"); sfn != nil {
			dec_fn = sfn
			dec_name = "q6s"
		}
	}
	dec_sp := u32(1)
	if (sh.kind == .Q4_K || sh.kind == .Q6_K) && sh.rows > 0 && sh.rows <= 8192 {
		nb := sh.cols / 256
		if nb >= 4 {
			dec_sp = 4
			wn := (sh.rows + 3) / 4
			if nb / 2 >= 12 && wn < 80 * 32 {
				dec_sp = 2
			}
		} else if nb >= 2 {
			dec_sp = 2
		}
	}
	dec_ms: f64 = -1
	if dec_fn != nil {
		if sh.kind == .Q4_K || sh.kind == .Q5_K {
			args := [?]rawptr{&wb, &qs, &sc, &sm, &yy, &nn, &dd, &acc}
			if dec_sp > 1 {
				dec_ms = gemv_time_split(dec_fn, args[:], gx, dec_sp, bx, dy, sh.rows * 4, warm, iters)
			} else {
				dec_ms = gemv_time(dec_fn, args[:], gx, 1, bx, warm, iters)
			}
		} else {
			args := [?]rawptr{&wb, &qs, &sc, &yy, &nn, &dd, &acc}
			if dec_sp > 1 {
				dec_ms = gemv_time_split(dec_fn, args[:], gx, dec_sp, bx, dy, sh.rows * 4, warm, iters)
			} else {
				dec_ms = gemv_time(dec_fn, args[:], gx, 1, bx, warm, iters)
			}
		}
	}
	if dec_ms > 0 {
		_, dec_gbs := gemv_gbs(sh.kind, sh.rows, sh.cols, dec_ms)
		fmt.eprintfln("    dec  %.3f ms  %.1f GB/s  %.1f%% of peak  (%s gy={})", dec_ms, dec_gbs, dec_gbs / peak * 100, dec_name, dec_sp)
	}
	if sh.kind == .Q4_K && (sh.name == "9B w1" || sh.name == "27B w1") {
		pfn := s_fn("gemv2_q4_pair_k")
		dy2, ey2 := cu.alloc(sh.rows * 4)
		if pfn != nil && ey2 == cu.OK {
			two := gemv_time_twice(nw, wb, qs, sc, sm, yy, nn, dd, acc, gx, bx, warm, iters)
			w0 := wb
			y2 := dy2
			pargs := [?]rawptr{&w0, &wb, &qs, &sc, &sm, &yy, &y2, &nn, &dd}
			pms := gemv_time(pfn, pargs[:], gx, 1, bx, warm, iters)
			if two > 0 && pms > 0 {
				fmt.eprintfln("    pair %.3f ms   two r4 %.3f ms  (same weights twice, one x)", pms, two)
			}
		}
		if ey2 == cu.OK {
			cu.free(dy2)
		}
	}
	fmt.eprintfln("    old  %.3f ms  %.1f GB/s  %.1f%% of peak", old_ms, old_gbs, old_gbs / peak * 100)
}

@(private)
gemv_time :: proc(fn: cu.Function, args: []rawptr, gx, gy, bx: u32, warm, iters: int) -> f64 {
	for _ in 0 ..< warm {
		cu.launch(fn, gx, gy, 1, bx, 1, 1, 0, args)
	}
	cu.sync()
	t0 := time.now()
	for _ in 0 ..< iters {
		if cu.launch(fn, gx, gy, 1, bx, 1, 1, 0, args) != cu.OK {
			return -1
		}
		cu.sync()
	}
	return f64(time.duration_milliseconds(time.since(t0))) / f64(iters)
}

// Split-K atomicAdds into y, so each sample starts from zero. The memset is
// inside the timed region because decode pays it too.
@(private)
gemv_time_split :: proc(fn: cu.Function, args: []rawptr, gx, gy, bx: u32, y: cu.Device_Ptr, bytes, warm, iters: int) -> f64 {
	for _ in 0 ..< warm {
		cu.memset(y, 0, bytes)
		cu.launch(fn, gx, gy, 1, bx, 1, 1, 0, args)
	}
	cu.sync()
	t0 := time.now()
	for _ in 0 ..< iters {
		if cu.memset(y, 0, bytes) != cu.OK {
			return -1
		}
		if cu.launch(fn, gx, gy, 1, bx, 1, 1, 0, args) != cu.OK {
			return -1
		}
		cu.sync()
	}
	return f64(time.duration_milliseconds(time.since(t0))) / f64(iters)
}

// Two Q4 GEMVs back to back, the gate+up decode pair before fusion.
@(private)
gemv_time_twice :: proc(fn: cu.Function, wb, qs, sc, sm, yy: cu.Device_Ptr, nn, dd, acc: i32, gx, bx: u32, warm, iters: int) -> f64 {
	w := wb
	q, s, m, y := qs, sc, sm, yy
	n, d, a := nn, dd, acc
	args := [?]rawptr{&w, &q, &s, &m, &y, &n, &d, &a}
	for _ in 0 ..< warm {
		cu.launch(fn, gx, 1, 1, bx, 1, 1, 0, args[:])
		cu.launch(fn, gx, 1, 1, bx, 1, 1, 0, args[:])
	}
	cu.sync()
	t0 := time.now()
	for _ in 0 ..< iters {
		if cu.launch(fn, gx, 1, 1, bx, 1, 1, 0, args[:]) != cu.OK {
			return -1
		}
		if cu.launch(fn, gx, 1, 1, bx, 1, 1, 0, args[:]) != cu.OK {
			return -1
		}
		cu.sync()
	}
	return f64(time.duration_milliseconds(time.since(t0))) / f64(iters)
}

} // GPU_CUDA
