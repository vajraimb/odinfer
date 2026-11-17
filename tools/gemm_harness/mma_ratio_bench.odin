#+build darwin

/* Raw simdgroup-mma throughput microbenchmark: f16xf16->f32 vs
   s8xs8->s32, back-to-back in registers, no memory traffic. Go/no-go gate
   for an int8 MMQ path: if the s8 rate is not clearly above f16, the int8
   epilogue (per-k-block scale correction) would eat the gain.

   odin run tools/gemm_harness/mma_ratio_bench.odin -file -o:speed \
     -collection:ggml=packages
*/

package mma_ratio_bench

import "core:fmt"
import "core:os"
import "core:strings"
import "core:time"
import NS "core:sys/darwin/Foundation"
import MTL "vendor:darwin/Metal"

MSL :: `
#include <metal_stdlib>
using namespace metal;

kernel void mma_f16(device float *out [[buffer(0)]], constant uint &iters [[buffer(1)]],
    uint tg [[threadgroup_position_in_grid]], uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]) {
    threadgroup float red[16 * 64];
    simdgroup_half8x8 a0 = make_filled_simdgroup_matrix<half, 8>((half)1.5);
    simdgroup_half8x8 b0 = make_filled_simdgroup_matrix<half, 8>((half)1.0);
    simdgroup_float8x8 acc0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 acc1 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 acc2 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    simdgroup_float8x8 acc3 = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint i = 0; i < iters; ++i) {
        simdgroup_multiply_accumulate(acc0, a0, b0, acc0);
        simdgroup_multiply_accumulate(acc1, a0, b0, acc1);
        simdgroup_multiply_accumulate(acc2, a0, b0, acc2);
        simdgroup_multiply_accumulate(acc3, a0, b0, acc3);
    }
    simdgroup_store(acc0, red + sgitg * 8, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0 && red[sgitg * 8] + red[sgitg * 8 + 63] == 12345.678f) out[tg] = 1.0f; // never true
}

`

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

	src_ns := NS.String.alloc()->initWithOdinString(MSL)
	defer src_ns->release()
	opts := MTL.CompileOptions.alloc()->init()
	defer opts->release()
	opts->setFastMathEnabled(true)
	lib, err := dev->newLibraryWithSource(src_ns, opts)
	if err != nil {
		fmt.eprintfln("compile: {}", err->localizedDescription()->odinString())
		os.exit(1)
	}
	defer lib->release()

	ob := dev->newBufferWithLength(4096, MTL.ResourceStorageModeShared)
	defer ob->release()

	run2 :: proc(dev: ^MTL.Device, queue: ^MTL.CommandQueue, lib: ^MTL.Library, name: string, ob: ^MTL.Buffer, iters: int) -> f64 {
		ns := NS.String.alloc()->initWithOdinString(name)
		defer ns->release()
		fn_ := lib->newFunctionWithName(ns)
		if fn_ == nil {
			fmt.eprintfln("missing {}", name)
			os.exit(1)
		}
		defer fn_->release()
		pso, perr := dev->newComputePipelineStateWithFunction(fn_)
		if perr != nil {
			fmt.eprintfln("pso {}: {}", name, perr->localizedDescription()->odinString())
			os.exit(1)
		}
		defer pso->release()
		cmd := queue->commandBuffer()
		enc := cmd->computeCommandEncoder()
		enc->setComputePipelineState(pso)
		enc->setBuffer(ob, 0, 0)
		it := u32(iters)
		enc->setBytes(([^]u8)(&it)[:4], 1)
		enc->dispatchThreadgroups(MTL.Size{64, 1, 1}, MTL.Size{512, 1, 1})
		enc->endEncoding()
		t0 := time.tick_now()
		cmd->commit()
		cmd->waitUntilCompleted()
		return f64(time.tick_since(t0)) / 1e6
	}

	ITERS := 20000
	for _ in 0 ..< 2 {
		_ = run2(dev, queue, lib, "mma_f16", ob, 1000)
	}
	best_f16: f64 = 1e18
	for _ in 0 ..< 5 {
		ms := run2(dev, queue, lib, "mma_f16", ob, ITERS)
		if ms < best_f16 { best_f16 = ms }
	}
	// 64 TGs x 16 SGs x 4 accs x 128 MACs x ITERS, x2 for FLOP counting
	flop := 2.0 * 64.0 * 16.0 * 4.0 * 128.0 * f64(ITERS)
	fmt.eprintfln("f16 simdgroup mma: {:7.2f} ms   {:6.2f} TFLOPS (device peak reference)",
		best_f16, flop / (best_f16 * 1e-3) / 1e12)
	fmt.eprintln("s8 path: simdgroup_matrix<int8_t/int32_t,8,8> fails is_simdgroup_matrix_element")
	fmt.eprintln("          static_assert on this toolchain — Metal simdgroup mma is f16/f32 only.")
	fmt.eprintln("          No dp4a in MSL either -> int8 MMQ is NOT available on Apple/Metal.")
}
