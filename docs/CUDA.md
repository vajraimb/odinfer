# CUDA backend

[中文](CUDA.zh.md)

Linux x86_64 backend for the Qwen3.5 engine. It targets **sm_70 and newer**
(validated for Tesla V100S, compute capability 7.0, driver CUDA 13.0).
Accumulation is fp32. There is no bf16. Prefill GEMM (`T>1`) can use Volta
tensor cores (WMMA 16×16×16, fp16 operands, fp32 accumulate). Decode (`T==1`)
uses a bandwidth GEMV: 128-bit loads, `dp4a`, and the activation quantized to
q8 once per vector. Short Q4_K / Q6_K matrices use one row per warp; taller
ones keep four. `ODINFER_CUDA_GEMV=old` restores the scalar kernel. Dense
decode stays eager unless `ODINFER_CUDA_GRAPH=1`.

macOS builds are unchanged: the default backend is still Metal. A single
binary never contains both hot paths.

## Build

Odin compile does not need `nvcc`, the CUDA toolkit headers, or a GPU.
`libcuda.so.1` and NVRTC are `dlopen`'d at runtime.

```sh
# CPU Linux build (no Metal frameworks). Same entry point as macOS.
./build.sh

# CUDA engine
odin build server/serve_harness/serve.odin -file -out:bin/odin-serve \
  -o:speed -no-bounds-check -disable-assert \
  -define:ODINFER_BACKEND="cuda" \
  -collection:ggml=packages -collection:infer=packages \
  -collection:qwen3_5=packages -collection:qwen3_5_tokenizer=packages \
  -collection:sampler=packages -collection:tokenizer=packages \
  -collection:cuda=packages

bin/odin-serve model.gguf --cuda --max-ctx 2048
# --metal is accepted as the same "use the GPU" switch on a CUDA build.
```

`-define:ODINFER_BACKEND="cuda"` must be quoted. The string is compared to
`"cuda"`; the bare word `default` (the unset value) keeps Metal on Darwin and
CPU everywhere else.

The define is read in `packages/qwen3_5/backend.odin` (`GPU_METAL` /
`GPU_CUDA`). Forward, KV save/load, and reset are `when` branches, so the
compiled hot path has no Metal-vs-CUDA test.

## NVRTC

`ODINFER_NVRTC_PATH` is an exclusive override: a full path to
`libnvrtc.so.12` (or `.13`), or a directory that contains it. When it is
set, nothing else is searched. `NVRTC_DIR` is searched first when the
override is unset, and the search then continues.

Otherwise the loader tries versioned sonames before the bare name
(`libnvrtc.so.13`, then `.12`, then `libnvrtc.so`), then a pip
`nvidia-cuda-nvrtc-cu12` tree
(`~/.local/lib/python3.*/site-packages/nvidia/cuda_nvrtc/lib`). The matching
`libnvrtc-builtins.so.<major>.<minor>` (the wheel ships
`libnvrtc-builtins.so.12.9`, not `.so.12`) is preloaded with `RTLD_GLOBAL`.
The `.alt.` copy is skipped.

CUDA 13 NVRTC cannot target `compute_70`. Archs below sm_75 were removed,
and the bare `libnvrtc.so` on a CUDA 13 machine is that 13.x library. A
13.x driver can still JIT PTX produced by NVRTC 12.9. When the device is
sm_70 / sm_72, or a `compute_70` compile fails on a 13.x library, the
runtime loads NVRTC 12.x only and prints the pip path if none is found.
NVRTC 12.9 warns that archs before sm_75 are deprecated. That warning is
expected on a V100.

```sh
python3 -m pip install --user nvidia-cuda-nvrtc-cu12
export ODINFER_NVRTC_PATH="$HOME/.local/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12"
```

Every compile passes `--gpu-architecture=compute_XX` from the device.
`CUDA_DEVICE` selects the ordinal (default 0).

## Tensor cores

Prefill and other batched GEMMs (`Y[T, rows] = X[T, cols] · W[rows, cols]`)
dequantize a 64×32 fp16 tile of `W` into shared memory, narrow `X` to fp16,
and issue `wmma.mma.sync.aligned.m16n16k16` (fp16 in, fp32 accumulate). The
PTX is inline. NVRTC does not ship `mma.h`, and `mma.sync.m16n16k16` is not
the Volta instruction; the WMMA form is what sm_70 JITs. `ldmatrix` is not
used.

The quant kinds with a tensor-core kernel are the same set as MMQ: F32, F16,
Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K. The 9B
Q4_K_M prefill is Q4_K (up/gate/attention) and Q6_K (down). The 0.8B MoE
projections that are batched use Q4_K, Q8_0, and Q5_0. Expert GEMVs stay on
the GEMV kernels.

Selection, read at `cuda_init`:

| `ODINFER_CUDA_TC` | Behaviour |
|-------------------|-----------|
| unset | tensor cores when compute capability ≥ 7.0 and `T ≥ 16` |
| `1` | tensor cores for every `T > 1` (still GEMV at `T == 1`) |
| `0` | MMQ for `T > 1`, GEMV for decode |

`T == 1` is always GEMV. A missing tensor-core kernel falls back to MMQ.

Correctness (`max_abs`, the tolerance at that element, `atol`, `rtol`) and
the microbenchmark:

```sh
# build is the same command as bin/cuda_kernel_test in the check script
bin/cuda_kernel_test
bin/cuda_kernel_test --bench
```

`cuda_selftest` compares tensor-core GEMM to the CPU fp32 dot for F32, F16,
Q8_0, Q4_0, Q5_0, Q4_K, Q6_K, Q5_K, Q8_K at `T=32`, `rows=64`, `cols=256`,
plus partial tiles (`T=20`, `rows=40`) for Q4_K, Q8_0, and Q5_0. fp16
rounding uses `atol=2e-2 rtol=1e-2` for F32/F16 and `atol=5e-2 rtol=5e-2`
for the quantized kinds. A wrong tile mapping is far outside that.

`--bench` prints milliseconds and TFLOPS (2·T·rows·cols) for tensor cores
and MMQ:

| Name | Kind | rows × T × cols |
|------|------|-----------------|
| 9B w1/w3 | Q4_K | 12288 × 64 × 4096 |
| 9B wq | Q4_K | 8192 × 64 × 4096 |
| 9B wo | Q4_K | 4096 × 64 × 4096 |
| 9B w2 | Q6_K | 4096 × 64 × 12288 |
| 0.8B Q8_0 | Q8_0 | 1024 × 32 × 1024 |
| 0.8B Q5_0 | Q5_0 | 1024 × 32 × 1024 |

End-to-end prefill, same prompt, tensor cores on and off. The smoke line
`prefill N tok` is the number that has to be at least 16 for the default
(auto) policy; `ODINFER_CUDA_TC=1` uses tensor cores for any `N > 1`.

```sh
ODINFER_CUDA_TC=0 bin/cuda_smoke "$MODEL" 8 2048
bin/cuda_smoke "$MODEL" 8 2048
ODINFER_CUDA_TC=1 bin/cuda_smoke "$MODEL" 8 2048
```

Decode tok/s should match across those three. The init line says
`tensor cores auto (T>=16)`, `off`, or `on for T>1`.

## Decode GEMV

`T==1` is a memory-bound GEMV. The scalar kernel (one warp per row, byte
loads) is the `ODINFER_CUDA_GEMV=old` fallback. The bandwidth kernel has two
shapes. Four rows per warp, eight threads per row, is the tall-matrix path
(FFN up, the 27B Q projection, the logits row). Those eight threads issue
one `ld.global.v4` each, so a row's quant payload is a single coalesced
128-byte read, and the dot is `dp4a.s32.s32`. Q6_K blocks are 210 bytes, so
the 16-byte load is a shift of two aligned windows; the weight allocation
is padded by 64 bytes and the load never emits a misaligned `v4` on sm_70.
Q6 codes are unpacked per byte. There is no `vsub4` in the PTX: NVRTC emits
`vsub4.s32.s32.s32`, and the sm_70 JIT rejects that opcode, which rejects
the whole module. Scales stay in registers.

Four rows per warp is the default for every Q4_K and Q6_K shape. One row
per warp was slower on the V100S at every size that was timed (9B decode
33 tok/s versus 62 for four rows; bench r1 was 8% of peak where r4 was
39%). `ODINFER_CUDA_GEMV_ROWS=1` still selects it. Total warps of one K range are
`rows / 4`. Short matrices split that K range across `gridDim.y` instead of
launching a second kernel.

Q4_K and Q6_K rows of 8192 or fewer split K across four blocks (`ODINFER_CUDA_GEMV_SPLIT=0` leaves them in one block). The partials are `atomicAdd`s, so the output is zeroed first unless the GEMV is accumulating into the residual.

Q4_K and Q6_K prefill with K a multiple of 256 uses a tensor-core tile only where it won. For `T<=64` that is the 32-token tile (profile name `tc2`): at T=19 on the V100S it was 176 ms, and the 128-token tile was 323 ms. For `T>64` the default is the scalar-dequant tile (profile name `tc`). On `f229ac0` that kernel was about 70–75% of long-prefill time. The 128-token tile (`tc2w`) was slower there too: 9B at 581 tokens 124.5 vs 138.9 tok/s, at 2000 tokens 71.9 vs 76.7; 27B at 581 tokens 50.9 vs 53.1. `ODINFER_CUDA_TC2_WIDE=1` selects that tile. `ODINFER_CUDA_TC2_BLOCK=1` selects `tc2m` for `T>64` only. On the V100S that tile was slower (9B at 581 tokens 151 vs 208 tok/s, at 2000 tokens 150 vs 203; 27B at 581 tokens 52 vs 71) and it did not match the scalar tile's tokens. Leave it off. `ODINFER_CUDA_TC2_REUSE=1` selects `tc2r` for `T>64`: the scalar tile's own dequant and WMMA, applied to both 64-token halves of a 128-token chunk so each weight is dequantized once. `T<=64` stays `tc2`. It has not been timed on a V100. `cuda_kernel_test` requires `tc2r` to match `tc` bit for bit on partial tiles; that is not a claim about sampled text. `ODINFER_CUDA_TC2=0` is the scalar tile at every T.

Q4_K and Q6_K prefill with `T>1` and K a multiple of 256 uses `mmqk` by default (profile names `mmqk_q` and `mmqk`). `pgemm` (a 64×64 tile, BK=128, fp16 dequant into shared memory, then WMMA) was correct on a V100 and slower, so it is not launched. 9B prefill tok/s at 905/1805/8000/16000 were 173.75/177.03/168.22/153.78 against 221.70/220.46/203.90/183.09 with `ODINFER_CUDA_PREFILL_GEMM=0`. The 27B at 905/8000/16000 was 63.79/62.45/56.85 against 74.19/69.52/62.71. A 9B chunk at position 7808 spent 584 ms in the linear bucket against 450 ms, and a 27B 128-token chunk spent about 1636 ms against 1423 ms. That kernel writes 32 fp16 rows at one K with a 256-byte stride, a 32-way shared-memory bank conflict, and a 64-token tile dequantizes every weight twice in a 128-token chunk. `mmqk` quantizes the activation tile to q8 once (`quant_q8_batch`, the same `amax/127` groups and 16-float sums as `quant_q8`) and dots the stored nibbles with `dp4a.s32.s32`. X is K-major, so one warp's 32 tokens are a coalesced uint32. The timed kernel gave each warp 4 rows and 64 tokens (the lane and the lane 32 later) and reloaded X from device memory for every warp. `ODINFER_CUDA_PREFILL_GEMM=0` keeps `tc2` for `T<=64` and the scalar `tc` tile for `T>64`. Other quant kinds stay on `tc` or MMQ. The token embedding is a lookup and the logits row is still a `T==1` GEMV. On a V100 with chunk 512, `mmqk` beat `ODINFER_CUDA_PREFILL_GEMM=0`. 9B prefill tok/s at 905/1805/8000/16000 were 303.23/305.95/279.30/245.95 against 253.87/256.53/237.89/212.92. The 27B at 905/8000/16000 was 100.85/93.45/82.70 against 82.97/79.63/71.90. `PHASE=1` linear time at 8000 tokens was 18378 ms on the 9B (23341 with the flag at 0) and 61546 ms on the 27B. A 14.5k request prefilled in 173.2 s (84.1 tok/s). Those numbers are the 4-row tile, which reloaded each q8 byte from device memory every 4 rows. Staging one superblock and reusing it across 32 rows (4 per warp, 22528 bytes) was faster on that V100: 9B at 905/1805/8000/16000 was 371.70/376.91/337.20/290.49 tok/s, the 27B at 905/8000/16000 was 124.00/112.97/97.43, and `PHASE=1` linear time at 8000 was 13425 ms on the 9B and 46627 ms on the 27B. A 14.5k request prefilled in 146.4 s (99.4 tok/s). The kernel in this tree is not timed. Q4 is 8 rows per warp (64 rows per block): sm_70 `launch_bounds(256, 2)` is 128 registers and no spills, so two blocks still fit. Q6 at 8 rows spilled 36 bytes stored and 68 loaded, so Q6 stays at 4 rows per warp. Gate+up and K+V share that staged q8 (`ODINFER_CUDA_PREFILL_FUSE=0` launches them separately). A later matmul of the same activation skips `quant_q8_batch`. KV dequant reads each Q8 scale once per 32 values. Weights are broadcast inside the warp, so staging them does not cut device traffic. A 128-token tile needs about 45 KiB and only one block fits in the 48 KiB default, so it is not the default. Greedy text matched, including `ODINFER_CUDA_GEMV_SPLIT=0`. `cuda_kernel_test` requires `mmqk` to match `gemv2` within `2e-3 + 2e-3*|ref|` (Q6 max abs on that machine was 7.6e-6). The fp32 CPU compare uses the same q8 allowance as `gemv2`, atol 1 and rtol 5e-2. A 2e-2 bound failed four Q6 shapes by 0.05-0.16 on a few elements even though those kernels matched `gemv2`. The shapes include a partial tile, K of 256, 512, and 1024, and a token count that is not a multiple of 64. The `pgemm` kernels stay in that test at `5e-2 + 5e-2*|ref|`. `ODINFER_CUDA_PREFILL_CHUNK` sets the prefill chunk (default 512; a multiple of 16 from 16 to 512). On the scalar tile, 9B at 16000 tokens was 183.09/206.62/213.01 tok/s for chunks 128/256/512. A size that does not fit falls back through 256, 128, 32, 8, and 1, and does not grow past the request. The WMMA attention workspace is sized for the chunk that was allocated.

Full attention in prefill is quadratic. Query t scores and mixes `t+1` keys. The one-block kernel uses one block per head, so a 16-head layer fills 16 of the V100's 80 SMs and its V loop walks every position on each thread. On the V100S that loop is almost the entire long-context decode cost (27B about 21.6 µs per context token per decode token at `9448e51`, linear/kv/norm/delta/other flat). For the default mixed-V cache, decode (`T==1`) with `npos>=128` uses `attention_q8v_part` plus `attention_q8v_reduce`: 32-token parts, one block per KV head, the query heads in the group share the staged K/V, and a second launch merges the partial softmaxes. Prefill (`T>1`) uses `attention_q8v_wmma` when head_dim is 256 on sm_70 or newer. `ODINFER_CUDA_PREFILL_FLASH=0` keeps `attention_q8v`. `ODINFER_CUDA_PREFILL_FLASH=1` is `attention_q8v_flash`: query tiles of `br` tokens (largest of 4, 3, 2, 1 that fits in 48 KB; 2 on the 27B, where a group is 6 query heads; 3 on the 9B, where a group is 4), 32-token KV tiles, online fp32 softmax, fp32 warp dots. On the 9B V100S that kernel was slower than `attention_q8v` (prefill tok/s at 905 / 1805 / 8000 / 16000: 201.8 vs 213.3, 186.7 vs 203.9, 120.0 vs 153.2, 80.7 vs 114.2). A T=128 chunk at position 15872 spent 2074 ms in attention versus 1147 ms. `=2` is `attention_q8v_wmma`. One block owns 16 query tokens and one KV head. It dequantizes each 32-token Q8_0 K tile and mixed-V tile into fp16 shared memory once and reuses that tile for every query head in the group (16 tokens times 4 heads is 64 queries on the 9B; times 6 is 96 on the 27B). `S = Q K^T` and `O += P V` are WMMA 16×16×16 fp16 with fp32 accumulate. Softmax is online in fp32. The KV range is split on `blockIdx.z` so the grid covers the 80 SMs; `attention_q8v_wmma_reduce` merges the partial max, sum, and output. Each K/V byte is loaded once per 16-token query tile. Causal masking is applied to any score whose key is past that query, which is the diagonal tile and any partial tail. head_dim must be 256. On the V100 it is the default for that shape: 9B prefill tok/s at 905/1805/8000/16000/30000 were 221.3/219.9/203.2/182.8/154.7 against 213.3/203.9/153.2/114.2 for `attention_q8v` (30000 timed out on the old kernel), and the 27B at 905/8000/16000 was 74.0/69.3/62.6 against 71.4/53.7/40.9. Greedy text matched. `=0` is `attention_q8v`. `=2` forces the same WMMA kernel. The per-token split inside a T=128 chunk stays off (it was about 10% slower on the V100S: 27B 905 tokens 63.6 vs 71.7 tok/s). Shorter decode, f16 KV, and symmetric Q8 stay on it too. `ODINFER_CUDA_ATTN_SPLIT=0` forces that kernel. A short chunk still stores every token's KV and launches once with a grid of heads by tokens (`ODINFER_CUDA_ATTN_BATCH=0` restores the per-token launch). The flops stay O(N^2); a 2000-token prompt does not become a linear GEMM. `delta_recurrent_chunk` is linear in T. Token t needs the gated state from t-1, so the loop stays inside the block, one block per value head, one launch per linear layer. It is not quadratic, and a parallel scan is a different algorithm. When `head_k_dim * head_v_dim * 4` is at most 64 KiB the chunk keeps that matrix in shared memory (`delta_sm`) and writes it back at the end. Qwen3.5's 128×128 state is 64 KiB, past the 48 KiB dynamic default, so init calls `cuFuncSetAttribute`; if that fails the global-memory kernel is used. The recurrence math is unchanged. Decode (`T==1`) still launches `delta_recurrent` once. Decode also folds the q8 cache into the attention gate, SwiGLU, and gated RMSNorm, so those GEMVs skip `quant_q8`.

`--max-ctx` is `min(requested, GGUF context_length)`. Qwen3.5-27B's native length is 262144. Nothing else in the CUDA path caps the window at 8192. KV is Q8_0 K (34 B per 32 elements) plus mixed V (40 B). For the 27B (16 full layers, 4 KV heads, head_dim 256, kv_dim 1024) that is 37888 bytes per token: 296 MiB at 8192, 1.16 GiB at 32768, 9.25 GiB at 262144. The score buffer is one row of `n_heads * seq_len` floats per query token in the launch, capped at 256 MiB, so a long context uses fewer tokens per attention launch instead of refusing to start. Split partials are `ceil(seq/32) * n_heads * head_dim` floats (24 MiB at 32768, 192 MiB at 262144). Shared memory in the split kernel is one 32-token tile, about 25 KB on the 27B, not the whole sequence. RoPE is computed from the position; there is no position table. The `.oikv` header stores `seq_len` as u32 and must match `--max-ctx`. Kernel `pos`/`seq_len` are 32-bit; split `grid.y` is at most 65535 parts (sequence 2097120). A 27B Q4 at ~16 GiB plus this scratch fits the native 262144 on a 32 GiB V100S with about 6 GiB left. The 9B (8 full layers, same kv_dim) is half that per token, 4.6 GiB of KV at 262144. Init prints `cuda: ctx=… kv=…`. A prefix snapshot copies the first `n_pos` Q8 positions of each full layer (the bytes are position-major) and zeros the tail on restore, so a short system-boundary hit does not move the whole max-ctx allocation. At 262144 the snapshot is still about 10 GiB of host RAM when `n_pos` is the full window. Prefill attention is still quadratic in flops. On `attention_q8v` the measured 27B T=128 chunk at position 1664 cost 324 ms of attention and 1423 ms of linear layers, so a 30k prefill on that kernel is on the order of 11 minutes of attention and 6 minutes of linear work, and attention is the larger term past roughly 7k tokens. The fp32 flash kernel is not the default. On the 9B it reloaded each KV byte about 43 times per 128-token chunk and lost to `attention_q8v`, which already costs about 72 µs per KV position (1147 ms at position 15872). `attention_q8v_wmma` (the default when head_dim is 256 on sm_70+; `ODINFER_CUDA_PREFILL_FLASH=0` is the fallback) loads each byte once per 16-token query tile, eight times across that chunk, and does the products on tensor cores. A napkin for one T=128 chunk on the 9B at position 16000, 8 full layers: `2 * 2 * 8 * 128 * 16 * 16000 * 256` = 268 GFLOP. V100 fp16 tensor peak is about 125 TFLOPS, so every SM busy is about 2.1 ms. KV traffic is about `16000 * 4 * (272 + 320)` = 38 MB per layer, times 8 query tiles, times 8 layers, about 2.4 GB, about 3 ms at 900 GB/s. Softmax and the partial-output writes sit on top. Ten to twenty-five percent of peak plus that traffic is about 15–40 ms, centered near 25 ms, against the measured 1147 ms. That is an estimate, not a V100 timing.

On a dense `T==1` step the attention out, the linear out, and the FFN down atomic-add straight into the residual, and the next RMSNorm is its own launch (`rmsnorm_q8`, 1.41 ms on the 9B, 3.53 ms on the 27B). `ODINFER_CUDA_FUSE_NORM=1` writes that GEMV into a side buffer and runs `residual_rmsnorm_q8` instead. On the V100S that was slower (9B 1.99 ms across 64 launches, 27B 4.96 ms) and decode dropped from 68.4 / 26.05 tok/s to 63.6 / 23.4, so it stays off. The 9B Q4 GEMVs read about 3.0 GB of weights (8 full layers, 24 linear layers, 32 MLPs of w1 and w3). 6.01 ms is about 500 GB/s, 44% of the 1134 GB/s peak. Counting 4.6 GB instead would be 765 GB/s. The default Q4 load is a plain `ld.global` (`*(const uint4*)`), the same as 8975985. `ld.global.nc` was slower (9B q4 6.01 → 6.50 ms, 27B 19.57 → 22.4) and is `ODINFER_CUDA_Q4_NC=1` only. `ODINFER_CUDA_Q4_ILP=1` selects `gemv2_q4x`: 8 rows per warp, coherent 128-bit loads, two super-blocks per iteration. `--bench-gemv` prints that line as `r8` next to `r4` and `nc`. It is not the decode default.

Q6 GEMV stages a block through aligned loads only when K is at least 16384 or the matrix is taller than 8192 rows (the 27B down projection and the logits). The 9B down projection (4096×12288) stays on the direct load. `ODINFER_CUDA_Q6_STAGE=1` stages every Q6 GEMV; `=0` stages none. The staged profile name is `gemv2_q6s`.

Q8_0 and Q5_0 matrices under 16 MiB stay on the scalar kernel. At
2048×2048 the new kernel was slower (Q8_0 118.5 vs 121.8 GB/s, Q5_0 49.2
vs 69.7). The 9B and 27B hot weights are Q4_K and Q6_K, so they are not
on this path. `ODINFER_CUDA_GEMV=new` keeps the new kernel for those small
shapes too.

The activation is quantized to q8 (groups of 32, signed int8, scale
`amax/127`) once per vector. Q4_K and Q5_K also keep the exact sum of each
16-float half so the `dmin` term is not quantized. A one-row RMSNorm writes
that q8 in the same kernel; the Q, K, and V projections, and the two FFN
up projections, reuse it. SwiGLU and a multi-row norm drop the cache.
Kinds without a bandwidth kernel (F32, Q2_K, Q3_K, Q8_K, the Q4_0 family)
stay on the scalar GEMV. So does a K that is not a multiple of the quant
block.

Block size is `ODINFER_CUDA_GEMV_WARPS` warps (default 8, range 1..16).
With four rows per warp that is 32 rows per block. With one row per warp
it is one row per warp in the block. V100S has 80 SMs and 64 warp slots
per SM.

On a dense `T==1` step the attention out, the FFN down, and the linear
out add straight into the residual. `ODINFER_CUDA_FUSE_NORM=1` sends them
through `residual_rmsnorm_q8` instead (see above). Prefill (`T>1`) and the
scalar fallback still add with their own kernel. The logits GEMV does not
accumulate.

| Variable | Default | Effect |
|----------|---------|--------|
| `ODINFER_CUDA_GEMV` | auto | `old` or `0`: scalar GEMV. `new`: bandwidth kernel even for small Q5_0 / Q8_0. Unset: bandwidth, except Q5_0 / Q8_0 under 16 MiB |
| `ODINFER_CUDA_GEMV_ROWS` | `4` | Rows per warp for Q4_K and Q6_K. Unset is four. `1` selects the slower one-row kernel |
| `ODINFER_CUDA_GEMV_WARPS` | `8` | Warps per bandwidth-GEMV block (1..16) |
| `ODINFER_CUDA_GEMV_SPLIT` | on | Q4_K / Q6_K with `rows≤8192` split K four ways. `=0` is one K range |
| `ODINFER_CUDA_TC2` | on | Q4_K / Q6_K prefill. `T<=64` uses the 32-token fp16 tile. `T>64` uses the scalar-dequant tile. `=0` is the scalar tile at every T |
| `ODINFER_CUDA_TC2_WIDE` | off | `=1` uses the 128-token tile for `T>64`. It was slower than the scalar tile at 581 and 2000 tokens |
| `ODINFER_CUDA_TC2_BLOCK` | off | `=1` uses `tc2m`. Slower on the V100S (about −26% at 581 and 2000 tokens) and it changed a sampled token. Leave off |
| `ODINFER_CUDA_TC2_REUSE` | off | `=1` uses `tc2r` for `T>64`: one scalar weight tile covers 128 tokens. Not timed. `T<=64` stays the 32-token tile. Wins over `TC2_BLOCK` and `TC2_WIDE` if several are set |
| `ODINFER_CUDA_PREFILL_GEMM` | on | Q4_K / Q6_K prefill (`T>1`, K multiple of 256) uses `mmqk`. The 32-row tile, chunk 512, on a V100: 9B at 905/1805/8000/16000 was 371.70/376.91/337.20/290.49 tok/s, 27B at 905/8000/16000 was 124.00/112.97/97.43. A 64-row Q4 tile was slower (9B 364.44/367.80/333.11/290.06) and is not the default. `=0` keeps `tc2` / scalar `tc`. Tolerance versus `gemv2` is `2e-3 + 2e-3*|ref|`. The fp32 CPU compare uses atol 1 and rtol 5e-2 |
| `ODINFER_CUDA_PREFILL_FUSE` | off | `=1` makes gate+up and K+V share one q8 stage. On the 64-row build it was neutral (9B at 8000 tokens, 332 with it off vs 333 on) and the linear bucket was slower, so it stays off |
| `ODINFER_CUDA_MMQK_ROWS` | `4` | Default Q4 tile is 32 rows per block. `=8` selects 64 rows, which was slower on a V100. Q6 stays at 32 either way |
| `ODINFER_CUDA_ATTN_LONG` | on | Mixed-V decode with `npos>=2048`, head_dim 256, and 4 or 6 query heads per KV head. 128 positions per block, K dequantized once per group, V once per position, then the same reduce as the 32-wide split. `=0` keeps 32-token parts at every length. Not bit-exact: same `2e-2 + 1e-3*|ref|` allowance |
| `ODINFER_CUDA_DELTA_WMMA` | off | `=1` runs a 16-token gated-delta chunk (WY solve in fp32, intra-chunk state products on WMMA fp16) for 128×128 prefill state. Default stays the recurrent kernel, which is the text-matched path. `cuda_selftest` allows 2e-3 against that recurrence |
| `ODINFER_CUDA_PREFILL_CHUNK` | `512` | Prefill tokens per forward. A multiple of 16 from 16 to 512. On the scalar tile, 9B at 16000 tokens was 183/207/213 tok/s for 128/256/512. A size that does not fit falls back through 256, 128, 32, 8, 1, and does not grow past the request |
| `ODINFER_CUDA_ATTN_BATCH` | on | Prefill full attention is one launch per chunk. `=0` is one launch per token |
| `ODINFER_CUDA_ATTN_SPLIT` | on | Mixed-V decode (`T==1`) with `npos>=128`. 32-token parts, one block per KV head, then a reduce. GQA query heads in a group share that K/V. Does not select the prefill kernel. `=0` is the one-block kernel for decode. Not bit-exact: `cuda_kernel_test` allows `2e-2 + 1e-3*|ref|` against `attention_q8v`, including npos 300, 1800, 8192, and 32768. f16 and symmetric Q8 stay on the old kernel. Turns the CUDA graph off |
| `ODINFER_CUDA_PREFILL_FLASH` | WMMA | Mixed-V prefill (`T>1`). Unset on sm_70+ with head_dim 256 is `attention_q8v_wmma`. `=0` is `attention_q8v`. `=1` is `attention_q8v_flash` (fp32 online softmax, `br` query tokens, 32-token KV tiles). On the 9B V100S that was slower (16000 tokens 80.7 vs 114.2 tok/s). `=2` forces `attention_q8v_wmma`: 16 query tokens, 32-token fp16 K/V tiles, WMMA 16×16×16 fp16 with fp32 accumulate, online fp32 softmax, one KV load per query tile, KV split across blocks so 80 SMs are busy. head_dim 256. Not bit-exact. The fp32 flash allows `2e-2 + 1e-3*|ref|`. The WMMA kernel allows `1e-1 + 5e-2*|ref|` (fp16 products) at kv_mul 4, npos 128, 300, 1800, 8192, 32768, and at kv_mul 6 for npos 128 and 1800. Greedy text on the smoke prompts is the acceptance check. Decode stays on the split kernels |
| `ODINFER_CUDA_FUSE_NORM` | off | `=1` adds the projection in `residual_rmsnorm_q8` and skips the next RMSNorm. Slower than the accum GEMV on the V100S (9B 1.99 vs 1.41 ms) |
| `ODINFER_CUDA_Q4_NC` | off | `=1` loads Q4 weights and the q8 vector with `ld.global.nc`. Slower on the V100S (9B 6.01 → 6.50 ms) |
| `ODINFER_CUDA_Q4_ILP` | off | `=1` uses `gemv2_q4x`: 8 rows/warp, coherent 128-bit loads, two super-blocks per iteration. `--bench-gemv` prints it as `r8` |
| `ODINFER_CUDA_Q6_STAGE` | auto | `=1` stages every Q6 GEMV through aligned loads. `=0` never stages. Unset stages only when K≥16384 or rows>8192 |
| `ODINFER_CUDA_SMOKE_PREFILL` | unset | `cuda_smoke` only. `=N` prefills N tokens |
| `ODINFER_CUDA_LAUNCHES` | off | `=1` prints the kernel launch count for each forward |
| `ODINFER_CUDA_PEAK_GBS` | `1134` | Peak used by `--bench-gemv` to print % of HBM |
| `ODINFER_CUDA_GRAPH` | off | `=1` captures dense `T==1` as one CUDA graph. The position is an async copy on the graph stream. MoE and prefill stay eager. `ODINFER_CUDA_PROFILE=1` turns the graph off |
| `ODINFER_CUDA_PROFILE` | off | `=1` prints per-kernel GPU time for that forward, slowest first. The 8192-event ring is folded into name totals as it fills, so a long prefill is not truncated |
| `ODINFER_CUDA_PHASE` | off | `=1` prints one line per forward: `linear attn kv norm delta other` milliseconds, from the same events as `PROFILE`, plus one `cuda: phase prefill` total for the whole prefill. Tok/s in that process is not the production number. Does not change kernels |
| `ODINFER_PHASE` | off | `=1` prints CPU milliseconds for the JSON grammar mask and the tool-call grammar mask on each sample. Zero when no grammar is installed |

`cuda_selftest` also runs `residual_rmsnorm_q8` against a CPU residual add plus RMSNorm. Q4_K also runs `gemv2_q4nc_k` and `gemv2_q4x_k` (8 rows/warp, including split-K when K has at least two blocks). It runs the four-row kernel against the CPU fp32 dot for Q4_K,
Q6_K, Q8_0, Q5_0, and Q5_K at 64×512 and at 6×256 (the tail that is not a
multiple of four rows). Q4_K and Q6_K also run the one-row kernel, including
`accumulate=1` (output starts at 0.25, expect `ref+0.25`). Q6 also runs the staged kernel. The 64×512 cases also launch split-K with `gridDim.y=2`. `tc2s_*` (32-token tile) and `tc2_*` (128-token tile) run at 64×256×32 and at a partial tile; the wide tile also runs at T=80 and T=160. `delta_recurrent_chunk` runs two tokens and two value heads against `delta_recurrent_step`. Allowance is
`1 + 0.05*|ref|`. The gap against the CPU is the q8 of `x`; a swapped
nibble or scale is on the order of `|ref|`.

```sh
bin/cuda_kernel_test
bin/cuda_kernel_test --bench-gemv
ODINFER_CUDA_GEMV_WARPS=4 bin/cuda_kernel_test --bench-gemv
```

`--bench` still prints the tensor-core TFLOPS table, then the same GEMV
table. `--bench-gemv` prints only the GEMV table: milliseconds, GB/s, and
percent of peak, for the four-row kernel (`r4`), the one-row kernel where
it exists (`r1`), and the scalar kernel (`old`).

| Name | Kind | rows × cols | Where it shows up |
|------|------|-------------|-------------------|
| 9B w1 | Q4_K | 12288 × 4096 | FFN up / gate |
| 9B wq | Q4_K | 8192 × 4096 | Q projection |
| 9B wo | Q4_K | 4096 × 4096 | output projection of attention |
| 9B w2 | Q6_K | 4096 × 12288 | FFN down |
| 9B out | Q6_K | 248320 × 4096 | logits |
| 27B w1 | Q4_K | 17408 × 5120 | FFN up / gate |
| 27B wq | Q4_K | 12288 × 5120 | Q projection |
| 27B wo | Q4_K | 5120 × 6144 | attention out |
| 27B w2 | Q6_K | 5120 × 17408 | FFN down |
| 27B out | Q6_K | 248320 × 5120 | logits |
| q8_0 / q5_0 | | 2048 × 2048 | 0.8B MoE kinds |
| q5_k | Q5_K | 5120 × 5120 | mixed Q4_K_M tensors |

GB/s counts the weight bytes of one launch. The q8 of `x` is outside the
timed loop, which is how a cached activation behaves. The two output rows
are about 1 GB; a failed alloc is printed and the rest of the table still
runs. `ODINFER_CUDA_PEAK_GBS` overrides 1134 when the device is not a V100S.

End-to-end decode, scalar kernel against the default. The smoke prompt is
short, so this is the short-context number (the 9B figure that was 32.7
tok/s before this kernel). A 990-token prefill is a separate measurement:
the KV read at that length is a few MB per full-attention layer, and the
weight stream is still the whole model.

```sh
ODINFER_CUDA_GEMV=old bin/cuda_smoke "$MODEL9B" 16 2048
bin/cuda_smoke "$MODEL9B" 16 2048
ODINFER_CUDA_GEMV=old bin/cuda_smoke "$MODEL27B" 16 2048
bin/cuda_smoke "$MODEL27B" 16 2048
ODINFER_CUDA_LAUNCHES=1 bin/cuda_smoke "$MODEL9B" 8 2048
```

Init prints the GEMV row policy and `graph off` unless `ODINFER_CUDA_GRAPH=1`.

Dense decode (`T==1`, not MoE) stays eager. On the V100S a captured graph
was slower than eager (9B 57.6 tok/s versus 63.0): the position update was
`cuMemcpyHtoD` on the legacy stream, which syncs the device every token,
and capture left split-K out because the zeroing memset was not on the
capture stream. `ODINFER_CUDA_GRAPH=1` opts in. The position is then an
async copy on the graph stream, and split-K is recorded with
`cuMemsetD8Async` when the driver has that symbol. A missing
`g_decode_pos` symbol uses a device int instead of turning the graph off.
Prefill passes the real position and stays eager. If capture or instantiate
fails, that token runs eager and graphs stay off. `ODINFER_CUDA_PROFILE=1`
disables the graph and prints each kernel's GPU time (name, count,
milliseconds), slowest first, plus wall time against the sum. Sampling
stays on the CPU, after the logits copy. The graph and event symbols are
looked up on their own; a driver without them disables graphs and the
profiler and still loads the GEMV.

## Check script

`scripts/cuda_check.sh` builds the CUDA test and smoke binaries, compiles the
kernel source for `compute_70` (no GPU required for that step), and, when a
GPU is visible, runs the kernel self-test and an end-to-end generation.

```sh
./scripts/cuda_check.sh
```

Model search, first hit wins (`MODEL` overrides):

| Path | Size | What it exercises |
|------|------|-------------------|
| `$MODEL` | any Qwen3.5 GGUF the loader accepts | whatever you point at |
| `models/Qwen3.5-0.8B-MoE-A0.6B-Q4_K_M.gguf` | 524 MB | MoE, mixed Q4_K / Q8_0 / Q5_0 |
| `$HOME/.odinfer/models/Qwen3.5-0.8B-MoE-A0.6B-Q4_K_M.gguf` | 524 MB | same, installer layout |
| `models/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf` | Mac benchmark, sha256 prefix `a692de95` | dense Q4_K_M (Q4_K + Q6_K + F32 norms) |

Smoke arguments (overridable): `SMOKE_TOKENS` default 16, `SMOKE_CTX` default
2048. If no model file is present the kernel test still runs and smoke is
skipped, unless `CUDA_CHECK_REQUIRE_SMOKE=1`.

The script prints `sha256 <hex> <path>` for the file it found. The Mac
benchmark 9B starts with `a692de95`. Current Hugging Face `main` of
`mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-GGUF`
(commits `29319295d35e` through `a713868aa6cd`) is a different quant:
sha256 prefix `b4363b57`, 5780092768 bytes. That revision is not pinned,
because pinning it would not match the Mac file, and the `a692de95` blob
is not in the surviving history of that repo. `cuda_check.sh` refuses to
smoke a heretic 9B whose hash does not start with `a692de95` unless
`CUDA_CHECK_ALLOW_REQUANT=1`. `scripts/install.sh --model qwen9b` still
fetches HF `main`, prints the sha256, and warns when it is not
`a692de95`.

Direct commands the script wraps:

```sh
odin build tools/cuda_kernel_test.odin -file -out:bin/cuda_kernel_test \
  -define:ODINFER_BACKEND="cuda" \
  -collection:ggml=packages -collection:infer=packages \
  -collection:qwen3_5=packages -collection:qwen3_5_tokenizer=packages \
  -collection:sampler=packages -collection:tokenizer=packages \
  -collection:cuda=packages

bin/cuda_kernel_test --nvrtc-only compute_70
bin/cuda_kernel_test

odin build tools/cuda_smoke.odin -file -out:bin/cuda_smoke \
  -define:ODINFER_BACKEND="cuda" \
  -collection:ggml=packages -collection:infer=packages \
  -collection:qwen3_5=packages -collection:qwen3_5_tokenizer=packages \
  -collection:sampler=packages -collection:tokenizer=packages \
  -collection:cuda=packages

bin/cuda_smoke models/Qwen3.5-0.8B-MoE-A0.6B-Q4_K_M.gguf 16 2048
```

`cuda_kernel_test` compares GEMV/MMQ, RMSNorm, RoPE, SwiGLU, mixed-V KV
store + attention, and the gated-delta step to the CPU reference in
`packages/ggml` and `packages/qwen3_5/ops.odin`. `cuda_smoke` prints prefill
and decode tok/s. Sampling stays on the existing CPU sampler (temperature 0
in the smoke tool).

## What is ported

GEMV (decode) and MMQ (prefill, tile of up to 8 tokens, any `T`,
token-major output), fp32 accumulate. The default decode GEMV is the
bandwidth kernel for Q4_K, Q5_K, Q6_K, Q8_0, and Q5_0; every kind still
has the scalar one-warp-per-row kernel:

F32, F16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K.

Also: embedding via the CPU dequantizer plus an upload, RMSNorm, per-head
Q RMSNorm, partial RoPE, attention decode and prefill, KV store and attention
for f16, symmetric Q8_0, and the 40-byte mixed V block, attention gate,
SwiGLU, residual, depthwise conv1d+SiLU, l2norm+scale, per-token gated delta,
RMSNormGated, MoE softmax top-k plus per-expert GEMV and the optional shared
expert.

The default 9B Q4_K_M file (Q4_K, Q6_K, F32 norms, often Q8_0 or Q6_K
embeddings) and the 0.8B MoE file (Q4_K, Q8_0, Q5_0, F32) are in that set.
Init scans every projection and refuses a kind that is not in the list,
naming the tensor. Embeddings may be any kind `ggml` can dequant on the CPU,
including IQ; the output projection may not.

## What is not ported

IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q8_1.

Also not in this backend: Metal's shape-specialized GEMV (`QGEMVSPEC`),
wide-N autotune and the tune cache, fast-math, MoE grouped GEMM
(`forward_gpu_batch_moe`), and `engine_verify_batch`
(speculative verify). A one-row RMSNorm on CUDA does feed the q8 cache
the next GEMV reads. MoE prefill runs the attention projections as MMQ
and the experts as per-token GEMV. Q4_K / Q6_K attention projections use the same prefill GEMM as a dense model. Dense prefill chunks by
`ODINFER_CUDA_PREFILL_CHUNK` (default 512, smaller if the activation alloc fails). Decode is
GEMV, not MMQ.

Weights are copied to the device once (`cuMemcpyHtoD` of the GGUF mmap, plus
the host-dequantized MoE router and shared-expert matrices). A 5.6 GB Q4_K_M
9B fits in a 32 GB V100S; the resident KV cache is extra and follows the same
`KV_Q8` / `KV_V` layout as Metal, so snapshots interchange.

Sigmoid and softplus in the CUDA kernels match `ops.odin` (stable sigmoid,
softplus clamped at ±20). Metal's fast-math `1/(1+exp(-x))` can differ by a
few ulps. Kernel tests compare against the CPU reference, not against Metal
golden files. The scalar GEMV tolerance is about `2e-2 + 1e-3 * |ref|`
because the warp reduction order is not the CPU's left-to-right sum. The
bandwidth GEMV allowance is `1 + 0.05 * |ref|` because `x` is q8.
