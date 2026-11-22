<!--
Internal engineering log, kept in its original language (中文内部工程记录，保留原文).
The user-facing documentation (README / AGENTS / POSITIONING / docs/MCP·MEMORY·THERMAL) is bilingual.
面向用户的文档（README/AGENTS/POSITIONING/docs 下三篇）为中英双语；本文件为内部工程日志。
-->

# Environment variables

All tunables recognized by `odin-infer` (library) and `odin-infeer` (CLI).
Defaults apply when the variable is unset.

Inspect current values on your machine:

```sh
odin run doctor_harness/doctor.odin -file -- env \
  -collection:ggml=. -collection:infer=. \
  -collection:tokenizer=. -collection:sampler=.
```

## Inference / Metal backend

Sourced from `qwen3_5/metal.odin` (read once during `metal_init`). The Qwen3
Metal backend (`infer/metal.odin`) currently has no env-var knobs.

| Variable       | Default | Effect                                                                 |
|----------------|---------|------------------------------------------------------------------------|
| `QFASTMATH`    | `ON`    | Metal shader fast-math enable. Set `=0` to disable for precision debug |
| `QTIMING`      | off     | Set `=1` to print per-prefill GPU ms                                  |
| `QPROF_NOLIN`  | off     | Ablation: skip linear-layer per-token stateful loop (output garbage)  |
| `QPROF_NOFULL` | off     | Ablation: skip full-attention per-token block                          |
| `KV_Q8`        | `ON`    | Q8_0 KV cache. Set `=0` for the legacy f16 Metal KV (A/B, precision)   |
| `KV_V`         | `mix`   | V cache format when `KV_Q8` is on. Default stores the peak value of each 32-block as f16 and Q8-quantizes the rest. Set `=q8` for symmetric Q8 V |
| `QGEMVSPEC`    | `ON`    | Per-tensor GEMV shape kernels for Q4_K/Q6_K and the IQ family. Set `=0` to use the generic kernels |
| `QMMQN`        | `ON`    | Wide-N prefill MMQ. Q4_K is 32×32×64, Q6_K is 16×64×64, and the eight IQ kinds are 32×32×64, when M/N/K divide. Set `=0` for the BN=8 tiles |
| `QGEMVEXACT`   | off     | Dense prefill only. Set `=1` to recompute attention and linear projections with per-token GEMV so they match decode bit for bit. MoE prefill always does that overwrite |
| `METAL_FALLBACK` | off   | Set `=1` to let a failed Metal init fall back to CPU. Default: `--metal`/`-g 1` + failed init = hard error |

`QPROF_*` are GPU-kernel timing ablations — they produce wrong output but let
you isolate per-stage GPU time. Never enable in production.

## CUDA backend (Linux, `-define:ODINFER_BACKEND="cuda"`)

| Variable | Default | Effect |
|----------|---------|--------|
| `ODINFER_NVRTC_PATH` | unset | Exclusive path to `libnvrtc.so.12` (sm_70) or `.13`, or the directory that contains it. CUDA 13 NVRTC cannot target compute_70; a V100 needs NVRTC 12.x (`pip install nvidia-cuda-nvrtc-cu12`). |
| `NVRTC_DIR` | unset | Directory or soname path searched first when `ODINFER_NVRTC_PATH` is unset. For sm_70 a 13.x library there is skipped and the search continues to pip `libnvrtc.so.12` and the loader. |
| `CUDA_DEVICE` | `0` | Device ordinal. |
| `ODINFER_CUDA_TC` | auto | Prefill GEMM. Unset: Volta WMMA when `T≥16`. `1`: WMMA for every `T>1`. `0`: MMQ. Decode (`T==1`) stays GEMV. |
| `ODINFER_CUDA_GEMV` | auto | Decode GEMV. Unset: dp4a bandwidth kernel; Q5_0 and Q8_0 under 16 MiB stay on the scalar kernel. `new`: bandwidth kernel for those small shapes too. `old` or `0`: scalar GEMV for everything. |
| `ODINFER_CUDA_GEMV_ROWS` | `4` | Q4_K / Q6_K rows per warp. Unset is four. `1` selects the one-row kernel, which was slower on the V100S. |
| `ODINFER_CUDA_GEMV_WARPS` | `8` | Warps per bandwidth-GEMV block, 1..16. Unset shrinks a grid that is under one block per SM. A set value pins the count. Q5_K decode uses the same q8 `dp4a` kernel and splits K when rows are at most 8192. |
| `ODINFER_CUDA_GEMV_SPLIT` | on | Q4_K / Q6_K decode with `rows≤8192` splits K and `atomicAdd`s the partials. `=0` is one K range. |
| `ODINFER_CUDA_GEMV_SPLITS` | auto | Unset uses 2-way on short-row matrices whose K still has 12 superblocks per half, and 4-way otherwise. 27B w2 is staged plus that 2-way (bench `q6s2`). 9B w2 is direct 2-way (bench `s2`). `r4` is not the decode launch. `=4` is the flat split. |
| `ODINFER_CUDA_GEMV_FUSE` | auto | Q4 gate+up with K≥5120 is one kernel (`gemv2_q4p`). 9B (K=4096) stays two launches. `=1` forces the pair. `=0` is two GEMVs. `--bench-gemv` prints `pair` versus `two r4` on the w1 shapes. |
| `ODINFER_CUDA_TC2` | on | Q4_K / Q6_K prefill (K multiple of 256). `T<=64` uses the 32-token fp16 tile. `T>64` uses the scalar-dequant tile. `=0` uses the scalar tile at every T. |
| `ODINFER_CUDA_TC2_WIDE` | off | `=1` uses the 128-token tile for `T>64`. Slower than the scalar tile at 581 and 2000 tokens on the V100S. |
| `ODINFER_CUDA_TC2_BLOCK` | off | `=1` uses `tc2m`. Measured slower on the V100S and it changed a sampled token. Leave off. |
| `ODINFER_CUDA_TC2_REUSE` | off | `=1` uses `tc2r` for `T>64` (profile name `tc2r`). The scalar tile's weight dequant is reused across 128 tokens. Unmeasured. If several `TC2_*` flags are set, this one is used. |
| `ODINFER_CUDA_PREFILL_GEMM` | on | Q4_K / Q5_K / Q6_K prefill (`T>1`, K multiple of 256). Default is `wgemm` once `T` reaches `WGEMM_MIN`: one fp16 dequant of the weight tile, fp16 activations, a 128×128 WMMA tile. Shorter T, including Q5_K, is `mmqk` (`mmqk_q5_k`). `ODINFER_CUDA_PREFILL_LINEAR=mmqk` keeps `mmqk` at every T (q8, one staged superblock, int8 dp4a). On a V100 with chunk 512 the older Q4/Q6 tile was 9B at 905/1805/8000/16000 371.70/376.91/337.20/290.49 tok/s and the 27B at 905/8000/16000 124.00/112.97/97.43. `=0` keeps `tc2` for `T<=64` and the scalar tile for `T>64`. `wgemm` allows `5e-2 + 5e-2*|ref|` against the fp32 CPU GEMV. `mmqk` allows `2e-3 + 2e-3*|ref|` against `gemv2`. |
| `ODINFER_CUDA_PREFILL_LINEAR` | `wgemm` | `=mmqk` selects the dp4a tile. Unset is `wgemm` for `T` at or above `ODINFER_CUDA_WGEMM_MIN`. Ignored when `ODINFER_CUDA_PREFILL_GEMM=0`. |
| `ODINFER_CUDA_WGEMM_MIN` | `128` | Q4_K / Q5_K / Q6_K prefill below this token count uses `mmqk`. Q5_K used to fall through to scalar `tc_q5k`. A 24-token prefill was 1661/824 ms on `wgemm` against 896/452 on `mmqk`. `=0` keeps `wgemm` for every `T>1`. |
| `ODINFER_CUDA_PREFILL_FUSE` | off | `=1` makes gate+up and K+V share one q8 stage. It was neutral on a V100 (9B at 8000 tokens, 332 off vs 333 on) and the linear bucket was slower. |
| `ODINFER_CUDA_MMQK_ROWS` | `4` | Default Q4 tile is 32 rows per block. `=8` selects the 64-row tile, which was slower on a V100. Q6 stays at 32 either way. |
| `ODINFER_CUDA_ATTN_LONG` | on | Mixed-V decode, `npos>=2048`, head_dim 256, 4 or 6 query heads per KV head. 128 positions per block. `=0` keeps 32-token parts. |
| `ODINFER_CUDA_DELTA_WMMA` | on | 128×128 prefill state uses a 16-token WMMA gated-delta chunk. The scalar passes hoist the per-column scale. The earlier chunk dropped the 27B 8000-token delta bucket from 13237 ms to 11057. `=0` is the fp32 recurrence. |
| `ODINFER_CUDA_PREFILL_CHUNK` | `512` | Tokens per prefill forward. A multiple of 16 from 16 to 512. Anything else is ignored. On the scalar tile, 9B at 16000 tokens was 183/207/213 tok/s for 128/256/512. A size that does not fit falls back through 256, 128, 32, 8, 1, and does not grow past the request. |
| `ODINFER_CUDA_ATTN_BATCH` | on | Prefill full attention launches once per chunk (heads by tokens). `=0` launches once per token. |
| `ODINFER_CUDA_ATTN_SPLIT` | on | Mixed-V decode (`T==1`, `npos>=128`) uses 32-token parts and a reduce (`attn_part`, `attn_red`). Does not select the prefill kernel. `=0` is the one-block kernel for decode. fp32 merge tolerance versus `attention_q8v` is `2e-2 + 1e-3*|ref|`. Disables the CUDA graph. |
| `ODINFER_CUDA_PREFILL_FLASH` | WMMA | Mixed-V prefill (`T>1`). Unset on sm_70+ with head_dim 256 and at most 8 query heads per KV head is `attention_q8v_wmma`. `=0` is `attention_q8v`. `=1` is `attention_q8v_flash` (fp32 online softmax, `br` query tokens, 32-token KV tiles). On the 9B V100S that was slower (16000 tokens 80.7 vs 114.2 tok/s). `=2` forces the WMMA kernel. The kernel in this tree owns 32 query tokens and one 64-token fp16 K tile, reused across the query group, then V replaces K in the same buffer. Each of 8 warps owns a 16×16 score tile and accumulates the full head dimension. A tile that starts past the last query is skipped. Shared memory is 91136 bytes. It has not been timed. The previous 16-query, 32-key kernel, which these commands replace, was the measured default: 9B at 16000 tokens 182.8 vs 114.2 for `attention_q8v`, 27B 62.6 vs 40.9, and on e452657 the 27B 16000 prefill attention bucket was 43557 ms. Tolerance versus `attention_q8v` is `1e-1 + 5e-2*|ref|`. Decode is unchanged |
| `ODINFER_CUDA_FUSE_NORM` | off | `=1` writes the projection to a side buffer and fuses the add with the next RMSNorm and its q8 store. Slower than the accum GEMV plus `rmsnorm_q8` on the V100S. |
| `ODINFER_CUDA_Q4_NC` | off | `=1` uses `ld.global.nc` for the Q4 GEMV. Slower than plain `ld.global` on the V100S. |
| `ODINFER_CUDA_Q4_ILP` | off | `=1` uses `gemv2_q4x` (8 rows/warp, coherent 128-bit loads, two super-blocks per iteration). `--bench-gemv` prints it as `r8`. Not the decode default. |
| `ODINFER_CUDA_Q6_STAGE` | auto | Unset stages a Q6 GEMV only when K≥16384 or rows>8192. `=1` stages every Q6 GEMV. `=0` uses direct loads. |
| `ODINFER_CUDA_SMOKE_PREFILL` | unset | `cuda_smoke` only. `=N` prefills N tokens of a fixed repeated sentence instead of the short 2+2 prompt. |
| `ODINFER_CUDA_LOGITS` | unset | `cuda_smoke` only. `=1` prints the top-5 logits at the first decode position and the next one, and the margin to EOS 248046. |
| `ODINFER_CUDA_GRAPH` | off | `=1` captures dense `T==1` decode as one CUDA graph and recaptures when the split-attention part count changes. MoE and prefill stay eager. Off while `ODINFER_CUDA_PROFILE=1`. |
| `ODINFER_CUDA_PROFILE` | off | `=1` prints per-kernel GPU time for each forward (disables the graph). The 8192-event ring is aggregated by kernel name as it fills, so a long prefill is complete. |
| `ODINFER_CUDA_PHASE` | off | `=1` prints one `cuda: phase` line per forward (linear, attn, kv, norm, delta, other, nongemv, gap), a `cuda: phase kernels` line of the eight largest kernel totals, and one `cuda: phase prefill` total across the prefill chunks. `nongemv` excludes linear. `gap` is wall minus the events. Same events as `PROFILE`. Not a production tok/s. |
| `ODINFER_CUDA_WGEMM_X` | on | Pack each wgemm activation to fp16 once. `=0` loads fp32 inside the tile. Q5_K prefill uses the same tile. |
| `ODINFER_CUDA_GEMV_PIPE` | off | `=1` prefetches the next Q4/Q6 super-block. Unset is one block in flight and a per-row x load. |
| `ODINFER_CUDA_DELTA_SM` | on | `=0` keeps the decode delta state in global memory. |
| `ODINFER_PHASE` | off | `=1` prints CPU time for the grammar mask and the tool-call grammar mask on each sample. |
| `ODINFER_CUDA_LAUNCHES` | off | `=1` prints the kernel launch count of each CUDA forward on stderr. A graph replay prints the captured count. |
| `ODINFER_CUDA_PEAK_GBS` | `1134` | HBM peak for `cuda_kernel_test --bench-gemv` (percent of peak). |
| `QTIMING` | off | `=1` prints per-chunk CUDA wall time on stderr. |
| `KV_Q8` / `KV_V` | Q8 K + mixed V | Same switches as Metal. The on-disk KV snapshot format is shared. |
| `METAL_FALLBACK` | off | `=1` lets a failed CUDA init fall back to CPU. `--cuda` / `--metal` / `-g 1` otherwise hard-error. |
| `MOE_CPU` | off | `=1` forces the MoE path onto CPU. |

`QGEMVSPEC`, `QMMQN`, `QFASTMATH`, and the tune cache are Metal-only. The CUDA build does not read them.

## CLI (`odin-infeer`)

| Variable | Default | Effect                                  |
|----------|---------|-----------------------------------------|
| `QDBG`   | off     | Set `=1` to dump token encode/decode to stderr |

## Compile-time constants (not env vars, but adjacent)

These are baked at build time; changing them requires a rebuild.

| Constant                 | Where                  | Default | Notes                                            |
|--------------------------|------------------------|---------|--------------------------------------------------|
| `DEFAULT_MAX_CONTEXT`    | `infer/model.odin`     | 4096    | Cap on KV cache; CLI overrides with `-c`         |
| `MAX_MATMUL_N`           | `infer/matmul.odin`    | 65536   | Per-thread dequant scratch length                |
| `MAX_BATCH_T`            | `qwen3_5/metal.odin`   | 512     | Max tokens per batched-prefill chunk (Stage 1a)  |
| `MAX_VOCAB` / `MAX_MERGES` | `tokenizer/tokenizer.odin` | 151936 / 151386 | Tokenizer buffer ceilings  |

## CLI flags (not env, but worth documenting here)

`odin-cli <model.gguf> [flags]`:

| Flag | Meaning                                  | Default                |
|------|------------------------------------------|------------------------|
| `-t` | Temperature                              | 0.6                    |
| `-p` | Top-p                                    | 0.95                   |
| `-s` | RNG seed (0 = time-based)                | 0                      |
| `-m` | Multi-turn (1 = on)                      | off                    |
| `-k` | Thinking on (1 = on, Qwen3 reasoning)    | off                    |
| `-x` | Repetition penalty                       | 1.0 (off)              |
| `-r` | Print tokens-per-sec                     | off                    |
| `-f` | Print TTFT                               | off                    |
| `-j` | Thread count                             | physical core count    |
| `-c` | Max context                              | `DEFAULT_MAX_CONTEXT`   |
| `-g` | Metal GPU (1 = on)                       | off                    |

Special: `<model.gguf> --dump` runs `dump_gguf` (metadata dump) instead of chat.

## Adding a new env var

1. Read it in the relevant package with `os.get_env(NAME, context.temp_allocator)`.
2. Add a row to the table above.
3. Add an entry to `ENV_VARS` in `doctor_harness/doctor.odin` so `doctor env`
   picks it up automatically.
