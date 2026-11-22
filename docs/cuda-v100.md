# CUDA V100

This branch is the V100 build of the Qwen3.5 CUDA engine. The code is commit
`965afd105e4923c18c9f766f41d03489c19d2943` (PR #25). This file is the only
change on top of that tree. It does not contain the later attention
experiments.

Target: Tesla V100 / V100S, sm_70. The numbers below were measured on a
V100S 32GB with Qwen3.5-27B Q4_K_M.

Full backend notes: [CUDA.md](CUDA.md).

## Build and run

Odin does not need `nvcc` or a GPU. `libcuda.so.1` and NVRTC are loaded at
runtime. On sm_70 use NVRTC 12.x (CUDA 13 NVRTC cannot emit `compute_70`).

```sh
python3 -m pip install --user nvidia-cuda-nvrtc-cu12
export ODINFER_NVRTC_PATH="$HOME/.local/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12"

odin build server/serve_harness/serve.odin -file -out:bin/odin-serve \
  -o:speed -no-bounds-check -disable-assert \
  -define:ODINFER_BACKEND="cuda" \
  -collection:ggml=packages -collection:infer=packages \
  -collection:qwen3_5=packages -collection:qwen3_5_tokenizer=packages \
  -collection:sampler=packages -collection:tokenizer=packages \
  -collection:cuda=packages

bin/odin-serve model.gguf --cuda --max-ctx 32768
```

`-define:ODINFER_BACKEND="cuda"` must be quoted.

## Recommended environment

These are the defaults of this commit on a V100 with `head_dim` 256. Set
them only when the process environment might override them.

```sh
export ODINFER_CUDA_PREFILL_FLASH=2
export ODINFER_CUDA_PREFILL_CHUNK=512
```

- `ODINFER_CUDA_GEMV_MMVQ` stays on. `=0` turns it off.
- Do not set `ODINFER_CUDA_WGEMM_BK`. Unset is BK=64. `=32` is slower and stays off.
- Do not set `ODINFER_CUDA_ATTN_PARTS`. Unset keeps the planner.
- `ODINFER_CUDA_ATTN_QTILE` is not in this tree. The 4-query-tile kernel was
  measured slower and is not part of this branch.
- `ODINFER_CUDA_GRAPH` stays unset.

## Measured on V100S 32GB, Qwen3.5-27B Q4_K_M

Repeated A/B, previous production binary to this commit. Deployed 2026-11-22
17:31 Beijing. Texts were identical. `cuda_kernel_test`: 302 PASS.

| Run | Previous | This commit |
|---|---:|---:|
| 16000 prefill | 214.16 tok/s | 218.55 tok/s (+2.05%) |
| 16000 decode | 29.49 tok/s | 29.54 tok/s |
| 905 prefill / decode | 234.44 / 32.14 tok/s | 234.65 / 32.60 tok/s |

`ODINFER_CUDA_WGEMM_BK=32` and a 4-query-tile attention (`ATTN_QTILE=4`) were
both slower on this GPU and are off. This branch does not build the 4-tile
kernel.
