# CUDA 后端

[English](CUDA.md)

Qwen3.5 引擎的 Linux x86_64 后端。目标是 **sm_70 及以上**（按 Tesla V100S、
计算能力 7.0、驱动 CUDA 13.0 来写）。累加是 fp32，没有 bf16。预填充 GEMM
（`T>1`）可以用 Volta tensor core（WMMA 16×16×16，fp16 乘、fp32 累加）。
解码（`T==1`）走带宽 GEMV：128 位加载、`dp4a`，激活每条向量只量化成 q8
一次。短的 Q4_K / Q6_K 每个 warp 一行，更高的矩阵仍是四行。
`ODINFER_CUDA_GEMV=old` 回到标量内核。稠密解码默认即时启动，
`ODINFER_CUDA_GRAPH=1` 才捕获 CUDA graph。

macOS 构建不变，默认仍是 Metal。同一个二进制不会同时带上两条热路径。

## 构建

Odin 编译不需要 `nvcc`、CUDA toolkit 头文件，也不需要 GPU。`libcuda.so.1`
和 NVRTC 在运行时 `dlopen`。

```sh
# Linux CPU 构建（不链 Metal）。入口和 macOS 一样。
./build.sh

# CUDA 引擎
odin build server/serve_harness/serve.odin -file -out:bin/odin-serve \
  -o:speed -no-bounds-check -disable-assert \
  -define:ODINFER_BACKEND="cuda" \
  -collection:ggml=packages -collection:infer=packages \
  -collection:qwen3_5=packages -collection:qwen3_5_tokenizer=packages \
  -collection:sampler=packages -collection:tokenizer=packages \
  -collection:cuda=packages

bin/odin-serve model.gguf --cuda --max-ctx 2048
# CUDA 构建里 --metal 和 --cuda 都表示「用 GPU」。
```

`-define:ODINFER_BACKEND="cuda"` 要带引号。未设置时是 `"default"`：Darwin
上仍是 Metal，其它系统是 CPU。

开关在 `packages/qwen3_5/backend.odin`（`GPU_METAL` / `GPU_CUDA`）。前向、
KV 存取、复位都是 `when`，编译出来的热路径没有 Metal/CUDA 分支。

## NVRTC

`ODINFER_NVRTC_PATH` 是独占覆盖：指向 `libnvrtc.so.12`（或 `.13`）的完整路径，
或包含它的目录。设了它就不再搜索其它位置。没设覆盖时先找 `NVRTC_DIR`，然后
继续往下找。

否则加载器先试带版本的 soname，再试不带版本的名字（`libnvrtc.so.13`、`.12`、
`libnvrtc.so`），然后找 pip 的 `nvidia-cuda-nvrtc-cu12`
（`~/.local/lib/python3.*/site-packages/nvidia/cuda_nvrtc/lib`）。对应的
`libnvrtc-builtins.so.<主版本>.<次版本>`（wheel 里是
`libnvrtc-builtins.so.12.9`，没有 `.so.12`）用 `RTLD_GLOBAL` 预加载。
`.alt.` 那份会跳过。

CUDA 13 的 NVRTC 不能编译 `compute_70`。低于 sm_75 的架构被删了，CUDA 13
机器上不带版本的 `libnvrtc.so` 就是这个 13.x 库。13.x 的驱动仍然可以 JIT
NVRTC 12.9 产出的 PTX。设备是 sm_70 / sm_72，或者 `compute_70` 在 13.x 库上
编译失败时，运行时只加载 NVRTC 12.x，找不到就打印 pip 路径。NVRTC 12.9 会
警告 sm_75 之前的架构已弃用，在 V100 上这是预期的。

```sh
python3 -m pip install --user nvidia-cuda-nvrtc-cu12
export ODINFER_NVRTC_PATH="$HOME/.local/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/libnvrtc.so.12"
```

每次编译都带上从设备算力得到的 `--gpu-architecture=compute_XX`。
`CUDA_DEVICE` 选设备序号，默认 0。

## Tensor core

预填充和其它批量 GEMM（`Y[T, rows] = X[T, cols] · W[rows, cols]`）把 `W`
反量化成 shared memory 里的 fp16 小块（64×32），`X` 收成 fp16，再发
`wmma.mma.sync.aligned.m16n16k16`（fp16 进、fp32 累加）。PTX 是内联的。
NVRTC 不带 `mma.h`。`mma.sync.m16n16k16` 不是 Volta 的指令；sm_70 能 JIT 的
是 WMMA 这条。不用 `ldmatrix`。

有 tensor core 内核的量化种类和 MMQ 相同：F32、F16、Q4_0、Q4_1、Q5_0、Q5_1、
Q8_0、Q2_K、Q3_K、Q4_K、Q5_K、Q6_K、Q8_K。9B Q4_K_M 的预填充是 Q4_K（up、
gate、attention）和 Q6_K（down）。0.8B MoE 里走批量 GEMM 的投影是 Q4_K、
Q8_0、Q5_0。专家 GEMV 仍是 GEMV。

`cuda_init` 时读 `ODINFER_CUDA_TC`：

| 值 | 行为 |
|----|------|
| 不设 | 算力 ≥ 7.0 且 `T ≥ 16` 时用 tensor core |
| `1` | 所有 `T > 1` 用 tensor core（`T == 1` 仍是 GEMV） |
| `0` | `T > 1` 用 MMQ，解码用 GEMV |

`T == 1` 永远是 GEMV。没有对应内核时退回 MMQ。

正确性（打印 `max_abs`、该元素的 tol、`atol`、`rtol`）和微基准：

```sh
bin/cuda_kernel_test
bin/cuda_kernel_test --bench
```

`--bench` 对 9B 的 w1/w3、wq、wo（Q4_K）和 w2（Q6_K），以及 0.8B 规模的
Q8_0、Q5_0，分别打印 tensor core 和 MMQ 的毫秒数和 TFLOPS（2·T·rows·cols）。

端到端预填充对比（同一条 prompt）。`prefill N tok` 里的 N 至少要 16，默认
策略才会走上 tensor core；`ODINFER_CUDA_TC=1` 则只要 N>1。

```sh
ODINFER_CUDA_TC=0 bin/cuda_smoke "$MODEL" 8 2048
bin/cuda_smoke "$MODEL" 8 2048
ODINFER_CUDA_TC=1 bin/cuda_smoke "$MODEL" 8 2048
```

这三次的解码 tok/s 应当一致。初始化日志会写 `tensor cores auto (T>=16)`、
`off` 或 `on for T>1`。

## 解码 GEMV

`T==1` 是访存 GEMV。标量内核（一行一个 warp、按字节加载）留在
`ODINFER_CUDA_GEMV=old`。带宽内核有两种形状。四行一个 warp、每行 8 个线程，
用于高矩阵（FFN up、27B 的 Q、logits）。这 8 个线程各发一次 `ld.global.v4`，
一行的量化数据是一次 128 字节的合并读，点积是 `dp4a.s32.s32`。Q6_K 的块是
210 字节，16 字节加载是两个对齐窗口的移位；权重分配多留 64 字节，sm_70 上
不会发出未对齐的 `v4`。Q6 按字节拆码，PTX 里没有 `vsub4`：NVRTC 会发出
`vsub4.s32.s32.s32`，sm_70 的 JIT 拒绝这条指令，整模块都加载失败。scale
留在寄存器里。

Q4_K 和 Q6_K 默认都是四行一个 warp。一行一个 warp 在 V100S 上每个测过的
形状都更慢（9B 解码 33 tok/s，四行是 62；bench 上 r1 只有峰值的 8%，r4 是
39%）。`ODINFER_CUDA_GEMV_ROWS=1` 仍可选它。一段 K 上的总 warp 数是 `rows / 4`。
行数不超过 8192 的 Q4_K / Q6_K 把这段 K 拆到 `gridDim.y`（默认 4 路，
`atomicAdd` 汇总）。`ODINFER_CUDA_GEMV_SPLIT=0` 关掉。K 是 256 倍数的
Q4_K / Q6_K 预填充只在测得更快的长度上走 tensor core。`T<=64` 用 32-token 的块（profile 名 `tc2`）：V100S 上 T=19 是 176 ms，128-token 的块是 323 ms。`T>64` 默认用逐元素反量化（profile 名 `tc`）。128-token 的块（`tc2w`）在长 prompt 上更慢：9B 581 token 124.5 对 138.9 tok/s，2000 token 71.9 对 76.7；27B 581 token 50.9 对 53.1。`ODINFER_CUDA_TC2_WIDE=1` 才选它。`ODINFER_CUDA_TC2_BLOCK=1` 的 `tc2m` 在 V100S 上更慢（9B 581 token 151 对 208 tok/s），而且采样 token 和标量块不一致，保持关闭。`ODINFER_CUDA_TC2_REUSE=1` 在 `T>64` 用 `tc2r`：还是标量块那套反量化和 WMMA，一个权重块服务 128 个 token 的两半。`T<=64` 仍是 32-token 块。还没有 V100 计时。`cuda_kernel_test` 要求它和 `tc` 逐位一致，这不是采样文本一致的结论。`ODINFER_CUDA_TC2=0` 在任何 T 都用标量块。

`T>1` 且 K 是 256 倍数的 Q4_K / Q6_K 预填充默认走 `mmqk`（profile 名 `mmqk_q` 和 `mmqk`）。`pgemm`（64×64 块，BK=128，反量化到 fp16 共享内存再做 WMMA）在 V100 上结果正确但更慢，所以不再启动。9B 在 905/1805/8000/16000 的预填充 tok/s 是 173.75/177.03/168.22/153.78，对照 `ODINFER_CUDA_PREFILL_GEMM=0` 的 221.70/220.46/203.90/183.09。27B 在 905/8000/16000 是 63.79/62.45/56.85，对照 74.19/69.52/62.71。9B 位置 7808 的块，线性层 584 ms 对 450 ms；27B 每 128 token 大约 1636 ms 对 1423 ms。那个内核在同一个 K 上写 32 行 fp16，行距 256 字节，是 32 路共享内存 bank 冲突；64 token 的块在 128 token 的 chunk 里把每个权重反量化两遍。`mmqk` 先把这一块激活量化成 q8（`quant_q8_batch`，和 `quant_q8` 一样的 `amax/127` 分组和每 16 个 float 的精确和），再用 `dp4a.s32.s32` 点存储的 nibble。X 按 K 主序存放，一个 warp 的 32 个 token 是一次合并的 uint32。计过时的内核每个 warp 负责 4 行、64 个 token（lane 和 lane+32），每个 warp 都从显存重读 X。`ODINFER_CUDA_PREFILL_GEMM=0` 回到 `T<=64` 的 `tc2` 和 `T>64` 的逐元素 `tc`。别的量化种类仍走 `tc` 或 MMQ。token embedding 是查表，logits 仍是 `T==1` 的 GEMV。V100 上 chunk 512 时 `mmqk` 快于 `ODINFER_CUDA_PREFILL_GEMM=0`。9B 在 905/1805/8000/16000 的预填充 tok/s 是 303.23/305.95/279.30/245.95，对照 253.87/256.53/237.89/212.92。27B 在 905/8000/16000 是 100.85/93.45/82.70，对照 82.97/79.63/71.90。`PHASE=1` 在 8000 token 的线性层合计：9B 18378 ms（开关为 0 时 23341），27B 61546 ms。一次 14.5k 请求的预填充是 173.2 s（84.1 tok/s）。这些数字是 4 行的块：每个 q8 字节每 4 行就从显存重读一次。把一个 superblock 放进共享内存、32 行共用（每个 warp 4 行，22528 字节）在那台 V100 上更快：9B 在 905/1805/8000/16000 是 371.70/376.91/337.20/290.49 tok/s，27B 在 905/8000/16000 是 124.00/112.97/97.43，`PHASE=1` 在 8000 token 的线性层 9B 是 13425 ms、27B 是 46627 ms。一次 14.5k 请求的预填充是 146.4 s（99.4 tok/s）。这棵树里的内核还没有计时。Q4 是每个 warp 8 行（一块 64 行）：sm_70 的 `launch_bounds(256, 2)` 是 128 个寄存器、没有 spill，一个 SM 仍能放两个 block。Q6 在 8 行时 spill 了 36 字节写和 68 字节读，所以 Q6 仍是每个 warp 4 行。gate+up 以及 K+V 共用这一份 q8（`ODINFER_CUDA_PREFILL_FUSE=0` 分开启动）。同一份激活的下一次矩阵乘跳过 `quant_q8_batch`。KV 反量化每个 32 值只读一次 scale。权重在 warp 内是广播，放进共享内存减不了显存流量。128 token 的块大约 45 KiB，48 KiB 的默认上限里只能放一个 block，所以不是默认。贪心文本一致，含 `ODINFER_CUDA_GEMV_SPLIT=0`。`cuda_kernel_test` 要求它和 `gemv2` 的差在 `2e-3 + 2e-3*|ref|` 以内（那台机器上 Q6 的 max abs 是 7.6e-6）。相对 fp32 CPU GEMV 用和 `gemv2` 相同的 q8 容差：atol 1、rtol 5e-2。2e-2 的界限让四个 Q6 形状在少数元素上超出 0.05–0.16，但这些核和 `gemv2` 是对齐的。形状含不满的块、K=256/512/1024，以及 token 数不是 64 的倍数。`pgemm` 内核仍留在测试里，容差是 `5e-2 + 5e-2*|ref|`。`ODINFER_CUDA_PREFILL_CHUNK` 设定预填充块大小（默认 512，16 的倍数，范围 16 到 512）。标量路径上 9B 在 16000 token、块大小 128/256/512 是 183.09/206.62/213.01 tok/s。放不下就按 256、128、32、8、1 退，不会比请求的更大。WMMA 注意力的工作区按实际分配到的块来算。

这棵树里 `T>1` 且 K 是 256 倍数的 Q4_K / Q6_K 预填充默认启动 `wgemm`，不是 `pgemm`（`pgemm` 仍然不启动）。一个 block 负责 128 个 token、128 行。权重块只反量化一次到 fp16，行距 80（160 字节，16 字节对齐，相邻行相差 8 个 bank）。8 个 warp，每个拿 4×2 个 WMMA 16×16×16 片段。激活是 fp16，不是 q8。下一截 K 先放进寄存器，再写到另一块共享内存。共享内存 81920 字节，一个 SM 只能放一个 block。profile 名是 `wgemm`，算在线性层。`0c7befd` 上 27B 16000 的线性层是 84816 ms，大约占 121 s。这一段受延迟限制：85 s 里大约读 1.5 TB 权重，在 1134 GB/s 的 V100 上只有约 18 GB/s。X 是 fp32，每个行块都重读（一块 128×5120 是 2.6 MB，大约是这块 Q4 权重的 7 倍）。把 Q4 面板写成 fp16 给 512 chunk 的四个 token 块复用会增加流量（368 KB 的 Q4 对 1.31 MB 的 fp16，还要写一次）。BK=32、256 行的块，或每个 SM 两个 block，都放不进 96 KB（两块权重面板加上现有双缓冲是 122880 字节）。启动的块没有改。`ODINFER_CUDA_WGEMM_X`（默认开）把这份激活打成 fp16 一次，每个行块重读的字节减半，核内不再做转换。`=0` 仍在核内读 fp32。一个 block 做两行块需要第二套累加器，会 spill，所以网格仍是 128 行。`ODINFER_CUDA_PREFILL_LINEAR=mmqk` 回到 dp4a。`ODINFER_CUDA_PREFILL_GEMM=0` 仍是 `tc2` / 逐元素 `tc`。`cuda_kernel_test` 相对 fp32 CPU GEMV 的容差是 `5e-2 + 5e-2*|ref|`，和 `pgemm` 一样。fp16 WMMA 不是逐位一致。`ODINFER_CUDA_LOGITS=1` 让 `cuda_smoke` 在第一个解码位置和下一位置打印 top-5 logits，以及相对 EOS 248046 的差值。9B 905 上默认（以及 `DELTA_WMMA=0`）只吐出一个 token，`PREFILL_GEMM=0` 和 e452657 的默认是整句；用这份 top-5 对拍，不要靠放宽容差。

完整注意力的预填充是 prompt 长度的平方。第 t 个 query 要扫过 t+1 个 key。旧内核每个 query head 一个 block，V 的累加在每个线程里按位置串行，所以长上下文的解码时间几乎全在这里（`9448e51` 上 27B 大约每个上下文 token、每个解码 token 21.6 µs，linear/kv/norm/delta/other 是平的）。默认 mixed-V 的解码（`T==1` 且 `npos>=128`）改用 `attention_q8v_part` 加 `attention_q8v_reduce`：32 个 token 一块，每个 KV head 一块，组内 query head 共用 K/V。预填充（`T>1`）在 sm_70 及以上且 head_dim 为 256 时走 `attention_q8v_wmma`。`ODINFER_CUDA_PREFILL_FLASH=0` 仍走 `attention_q8v`。`ODINFER_CUDA_PREFILL_FLASH=1` 是 `attention_q8v_flash`：块按几个 query token 分片，取 4、3、2、1 里能放进 48 KB 共享内存的最大一个（27B 一组 6 个 query head，用 2；9B 一组 4 个，用 3），KV 32 token 一块，fp32 在线 softmax，fp32 warp 点积。9B V100S 上它比 `attention_q8v` 慢（905 / 1805 / 8000 / 16000 的预填充 tok/s：201.8 对 213.3，186.7 对 203.9，120.0 对 153.2，80.7 对 114.2）。位置 15872 的 T=128 块，注意力 2074 ms 对 1147 ms。`=2` 是 `attention_q8v_wmma`。下面的 tok/s 是上一版：一个 block 负责 16 个 query token，KV 32 token 一块。这一版是 32 个 query token、64 个 key，K 进 fp16 共享内存后整组复用，再换成 V；超过最后一个 query 的 tile 直接跳过。共享内存 91136 字节，还没计时。kv_mul 大于 8 仍走 `attention_q8v`。`S = Q K^T` 和 `O += P V` 用 WMMA 16×16×16 fp16、fp32 累加。softmax 仍在 fp32 里在线算。KV 范围按 `blockIdx.z` 切开，让 80 个 SM 都有块；`attention_q8v_wmma_reduce` 合并部分 max、sum 和输出。每个 K/V 字节在每个 16 token 的 query tile 里只读一次。因果掩码打在任何越过该 query 的 key 上，也就是对角块和不满的尾巴。head_dim 必须是 256。V100 上这个形状已经是默认：9B 在 905/1805/8000/16000/30000 的预填充 tok/s 是 221.3/219.9/203.2/182.8/154.7，对照 `attention_q8v` 的 213.3/203.9/153.2/114.2（旧内核 30000 超时）；27B 在 905/8000/16000 是 74.0/69.3/62.6，对照 71.4/53.7/40.9。贪心文本一致。`=0` 是 `attention_q8v`。`=2` 强制同一个 WMMA 内核。按 token 把切分套进 T=128 的块仍然关闭（V100S 上大约慢 10%，27B 905 token 63.6 对 71.7 tok/s）。短解码、f16、对称 Q8 也是旧内核。`ODINFER_CUDA_ATTN_SPLIT=0` 强制旧内核。短块仍是先写完 KV，再按 head × token 启动一次（`ODINFER_CUDA_ATTN_BATCH=0` 恢复逐 token）。计算量仍是 O(N^2)。`delta_recurrent_chunk` 对 T 是线性的：token t 依赖 t-1 的门控状态，循环留在 block 里面，每个 value head 一个 block，不是平方。并行 scan 是另一套算法。`head_k_dim * head_v_dim * 4` 不超过 64 KiB 时，这一块状态留在共享内存里（`delta_sm`），块结束再写回。Qwen3.5 的 128×128 状态是 64 KiB，超过默认的 48 KiB 动态上限，所以初始化会调 `cuFuncSetAttribute`；调用失败就继续用全局内存的内核。递推本身没改。解码 `T==1` 仍是单 token 内核。

`--max-ctx` 取 `min(请求值, GGUF context_length)`。Qwen3.5-27B 的原生长度是 262144。CUDA 路径里没有别的 8192 上限。KV 是 Q8_0 的 K（每 32 元 34 字节）加混合 V（40 字节）。27B（16 个完整注意力层，4 个 KV head，head_dim 256，kv_dim 1024）每个 token 37888 字节：8192 是 296 MiB，32768 是 1.16 GiB，262144 是 9.25 GiB。分数缓冲是每次启动里每个 query token 一行 `n_heads * seq_len` 个 float，上限 256 MiB，所以长上下文减少同一次启动的 token 数，而不是拒绝启动。切分的部分和是 `ceil(seq/32) * n_heads * head_dim` 个 float（32768 为 24 MiB，262144 为 192 MiB）。切分内核的共享内存是一块 32 token，27B 大约 25 KB，不是整段序列。RoPE 按位置现算，没有位置表。`.oikv` 头里的 `seq_len` 是 u32，必须和 `--max-ctx` 一致。内核的 `pos`/`seq_len` 是 32 位；切分的 `grid.y` 最多 65535 段（序列 2097120）。27B Q4 大约 16 GiB，加上这些缓冲，原生 262144 在 32 GiB 的 V100S 上还剩大约 6 GiB。9B（8 个完整层，同样的 kv_dim）每 token 减半，262144 的 KV 是 4.6 GiB。初始化会打印 `cuda: ctx=… kv=…`。前缀快照只拷每个完整注意力层里前 `n_pos` 个 Q8 位置（按位置排列，所以是连续字节），恢复时把尾巴清零。短的系统提示命中不会再搬整段 max-ctx。`n_pos` 等于 262144 时主机上仍大约是 10 GiB。预填充注意力的计算量仍是平方。在 `attention_q8v` 上，测得的 27B T=128 块在位置 1664 是注意力 324 ms、线性层 1423 ms，所以这个旧内核做 3 万 token 的预填充大约是 11 分钟注意力加 6 分钟线性层，大约 7k 之后注意力成为更大的一项。fp32 flash 不是默认。9B 上它在每个 128 token 的块里把每个 KV 字节大约重读 43 次，输给了 `attention_q8v`。旧内核在位置 15872 大约每个 KV 位置 72 µs（1147 ms）。`attention_q8v_wmma`（head_dim 256 且 sm_70 及以上时的默认；`ODINFER_CUDA_PREFILL_FLASH=0` 是回退）每个 16 token 的 query tile 只读一次，128 token 的块读 8 次，乘积走 tensor core。9B 位置 16000、8 个完整层、一块 T=128 的估算：`2 * 2 * 8 * 128 * 16 * 16000 * 256` = 268 GFLOP。V100 fp16 tensor 峰值大约 125 TFLOPS，80 个 SM 打满大约 2.1 ms。KV 流量大约 `16000 * 4 * (272 + 320)` = 每层 38 MB，乘 8 个 query tile、乘 8 层，大约 2.4 GB，900 GB/s 下大约 3 ms。softmax 和部分输出的写在这之上。峰值的 10% 到 25% 加上这部分流量大约 15–40 ms，中心约 25 ms，对照测得的 1147 ms。这是估算，不是 V100 计时。

稠密解码默认把注意力输出、线性层输出和 FFN down 直接累加进残差，下一次 RMSNorm 仍是单独的 `rmsnorm_q8`（9B 1.41 ms，27B 3.53 ms）。`ODINFER_CUDA_FUSE_NORM=1` 才写到旁路缓冲再跑 `residual_rmsnorm_q8`。V100S 上这条更慢（9B 1.99 ms，27B 4.96 ms），解码从 68.4 / 26.05 掉到 63.6 / 23.4 tok/s，所以默认关闭。RoPE 是 `qnorm_rope` 和 `rmsnorm_rope` 的收尾（旧的 `qnorm_extract`、`rmsnorm_k`、`rope_partial` 留给自测）。KV 写入、线性层 conv、L2 和 gated RMSNorm 整段 chunk 一次启动（`grid.y` 是 token；conv 在线程内按通道走完 T，因为状态要按顺序移位）。`0c7befd` 上每个解码 token 930 次启动，24 token 预填充 6178 次。`ODINFER_CUDA_PHASE=1` 会打 `nongemv`、`gap` 和 `cuda: phase kernels`（最大的八个内核合计）。`8cd511d` 把 Q4/Q6 的 x 在 warp 里 shuffle 广播。短上下文的 linear 从 28.4 ms 变成 28.6 ms，905 token 的解码从 27.58 掉到 26.42 tok/s，所以默认又改回每行自己加载 x。`ODINFER_CUDA_GEMV_PIPE=1` 是另一个核：在算当前 super-block 之前发出下一块，仍然是每 warp 4 行。`Q4_NC` 和 `Q4_ILP` 仍关闭。`rmsnorm_quant_q8` 保持 128 线程的平方和。每个线程接着归一化一组 32 个数、写回，并从这些寄存器写出 q8，不再回读 `out`。#23 上这个内核是 2.06 ms/token。#24 上 T=1 的 nongemv 是 wall 减线性层，30.40 − 23.36 = 7.04 ms。phase 里最大的几项仍是 `rmsnorm_q8`（128 线程一个 block，求和顺序没变）、`delta_sm`、注意力。`delta_sm` 现在把 128 步的状态循环按 4 展开，加法顺序不变。把 norm 合进 GEMV（`FUSE_NORM`）和关掉 smem delta 都更慢，所以保持原样。27B 解码线性层 23.36 ms 大约读 16.2 GB 的 Q4/Q6 权重，约 694 GB/s，是 1134 GB/s 峰值的 61%。Q5_K 预填充走和 Q4/Q6 一样的 128×128 wgemm。其余种类仍走逐元素的 `tc_*`，profile 名字是 `tc_q5k`、`tc_q8`、`tc_f16` 或 `tc`。`ODINFER_CUDA_DELTA_SM=0` 让解码 delta 回到全局内存。9B 的 Q4 GEMV 大约读 3.0 GB 权重，6.01 ms 约 500 GB/s，是 1134 GB/s 峰值的 44%。默认 Q4 加载是普通 `ld.global`（与 8975985 相同）。`ld.global.nc` 更慢（9B 6.01 → 6.50 ms，27B 19.57 → 22.4），只在 `ODINFER_CUDA_Q4_NC=1` 时启用。`ODINFER_CUDA_Q4_ILP=1` 选择 `gemv2_q4x`：每 warp 8 行、普通 128 位加载、每次迭代两个 super-block。`--bench-gemv` 把它印成 `r8`。它不是解码默认。27B 在 16000 的一步解码是 47.1 ms，线性层大约 30 ms（`6cb0e7e`）。短上下文 24.7 tok/s，大约 40 ms。16 GB 权重用 30 ms 大约是 533 GB/s，900 GB/s 的下限大约是 18 ms。这和 9B Q4 GEMV 只到 1134 GB/s 峰值 44% 是同一档。`Q4_NC`、`Q4_ILP`、`GEMV_ROWS=1`、`FUSE_NORM` 都已经测过更慢，所以都不做默认。Q6 只在 K≥16384 或行数>8192 时把量化块经对齐加载放进共享内存（27B 的 down 和 logits）；9B 的 down（4096×12288）仍直接加载。`ODINFER_CUDA_Q6_STAGE=0` 关掉，`=1` 全开。

16 MiB 以下的 Q8_0 和 Q5_0 留在标量内核。2048×2048 上新内核更慢（Q8_0
118.5 对 121.8 GB/s，Q5_0 49.2 对 69.7）。9B 和 27B 的热权重是 Q4_K 和
Q6_K，不走这条。`ODINFER_CUDA_GEMV=new` 让这些小形状也用新内核。

激活按 32 个一组量化成 q8（有符号 int8，scale 是 `amax/127`），每条向量一次。
Q4_K 和 Q5_K 另外记下每 16 个 float 的精确和，`dmin` 那一项不量化。单行
RMSNorm 在同一个内核里写出这份 q8；Q、K、V 和两路 FFN up 共用它。SwiGLU
和多行 norm 会使缓存失效。没有带宽内核的种类（F32、Q2_K、Q3_K、Q8_K、
Q4_0 这一族）仍走标量 GEMV。K 不是量化块的整数倍时也走标量。

块大小是 `ODINFER_CUDA_GEMV_WARPS` 个 warp（默认 8，范围 1..16）。四行时
每块 32 行；一行时每块的每个 warp 一行。V100S 有 80 个 SM、每个 SM 64 个
warp 槽。按网格自动减少 warp 时，把 `GEMV_WARPS=8` 钉死会让 27B 解码在第 7
个 token 遇到 EOS，解码 tok/s 也没有变化，所以 warp 数就是这个环境变量。
Q5_K 解码同样是 q8 激活加 `dp4a`（`gemv2_q5_k`），不切分 K。

稠密 `T==1` 时，注意力输出、FFN down 和线性层输出直接加进残差。
`ODINFER_CUDA_FUSE_NORM=1` 才改走 `residual_rmsnorm_q8`。
预填充（`T>1`）和标量回退仍用单独的加。logits 的 GEMV 不累加。

| 变量 | 默认 | 作用 |
|------|------|------|
| `ODINFER_CUDA_GEMV` | auto | `old` 或 `0`：标量 GEMV。`new`：小 Q5_0 / Q8_0 也用带宽内核。不设：带宽内核，但 16 MiB 以下的 Q5_0 / Q8_0 用标量 |
| `ODINFER_CUDA_GEMV_ROWS` | `4` | Q4_K / Q6_K 每个 warp 的行数。不设是四行。`1` 选更慢的一行内核 |
| `ODINFER_CUDA_GEMV_WARPS` | `8` | 带宽 GEMV 每块的 warp 数（1..16） |
| `ODINFER_CUDA_GEMV_MMVQ` | 开 | Q4_K（`gemv2_q4w`）、Q6_K（`gemv2_q6w`）、Q5_K（`gemv2_q5w`）每个行一个 warp，沿 K 做 128 位加载，不切分 K。K≥5120 的 Q4 gate+up 是 `gemv2_q4wp`。9B 的 K=4096 仍是两次 `q4w`。#24 上 T=1 wall 30.40 ms，线性层 23.36；27B 解码 905/8000/16000 是 32.45/31.09/29.54。`=0` 回到 `gemv2_q4` / `gemv2_q4p` / `gemv2_q6s` / `gemv2_q5_k` |
| `ODINFER_CUDA_GEMV_Q5` | `w` | MMVQ 开着时的 Q5 路径。`w` 是对齐的 `gemv2_q5w`。`k` 是 4 行的 `gemv2_q5_k`。`w2` 是 `gemv2_q5w2`：连续两行，x 只加载一次。V100 复测 `q5k` 2.94–2.95 ms，`q5w` 3.10–3.13，`q5w2` 3.04–3.09，解码 tok/s 没有变化。K≥5120 的 Q5 gate+up 走 `gemv2_q5wp`，除非设成 `k`。27B 的 gate+up 是 Q4，不会走到这个配对。`--bench-gemv` 会打 `q5w`、`q5w2`，以及 `q5wp` 对两次 `q5w` |
| `ODINFER_CUDA_GEMV_SPLITS` | auto | 不设：行数不超过 8192 时 4 路；若拆成两半仍有 12 个 superblock，并且 warp 数不到每 SM 32 个，则改成 2 路。27B w2 是暂存加这路 2 分（bench `q6s2`）。9B w2 是直接加载的 2 路（`s2`）。`r4` 不是解码启动。`=4` 恢复固定 4 路 |
| `ODINFER_CUDA_GEMV_FUSE` | 自动 | 行数超过 8192 且 K≥5120 的 Q4 gate+up 一次扫完 K（`gemv2_q4p`，27B 的 w1）。9B 的 w1 是 K=4096，仍是两次启动：配对把 16k 解码从 58.84 降到 57.11 tok/s。`=1` 强制所有高行数 Q4 都配对。`=0` 是两次 GEMV。`--bench-gemv` 在 w1 上打印 `pair` 和 `two r4` |
| `ODINFER_CUDA_LAUNCHES` | 关 | `=1` 时每次 forward 打印内核启动次数 |
| `ODINFER_CUDA_PEAK_GBS` | `1134` | `--bench-gemv` 用来算峰值百分比 |
| `ODINFER_CUDA_GRAPH` | 关 | `=1` 把稠密 `T==1` 收成一张 CUDA graph，注意力分块数变化时重捕获。位置用图所在 stream 上的异步拷贝。MoE 和预填充仍即时。`ODINFER_CUDA_PROFILE=1` 会关掉 graph |
| `ODINFER_CUDA_PROFILE` | 关 | `=1` 打印这次 forward 每个内核的 GPU 时间，最慢的在前。8192 个 event 的环在写满时按内核名汇总，长 prompt 不会被截断 |
| `ODINFER_CUDA_PHASE` | 关 | `=1` 每次 forward 打一行 `linear attn kv norm delta other nongemv gap` 毫秒，再打一行 `cuda: phase kernels`（最大的八个内核），整段预填充再打一行带 `nongemv` 的 `cuda: phase prefill` 合计。`nongemv` 是线性层以外的 GPU 时间。`gap` 是 wall 减去 event（启动和主机时间）。`T=1` 是一个解码 token。和 `PROFILE` 共用 event。这一进程的 tok/s 不是生产数字。不改内核 |
| `ODINFER_CUDA_ATTN_SPLIT` | 开 | 只用于 mixed-V 解码（`T==1` 且 `npos>=128`）。32 token 一块，每个 KV head 一块，组内 query head 共用 K/V，再合并 softmax。不决定预填充内核。`=0` 解码走单 block 内核。相对 `attention_q8v` 允许 `2e-2 + 1e-3*|ref|`，含 npos 300、1800、8192、32768。CUDA graph 在分块数变化时重捕获 |
| `ODINFER_CUDA_PREFILL_FLASH` | WMMA | mixed-V 预填充（`T>1`）。sm_70 及以上、head_dim 256、每个 KV 头最多 8 个 query 头时，不设置就是 `attention_q8v_wmma`。`=0` 是 `attention_q8v`。`=1` 是 `attention_q8v_flash`（fp32 在线 softmax）。9B V100S 上它更慢（16000 token 80.7 对 114.2 tok/s）。`=2` 强制 WMMA。这一版一个 block 负责 32 个 query token 和 64 个 key：K 反量化成 fp16 后给整组 query 用，然后同一块缓冲换成 V。每个 Q8_0 K 块和每个混合 V 块是两次 16 字节加载，按元素顺序写回；混合 V 的 outlier 仍在缩放之后写回它的下标。8 个 warp 各算一块 16×16 的分数并沿整个 head 维累加。起点已经超过本组最后一个 query 的 tile 不再加载。共享内存 91136 字节。还没在 V100 上计时。上一版 16 query、32 key 的数字是：9B 16000 token 182.8 对 114.2，27B 62.6 对 40.9；e452657 上 27B 16000 的注意力桶是 43557 ms。相对 `attention_q8v` 允许 `1e-1 + 5e-2*|ref|`。解码仍走切分内核 |
| `ODINFER_CUDA_PREFILL_GEMM` | 开 | Q4_K / Q6_K 预填充（`T>1`，K 是 256 的倍数），Q5_K 在 `T` 达到 `WGEMM_MIN` 之后也走 `wgemm`（fp16 WMMA，128×128）。更短的 Q4/Q6 走 `mmqk`。更短的 Q5_K 仍是逐元素 `tc_q5k`：`mmqk_q5_k` 把 T=19 的线性层从 406.76 ms 拉到 441.98（Q5 那段从 58.40 到大约 94）。`ODINFER_CUDA_PREFILL_LINEAR=mmqk` 只让 Q4/Q6 走 dp4a。旧的 Q4/Q6 `mmqk` 在 V100、chunk 512 上：9B 905/1805/8000/16000 是 371.70/376.91/337.20/290.49 tok/s，27B 905/8000/16000 是 124.00/112.97/97.43。`=0` 跳过两者，仍是 `tc2` / 逐元素 `tc`。`wgemm` 相对 fp32 CPU GEMV 允许 `5e-2 + 5e-2*|ref|`。`mmqk` 相对 `gemv2` 允许 `2e-3 + 2e-3*|ref|` |
| `ODINFER_CUDA_PREFILL_LINEAR` | `wgemm` | `=mmqk` 选择 q8 dp4a。不设时，`T` 达到 `ODINFER_CUDA_WGEMM_MIN` 才走 `wgemm`。`ODINFER_CUDA_PREFILL_GEMM=0` 时忽略 |
| `ODINFER_CUDA_WGEMM_MIN` | `128` | 低于这个 token 数的 Q4_K / Q6_K 预填充走 `mmqk`。Q5_K 仍走 `tc_q5k`（#24 上 T=19 是 58.41 ms）。不满的块会跳过起点已经超过 `T` 的 16 列 MMA。`=0` 让所有 `T>1`（含 Q5_K）都走 `wgemm`。256 token 的双缓冲放不进 96 KB |
| `ODINFER_CUDA_WGEMM_BK` | `64` | `=32` 启动 BK=32（步长 40）。K 缓冲是 40960 字节，但收尾要把 65536 字节写进同一块共享内存，所以启动大小是 65536。`36447cb` 只给了 40960，写越界。65536 在 96 KB 里放不下两个 block。默认 BK=64 内核回到 #24 的源码：模板化的那个版本就是 16k 线性层从 51064 ms 到 52588 ms 的回退。512 token 一块大约 24.9 TFLOP，#24 上 1614 ms，约 15.4 TFLOPS，是 125 TFLOPS fp16 tensor 峰值的 12%。差距在 HMMA 前面的 fp16 反量化 |
| `ODINFER_CUDA_ATTN_PARTS` | 关 | 不设时仍按 80 个 block 的目标，并且一段 KV 超过 2048 token 就再切，上限是 8 和 KV tile 数。上下文开头的 512 token 仍是 2 段。落到 16000、4 个 KV head 的 512 token 块是 8 段。整段 16k 预填充强制 8 段时，注意力从 14137 ms 降到 12467 ms。`=4` 或 `=8` 仍是强制段数 |
| `ODINFER_CUDA_ATTN_QTILE` | `1` | 每个 KV tile 一个 query tile（`attn_wmma`），就是 `965afd1`。`=4` 让四个 tile 共用一次反量化（`attn_wqr`）。V100 上结果精确（max abs 0）但更慢：16k 注意力 12643 → 15798 ms，pos 0 的 512 token 块 23.45 → 55.63 ms，pos 15360 的块 767 → 832 ms。16k 预填充 218.52 → 209.52 tok/s。`=1` 回到 218.48 / 29.53，注意力 12645。`=2` 没有跑过。只有一个 query tile 的块无论设多少都走 `attn_wmma` |
| `ODINFER_CUDA_WGEMM_X` | 开 | 每个 wgemm 的激活打成 fp16 一次。`=0` 在核内读 fp32。Q5_K 也走这块 |
| `ODINFER_CUDA_GEMV_PIPE` | 关 | `=1` 在算当前 Q4/Q6 super-block 之前发出下一块。不设则每行自己加载 x |
| `ODINFER_CUDA_DELTA_SM` | 开 | 解码 delta 把 128×128 状态放进共享内存。`=0` 走全局内存 |
| `ODINFER_CUDA_PREFILL_FUSE` | 关 | `=1` 让 gate+up 和 K+V 共用一份 q8。在 V100 上差不多持平（9B 8000 token，关掉 332，打开 333），线性层还更慢，所以默认关 |
| `ODINFER_CUDA_MMQK_ROWS` | `4` | 默认 Q4 一块 32 行。`=8` 选 64 行，在 V100 上更慢。Q6 两种都是 32 行 |
| `ODINFER_CUDA_ATTN_LONG` | 开 | 混合 V 解码，`npos>=2048`，head_dim 256，每个 KV 头 4 或 6 个 query 头。一块 128 个位置，K 对这一组只反量化一次，V 每个位置一次，然后走和 32 宽拆分一样的 reduce。`=0` 在任何长度都用 32 token 的块 |
| `ODINFER_CUDA_DELTA_WMMA` | 开 | 128×128 的预填充状态走 16 token 的门控 delta 分块（WY 用 fp32，块内状态乘用 WMMA fp16）。27B 上文本一致，8000 token 的 delta 桶从 13237 ms 降到 11057，预填充 115.27 / 100.09 tok/s（8000 / 16000），对照关掉时的 112.05 / 97.61。`=0` 是 fp32 递推。自测以前喂的是没归一化的 k（范数约 40），WY 代入溢成 NaN；现在先做 L2 归一化并把 q 乘 `1/sqrt(head_k)`，允许误差仍是 2e-3 |
| `ODINFER_CUDA_PREFILL_CHUNK` | `512` | 每次预填充 forward 的 token 数。16 的倍数，16 到 512。标量路径上 9B 16000 token、块 128/256/512 是 183/207/213 tok/s。放不下就按 256、128、32、8、1 退，不会比请求的更大 |
| `ODINFER_PHASE` | 关 | `=1` 每次采样打印 JSON grammar 和 tool-call grammar 的 CPU 毫秒。没装 grammar 时为 0 |
| `ODINFER_CUDA_FUSE_NORM` | 关 | `=1` 才把投影加进 `residual_rmsnorm_q8`。V100S 上比累加 GEMV 慢 |
| `ODINFER_CUDA_Q4_NC` | 关 | `=1` 用 `ld.global.nc` 加载 Q4。V100S 上比普通 `ld.global` 慢 |
| `ODINFER_CUDA_Q4_ILP` | 关 | `=1` 用 `gemv2_q4x`（每 warp 8 行、128 位加载、每次两个 super-block）。`--bench-gemv` 印成 `r8` |

`cuda_selftest` 用 CPU fp32 点积对四行内核做 Q4_K、Q6_K、Q8_0、Q5_0、Q5_K
的 64×512 和 6×256（行数不是 4 的倍数）。Q4_K 再跑 `gemv2_q4nc_k` 和 `gemv2_q4x_k`（8 行，K 至少两块时含 split-K）。Q4_K 和 Q6_K 再跑一行内核，包括
`accumulate=1`（输出先写成 0.25，期望 `ref+0.25`）。容差是 `1 + 0.05*|ref|`。
和 CPU 的差来自 `x` 的 q8；半字节或 scale 对调会差到 `|ref|` 这一量级。

```sh
bin/cuda_kernel_test
bin/cuda_kernel_test --bench-gemv
ODINFER_CUDA_GEMV_WARPS=4 bin/cuda_kernel_test --bench-gemv
```

`--bench` 先打 tensor core 的 TFLOPS，再打同一张 GEMV 表。`--bench-gemv`
只打 GEMV：毫秒、GB/s、占峰值的百分比。四行内核是 `r4`，有一行内核的形状再打
`r1`，标量内核是 `old`。形状是 `T=1` 的 9B / 27B（w1、wq、wo、w2、logits）
以及 2048 的 Q8_0 / Q5_0 和 5120 的 Q5_K。
GB/s 只算这一次启动读的权重字节。`x` 的 q8 在计时循环外面，和缓存命中一样。
两行 logits 大约 1 GB，分配失败会打印并继续其余形状。设备不是 V100S 时用
`ODINFER_CUDA_PEAK_GBS` 改峰值。

端到端解码，标量内核对比默认。冒烟的 prompt 很短，这是短上下文的数（9B
上以前是 32.7 tok/s）。990 token 的预填充是另一次测量：那个长度上每个完整
注意力层的 KV 只有几 MB，权重大小仍然是整个模型。

```sh
ODINFER_CUDA_GEMV=old bin/cuda_smoke "$MODEL9B" 16 2048
bin/cuda_smoke "$MODEL9B" 16 2048
ODINFER_CUDA_GEMV=old bin/cuda_smoke "$MODEL27B" 16 2048
bin/cuda_smoke "$MODEL27B" 16 2048
ODINFER_CUDA_LAUNCHES=1 bin/cuda_smoke "$MODEL9B" 8 2048
```

初始化日志会写 GEMV 的行策略，以及 `graph off`，除非设了 `ODINFER_CUDA_GRAPH=1`。

稠密解码（`T==1`，不是 MoE）默认即时启动。V100S 上捕获的 graph 比即时慢
（9B 57.6 对 63.0 tok/s）：位置更新走的是 legacy stream 上的
`cuMemcpyHtoD`，每个 token 都同步整卡，而且捕获时清零不在捕获 stream 上，
split-K 被丢掉了。`ODINFER_CUDA_GRAPH=1` 才打开。打开之后位置是 graph
所在 stream 上的异步拷贝，驱动有 `cuMemsetD8Async` 时 split-K 会进图。
`g_decode_pos` 找不到时改用一个设备 int，不再因此关掉 graph。预填充传入
真实位置，并且保持即时启动。捕获或实例化失败时，这个 token 改走即时路径，
之后也不再捕获。`ODINFER_CUDA_PROFILE=1` 关掉 graph，并按从慢到快打印每个
内核的 GPU 时间（名字、次数、毫秒），以及墙钟和内核时间之和。采样仍在
CPU 上，在 logits 拷回之后。graph 和 event 符号是单独查找的；驱动里没有
它们时，graph 和 profiler 关掉，GEMV 照常加载。

## 检查脚本

`scripts/cuda_check.sh` 会编 CUDA 测试和冒烟二进制，把内核源编译成
`compute_70` 的 PTX（这一步不需要 GPU），机器上能看到 GPU 时再跑内核自测和
端到端生成。

```sh
./scripts/cuda_check.sh
```

模型按下面的顺序找，命中即停（`MODEL` 优先）：

| 路径 | 大小 | 覆盖 |
|------|------|------|
| `$MODEL` | 加载器能读的任意 Qwen3.5 GGUF | 你指定的文件 |
| `models/Qwen3.5-0.8B-MoE-A0.6B-Q4_K_M.gguf` | 524 MB | MoE，混合 Q4_K / Q8_0 / Q5_0 |
| `$HOME/.odinfer/models/Qwen3.5-0.8B-MoE-A0.6B-Q4_K_M.gguf` | 524 MB | 同上，安装器目录 |
| `models/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf` | Mac 基准，sha256 前缀 `a692de95` | 稠密 Q4_K_M（Q4_K + Q6_K + F32 norm） |

冒烟参数：`SMOKE_TOKENS` 默认 16，`SMOKE_CTX` 默认 2048。没有模型文件时内核
测试照跑，冒烟跳过，除非 `CUDA_CHECK_REQUIRE_SMOKE=1`。

脚本会打印 `sha256 <十六进制> <路径>`。Mac 上的 9B 基准文件以前缀
`a692de95` 开头。`mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-GGUF`
当前的 Hugging Face `main`（提交 `29319295d35e` 到 `a713868aa6cd`）是另一份
量化：sha256 前缀 `b4363b57`，5780092768 字节。不能把这个提交钉成基准，
因为 `a692de95` 已经不在该仓库现存历史里。heretic 9B 的哈希不是
`a692de95` 时，`cuda_check.sh` 拒绝冒烟，除非 `CUDA_CHECK_ALLOW_REQUANT=1`。
`scripts/install.sh --model qwen9b` 仍会拉 HF `main`，打印 sha256，并在
不是 `a692de95` 时给出警告。

脚本包住的命令：

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

`cuda_kernel_test` 把 GEMV/MMQ、RMSNorm、RoPE、SwiGLU、混合 V 的 KV 写入和
注意力、门控 delta 步，和 `packages/ggml` 与 `packages/qwen3_5/ops.odin` 的
CPU 参考对比。`cuda_smoke` 打印 prefill 和 decode 的 tok/s。采样仍用现有的
CPU sampler（冒烟工具温度是 0）。

## 已移植

GEMV（解码）和 MMQ（预填，最多 8 个 token 一块，任意 `T`，输出按 token
主序），fp32 累加。默认解码 GEMV 是 Q4_K、Q5_K、Q6_K、Q8_0、Q5_0 的带宽
内核；每种量化仍保留一行一个 warp 的标量内核：

F32、F16、Q8_0、Q4_0、Q4_1、Q5_0、Q5_1、Q2_K、Q3_K、Q4_K、Q5_K、Q6_K、Q8_K。

另外：embedding 用 CPU 反量化再上传，RMSNorm，按头的 Q RMSNorm，部分 RoPE，
解码和预填的注意力，f16 / 对称 Q8_0 / 40 字节混合 V 的 KV 写入和注意力，
注意力门，SwiGLU，残差，深度卷积 conv1d+SiLU，l2norm+缩放，逐 token 门控
delta，RMSNormGated，MoE softmax top-k、逐专家 GEMV，以及可选的共享专家。

默认的 9B Q4_K_M（Q4_K、Q6_K、F32 norm，embedding 常见 Q8_0 或 Q6_K）和
0.8B MoE（Q4_K、Q8_0、Q5_0、F32）都在这个集合里。初始化会扫每一层投影，
遇到名单外的量化种类就失败，并打出张量名。Embedding 可以是 `ggml` 在 CPU
上能反量化的任意种类，包括 IQ；输出投影不行。

## 未移植

IQ1_S、IQ1_M、IQ2_XXS、IQ2_XS、IQ2_S、IQ3_XXS、IQ3_S、IQ4_NL、IQ4_XS、Q8_1。

也不在这个后端里：Metal 按形状特化的 GEMV（`QGEMVSPEC`）、宽 N 自动调优和
tune 缓存、fast-math、MoE 分组 GEMM（`forward_gpu_batch_moe`）、
`engine_verify_batch`（投机校验）。CUDA 的单行 RMSNorm 会写下一次 GEMV 读的
q8 缓存。MoE 预填的专家走逐 token GEMV；Q4_K / Q6_K 的注意力投影和稠密模型用同一套预填充 GEMM。稠密预填按
`ODINFER_CUDA_PREFILL_CHUNK`（默认 512，激活缓冲分配失败时更小）分块。解码用 GEMV，不用 MMQ。

权重在初始化时拷上设备一次（GGUF mmap 的 `cuMemcpyHtoD`，再加上主机上已经
反量化的 MoE router 和共享专家）。5.6 GB 的 Q4_K_M 9B 放得进 32 GB 的
V100S；KV 另算，布局和 Metal 的 `KV_Q8` / `KV_V` 相同，快照可以互换。

CUDA 里的 sigmoid 和 softplus 对齐 `ops.odin`（稳定 sigmoid，softplus 在
±20 截断）。Metal fast-math 的 `1/(1+exp(-x))` 可能差几个 ulp。内核测试对
的是 CPU 参考，不是 Metal 的 golden。GEMV 容差大约是
`2e-2 + 1e-3 * |ref|`，因为 warp 归约的求和顺序和 CPU 从左到右不一样。
