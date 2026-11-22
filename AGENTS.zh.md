# AGENTS.zh.md — AI agent 速查（构建 / 测试 / 使用）

这是给 AI coding agent（Cursor / Claude Code 等）的项目速查。读完即可独立
构建、测试、运行和改动本项目。

[English version / 英文版](AGENTS.md)

## 一句话架构

Apple Silicon 上的 Qwen3.5 级推理引擎（纯 Odin，无 Python 运行时依赖）。
**产品定位：Qwen3.5 及以上**——服务端（主形态）只跑 `packages/qwen3_5` 引擎
（混合注意力 + gated delta），这是有意为之。`packages/infer`（Qwen3 密集）
仅作 CLI 的历史兼容路径，不要在产品文档中扩展。共享 `packages/ggml`
（GGUF 解析/量化）。服务端在 `server/serve_harness/serve.odin`，CLI 在
`cli/main.odin`。

## 发布

- 零编译二进制由 GitHub Actions 构建：`.github/workflows/release.yml`
  （macos-15 arm64 runner，Odin 锁定 dev-2026-07，`./build.sh`，推 `v*` tag
  自动 `gh release create` 发布 bin/odin-serve + bin/odin-cli）。
- 改发布流程后：push tag 触发；`gh run watch` 观察。

## 安装与运行（零编译——agent 先读这里）

无需工具链：安装脚本下载 release 二进制（可选加模型）并打印可直接运行的命令。

```sh
bash scripts/install.sh --model moe   # 二进制 + 524MB MoE 测试模型
export PATH="$PATH:$HOME/.odinfer/bin"
odin-serve "$HOME/.odinfer/models/Qwen3.5-0.8B-MoE-A0.6B-Q4_K_M.gguf" --mcp --metal
```

- `--model qwen9b` 改下载 5.6GB 的 heretic 模型。
- `--prefix` / `MODEL_DIR` 覆盖安装路径。
- 手动等价做法：curl 下载
  `https://github.com/vajraimb/odinfer/releases/latest/download/` 下的两个二进制。

下面的源码构建仅供开发用。

## 构建与测试（必跑顺序）

```sh
./build.sh        # 全量：bin/odin-serve + bin/odin-cli + 全部单测（约 1-2 分钟）
```

> ⚠️ 本机（vajra 的 M3 Air）Homebrew 默认 `odin` 是 `2026-11a`：其
> `vendor:darwin/Metal` 的 `MTL.Size` 字段从 `Integer` 改成了 `UInteger`，
> 本仓库源码（锁定 `dev-2026-07`）用它编译直接报错。本机构建/测试请用
> `PATH=/opt/homebrew/Cellar/odin/2026-07a/bin:$PATH ./build.sh`（该 keg
> 已装好）；test_mcp.py 等会自行重建二进制的脚本同样要带这个 PATH。

单测单独跑：

```sh
( cd packages/qwen3_5 && ./build.sh )        # 11 个单测（含 3 个 MoE 数学测试）
( cd packages/tokenizer && odin test . -define:ODIN_TEST_THREADS=1 )
# ⚠️ tokenizer 测试必须单线程：5 个并行 runner 各建 248k 词表映射，内存紧张时
#    会 OOM 段错误——这是环境性已知问题，不是代码 bug。
```

集成测试（会加载真实模型，每个 ~30-60s）：

```sh
python3 server/serve_harness/test_mcp.py    # MCP 全握手（需 Qwen3.5 GGUF）
python3 server/serve_harness/test_tune.py   # 首启调优 + 缓存复用
# 通用环境变量：MODEL=<路径> 覆盖默认模型
```

> 集成测试的二进制缓存在 `server/serve_harness/.test-serve.bin`；**只在
> serve.odin 或 packages/*.odin 更新时自动重建**（按最新 mtime 判断）。

Linux 上 `./build.sh` 走 CPU（`metal.odin` 带 `#+build darwin`）。CUDA 是单独的编译开关，不是默认：

```sh
./scripts/cuda_check.sh
# 或 -define:ODINFER_BACKEND="cuda"  -collection:cuda=packages
```

见 [docs/CUDA.zh.md](docs/CUDA.zh.md)。热路径是 `when GPU_METAL` / `when GPU_CUDA`，同一个二进制不会在两者之间分支。

## 运行

```sh
# HTTP 服务（开发模式）
MODEL=./models/<qwen3.5-gguf>.gguf ./serve.sh 9748 8192
# 端点：POST /v1/chat/completions、/api/chat、/v1/responses(SSE)、GET /api/state

# MCP 模式（stdout 是协议通道，禁止往 stdout 打日志）
bin/odin-serve <model.gguf> --mcp --metal --max-ctx 4096

# CLI（自动识别 Qwen3 / Qwen3.5）
bin/odin-cli <model.gguf> -g 1 -c 2048
```

## 模型文件

- `models/` 是 gitignored 软链目录（见 models/README.zh.md）。
- 默认基准模型：`models/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf`
  （→ ollama blob 软链，Qwen3.5，5.6GB）。
- MoE 测试模型：`models/Qwen3.5-0.8B-MoE-A0.6B-Q4_K_M.gguf`（524MB，
  bayliner1980）——tools/moe_smoke.odin 用它。
- 快速冒烟用 0.6B：任意 `Qwen3-0.6B-Q4_K_M.gguf`（Qwen3，历史兼容路径，
  加载 ~1s）——仅 CLI；serve 只认 Qwen3.5，产品定位即 3.5+。

## 关键约定（改代码前必读）

1. **诊断一律 stderr**：`fmt.eprint*`。`--mcp` 模式下 stdout 只允许协议消息；
   库层启动打印已迁到 stderr，不要回退。
2. **fmt.tprintf 的 `{`/`}` 是动词**：JSON 模板必须写 `{{`/`}}`。
3. **json_escape 自带外层引号**（输出 `"escaped"`），调用方不要再加引号。
4. **集合导入**：`import q35 "qwen3_5:qwen3_5"`，构建必须挂 6 个
   `-collection:<name>=packages`（ggml/infer/qwen3_5/qwen3_5_tokenizer/
   sampler/tokenizer），见 build.sh 的 COLLECTIONS。
5. **内核 A/B 开关**（env，默认全关，均为实测回归项）：`QGEMV3=1`
   （llama.cpp 行切块移植）、`QGEMV2R=1`（2 行/线程）、`QGEMVNORM=1`
   （norm 融合）。
6. **Profiling/消融构建**（`when #config(ODIN_PROF, false)`）：QPROF_NOLIN/
   QPROF_NOFULL 环境开关与 `/tmp/qprof_flags` 免重启切换文件（noout/nomlp/
   nosmall/nonorm/noattn/nodelta，每 token 读一次）只存在于
   `-define:ODIN_PROF=true` 构建——发布二进制零消融代码（strings 已验证）。
   测性能时用 profiling 构建；不要依赖这些开关在发布版生效。
6. **调优缓存**：`~/.odin-infer/<model>.tune`（`ODIN_TUNE_DIR` 覆盖，
   `ODIN_TUNE=0` 关闭）；改了内核性能特征后删除缓存强制重测。
7. **KV 持久化**：`KV_PERSIST=0` 关闭，`KV_PERSIST_DIR` 改目录；跨 max-ctx
   不兼容（seq_len mismatch 属正常日志）。
8. **不要提交二进制**：构建产物在 `bin/`（已 ignore）；历史上误提交过
   `tokenizer`/`serve` 两个编译产物，已清理并加了 `/tokenizer`、`/serve`
   到 .gitignore。

## MoE（qwen35moe）

- CPU + Metal 双路径已支持。Metal MoE：GPU router（moe_gemv_f32 +
  moe_softmax_topk，每层一次同步）、专家 GEMV 复用生产内核（张量基址+专家
  偏移）、moe_combine 加权求和；共享专家经 F32 上传走 moe_gemv_f32。
  MOE_CPU=1 强制走 CPU。MoE 批量 prefill（forward_gpu_batch_moe，分组
  GEMM）自 v0.3.0 起为默认 Metal 路径（三处叠加 bug 已修，与逐 token
  max_abs 8e-4），MOE_BATCH=0 退回逐 token 做 A/B。
- Metal IQ/k-quant GEMV 内核：gemv_q2_k/q3_k/q5_k/q8_k + 9 个 IQ 内核
  （生产形状）。codebook 表在 packages/qwen3_5/msl_tables.odin，由
  tools/gen_msl_tables.py 从 packages/ggml/tables.odin 生成（唯一事实源）；
  build.sh 自动重生成，CI 用 --check 防漂移。不要手改 msl_tables.odin。Qwen3.8-27B 可 GPU 解码
  （~0.4-1.3s/token）；ctx 8192 已验证（Q8_0 KV 0.27GB，总占用 7.7GB），
  ctx 32768 kv 1.14GB（总占用 8.56GB）。
- KV 缓存默认 K=Q8_0、V=混合块（KV_Q8=0 退回旧 f16；KV_V=q8 让 V 也走
  对称 Q8）。K 布局 position-major [n_full][seq_len][kv_dim/32 块][34B]。
  V 混合块 40B：块内绝对值最大的一个元素存 f16，其余 31 个按剩余 amax 做
  Q8——避免一个离群值（V amax≈85）把整块尺度拉开。CPU Run_State 保持
  fp32、在快照边界转换（与 store 内核逐位一致）。.oikv 仍是 KV_VERSION=2
  （磁盘体是 f32；v1 文件明确拒绝）。对称 Q8 V 的 logits 噪声 ~1e-1 仍是
  KV_V=q8 的预期行为。
- dense 批量 prefill（enc_mm）仅支持 Q4_K/Q6_K；model_batch_mm_ok 把关，
  混合量化模型回退逐 token Metal。IQ 批量 prefill 已实现（见下一条）。
- dense 批量 prefill（enc_mm）现已支持 13 种量化——Q4_K/Q6_K/Q8_0/Q2_K/Q3_K
  + IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS（v3 形状 MMQ
  内核，dequant 与 GEMV 一致）。Q4_K/Q6_K 在 token 数能整除时走宽 N tile
  （Q4 32×32×64、Q6 16×64×64；QMMQN=0 退回 BN=8）。dense 预填充默认直接采用
  MMQ 投影（QGEMVEXACT=1 才用逐 token GEMV 覆盖，换回与解码逐位一致）。27B 的 Dynamic 混合量化 prefill 已批量化
  （~1.8x GPU，30-token prompt 227 vs 420 ms/tok）；批量 vs 逐 token prefill
  max_abs ~5.7e-3、解码步 ~6.6e-3（64 层 MLP 舍入残留，argmax 一致）。
  MoE 预填充仍逐 token GEMV 覆盖注意力列。八种 IQ 在 M/N/K 整除 32/32/64
  时走 BM=32 BN=32 BK=64（QMMQN=0 退回 BN=8）。27B IQ2_XXS：448 token 对齐块
  60.7→53.6 ms/tok，499 token 预填充墙钟 31.0→28.1 s；贪心两边都是 Four / fox。
  GGUF 检查工具：tools/inspect_iq_gguf.py + tools/gguf_meta.py。
- 前缀缓存/会话连续性：两条 HTTP 路径（OpenAI 与 Ollama）都跨请求复用引擎
  状态——延续（prompt 是上次 token 序列的精确前缀延伸时只 prefill 新尾）、
  系统边界快照（恢复后只 prefill 后缀）、否则全量 reset。OpenAI 路径在
  v0.3.x 周期补齐（MCP 的 odin.chat 自动继承）。计数器：/api/state 的
  pc_continuation_hits / pc_snapshot_hits。
- 专家张量布局（已对真实 unsloth 35B-A3B 头部验证）：`ffn_gate_inp`
  [dim, n_expert]，`ffn_up_exps`/`ffn_gate_exps` [dim, n_ff_exp, n_expert]，
  `ffn_down_exps` [n_ff_exp, dim, n_expert]——ne0 是收缩维，量化粒度 = 一行
  ne0。
- 路由数学与 llama.cpp build_moe_ffn 一致（softmax → top-k → 权重归一化，
  除数下限 6.1035e-5 → expert_weights_scale → 每专家 silu(gate)×up →
  sigmoid 门控共享专家）。
- 验证：`tools/make_moe_gguf.py` 写合成全 F32 MoE GGUF + Python 参考；
  `tools/moe_load_test.odin` 对拍 logits（当前逐位一致）。`./build.sh` 自动
  执行。
- 真实模型冒烟：`tools/moe_smoke.odin <model.gguf> [steps]`（CPU）。已在
  `bayliner1980/Qwen3.5-0.8B-MoE-A0.6B-GGUF` Q4_K_M（524MB，24 层，28 专家，
  top-14，248k 词表）验证——加载、预填充、解码连贯（CPU ~1.8s/token）。
  注意：我们的加载器接受任意层数（llama.cpp 的 qwen35moe 类型表只认
  40/48/60 层）。

## 已验证的坑（不要再踩）

- CUDA 预填充的 `pgemm`（fp16 反量化进共享内存再 WMMA）在 V100 上结果正确但比标量块慢（9B 16000 token 154 对 183 tok/s）。同一个 K 上写 32 行 fp16 是 32 路 bank 冲突，64 token 的 N 块在 128 token 的 chunk 里把每个权重反量化两遍。不再启动。`mmqk`（q8 + dp4a）是默认。全局 X 的块在那台 V100 上更快（9B 16000 token 246 对 213 tok/s，27B 8000 token 93 对 80）。32 行的块更快（9B 16000 token 290 对 246 tok/s，27B 8000 token 113 对 93）。后来的 64 行 Q4 块更慢（9B 8000 token 333 对 337），只在 `ODINFER_CUDA_MMQK_ROWS=8` 时启用。Q6 在 8 行时会 spill，所以仍是每个 warp 4 行。gate+up 和 K+V 融合差不多持平，只在 `ODINFER_CUDA_PREFILL_FUSE=1` 时启用。预填充注意力里向量化的 KV 反量化留下了（9B 8000 的 attn 3872 对 4161 ms）。
- GEMV 内核形状：本机（M3）最优 = 256 线程/32 行/8 lanes；其余多行/多线程/
  融合变体实测回归（数据在 packages/qwen3_5/README.md 与 gemm_harness）。
- **内核 A/B 必须同窗交错计时（2026-11-17 实证）**：M3 Air 被动散热，GPU 时钟
  随热状态在会话内漂移 ~2×。同一 harness 内**顺序**逐变体计时会得出完全相反的
  结论——实测 q4k ks8r2 变体顺序计时"赢 44%"（0.755→0.524ms），交错计时（每轮
  各变体各一个 command buffer 轮转 12 轮取 min/median）实为**慢 20%**；q6k
  顺序计时还出过 189 GB/s 的假带宽（>DRAM 物理上限，实为只算了 1/3 的 shape
  配错）。正确做法见 `gemm_harness/gemv_9b_ks.odin`（CHECK=1 先对拍，默认
  交错计时；含 Q8_0 同 shape 带宽天花板探针）。凡"新内核更快"的结论，先过
  这道闸。
- **9B decode 的 Q4_K/Q6_K GEMV 已在带宽墙上（2026-11-17 交错实测）**：同窗下
  generic/spec(特化)/ks16/ks8r2/ks8r4/ks32/uchar4 手写向量化（v4，逐位一致）
  在 up/gate 12288×4096、down 4096×12288、qkv 8192×4096、output 248320×4096
  上全部落在 55–75 GB/s（热态 median ~50-55，凉态 best ~72）；Q8_0 探针（解量化
  最便宜）同 shape 也只有 66–86 GB/s。结构与 ALU 微优化均无增益，编译器已生成
  等价代码。decode 剩余空间不在 GEMV 内核里；engine 端墙钟 87.6ms vs GPU
  85.4ms/token，CPU 间隙仅 ~2.5%，也不是杠杆。**decode 已封顶，别再投。**
- **IQ ks 内核（K-split 双行）经交错复验为真**：IQ2_XXS down 5120×17408
  generic 41 GB/s vs `gemv_iq2_xxs_ks_l` 52 GB/s（同窗 +27%，median 亦 +），
  IQ1_S 31→36 GB/s。这类内核离墙很远（IQ 解量化 ALU 重），结构改动有效——
  与 Q4_K/Q6_K 的"无增益"结论不矛盾。
- **prefill 的逐 token 状态循环已批量化（2026-11-18，dense + MoE 路径）**：全注意力块
  走 9 个 batch 内核一次 launch 处理整 chunk（`QATTNB=0` 回退逐 token）；线性层
  l2norm/delta/rmsgated 用 setBuffer 偏移直读 _t 行（无拷贝）。两者与逐 token
  路径逐位一致（三条 KV 路径都 A/B 验证过）。MoE prefill 路径(forward_gpu_batch_moe)也已移植同样的批量化
  （0.8B MoE GPU 输出与 QATTNB=0 逐位一致）；其 GEMV 覆写保持原样（数值决策）。
  线性循环 = 批量 l2norm/rmsgated（Stage 2 遗产内核）+ 逐 token delta；l2norm 批量版
  的乘法顺序已对齐逐 token 版（iq 不折叠 scale），改内核时保持。
- **int8 MMQ 在 Apple/Metal 上不存在（2026-11-18 实证，别再评估）**：
  `simdgroup_matrix<int8_t/int32_t>` 触发 static_assert——simdgroup mma 只有
  f16/f32，MSL 也没有 dp4a（那是 CUDA）。llama.cpp 的 int8 MMQ 不走 Metal。
  本机 f16 mma 微基准（寄存器内）≈ 2.25 TFLOPS（热态），我们的宽-N MMQ 已达
  其 ~89%——内核侧在硬件墙。工具：`gemm_harness/mma_ratio_bench.odin`。
- **chunked delta 已接线但默认关（QCHUNKD=1 开，2026-11-18）**：C=8，核内
  cp≥1e-20 下限修 0/0 NaN 根因。交错实测 −0.4~−0.9 ms/tok（6-8% prefill）。
  数值是逐 token 递归的重排（非逐位一致）——默认口径的决定留给维护者。
- **MMQ 的 A 级 dequant 必须向量化（2026-11-17 已交付）**：宽-N MMQ
  （`gemm_q4k_f32_n32`/`gemm_q6k_f32_n64`）的 dequant 逐元素标量写法是 ALU
  瓶颈；uchar4 加载 + 向量 nibble 运算 + half4 写（B 级 half4）后 Q4 +22%、
  Q6 ~3×（交错实测，`gemm_harness/mmq_v4_sweep.odin`），数值逐位一致，端到端
  prefill 15.50→12.59 ms/tok。重写这些 kernel 时不要退回标量 dequant；
  BK=128 已实测更慢，别加宽 K。
- 内核计时必须**墙钟 + 正确性验证**：GPU 计时器（GPUEndTime）对短 dispatch
  会给出假数据（曾误判 22.5ms 输出层，实际 6.4ms）。
- Odin 语法小坑：for 循环里不能直接放数组字面量（先命名变量）；proc 参数取
  地址需先拷到局部变量；`os.to_stream(f)` 用于写文件流。
- MoE SwiGLU：silu 作用于 GATE 投影（silu(gate)×up），不是 up——对称测试
  数据会掩盖此错。
- GGUF 张量维度是 ggml ne 序（ne0 最快）= numpy shape 的反转；手写 GGUF
  写入器要反转 dims 且数据段 32 字节对齐。
- MTL setBuffer 签名是 (buffer, offset, index)——第二个参数当 index 写会静默
  重绑 index 0（曾导致 router 权重写进 logits 缓冲区）。
- 专家 GEMV 偏移是张量内相对偏移；m_weights 是整模型 mmap，必须加
  woff(张量) 基址。
- Q5_0 高比特：元素 j+16 用的是 qh 第 j+16 位（CPU 解量化是 >>(j+12) 后掩
  0x10）。
- 社区 MoE GGUF 逐层混合量化种类（0.8B-MoE 的 down_exps 在 Q8_0/Q5_0 间
  交替）——按张量 kind 分派，不要假设。
- GEMV 形状特化（Tier 3.5）：collect_gemv_shapes 扫描 GEMV 分派张量。
  Q4_K/Q6_K 仍要求 n%256==0 && d%256==0；IQ 超块要求 n%256==0（IQ4_NL
  为 n%32==0），d 为 GEMV_ROWS 的倍数。按 n*d*count 排序，上限 64。
  Q4_K/Q6_K 手写体；IQ 从通用内核改写（去 Dims、去行守卫、nsb/nb 编译期），
  数学与通用核相同。默认开，QGEMVSPEC=0 关。9B 的 Q4/Q6 集合不变（实测
  解码 +5.6%）。MoE 模型不收集形状（惰性）。
- Tier 2 热路径纪律：每 token 每层的分派零模式分支——enc_store_attn_gate
  拆成四个具体 proc（sag_f16/f16c/q8/q8c）+ 预选 g_gate_pso/g_gate_needs_hd；
  metal_init 在 kv_dim 校验后一次性选 g_sag。新增 KV 格式或 QCONST 组相关
  分派时，放进初始化期选择，不要写热分支。
- 加载期特化：metal_init 生成 MSL prelude，把模型常量（DIM/HEAD_DIM/KV_DIM/
  KEY_DIM/VALUE_DIM/CONV_DIM/rotary/ROPE_FREQ/RMS_EPS/N_VH/N_KH/CONV_KERNEL/
  L2_SCALE/KV_POS_BYTES）写成 constant；特化内核为 *_c 变体（#ifdef
  QCONST_ON）。QCONST=0 编译旧运行时参数内核（smoke 76/76 vs 96/96，双
  fastMath）。实测：27B 解码 −0.8%（最佳 −2.5%，a 组），9B 持平（解码 97%
  是 MLP GEMV）。GEMV 的 Dims n/d 保持运行时（逐张量）。
- 对拍门禁与 KV 格式相关：紧门禁（moe_smoke --compare ~3e-3、q38
  --compare ~1.4e-3）须用 KV_Q8=0 跑；默认 Q8 KV 时 Metal 腿相对 CPU fp32
  腿带 ~1e-1 量化噪声（MoE 0.177、27B 1.1e-1）——预期行为，不是 bug。
- MSL 保留字：变量名 'half' 会让整个 shader 库编译失败。过去 metal_init 失败
  会静默回退 CPU（所有模型）；现在 --metal / -g 1 下初始化失败直接报错退出
  （METAL_FALLBACK=1 恢复旧回退行为），build.sh 在 macOS 上自动跑
  tools/metal_smoke.odin：编译全部 kernel、逐个建 pipeline、核对 make_pso 名字。
- 已解决（v0.3.0）：MoE 分组 GEMM 批量 prefill 曾与逐 token 发散，根因是三处
  叠加 bug（M=16 ssm 张量 MMQ 零网格、top-k 跨 token 线程组竞争、MMQ 顺序
  放大误差），现已默认开启。

## 验证清单（改动后自检）

```sh
./build.sh                                          # 构建 + 单测全绿
python3 server/serve_harness/test_mcp.py            # MCP 握手全过
MODEL=./models/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf ./serve.sh &
curl -s http://127.0.0.1:9748/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"odinfer","messages":[{"role":"user","content":"What is 2+2? Answer in one word."}],"max_tokens":8}'
# 期望：content 为 "4"，stderr 有 mem: 行与（首次）tune: 行
