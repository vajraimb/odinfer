<!--
Internal engineering log, kept in its original language (中文内部工程记录，保留原文).
The user-facing documentation (README / AGENTS / POSITIONING / docs/MCP·MEMORY·THERMAL) is bilingual.
面向用户的文档（README/AGENTS/POSITIONING/docs 下三篇）为中英双语；本文件为内部工程日志。
-->

# Roofline / Profiling Notes (数据驱动的优化空间封顶)

记录对本引擎(Ornith-1.0-9B Q4_K_M,Apple M3 Air)prefill/decode 的 roofline 分析,
用于决定"继续优化 vs 收工"。结论:**prefill 已在 MMQ 算力墙上,decode 已在带宽墙上;
剩余空间封顶 ~2×,且只有"死磕 MMQ dequant"一条硬路。**

## 关键事实(实测 + 算术)

模型 **dense**(非 MoE,单 MLP/层),~9B 参数,Q4_K_M ~5.6GB。M3 Air 统一内存 ~100GB/s,
GPU f32 ~2.15 TFLOPS / fp16 ~4.3 TFLOPS。

### Prefill(批量,Stage 1b + tuned MMQ,当前 ~15-22 ms/tok)
- 每 chunk(T=512)投影 FLOP = **7.085e12**(dense,全参数激活)。
- 实测 MMQ 吞吐 **~900 GFLOPS**(isolated benchmark,引擎实测匹配)。
- 7.085e12 / 900e9 = **15.4 ms/tok** ← 精确匹配实测 15-20ms/tok。
- **= 投影 COMPUTE @ MMQ 算力墙**(dequant-ALU 受限)。不是带宽(带宽地板 0.11ms/tok
  需 127 TFLOPS,物理不可能),不是 dispatch(~1248 dispatch/chunk × 5µs = 0.012ms/tok,
  可忽略),不是逐 token 状态循环(消融实测线性层 ~2ms/tok、全注意力 ~1-4ms/tok)。
- **引擎 MMQ 已达 isolated benchmark 的 900 GFLOPS** → 没有"engine dispatch 形态效率损失",
  已在 kernel 物理上限。继续推只能重写 dequant kernel 本身,不是改 dispatch。

### Decode(逐 token 生成,~91 ms/tok)
- 每 token 读全 5.6GB 权重 → 5.6e9/100e9 = **56ms/tok 带宽地板**。
- 实测 ~91ms/tok → ~60% 带宽效率,**已接近带宽墙**,没什么可优化。

## 优化空间封顶

| 路径 | 收益 | 状态 |
|---|---|---|
| Stage 1b 批量投影 + tuned MMQ | prefill 122→23 ms/tok(**4.9×**) | ✅ 已交付 |
| Stage 2 chunked delta(线性层递推) | ~2 ms/tok | ❌ 数据否决(低 ROI),kernel 验证过但未接 |
| MMQ dequant/staging 往 1.5 TFLOPS 推 | prefill 15→~9-12 ms/tok(~1.5×) | 硬工程,唯一真杠杆,收益封顶 ~2× |
| tile sweep(真实 shape) | ~10-20%(900→1100) | ✅ 已交付并采纳进 engine:`gemm_harness/gemm_autotune.odin` + `metal.odin` 的 `gemm_q4k_f32_b64`(M%64 走 bm=64 bk=64,其余回退 v3)。Q4K isolated w1 874→1028(+18%);**engine 实测 prefill ~1.2×**(QTIMING 位置对齐 median 39.2→32.4 ms/tok)。Q6K b64 是 0.66× 回归已排除(w2 保持 v3 ~400 GFLOPS 上限);fm on/off 无差。2026-11-13 列方向加宽(BN 不再锁 8,`qn_sweep.odin`):Q4 `gemm_q4k_f32_n32` BM=32 BN=32 BK=64,9B 形状 ~1470 vs b64 ~1195 GFLOPS;Q6 `gemm_q6k_f32_n64` BM=16 BN=64 BK=64,~800 vs v3 ~480。Q6 的 BN=8 b64 回归仍然成立,不选。`QMMQN=0` 回到 BN=8。dense prefill 按 64 token 对齐切块。八种 IQ 同一块 BM=32 BN=32 BK=64（`iq_n_sweep.odin`，up 形状约 1.25–1.34×）。27B IQ2_XXS 端到端对照（QMMQN=0 vs 默认，两边都不做 GEMV 覆盖）：448 token 对齐块 60.7→53.6 ms/tok，499 token 预填充墙钟 31.0→28.1 s，贪心都是 Four / fox |
| fp16 累加路径 | 死路 | ❌ half 操作数已用;全 half 累加 K=4096 精度崩;dequant 是瓶颈非 MAC |
| 全注意力批量(Stage 1c) | ~1-4 ms/tok | 低 ROI(全注意力只 8 层) |
| Flash attention | ~5% 以下 | 跳过(Rigel:分解式优于融合) |

**prefill 现实极限 ~8-9 ms/tok**(MMQ 推到 ~1.8 TFLOPS),decode 已到墙。整个项目优化空间
已封顶:从当前 4.9× 出发,最多再 ~2×(prefill),且需重写 dequant kernel。

## Profiling 工具(env 控制,默认行为不变)
- `QFASTMATH=0`:CompileOptions 关 fast-math(precise 着色器,f32 漂移排查)。
- `QTIMING=1`:forward_gpu_batch 打印 `GPUStartTime/GPUEndTime` 精确 GPU ms。
- `QPROF_NOLIN=1` / `QPROF_NOFULL=1`:消融跳过线性层/全注意力逐 token 状态循环(输出垃圾,
  GPU 时间有效),用于归因。
- 启动打印 `maxThreadgroupMemoryLength`(M3=32KB,fused kernel 的硬预算)。

## 决策
4.9× prefill + bit-identical + 完整 profiling 工具链 + roofline 封顶 = 优化项目闭环完成。
是否继续推 MMQ 到 ~1.5 TFLOPS(再 ~2×、硬工程)是一个**有数据支撑的收工/继续决策**,
非直觉押注。

## Continuation 续写缓存收益实测(2026-10-31,A/B 双二进制)

修复前(c2591c0,已含 b64 MMQ)vs 修复后(HEAD,continuation 路径),同一 5 轮中文对话,
贪心、num_predict=32、冷启动,server 端 prefill= 计时:

| 轮次 | 修复前 prefill | 修复后 prefill | 路径变化 |
|---|---|---|---|
| 1 | 5209 ms(含一次性快照保存) | 2314 ms | full prefill(同一路径,差值为噪声) |
| 2 | 2066 ms | 1122 ms | snapshot HIT → continuation hit |
| 3 | 3405 ms | 1587 ms | 同上 |
| 4 | 3808 ms | 1340 ms | 同上 |
| 5 | 5568 ms | 802 ms(**6.9×**) | 同上 |

- 修复前:每轮 prefill 随会话长度**线性增长**(整段历史从系统边界重算,O(n²) 累计);
  修复后:每轮只 prefill 新回合 ~15–25 token,**基本恒定**(O(n) 累计)。
- 轮 2–5 合计:14.8s → 4.9s(**3.0×**);会话越长收益越大,轮 10 外推 ~10s → ~1s。
- 计数器佐证:修复前 4 次 snapshot hit,修复后 4 次 continuation hit。
- 保真度:continuation 命中与冷全量 prefill 逐字节相同(test_prefix_bitidentical.py 多轮阶段)。

## 客户端契约:prompt 字节必须跨轮稳定(2026-10-31 晚,真实流量教训)

continuation 修复后实测 pie 真实会话仍 100% miss。服务端 QDUMP_DIVERGE=1 打点定位:
**pie 每轮把工具 schema 经 json.Value(map) 重序列化,Odin map 遍历序不稳定 → system 块
~600 token 处起字节全变**,快照与 continuation 双双失效,每轮全量重算 1100–1800 token
(76–218s,热节流加剧)。修复在 pie-odin 侧(Raw_Json 逐字节透传,commit 6e8cda1):
修复后同 prompt 二次请求 PREFIX CACHE HIT(restored 1119/1137,prefill 81s→3.2s,**25×**)。

**契约**:任何客户端只要改变 system 块/历史消息的**字节**(哪怕语义等价的 JSON 键序),
前缀缓存即失效。稳定序列化是客户端责任;服务端 `QDUMP_DIVERGE=1` 可打点定位漂移段。

## Decode GEMV 带宽墙终验 + 交错计时方法学（2026-11-17）

问题:README 记录的"MLP GEMV 76 GB/s / attention 90 GB/s"与本次复测对不上,且
同一内核两次 sweep 差 2×。归因:**M3 Air 被动散热,会话内 GPU 时钟漂移**——顺序
逐变体计时的 A/B 会被热状态系统性带偏(实测 ks8r2 顺序"赢 44%",交错实为慢 20%;
q6k 假带宽 189 GB/s = shape 配错只算 1/3)。

方法学修正(`gemm_harness/gemv_9b_ks.odin`,CHECK=1 对拍 + 12 轮交错,每轮每变体
一个 command buffer×5 dispatch,取 min/median):

- **Q4_K/Q6_K GEMV 在墙上**:9B 四大流量 shape(up/gate 12288×4096 Q4_K ×66、
  down 4096×12288 Q6_K ×33、qkv 8192×4096 Q6_K ×33、output 248320×4096 Q6_K)
  上,generic ≈ Tier3.5 spec ≈ ks16/ks8r2/ks8r4/ks32 ≈ uchar4 向量化 v4(逐位
  一致),全部 55–75 GB/s;Q8_0 同 shape 探针 66–86 GB/s。差距 < 变体间噪声。
  结论:结构(8 种)与 ALU 微优化(手写 nibble 向量化)都已到头,编译器等价。
- **decode 端到端**:墙钟 87.6ms/tok vs GPU 85.4ms/tok(11.4 tok/s,热态),
  CPU 间隙 ~2.5%——双 buffer 预编码最多 +2.6%,低 ROI 不做。
- **IQ ks(K-split 双行)复验为真**:IQ2_XXS down generic 41→ks_l 52 GB/s(+27%,
  median 同向),IQ1_S 31→36。IQ 内核离墙远(重解量化 ALU),结构改动有效;
  与 Q4_K/Q6_K 无增益不矛盾。未提交的 ks/n16 系列可以放心提交。

**最终判定:本机 decode 优化封顶(带宽墙 + 2.5% CPU 间隙),prefill MMQ 死磕仍是
唯一 >10% 杠杆(roofline 原判断不变)。**

## 同机 ollama 对照(2026-11-17 晚,同 GGUF blob,交替同窗,贪心)

对照组:ollama 0.40.16(runner llamacpp,100% GPU,常驻),模型
`hf.co/mradermacher/...heretic-GGUF:Q4_K_M`——与 odinfer 用的 GGUF 是**同一个
blob**(sha256-a692de95),prompt/贪心/模板对齐(raw:true 复刻 odin 渲染),每轮
交替执行防热漂移;ollama 的 KV prompt 缓存用唯一 prompt 后缀击败(否则会测到
0.3-0.6 ms/tok 的假 prefill)。

| 指标 | odinfer(当日 build) | ollama | 差距 |
|---|---|---|---|
| decode tok/s(内部速率) | 11.3(85 ms/tok GPU) | 14.2-14.9(67-71 ms/tok) | ollama 1.27× |
| prefill ms/tok(T=448 块) | 11.0-12.6 | 6.0-6.4 | ollama 1.76× |
| 454-tok TTFT(模型常驻) | ~5.0 s(纯 prefill) | 2.8-3.0 s | — |

解读:ollama decode 全 token 折算 ~80 GB/s 有效带宽——等于我们隔离最优内核
(Q8_0 探针 82-86)的水平,说明差距不在单个大 GEMV 的带宽,而在(a)小内核/调度
开销(我们 ~85ms 里 ~14ms 非 GEMV),(b)混合 shape 上的逐形状 tile 选择
(llama.cpp 的 mul_mv_ext 按形状选参,我们固定 8-lane;autotune 框架
Tune_Cand 已有,可扩)。prefill 侧 llama.cpp MMQ 有效吞吐 ~2.2 TFLOPS vs 我们
v4 的 ~1.66——MMQ 死磕的下一目标明确:双缓冲 sa/sb 消 barrier/更大 tile。

## MMQ dequant 向量化交付(2026-11-17 下半场,"死磕 MMQ"的第一刀)

问题:宽-N MMQ 的 A 级(stage)dequant 是逐元素标量代码,Q6_K 每元素两次字节加载
+ or/shift,是 roofline 判定的 ALU 瓶颈本身。

改动(`gemm_q4k_f32_n32` / `gemm_q6k_f32_n64`,kernel 名与分派不变):A 级改
uchar4 加载 + 向量 nibble 运算 + half4 写 threadgroup,B 级改 half4 加载。
数值与标量路径**逐位一致**(gate max_rel 与标量版到数字位都相同)。

交错实测(`tools/gemm_harness/mmq_v4_sweep.odin`,10 轮,gate=f64 dot):

| shape(9B) | 基线 | v4 | 比 |
|---|---|---|---|
| Q4 up/gate 12288×64×4096 | 1356 GFLOPS | **1659** | +22% |
| Q6 down 4096×64×12288 | 465 | **1588** | 3.4× |
| Q6 qkv 8192×64×4096 | 533 | **1586** | 3.0× |
| Q6 output 248320×64×4096 | 512 | **1598** | 3.1×(254→81 ms/slab) |

BK=128 变体全面更慢,弃。

端到端(同机同会话,T=448 chunk):**15.50 → 12.59 ms/tok(+23% 吞吐)**,贪心输出
与基线逐字一致,build.sh 单测/metal_smoke 120 kernel/test_mcp 全绿。

新 roofline:prefill 已从收工时的 ~900 GFLOPS 推到 **~1600–1660**,原"硬工程
天花板 ~1.8 TFLOPS"已交付大半;残余 ~10% 在 mma/barrier 结构,dequant 不再是
瓶颈。

### 双缓冲交付(2026-11-18,同 harness)

sa/sb 双缓冲:tile k+1 装入另一块 buffer、与 tile k 的 simdgroup mma 并行,
每 K 步 barrier 从 2 个降到 1 个。三个生产内核换体(名字/分派门槛不变):
`gemm_q4k_f32_n32`(1348→**1874** GFLOPS)、`gemm_q6k_f32_n64`(498→**1707**),
新增 `gemm_q4k_f32_n64`(BM=32 BN=64,1024 线程,enc_q4k 在 N%64==0 时优先,
**1994** GFLOPS)。gate 全过,数值与单缓冲逐位一致。

端到端:T=448 chunk **12.59 → 11.23 ms/tok**(今日累计 15.50→11.23,+38%),
贪心输出逐字一致,单测/metal_smoke 121 kernel/test_mcp 全绿。对 llama.cpp
(~2.2 TFLOPS 有效)的内核级差距缩到 **~1.10×**(Q4 侧);端到端 prefill 差距
从 1.76× 预计缩到 ~1.6×(未重跑 ollama 对照,模型已按用户要求卸载)。


## v1.0 Release Status(收工基线)
- **主线交付**:prefill 122→23 ms/tok(**4.9×**),贪心解码 **bit-identical**
  (`2+2=4`、`Paris`、1062-token 摘要逐字一致),三仓库(origin/main)同步。
- 端到端体感:1500-tok prefill + 500-tok 生成 ≈ 78s(原 ~6 分钟)。继续死磕 MMQ 极限
  仅换 ~15% 端到端(78→67s),且 decode 已在带宽墙 → **杠杆率极低,决定收工**。
- profiling 工具(`QFASTMATH`/`QTIMING`/`QPROF_*`)保留,**默认关闭**(env 门控)。
- **C=16 chunked-delta 悬案(已结案,2026-10-31)**:engine 在 C=16 出 `!!!!!`,C≤8 正常。
  用 `delta_harness/engine_chunked_gpu.odin`(engine 内核逐字 + engine 维度/布局 + CPU 逐行双胞)
  做了判决性实验:
  - **fast-math 排除**:良性输入下 C∈{8,16} × fast-math{on,off} 四组合全过
    (GPU vs CPU 双胞 max|dOut| ~5e-9),engine 内核本身无辜。
  - **真凶定位(机制级)**:对抗用例(cp 累积衰减下溢:单个 head 内两个 token 的
    g=exp(a_dec·softplus)≈1e-14 → cp[t] 下溢为字面 0 → `cp[t]/cp[i]` = 0/0 → NaN),
    CPU 双胞与 GPU 产生**完全一致的 NaN 模式**(各 19968 个)。这不是 fast-math,也不是
    dispatch,而是 chunked 公式本身的脆弱点——tokenwise 路径从不构造 cp 比率所以免疫;
    C 越大,一个 chunk 内含多个极小-g token 的概率越高,与"C≤8 正常、C=16 炸"的症状吻合。
  - **对 fused kernel 的硬性设计要求**:必须稳定化 cp 比率——clamp 累积衰减下界
    (cp ≥ 1e-20 量级)或改 log-space 比率(exp(cumsum 差)并 clamp),否则 fused 版会
    在真实激活上随机炸 NaN。
  chunked delta 整体只值 ~2ms/tok,非优先。
