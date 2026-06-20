# Diffusion LLM 研究汇总：AR→Diffusion 转换 · Qwen3.5-9B 方案 · int8 量化

> **范围**：三轮独立调研的合并归档——(1) 能否把普通自回归(AR)模型权重转成 diffusion LM；(2) 把 Qwen3.5-9B 具体转成 dLLM 的可行性与 PoC 计划；(3) dLLM 能否适配 int8（尤其在 Ampere / 2×3090）。
> **生成方式**：3 个多智能体 ultracode 工作流（共 ~72 个 subagent、~290 万 token、~1350 次 web 工具调用、对抗式核实）。
> **日期**：2026-06-16。
> **⚠️ 重要诚实声明**：知识截止为 2026-01，本文大量结论建立在 2025–2026 的实时检索（含 post-cutoff 的 arXiv 号如 `2508.x` `2510.x` `2604.x` `2606.x` 和 DiffusionGemma 发布）之上。这些**无法用训练知识背书，落地前请亲自点开核对**（尤其 §1.1 DiffusionGemma、§2.3 SANA/WeDLM、§3 的 DLLMQuant/CoDA/3090-DiT 几篇）。
> **硬件语境**：2×RTX 3090（24GB，无 NVLink，Ampere sm_86），已有 patched Marlin 跑 AR 模型的 **W4A8（int4 权重 + int8 激活）**；主力是 ~27B Qwen3.6 W4A8 serving 栈 + Qwen3.5-9B 迁移评估中。

---

## 0. 执行摘要（先读这一段）

三条研究串起来指向同一个结论：**你手里攥着一条别人没走过、但门是开着的路线。**

| 研究 | 一句话结论 | 对你的含义 |
|---|---|---|
| **AR→diffusion 转换** | 可行且已是主流（DiffuLLaMA/Dream/RND1/DiffusionGemma 全是实证），但要数百 B token 的 continued-pretrain | 自己训练在 2×3090 上做不了；**该跑现成 dLLM，不要自训** |
| **Qwen3.5-9B→dLLM** | research-grade：8 个 softmax 层好转，但 75% 的 GatedDeltaNet 线性层要双向化、无 warm-start 先例 | 想试就走 §2.6 阶段 0（$0 forward-only go/no-go），优先 block-diffusion 折中 |
| **dLLM int8** | weight-int8 改改就行；**activation-int8 是全领域 defer、而你恰好有 edge 的空白** | 你的 Ampere int8-激活 Marlin 在硬件/计算区间/精度上三者全对，这是真正的护城河 |

**关键洞察（贯穿三研究）**：dLLM 的每步去噪是 **prefill-like、compute-bound**（整 canvas 重算），与 AR 的 **decode、bandwidth-bound** 正好相反。这把你 MEMORY 里"int8 prefill +15% / decode 0 gain"的发现**反转成利好**——dLLM 每一步都吃到 int8 prefill 收益，且 Ampere 上 int8 是唯一原生低精度 tensor-core 路径（无 FP8/FP4）。

**统一推荐路线（性价比/风险排序，见 §4）**：先 bf16 + Fast-dLLM 把现成 Dream-7B/LLaDA-8B 跑起来当 baseline → fake-quant 验证 W8A8 误差平坦 → 再把 Marlin int8 GEMM 以 W8A8 接进 denoiser。转换（§1/§2）当独立的更大赌注，别挡住量化这条近路。

---

# Part 1 — AR→Diffusion 权重转换可行性

## 1.1 先验核实：`google/diffusiongemma-26B-A4B-it` 是否存在

**存在，且是 Google 官方真实发布（2026-06-10，Apache 2.0）。** 多源一手收敛（活的 HF API 记录 + ai.google.dev/deepmind.google/blog.google + vLLM blog + The Register/VentureBeat 独立报道 + NVIDIA NGC），排除伪造。
- 本质：离散文本 diffusion LM（block-autoregressive multi-canvas denoising），放弃 AR 逐 token 串行解码。
- 架构：基于 **Gemma 4 26B-A4B MoE**（25.2B 总 / 3.8B 激活，128 routed experts 取 8 + 1 shared，30 层，256K ctx，262K 词表），多模态输入→文本输出。
- **关键：不是从零训练，而是从 AR Gemma 4 checkpoint fine-tune 改造（加 diffusion head）。** 但 token 预算/算力/denoising step 数 Google 全部未公开，无技术报告。
- 性能：比 AR 原版快 ~4×（H100 FP8 1000+ tok/s），但全 benchmark 略低（MMLU-Pro 77.6 vs 82.6，AIME 2026 69.1 vs 88.3）。vLLM 首个原生 dLLM。

**同类参照**：Gemini Diffusion（闭源）、LLaDA-8B / LLaDA-MoE / LLaDA2.0（从零开源）、Dream-7B（从 Qwen2.5-7B 适配）、Mercury（Inception Labs，首个商用 dLLM）。

## 1.2 结论：可行且主流

AR → masked/discrete diffusion 的"权重改造"原理成立、有大量带开源权重与配方的实证。**核心代价不是"能不能"，而是 continued-pretraining 的算力（数百 B token + 数据中心级 GPU 节点）。**

## 1.3 原理：为何权重可复用

AR transformer 与 masked diffusion LM **在权重张量层面几乎是同一模型**，差别集中三处且都不改权重形状：

| 维度 | AR | masked diffusion LM |
|---|---|---|
| 注意力 mask | causal | bidirectional（删 causal mask） |
| 训练目标 | next-token prediction | absorbing-state ELBO（只在 mask 位算 CE，按 1/t 重加权） |
| 时间条件 | 无 | **可省略**（DiffuLLaMA 证明：mask 数量隐式承载噪声，零新增参数） |
| shift | 位置 i 预测 token i+1 | **保留这个 shift** |

两个深层原因：① AR 本来就在每位置算 logits 只用了最后一个，diffusion denoiser 要的正是"所有位置 logits"；② masked diffusion ELBO 等价于加权 MLM 损失（MDLM, NeurIPS 2024），从"预测下一 token"切到"预测被掩码 token"是平滑过渡。

**易搞反的关键点（对抗核实确认）**：从 AR 适配要**保留 next-token shift**（DiffuLLaMA/Dream 都强调，"maximally architectural alignment with AR"）。区分两个家族：从零训练的 native dLLM（LLaDA/MDLM/SEDD）通常**不用 shift** + 带 timestep embedding；从 AR 适配（DiffuLLaMA/Dream/RND1）**保留 shift + 省 time embedding**。

## 1.4 已验证配方与数字

五步通用配方：(1) causal→bidirectional（可选 mask annealing）；(2) **保留 next-token shift**；(3) absorbing-state [MASK] masked diffusion；(4) NTP loss→只在 mask 位 1/t 重加权 CE；(5) time-embedding-free。**LR 是头号敏感旋钮**（太高冲掉 AR 知识，太低学不动）。

| 模型 | AR base | 类型 | 适配 token | 算力 | 能力保留 |
|---|---|---|---|---|---|
| DiffuGPT-S/M | GPT2 127M/355M | dense | ~30B | 8×A100-80G | 多数持平/超过 |
| DiffuLLaMA-7B | LLaMA2-7B | dense | ~65B | 64×GH200, LR 2e-5 常数 | GSM8K 63.1 vs 58.6↑；TriviaQA 18.5 vs 45.4 ↓59% |
| **Dream-7B** | Qwen2.5-7B | dense | **580B** | **96×H800×256h≈24.6k GPU-h** | MMLU 69.5 vs 71.9 持平；规划暴打 AR：Sudoku 81 vs 21 |
| Dream-Coder-7B | Qwen2.5-Coder-7B | dense | 322B | — | LiveCodeBench 21.4% pass@1 |
| **RND1-Base** | **Qwen3-30B-A3B (MoE)** | MoE | **500B** | 64×B200 | 超 Dream-7B/LLaDA-8B；仍低于 AR teacher |
| Fast-dLLM v2-7B | AR 7B | dense | **~1B（最省）** | 64×A100 | — |
| DiffusionGemma 26B-A4B | Gemma 4 26B-A4B (MoE) | MoE | 未公开 | 未公开 | 略低于 AR |
| LLaDA-8B（从零对照） | — | dense | 2.3T | 13 万 H800-h | 与 LLaMA3-8B 持平 |

**成本**：适配比从零省 ~4× token / ~5× GPU-h（Dream 580B/24.6k vs LLaDA 2.3T/130k），质量相当甚至更好。
**LoRA**：所有 base 转换都是**全参 FT**；"纯 LoRA 完成 base 转换"无任何先例，别假设。
**能力**：推理/代码/数学保留~改善；规划/infilling 暴打 AR；知识闭卷 QA 掉最狠（TriviaQA -59%，归因 token 预算小，未隔离证明）。

## 1.5 MoE / Gemma 特殊性

"A4B" = Active ~4B（总参 26B / 每 token 激活 4B，同 Qwen3-30B-A3B 约定）。Gemma 3 全 dense，26B-A4B MoE 是 Gemma 4 引入。
MoE 转 diffusion 已被证（DiffusionGemma + RND1 + LLaDA-MoE）但工程更难：① 遗忘更尖锐（知识在 expert，RND1 用**分参数组 LR**：attn 高 LR 3e-4，experts/norms/router 近冻结 1e-8+高 wd）；② 路由稳定性需 load-balancing+z-loss；③ 临界 batch 更大（masked diffusion 每步只监督 ~50% token）；④ 需 expert-parallel + 多节点。

## 1.6 对 2×3090 的可行性

- **自训转换 = 否决**：无 7B+ 适配在 <64 张数据中心卡上完成；27B 全参 ≈432GB 显存（48GB 的 9×）；W4A8 量化权重不能当初始化（须从 bf16 出发）。
- **该走的路 = 推理现成 dLLM**：Dream-7B(~14.6GB)/LLaDA-8B(~20.7GB) 单卡装得下，配 **Fast-dLLM**（training-free，最高 27.6×）。或实测 DiffusionGemma 26B-A4B（51GB 跨两卡，无 NVLink 用 **PP 非 TP**）。
- **拦路石**：W4A8/Marlin 不迁移 dLLM（详见 Part 3）；dLLM 低比特量化研究+仿真阶段；FP8/NVFP4 是 Hopper/Blackwell-only。

## 1.7 核实清单（Part 1）
- **Supported**：模型真实存在/官方/diffusion-based；AR 权重可经 continued-pretrain 转 dLLM；DiffusionGemma 由 AR MoE 适配（非从零）；RND1 由 Qwen3-30B-A3B 经 500B token 转成；AR→diffusion 可零新增参数；DiffuLLaMA/Dream 保留 shift。
- **Uncertain**：DiffusionGemma token/算力/step 全未公开；纯 LoRA base 转换无验证；MoE dLLM 路由稳定性无量化来源；知识退化多少可恢复未隔离。

---

# Part 2 — Qwen3.5-9B → Diffusion 转换方案

## 2.1 精确架构（命门，已 primary 核实）

来源：`Qwen/Qwen3.5-9B(-Base)` 的 `config.json` 逐字节相同 + 官方 transformers `model_doc/qwen3_5`。

**定性：DENSE（非 MoE）、原生 VL、hybrid 注意力。**（个别二手博客误标 9B 为 MoE，以 config 为准。）

| 字段 | 值 |
|---|---|
| num_hidden_layers | **32** |
| hidden_size / intermediate_size | 4096 / 12288 (dense) |
| heads / kv_heads / head_dim | 16 / 4 (GQA) / 256 |
| vocab_size | **248320**（eos 248044, bos None, `</think>` 248069）|
| tie_word_embeddings | false（lm_head 独立）|
| **full_attention_interval** | **4** |
| mtp_num_hidden_layers | 1（AR-only，转换丢弃）|
| attn_output_gate | true（softmax 层是 gated attention）|

**命门数字**：`full_attention_interval=4` + 显式 32 项 `layer_types` = **24 个 GatedDeltaNet 线性层 + 8 个 full-softmax 层（3:1）**，full-attn 在 0-indexed 层 **3,7,11,15,19,23,27,31**。**75% 是线性递归层。** 官方文档原话："3:1 hybrid attention stack — three Gated DeltaNet layers for every one Gated Attention layer"。

**GatedDeltaNet（`Qwen3NextGatedDeltaNet`，复用 Qwen3-Next）**：`linear_num_key_heads 16` / `linear_num_value_heads 32`（2:1 GVA）/ head_dim 128 / conv_kernel 4。因果藏**三处**：(1) depthwise **causal** conv1d（k=4, pad=3, 取 `[:L]`）；(2) chunk 下三角 decay 掩码；(3) 跨 chunk 前向-only state。依赖 `fla` + `causal_conv1d` kernel，否则慢 10× torch 回退。RoPE 仅 8 个 full-attn 层用（`rope_theta 1e7`, `partial_rotary_factor 0.25`, mRoPE）。

## 2.2 难度判定：比 Qwen2.5-7B→Dream 难多少

整体 **research-grade**，但非"从零发明数学"。

| 部分 | 占比 | 难度 | 先例 |
|---|---|---|---|
| 8 个 full-softmax 层 | 25% | **trivial（删掩码）** | Dream/DiffuLLaMA/RND1 全覆盖 |
| 24 个 GatedDeltaNet 层 | 75% | **困难（因果=递归方向，无掩码可删）** | 见 §2.3 |

**对抗核实修正的两点**：① "双向 GatedDeltaNet"**本身已发表**——NVIDIA SANA-WM/SANA-Streaming（视频域、从头训）、LION 附录给了 DeltaNet 双向数学；② 真正没人做过的是**从预训练 delta-rule hybrid AR 文本模型 warm-start、复用权重、转 token-wise 双向 masked text diffusion**。→ 这是"已验证零件的新颖集成"，最大真·未知是 warm-start 后知识能否便宜保住。

## 2.3 命门：GatedDeltaNet 双向化

softmax 因果性是可删的加性掩码；GDN **无掩码这个对象**，要看未来必须**真跑反向扫描**，且三处因果点都要处理。

- **方案 A — 双扫描加性融合**：`out = gdn(x) + flip_T(gdn(flip_T(x)))` + anti-causal conv1d。权重复用**最高**（fusion gate 零初始化使初始等价原 AR）。⚠️ LION Obs 3.1 证朴素 `y_F+y_B` **≠** 正确双向（重复计对角 + 错误 row-scaling）→ 须适配训练收敛。
- **方案 B — LION 单遍非因果**：联合归一化 + 减半对角，数学正确但工程大（反向 chunk 的 WY/Householder 要重推）。
- **方案 C — block-diffusion 折中** ★：GDN 跨 block 保持因果（state 可缓存），只 block 内双向化 8 个 softmax 层。**权重复用最高、最便宜（Fast-dLLM v2 纯 softmax 仅 ~1B token）、最契合你 W4A8 cudagraph serving 栈**。未验证风险：block 内 75% 因果线性层会不会卡死双向信息流——**无人回答，最高性价比首实验**。
- **WeDLM (arXiv 2512.22737)** 称 dLLM 可保留标准因果注意力——若对 hybrid 成立，可能整体绕开双向化，**下手前值得查**。

## 2.4 转换配方 + fork 目标

- 训练脚手架：`github.com/HKUNLP/DiffuLLaMA`（唯一可跑转换代码库；Dream 转换码未公开）。
- 模型代码：HF `modeling_qwen3_5.py` + `configuration_qwen3_5.py`。
- kernel：`fla` —— **关键发现：其 autograd 反向已调用 `chunk_local_cumsum(..., reverse=True)`，即内部已有 reverse-time gate-cumsum，先用 `torch.flip(dim=1)` 包两次现成 causal op 即可，不必写 Triton**。`fla-org/flash-bidirectional-linear-attention` 不含 (Gated)DeltaNet。

| 组件 | 复用? | 说明 |
|---|---|---|
| masked-diffusion loss | 直接复用 | DiffuLLaMA train.py 可抄 |
| next-token shift | 复用且必须保留 | — |
| time-embedding-free | 直接复用 | 零新参数 |
| attn-mask annealing→双向 | 仅作用 8 softmax 层 | DiffuLLaMA 实测可跳过 annealing |
| **GDN 改双向** | **必须新写** | DiffuLLaMA 对线性层 `NotImplementedError` |
| **anti-causal conv1d** | **必须新写** | 易漏的第二个因果点 |
| vision tower + MTP 头 | 转换前剥掉 | 文本-only |

## 2.5 算力 / 显存 / 成本

9B 全参 FT ≈ **144GB states**（16 B/param）+ 双扫描 +10~18% → **单台 8×H100-80GB FSDP 足够，不需多节点**。2×3090 只能 forward-only / kernel 单测 / ≤1.3B proxy，**做不了 9B 全参**（多卡 PP 不要 TP）。

| 级别 | token | 成本（含开销实际，spot $1–2.5/h）|
|---|---|---|
| 最小 PoC | 1–5B | ~$400–5k |
| 可信 PoC | 20B | ~$2.7k–6.7k |
| 质量级 | ~580B | ~$63k(spot)–195k |

下限参考：DiffuLLaMA 仅 ~60–65B token；Fast-dLLM v2 仅 ~1B token。**但 GDN 双向收敛成本无人测=最大变量。** 2026 H100 spot $1–2.5/GPU-h（Vast 低至 $0.34），8×H100 节点 RunPod ~$16–22/h。

## 2.6 分阶段 PoC（可证伪）

- **阶段 0（本机 2×3090，$0，数小时）**：text-only 载 `Qwen3_5ForCausalLM`，patch 一个 GDN 层为双扫描 + 8 softmax 层 `is_causal=False` + 反向 fusion gate 零初始化。判据：(a) masked logits 无 NaN、shape 对；**(b) 改右侧 token → 左侧 masked 位 logits 变化（核心 go/no-go）**；(c) masked 位预测 > 随机。失败即停，$0 沉没。
- **阶段 0.5（$0，1–2 天）**：反向 gate≈0 + 留因果时应近 bitwise 复现原 AR logits（验证权重管线）；探针对比反向扫描输出范数 vs 前向。
- **阶段 1（$0–几百）**：~0.5–1.3B 同构 hybrid proxy 收敛性，对比方案 A vs C 哪个稳。
- **阶段 2（租 8×H100，$400–5k，1–10 天）**：9B 全参 FSDP，1–20B token，低 LR。判据：infilling 胜 AR；若 1–5B token 内 loss 不接近 AR-init 困惑度 → 落昂贵区重评。
- **阶段 3（可选，$60k+）**：质量级 ~100–580B token（仅阶段 2 信号强才做）。

## 2.7 转换后能否 serve

**不能 drop-in vLLM/SGLang**：vLLM dLLM 后端只 softmax（TRITON_ATTN/FA4），对双向线性注意力/GDN 零支持。9B bf16 推理 ✅ 可行（~18GB 单卡，无增长 KV-cache），2×3090 用 PP 切层。⚠️ 双扫描使线性层 ~2× 算力且**抹掉 GDN 的 O(1) decode 优势**，hybrid 卖点部分蒸发。

## 2.8 核实清单（Part 2）
- **Supported**：Qwen3.5-9B = dense/VL/hybrid 3:1（24 GDN + 8 full）；softmax 层转换成熟便宜；**无任何发表把 hybrid 线性 AR 转 diffusion**；fla 无 drop-in 双向 GDN kernel；DiffuMamba/SANA/LION 双向线性递归全从头训。
- **Refuted**：❌"双向 GDN 从未发表"（SANA/LION 已有）；❌"朴素 y_F+y_B=正确双向"（LION Obs 3.1）。
- **Uncertain**：warm-start 因果 GDN→双向能否保知识（核心风险）；收敛成本落 ~1B 还是 ~580B；方案 C 能否去噪；反向扫描在前向 gate 下是否语义退化；WeDLM 是否对 hybrid 成立。

---

# Part 3 — dLLM int8 量化可行性

## 3.1 直接结论
- **weight-only int8（W8A16）**：能，基本无损、今天能跑——dLLM 比 AR 更鲁棒（CoDA 8-bit 零损失）。GPTQ/Marlin 8-bit 内核可复用，但**只省显存不加速**（dLLM compute-bound）。
- **activation-int8（W8A8/W4A8）**：W8A8 精度已证近乎无损（LLaDA -1.8~2.5%，instruct ≤0.5%），但**无任何已部署的真实 int8-激活 dLLM 内核（全 fake-quant）**。你的 W4A8 Marlin 是对的工具/硬件/区间，缺的只是接进 denoiser 的工程胶水 = 真正空白区。

## 3.2 weight-only int8（W8A16）：简单一半，已解决
W4A16 GPTQ 已近乎无损（LLaDA-8B -0.3%，Dream-7B -0.8%）；CoDA-1.7B 8-bit 权重 HumanEval 0.481→0.481 零损失，且 dLLM 低 bit 权重比 AR 更鲁棒（3-bit 时 Qwen3 全崩 CoDA 仍工作）。dLLM 上 **GPTQ > AWQ**（outlier 幅度低）。真 8-bit Marlin 经 GPTQModel 在 Ampere 可用；现成产物 unsloth DiffusionGemma Q8_0 GGUF（~25–27GB，near-lossless）。**W8A16 在 GEMM 前 dequant 回 fp16，不碰 int8 tensorcore，省显存不加速。**

## 3.3 activation int8（W8A8/W4A8）——硬骨头
**激活 outlier vs AR**：AR outlier 集中**少数固定 channel**（SmoothQuant/AWQ 所利用）；dLLM massive outlier **横跨更多 token**、集中 FFN down-proj、铺更开 → 削弱全局 clipping/scaling。
- **不能无脑搬 SmoothQuant/AWQ**：W4A4 套 LLaDA 掉 >16%，SmoothQuant W4A4 崩（GSM8K 69.7→0.3%）。
- 但 **8-bit 激活 AR 工具链能迁移**：W8A8 多方法一致近乎无损。**方法排序：weight-only GPTQ；weight-activation DuQuant > QuaRot >> SmoothQuant**（rotation 类混 channel 扛宽 outlier）。
- **难度墙在激活位宽（A8 安全 / A4 崩），非权重位宽。**
- **真实 demo：文本 dLLM 没有**。唯一系统 W8A8 研究（2508.14896）是 fake-quant，Limitations 直说 kernel 适配留 future work。DLLMQuant（2508.14090）唯一报真加速（A6000 LLaDA 1.71x）**但是 W4A4 + 不命名 kernel**，1.71x vs 3.24x 显存的不对称=显存压缩指纹，可信度中等。**W4A8（你的配置）在任何 dLLM 从未测过=处女地。**

## 3.4 diffusion 特有：去噪误差累积
误差累积真实且 dLLM 独有（每步输出=下步输入，"层数×步数"相乘，几何增长）。教训已从 image-diffusion（Q-Diffusion/PTQ4DM/TFMQ）重新发明到 text：Quant-dLLM 的 MCS、DLLMQuant 的 TMAS+IA-AQ+CGQ。但领先方案**多非 per-timestep 量化器**（STaR-Quant 用静态 step-shared + 注意力补偿）→ step-aware 可选非必需。
**关键好消息**：DLLMQuant Fig.2（LLaDA-8B 100 步累积 MSE）显示 **INT8 误差全程平、几何累积是 INT4 现象（末期 ~5–6x）**。**int8 坐安全线之上。** 但早晚 step 差异化容忍度文本 dLLM 没人测（future work），你要自测。

## 3.5 kernel/硬件 + Marlin 能否复用
- **vLLM dLLM 路径不支持 int8**：DiffusionGemma 只 ship FP8-dynamic + NVFP4，吞吐只 H100/H200 测。
- **Ampere int8 形势偏向你**：vLLM INT8 W8A8（CUTLASS）原生支持 CC>7.5 = Ampere sm_80/86（3090），per-channel int8 权重 + per-token dynamic int8 激活。反过来 **Ampere 无 FP8/FP4 tensorcore，INT8 在 Blackwell 被弃用** → **3090 上 int8 是唯一原生低精度选项。**
- **旁证**：2606.14598 在 RTX 3090（sm_86, mma.s8）fused W8A8 GEMM per-GEMM 2.79–4.18x vs bf16，且 A100 慢 1.38x、B200 慢 3.49x（consumer-Ampere 专属优势；但为图像 DiT 非文本 dLLM）。
- **计算区间反转（对你最有利）**：dLLM 每步去噪 = prefill-like 整 canvas 重算 = compute-bound（vLLM 原话 "trades memory bandwidth pressure for additional compute"）。对照你 MEMORY 的"int8 prefill +15% / decode 0 gain" → **dLLM 没有 bandwidth-bound 单 token decode，每步都吃 int8 prefill 收益。**

**Marlin 插进 denoiser 要改**：① 激活 scale **static→per-token dynamic**（可能每步重算；W4A8 静态 scale 在 dLLM 更宽 outlier 上会失败）；② causal→双向 + block KV 逻辑；③ **8-row decode tile 换 prefill tile**（dLLM 是 large-M）；④ mask/timestep-aware 校准（MCS/TMAS），权重侧叠 GPTQ+CGQ（drop-in 改 Hessian，kernel 无关）；⑤ attention 留 fp16 可能绕开 softmax×value 主误差源；⑥ MoE 三重未知（expert-routing × 迭代去噪 × int8 outlier）。

## 3.6 对 2×3090 的务实建议
- **今天能落地**：① 7–9B dLLM bf16 + **Fast-dLLM**（最高 27.6x / 6.18x over AR）；② W8A16 GGUF（省显存）。
- **要自己造**：W8A8/W4A8 激活-int8 真实 dLLM 内核（无引擎 ship）。
- **int8 有没有用**：有，且**比 AR 更有用**（每步 compute-bound）。预期 **~1.3–1.7x 计算加速**（端到端现实上限 ~1.1–1.7x，别信 DLLMQuant 的 1.71x）。
- **值不值得 vs bf16+Fast-dLLM**：想今天就快 → bf16+Fast-dLLM（**int8 必须打败的 baseline，很难打败**）；想用独有 edge 做 R&D → int8 才值得。
- **推荐 + 第一步**：先 **W8A8 不 W4A8**（W4A8 半边未验证 + AR W4A8 underflow 风险），用 **DuQuant rotation + per-token dynamic scale**。选 **LLaDA-8B 或 Dream-7B**（dense，Dream activation-int8 没人测过=你第一个），先 **fake-quant 复现 DLLMQuant Fig.2 INT8 平坦曲线**，过了再上真 kernel；**先别碰 MoE/DiffusionGemma**（最大的 build）。

## 3.7 核实清单（Part 3）
- **Supported**：W8A16 近乎无损 + 真 8-bit Marlin 权重 kernel 在 Ampere 可用；W8A8 精度近乎无损、W4A4 崩；dLLM 激活 outlier 横跨更多 token 需 mask-aware 重校准；无已部署 int8-激活文本 dLLM kernel；厂商只 FP8/NVFP4 需 Hopper/Blackwell，Ampere int8 唯一原生；3090 int8 GEMM 模式已验证（A100/B200 反输）；dLLM compute-bound int8 每步吃到。
- **Refuted**：❌某文本 dLLM 已有真 int8-激活硬件加速 demo；❌DLLMQuant 1.71x 来自真低 bit 激活 GEMM（=显存压缩指纹）。
- **Uncertain（你要自测）**：W4A8 在任何 dLLM 从未测；int8 误差真实 N 步是否真不累积（仅单图）；早晚 step 差异化容忍度；8-row decode tile 在 large-M prefill 下要否换 tile；dLLM 是否需 int8 KV-cache（vLLM #33480 open）；MoE dLLM int8 完全空白。

---

# Part 4 — 串联：对你这套栈的统一行动计划

**三研究关系**：转换（Part 1/2）和量化（Part 3）是两条独立的赌注。转换是"造一个新模型"（贵、research-grade）；量化是"让现成 dLLM 在你硬件上更快"（近、且正中你 edge）。**两者都不需要先做对方。**

**按性价比/风险排序的推荐路线：**

1. **【$0，当天】跑现成 dLLM baseline**：拉 Dream-7B 或 LLaDA-8B，bf16 + Fast-dLLM，测 3090 真实 tok/s（全网无此数据，本身有价值）。
2. **【$0，1–2 天】W8A8 fake-quant 验证**：denoiser 挂 W8A8 模拟量化，复现 DLLMQuant Fig.2——确认 8-bit 激活在 48–256 步去噪误差平坦。这是"要不要上真 kernel"的 go/no-go。
3. **【几天，build】真 W8A8 kernel**：把 Marlin int8 GEMM 以 W8A8 换进 linear 层，改 per-token dynamic scale + prefill tile。先 dense、先别碰 MoE。
4. **【可选并行，$0】转换 PoC**：若对 Qwen3.5-9B 转换有兴趣，跑 §2.6 阶段 0 的 forward-only go/no-go（同样 $0）；优先方案 C（block-diffusion）以复用量化栈。
5. **【$400–5k，仅前面信号强才做】**：租 8×H100 做 9B 小规模适配（§2.6 阶段 2）。

**一句话**：量化（Part 3）是近路且独有 edge，先做；转换（Part 1/2）是远路且烧钱，当 R&D 慢推；两条都别让它们挡住"bf16+Fast-dLLM 今天就能拿到的吞吐"这个 baseline。

---

# 附录 — 关键来源 URL

**DiffusionGemma / dLLM 模型**
- https://huggingface.co/google/diffusiongemma-26B-A4B-it · https://ai.google.dev/gemma/docs/diffusiongemma/explained · https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/ · https://vllm-project.github.io/2026/06/10/diffusion-gemma
- LLaDA https://arxiv.org/abs/2502.09992 · https://github.com/ML-GSAI/LLaDA ；LLaDA-MoE https://arxiv.org/abs/2509.24389

**AR→diffusion 转换**
- DiffuLLaMA/DiffuGPT https://arxiv.org/abs/2410.17891 · https://github.com/HKUNLP/DiffuLLaMA
- Dream-7B https://hkunlp.github.io/blog/2025/dream/ · https://arxiv.org/abs/2508.15487 · https://github.com/DreamLM/Dream
- RND1 https://www.radicalnumerics.ai/blog/rnd1 · https://github.com/RadicalNumerics/RND1
- MDLM https://arxiv.org/abs/2406.07524 ；Block Diffusion https://arxiv.org/pdf/2503.09573

**Qwen3.5-9B 架构 + GatedDeltaNet 双向化**
- config.json https://huggingface.co/Qwen/Qwen3.5-9B-Base/raw/main/config.json · docs https://huggingface.co/docs/transformers/en/model_doc/qwen3_5
- DiffuMamba/DiffuApriel https://arxiv.org/html/2511.15927 ；LION https://arxiv.org/abs/2502.16249
- SANA-WM https://arxiv.org/html/2605.15178v1 ；SANA-Streaming https://arxiv.org/html/2605.30409 ；WeDLM https://arxiv.org/abs/2512.22737
- fla https://github.com/fla-org/flash-linear-attention · https://github.com/fla-org/flash-bidirectional-linear-attention

**dLLM int8 量化**
- Quantization Meets dLLMs（核心，W8A8 无损/W4A4 崩/outlier）https://arxiv.org/abs/2508.14896
- DLLMQuant（Fig.2 误差累积）https://arxiv.org/abs/2508.14090 ；Quant-dLLM https://arxiv.org/abs/2510.03274
- CoDA 8-bit https://arxiv.org/html/2604.20079v1 ；3090 int8 DiT GEMM https://arxiv.org/html/2606.14598 ；STaR-Quant https://arxiv.org/abs/2606.04945
- Fast-dLLM https://arxiv.org/abs/2505.22618 · https://nvlabs.github.io/Fast-dLLM/ ；vLLM INT8 docs https://docs.vllm.ai/en/latest/features/quantization/int8/
- H100 2026 租价 https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/

---

*相关 memory：`project_ar_to_diffusion_conversion` · `project_qwen35_9b_diffusion_conversion` · `project_dllm_int8_quant`。本文为三者的合并展开归档。*
