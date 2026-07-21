export const meta = {
  name: 'ampere-kernel-throughput-directions',
  description: 'Collect deep kernel/engine-level throughput optimization DIRECTIONS for the vllm-ampere-optimized fork (no go/no-go verdicts)',
  phases: [
    { title: 'Map',         detail: 'ground-truth ledger + native-kernel inventory' },
    { title: 'Find',        detail: '18 kernel domains, parallel direction-collection' },
    { title: 'Challenge',   detail: 'per-domain novelty-check + file:line anchors + runnable experiment' },
    { title: 'Critic',      detail: '3 lenses: what kernel domain / technique class is entirely missing' },
    { title: 'Synthesize',  detail: 'dedup, cluster by layer, tag novelty, emit direction menu' },
  ],
}

// ----------------------------------------------------------------------------
const REPO = '/home/trevor/vllm-ampere-optimized'
const MEM  = '/home/trevor/.claude/projects/-home-trevor-vllm-ampere-optimized/memory'

const SCOPE = `Fork = vllm-ampere-optimized: vendored vLLM v0.23.0 + FlashInfer v0.6.12, built FROM SOURCE for Ampere ONLY (TORCH_CUDA_ARCH_LIST="8.0 8.6": sm_80=A100, sm_86=RTX3090/A40/A6000/A10). Native .cu/.cuh edits are allowed (this is a real source build, not a pip overlay). Target = MODERN model architectures on OLD Ampere silicon: hybrid linear-attn (GatedDeltaNet/Mamba2 + a few full-attn layers, head_dim ~256), modern MoE, MTP/spec-decode, and (dev-dllm branch) diffusion-LMs. Flagship quant = W4A8 (int4 weight + int8 DYNAMIC activation) via Marlin. Smem budget: sm_80=192KB, sm_86=100KB. Consumer GA10x (3090/A10/A40) fp16-input/fp16-ACCUM HMMA is 2x fp16/fp32-accum (a real TC lever); A100 GA100 fp16==fp32 accum rate. int4/int8 IMMA tensor cores exist on all Ampere; NO fp8 TC (needs sm89+), NO wgmma/TMA/tcgen05 (Hopper/Blackwell). Every direction must clear 3 BARS: (1) generalizes across the Ampere line, (2) shippable in the fork (patch / native kernel / pinned per-arch dep / tuned-config data / tooling), (3) serves modern arch — NOT old pure-full-attention dense, NOT per-deployment config (TP/PP/NVLink/gpu-mem-util/topology/max-num-seqs are OUT of scope). GOAL THIS RUN: COLLECT kernel/engine throughput optimization DIRECTIONS only. Do NOT render any go/no-go verdict — that is a later phase. Bias DEEP (kernel-level); shallow only if genuinely overlooked.`

const DIRECTIONS_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['dimension', 'directions'],
  properties: {
    dimension: { type: 'string' },
    directions: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['title','layer','mechanism','where','ampere_rationale','regime','novelty','relation_to_known','what_to_measure','external_refs'],
        properties: {
          title: { type: 'string', description: 'short imperative title of the optimization direction' },
          layer: { type: 'string', description: 'GEMM | attention | linear-attn | kv-cache | quant | moe | sampling | graph-runtime | dllm | memory | other' },
          mechanism: { type: 'string', description: 'the concrete kernel-level change / technique' },
          where: { type: 'string', description: 'exact code location (file:line in vllm/ or flashinfer/) or upstream kernel name, anchored to real source if possible' },
          ampere_rationale: { type: 'string', description: 'why this plausibly helps sm_80/sm_86 specifically — TC rates, bandwidth wall, occupancy, smem, accum-mode' },
          regime: { type: 'string', description: 'prefill / decode / batched / long-ctx / which model class it targets' },
          novelty: { type: 'string', description: 'new | nuance-of-known | re-examination' },
          relation_to_known: { type: 'string', description: 'how it relates to existing shipped / NO-GO / in-progress items in the ground-truth ledger' },
          what_to_measure: { type: 'string', description: 'the experiment that would resolve it (ncu metric / benchmark script / A-B) — NOT a verdict' },
          external_refs: { type: 'array', items: { type: 'string' }, description: 'papers / repos / vLLM PRs / kernels' },
        },
      },
    },
  },
}

// ============================================================ MAP
phase('Map')
const map = await parallel([
  () => agent(
    `You are a CUDA kernel archaeologist on the ${REPO} fork. Inventory EVERY native/custom kernel and hot-path kernel relevant to Ampere serving throughput. Read real source under ${REPO}/vllm/csrc (Marlin, attention, moe, quant, activation, norm, cache), ${REPO}/flashinfer, and the patch recipe ${REPO}/patches/. For each kernel: name, file path, what it does, its CURRENT state in this fork (stock / patched / custom), and any known Ampere tuning status (tile sizes, occupancy, accum mode, whether it has tuned JSON configs). Focus on: quantized GEMM (Marlin W4A8/W4A16, CUTLASS scaled_mm int8), attention (FlashAttention2, FlashInfer prefill/decode, paged attn), MoE (moe_wna16, fused_moe Triton), linear-attn (GatedDeltaNet/Mamba2 chunk-scan + fused_recurrent + causal_conv1d + selective_state_update), KV-cache reshape/copy, RMSNorm/activation/quant epilogue, lm_head/sampling. ${SCOPE}`,
    { label: 'map:kernel-inventory', phase: 'Map',
      schema: { type:'object', additionalProperties:false, required:['kernels','notes'],
        properties:{ kernels:{type:'array', items:{type:'object', additionalProperties:false, required:['name','file','role','state'], properties:{ name:{type:'string'}, file:{type:'string'}, role:{type:'string'}, state:{type:'string'} }}}, notes:{type:'string'} } } }
  ),
  () => agent(
    `You are building the canonical "already-explored" ledger for the ${REPO} fork so a direction-collection sweep does NOT re-collect dead ground. Read: ${REPO}/docs/ROADMAP.md, every ${REPO}/docs/RESEARCH-*.md, ${REPO}/patches/README.md, and the project memory files under ${MEM} (the project_*.md and feedback_*.md). Produce a TIGHT ledger: (a) SHIPPED kernel optimizations, (b) DEAD/NO-GO items WITH the one-line measured reason each died, (c) IN-PROGRESS / to-build, (d) the explicit "next investigation candidates" already named, (e) scope constraints. Be specific and terse — this feeds 18 downstream agents who must avoid re-proposing dead items as if new. ${SCOPE}`,
    { label: 'map:ledger', phase: 'Map',
      schema: { type:'object', additionalProperties:false, required:['shipped','dead','in_progress','next_candidates','scope'],
        properties:{ shipped:{type:'array',items:{type:'string'}}, dead:{type:'array',items:{type:'object',additionalProperties:false,required:['item','reason'],properties:{item:{type:'string'},reason:{type:'string'}}}}, in_progress:{type:'array',items:{type:'string'}}, next_candidates:{type:'array',items:{type:'string'}}, scope:{type:'string'} } } }
  ),
])
const groundStr = JSON.stringify({ kernel_inventory: map[0], ledger: map[1] }).slice(0, 7000)
log(`Map done: ${map[0]?.kernels?.length ?? 0} kernels inventoried; ${map[1]?.dead?.length ?? 0} dead items ledgered`)

// ============================================================ FIND DIMENSIONS
const DIMENSIONS = [
  { key:'marlin-gemm-internals', title:'Marlin W4A8/W4A16 GEMM kernel internals',
    focus:`Go BELOW the "int8-GEMM tapped out, 1.7% wall-clock gap" verdict. Examine Marlin tiling, cp.async pipeline stages/double-buffering, register & smem occupancy, L2 weight residency, dual-issue/ILP, the dequant tax, epilogue fusion. Is the ~68%-IMMA-vs-82% gap uniform across M, or only at one M? Sweep M-regimes esp. batched-decode M=8..64 and the 8-row decode tile (patch 0002). Stream-K / split-K / persistent-kernel for skinny GEMMs. QServe int8-domain dequant. Per-arch tile tuning (sm_80 192KB vs sm_86 100KB smem). Weight pre-swizzle/repack cost.` },
  { key:'lm-head-vocab-gemm', title:'lm_head / large-vocab output projection + sampling GEMM',
    focus:`Vocab ~248k -> the output projection is a big GEMM every decode step. int4/int8 lm_head (the named "lm_head-int4" candidate), fused dequant, vocab-parallel comm, skipping TP-padding vocab rows, fused argmax/top-k into the projection epilogue, speculative/partial lm_head, only-compute-needed-logits for spec-decode verify. Quantify per-step cost share at decode.` },
  { key:'moe-grouped-gemm', title:'Modern MoE on Ampere: grouped GEMM + routing kernels',
    focus:`moe_wna16 Marlin grouped GEMM (35B-A3B path), expert sort/permute/scatter-gather overhead, fused topk-softmax routing kernel, grouped-GEMM tiling for SMALL per-expert M (decode), shared-expert fusion, the per-expert int8 scale path (patch 0005). Is the gather/scatter or the GEMM the bottleneck at batch 1 vs batched? Better Ampere grouped-GEMM (CUTLASS C2X grouped, SGLang/ktransformers MoE kernels). The fused_moe Triton JSON config path.` },
  { key:'epilogue-activation-fusion', title:'Epilogue / activation / norm fusion to cut HBM round-trips',
    focus:`RMSNorm+quant fusion, SiLU/SwiGLU+mul fusion, residual-add+norm fusion, fused dequant in the GEMM epilogue, fusing the per-token int8 quant op INTO the preceding norm/residual so it is not a standalone HBM pass. Count kernels + HBM round-trips per transformer block on Ampere; which fusions are NOT yet done in v0.23 for the hybrid+W4A8 path. torch.compile fusion coverage vs hand-fused.` },
  { key:'fa2-prefill-ampere', title:'FlashAttention2 prefill kernel on Ampere (head_dim 256)',
    focus:`Prefill is compute-bound -> the lever. head_dim=256 path, occupancy, smem tiling, async softmax, warp-specialization limits on Ampere (no wgmma), the shipped fp16-accum-PV lever (GeForce-gated) — what is left BEYOND it? Better Q@K and P@V tiling, splitkv for prefill, register pressure at hd256, chunked-prefill kernel efficiency, FA2 vs FlashInfer prefill on Ampere. Persistent/stream-K attention.` },
  { key:'decode-attention-gemv', title:'Decode / paged attention GEMV + the bandwidth wall',
    focus:`Decode attention is a bandwidth-bound GEMV. KV layout, block size, the FA2 fwd_kvcache verify path (MTP fix), split-KV occupancy at q=1, GQA exploitation (group several q-heads per KV load), batched-decode multi-query kernel efficiency, KV prefetch / L2 residency, vectorized loads. Is there a kernel that better amortizes KV bandwidth across the batch? FlashDecoding / FlashInfer XQA (the famp XQA-verify finding). Decode is the hardest wall — characterize what specifically caps it.` },
  { key:'spec-decode-tree-attn', title:'MTP / spec-decode kernels (verify, tree-attn, draft)',
    focus:`MTP is the #1 decode lever here. Beyond the shipped verify-attention fwd_kvcache fix: tree/Medusa-mask batched attention, the multi-token MTP head GEMM, accept/reject sampling kernel, EAGLE-2/3-style dynamic draft trees, the draft lm_head->W4A16 idea, batching draft+verify, multi-branch verify. XQA un-gate for verify (q=1+K) per the famp finding. Where does spec-decode lose efficiency on Ampere at the kernel level?` },
  { key:'sparse-longctx-attn', title:'Sparse / long-context attention for the hybrid full-attn layers',
    focus:`The hybrid models have a few full-attn layers carrying long context. Native Sparse Attention (NSA), Quest, MInference, H2O/SnapKV-style KV pruning, block-sparse attention kernels, dynamic top-k KV selection — all as Ampere kernels. Long-ctx decode is where the full-attn layers dominate. What sparse-attn kernel could ship for sm_80/86 and serve hd256 hybrids? Named "next candidate".` },
  { key:'gdn-mamba-kernels', title:'GatedDeltaNet / Mamba2 linear-attn kernels (deeper)',
    focus:`Decode fused_recurrent is state-bandwidth-bound (launch-tuning DEAD; bf16 state already default). Go deeper: chunk-scan PREFILL tiling, conv1d fusion, fusing GDN sub-ops, chunk-size tuning, the Mamba2 selective_state_update Ampere tuned-JSON (Tier C, real gap for 7 Mamba2 families). Algorithmic kernel alternatives that REDUCE bytes moved for the recurrent state (the only thing that helps a bandwidth-bound kernel). CuteDSL/Triton chunk kernels portable to Ampere.` },
  { key:'kv-cache-mgmt', title:'KV-cache management kernels (copy / paging / prefix / cascade)',
    focus:`Paged copy/reshape kernels, block allocation, prefix-cache copy/dedup, hybrid-KV over-allocation (#37121), cascade/shared-prefix attention (one KV read serving many requests), KV-cache compression by eviction/merging (NOT quant) as kernels, the cache-reshape kernel cost on Ampere. RadixAttention-style prefix sharing. These cut bandwidth/capacity without touching numerics.` },
  { key:'quant-op-overhead', title:'Dynamic quantization op overhead (per-token fp16->int8)',
    focus:`The W4A8 path needs per-token dynamic int8 activation quant every layer. Is the quant (compute scale + quantize) a fused epilogue or a standalone HBM round-trip? Fuse it into the prior RMSNorm/residual. The Marlin input-dtype int8 quant prep cost. Per-token-vs-per-tensor scale kernel. Measure the quant op's share of the W4A8 layer. Group-size requant (g=-1, patch0001 line ~60).` },
  { key:'cudagraph-coverage', title:'CUDA-graph capture coverage (piecewise, hybrid, spec-decode)',
    focus:`Capture coverage is throughput on Ampere (launch amortization). Piecewise cudagraph for hybrid GDN+MTP (the 2080Ti-fork takeaway), what ops BREAK capture (CPU syncs like .tolist(), dynamic shapes), full vs piecewise, capture-size trimming, spec-decode/MTP capturability, the int8qk .tolist() capture-break lesson. Mamba/GDN in-place state-update vs capture. Which Ampere paths silently fall to eager?` },
  { key:'launch-fusion-decode', title:'Decode-loop kernel-launch overhead & op fusion',
    focus:`How many kernels per decode step for the hybrid+W4A8 model? Where cudagraph does NOT cover (spec-decode rejection, dynamic batch), host overhead, fused multi-op decode kernels (norm+qkv-quant+attn-prep), reducing the GDN decode op-count, torch.compile coverage on Ampere. The gap between captured and un-captured decode paths.` },
  { key:'sampling-logits-kernels', title:'Sampling / logits-processing kernels',
    focus:`top-k/top-p/min-p, temperature, repetition/presence penalties, the logits softmax/argmax, FUSED sampling, guided/structured-decoding mask application kernel cost, batched sampling efficiency, the penalty-tensor gather. At high batch the sampling+logits can be non-trivial. Flashinfer sampling kernels on Ampere.` },
  { key:'external-sota-2025', title:'External SOTA kernels (2025-2026) applicable to Ampere sm_80/86',
    focus:`Survey the LAST ~year of kernel work that could run on Ampere int4/int8 IMMA + fp16 HMMA (NO fp8/wgmma). New FlashAttention variants/forks, new int4/int8 GEMM kernels, new quant schemes WITH an Ampere kernel, academic kernels targeting consumer GPUs, new open-source CUDA the fork hasn't seen. USE WebSearch/WebFetch (load via ToolSearch). Filter hard for "actually has an Ampere sm_80/86 kernel", not Hopper-only. Cite repos/papers/PRs.` },
  { key:'cross-engine-survey', title:'Cross-engine kernel tricks the fork lacks',
    focus:`What kernel-level tricks do OTHER engines use on Ampere/consumer GPUs that this fork has NOT adopted? Survey SGLang, TensorRT-LLM, lmdeploy/TurboMind, MLC-LLM/TVM, ktransformers, exllamav2/v3, llama.cpp CUDA, the weicj/vLLM-2080Ti fork. Focus: quantized GEMM, attention, MoE, KV, spec-decode kernels portable to sm_80/86. USE WebSearch/WebFetch. Be concrete about the specific kernel/technique and where it lives.` },
  { key:'dllm-kernels', title:'Diffusion-LM serving kernels (dev-dllm / DiffusionGemma)',
    focus:`The active branch serves DiffusionGemma (AR-Gemma-MoE adapted to a diffusion LM; denoising = prefill-like compute-bound). Kernel levers UNIQUE to dLLM: bidirectional/block-diffusion attention kernels, parallel multi-token denoising-step kernels, confidence-based unmasking kernel, KV-cache-for-diffusion (Fast-dLLM dual-cache), int8-act on the prefill-like compute (helps MORE than AR decode). Where does the dLLM serving path leave Ampere throughput on the table at the kernel level?` },
  { key:'profiling-grounding', title:'Profiling-grounded bottleneck re-discovery (meta/diagnostic)',
    focus:`From first principles + the existing harness (benchmarks/bench_marlin_gemm_imma.py ncu, torch_prof_phase, prof_decode_batchsweep), what should be RE-PROFILED to find the TRUE current kernel-share bottleneck for a modern hybrid+MoE+MTP W4A8 model on Ampere — decode vs prefill, per-kernel %, comm-excluded, across batch & ctx-len? Identify which kernel domains the data would say to prioritize, and what NEW diagnostic (ncu metric / kineto slice) is missing. This is a direction-prioritization lever, not a verdict.` },
]

// ============================================================ FIND -> CHALLENGE (pipeline)
phase('Find')
const findPrompt = (d) => `You are a senior CUDA / GPU-kernel performance engineer collecting THROUGHPUT optimization DIRECTIONS (no verdicts) for the vllm-ampere-optimized fork.

${SCOPE}

GROUND-TRUTH LEDGER (already shipped / dead-NO-GO with reasons / in-progress / next-candidates). Do NOT re-collect a dead item as if new — only surface it if you found a GENUINE NUANCE the prior measured analysis missed, and say so explicitly:
${groundStr}

YOUR DIMENSION: ${d.title}
${d.focus}

INSTRUCTIONS:
- Be EXHAUSTIVE and technically concrete. Prefer DEEP kernel-level levers (tiling, accum mode, occupancy, fusion, bytes-moved, IMMA/HMMA, smem, cp.async, swizzle) over shallow config knobs.
- ANCHOR to real source: read ${REPO}/vllm and ${REPO}/flashinfer (Read/Grep/Glob) and cite file:line or the exact upstream kernel name in "where".
- For external SOTA, load WebSearch/WebFetch via ToolSearch and cite repos/papers/PRs. Filter for "has an actual Ampere sm_80/86 kernel" — reject Hopper/Blackwell-only.
- For each direction fill the schema fully. Set "novelty" honestly (new / nuance-of-known / re-examination) and explain "relation_to_known" against the ledger.
- "what_to_measure" = the experiment that would later resolve it (ncu metric / which benchmark script / A-B). DO NOT write a go/no-go verdict anywhere.
- Aim for 4-9 distinct directions. Quality + concreteness over count, but do not omit a real lever.`

const challengePrompt = (d, found) => `You are an INDEPENDENT kernel reviewer hardening a set of proposed optimization directions for dimension "${d.title}". Do NOT render go/no-go verdicts; your job is to make the directions accurate, anchored, and testable.

PROPOSED DIRECTIONS:
${JSON.stringify(found?.directions ?? []).slice(0, 9000)}

GROUND-TRUTH LEDGER:
${groundStr}

FOR EACH direction: (1) VERIFY the novelty claim against the ledger — downgrade to nuance-of-known / re-examination, or drop it, if it is actually a shipped or measured-dead item (note the ledger reason). (2) ADD a concrete kernel-level anchor: read ${REPO}/vllm and ${REPO}/flashinfer and put a real file:line or exact upstream-kernel name in "where". (3) SHARPEN "what_to_measure" into a runnable experiment (specific ncu metric / benchmark script under ${REPO}/benchmarks / A-B toggle). (4) You MAY ADD up to 2 adjacent directions this dimension missed.
Return the FULL augmented direction list (kept + added) via the schema. ${SCOPE}`

const dimResults = await pipeline(
  DIMENSIONS,
  (d) => agent(findPrompt(d), { label: `find:${d.key}`, phase: 'Find', schema: DIRECTIONS_SCHEMA }),
  (found, d) => agent(challengePrompt(d, found), { label: `challenge:${d.key}`, phase: 'Challenge', schema: DIRECTIONS_SCHEMA }),
)

const allDirections = dimResults.filter(Boolean).flatMap(r => (r.directions ?? []).map(x => ({ ...x, _dim: r.dimension })))
log(`Find+Challenge done: ${allDirections.length} directions across ${dimResults.filter(Boolean).length} dimensions`)

// ============================================================ CRITIC (completeness lenses)
phase('Critic')
const titleList = allDirections.map((x,i) => `${i}. [${x.layer}] ${x.title} (${x.novelty})`).join('\n').slice(0, 12000)
const CRITIC_LENSES = [
  { key:'domain-gap', q:`What entire KERNEL DOMAIN or sub-kernel is ABSENT from the collected directions below? Think across the full Ampere serving stack: GEMM, all attention variants, MoE, linear-attn/SSM, KV-cache, quant ops, norm/activation, sampling, embedding, communication-overlap (intra-kernel only, not topology), graph capture, memory allocator. Name what is missing and propose concrete kernel-level directions to fill it.` },
  { key:'modern-arch', q:`For THIS fork's modern-arch scope (hybrid GatedDeltaNet/Mamba2 + few hd256 full-attn layers + modern MoE + MTP spec-decode + diffusion-LM), what technique CLASS specific to these architectures was NOT covered? E.g. cross-layer fusion in hybrids, MoE+spec-decode interaction, diffusion-specific kernels, hd256-specific layouts. Propose concrete directions.` },
  { key:'first-principles', q:`From first-principles roofline reasoning on Ampere (decode = HBM-bandwidth-bound GEMV wall; prefill = IMMA/HMMA compute-bound; sm_86 100KB smem; fp16-accum 2x lever on GeForce), what HIGH-LEVERAGE kernel direction is implied that nobody listed? Challenge the collected set: what would a kernel expert say is the single biggest un-attacked throughput lever, and why?` },
]
const criticResults = await parallel(CRITIC_LENSES.map(l => () => agent(
  `You are a completeness critic for a kernel-throughput direction sweep on the vllm-ampere-optimized fork. ${SCOPE}

GROUND-TRUTH LEDGER:
${groundStr}

ALREADY-COLLECTED DIRECTIONS (titles):
${titleList}

YOUR LENS: ${l.q}

Surface ONLY genuinely-missing directions (not restatements of the list, not ledger-dead items). For each, fill the schema. No go/no-go verdicts.`,
  { label: `critic:${l.key}`, phase: 'Critic', schema: DIRECTIONS_SCHEMA })))

const criticDirections = criticResults.filter(Boolean).flatMap(r => (r.directions ?? []).map(x => ({ ...x, _dim: 'critic:' + r.dimension })))
log(`Critic done: ${criticDirections.length} additional gap-directions`)

// ============================================================ SYNTHESIZE
phase('Synthesize')
const corpus = JSON.stringify([...allDirections, ...criticDirections]).slice(0, 90000)
const report = await agent(
  `You are the lead author consolidating a kernel/engine-level THROUGHPUT optimization direction survey for the vllm-ampere-optimized fork. Produce a single well-structured Markdown report. ${SCOPE}

GROUND-TRUTH LEDGER (shipped / dead / in-progress):
${groundStr}

ALL COLLECTED DIRECTIONS (find + challenge + critic), as JSON:
${corpus}

WRITE THE REPORT:
- Title + a 4-6 line orientation (what was swept, the standing constraints, that this is a DIRECTION MENU with NO go/no-go).
- DEDUP and MERGE near-duplicate directions across dimensions (the same lever surfaced by multiple agents = ONE entry, note convergence).
- GROUP by kernel layer (GEMM/quant · Attention (full) · Linear-attn/SSM · MoE · KV-cache/memory · Spec-decode/MTP · CUDA-graph/runtime · Sampling/logits · Diffusion-LM · Diagnostics/tooling).
- Within each group, ONE subsection per direction with: **Mechanism**, **Where** (file:line / upstream kernel), **Ampere rationale**, **Regime**, **Novelty** (new / nuance-of-known / re-examination + relation to ledger), **What to measure** (the resolving experiment), **Refs**.
- Add a top "Direction index" table: # · layer · title · novelty · regime · primary-bar-risk, sorted so the most novel + highest-plausible-leverage DEEP kernel directions are first. Do NOT assign go/no-go — "primary-bar-risk" just names which of the 3 bars is most in question (generalizes / shippable / modern-arch).
- Add a short "Re-examinations of ledger-dead items" subsection ONLY where an agent found a genuine measured-nuance worth a second look, each with the original NO-GO reason quoted.
- End with a "Suggested first probes" list (cheapest experiments that would illuminate the most directions) — still NO verdicts, just sequencing of measurement.
Be technically dense and specific. This is for an expert maintainer. Return ONLY the Markdown.`,
  { label: 'synthesize', phase: 'Synthesize', effort: 'xhigh' }
)

return { report, counts: { dimensions: dimResults.filter(Boolean).length, directions: allDirections.length, critic_adds: criticDirections.length } }
