export const meta = {
  name: 'ampere-decode-optimization-directions',
  description: 'Collect the FULL menu of decode optimization directions for the Ampere W4A8 hybrid+MoE+MTP fork (no go/no-go, no impact-sizing, EXCLUDE lm_head quant)',
  phases: [
    { title: 'Sweep',      detail: '7 decode axes, collect directions broad (incl. long-tail), cross-ref existing verdicts' },
    { title: 'Enrich',     detail: 'anchor to file:line, mark status (new / already-verdicted / long-tail), sharpen what-to-measure' },
    { title: 'Critic',     detail: '2 lenses: missing decode axis / first-principles amortize-vs-bytes-vs-DRAM' },
    { title: 'Synthesize', detail: 'organized decode direction menu, no go/no-go' },
  ],
}

const F = '/home/trevor/vllm-ampere-optimized/vllm'
const MEM = '/home/trevor/.claude/projects/-home-trevor-vllm-ampere-optimized/memory'

const CTX = `Fork = \`vllm-ampere-optimized\` (vLLM v0.23.0, Ampere sm_80/sm_86), serving Qwen3.5/3.6 27B & 35B-A3B HYBRID (GatedDeltaNet linear-attn + few full-attn hd256 layers) + MoE + MTP, W4A8 (int4 weight + int8 act). Source at ${F}.

DECODE = HBM-WEIGHT-BANDWIDTH-BOUND. The 3-knob mental model for decode tok/s ≈ (achieved-DRAM-BW × tokens-per-weight-sweep) / bytes-per-sweep: **(A) AMORTIZE** — more tokens per weight-sweep (spec-decode/MTP, batching); **(B) FEWER BYTES** — weights/KV/activations per token; **(C) FASTER/FULLER DRAM** — peak (mem-OC), occupancy, fill idle. Plus a **(D) SYSTEM/SCHEDULING** layer (overlap, host-overhead, sampling).

TASK: COLLECT the FULL MENU of decode optimization DIRECTIONS. The user explicitly said: do NOT render go/no-go, do NOT size the impact, and breadth > filtering — include uncertain, small, and long-tail/speculative ideas. **EXCLUDE lm_head quantization** (already covered: ${MEM}/project_mtp_draft_lmhead_quant.md + kernel-directions D12-16).

DO cross-reference existing memory verdicts (read the relevant ${MEM}/project_*.md) and MARK each direction's status — new / already-has-a-verdict / long-tail-speculative — so we don't double-count, WITHOUT judging it dead. Existing decode-relevant verdicts to cross-ref (list-but-mark, don't re-pitch as new): MTP+patch-A fwd_kvcache (the current #1 decode lever; project_spec_decode_ampere + project_mtp_verify_attention_fix); mem-OC (+3.8% validated, project_autotune_gpu_oc); KV-quant = capacity-not-speed / fp8-KV (project_kv_quant_ampere_verdict + project_int8_qk_prefill_ampere); GDN recurrent launch-tuning DEAD / state-bandwidth-bound (project_general_ampere_patch_roadmap / ROADMAP); cudagraph = decode already FULL-graph + DRAM-SOL, one MoE phantom-expert lever (project_cudagraph_decode_bandwidth); int8-GEMM TAPPED-OUT (project_int8_path_roadmap_ampere); the kernel-directions D-menu (project_kernel_throughput_directions). Prefer shippable general sm80/86 fork value, but INCLUDE per-deployment/system ideas too — just label scope.`

const SCHEMA = {
  type:'object', additionalProperties:false, required:['axis','directions','takeaways'],
  properties:{
    axis:{type:'string'},
    directions:{ type:'array', items:{ type:'object', additionalProperties:false,
      required:['title','knob','mechanism','where','regime','status','relation_to_known','what_to_measure'],
      properties:{
        title:{type:'string'},
        knob:{type:'string', description:'A-amortize | B-fewer-bytes | C-faster-DRAM | D-system'},
        mechanism:{type:'string', description:'the concrete decode optimization'},
        where:{type:'string', description:'file:line in the fork, or external paper/repo/PR'},
        regime:{type:'string', description:'b1 single-stream / batched / long-ctx / MoE / hybrid-GDN / which'},
        status:{type:'string', description:'new | already-verdicted (cite the memory verdict) | long-tail-speculative'},
        relation_to_known:{type:'string', description:'how it relates to existing levers/verdicts'},
        what_to_measure:{type:'string', description:'the experiment that would later resolve it (NOT a verdict)'},
      } } },
    takeaways:{ type:'array', items:{type:'string'} },
  },
}

const DIMS = [
  { key:'specdecode-amortize', title:'Spec-decode / amortization beyond basic MTP (the #1 decode axis)',
    focus:`The biggest decode lever is reading the weight sweep once and emitting MORE tokens. Beyond the shipped MTP+patch-A: TREE / multi-candidate drafting (EAGLE-2/3 dynamic trees, Medusa-style multi-head, token-tree verify); adaptive/dynamic K (speculation length from running acceptance); multi-token MTP heads / stacked draft layers; draft-model-FREE methods (n-gram / prompt-lookahead / retrieval / Lookahead-decoding Jacobi) good for code/repetitive; parallel vs sequential drafting; the draft NETWORK itself (its attention/MLP, NOT its lm_head which is excluded); speculative cascades; self-speculation / layer-skip draft. Read ${F}/vllm/v1/spec_decode + worker/gpu/spec_decode. Cross-ref project_spec_decode_ampere (K=2 sweet, accept ~2.2) + project_mtp_verify_attention_fix. What raises accept-len or tokens/step?` },
  { key:'kv-attention-decode', title:'KV-cache & attention decode bandwidth (full-attn layers + paged)',
    focus:`Decode attention + KV reads. GQA multi-query packing (fold q-heads-per-kv into the decode tile — the 1Cat XQA group-pack); cascade / shared-prefix attention (one KV read serves many requests, RadixAttention-style); KV eviction / sparse decode (H2O / SnapKV / Quest / StreamingLLM) to cut long-ctx KV bytes; KV layout / block-size / coalescing for the paged gather; fp8-KV (capacity, has verdict); decode-attention kernel (FA2 fwd_kvcache / the verify path) occupancy at q=1. Read ${F}/vllm/v1/attention. Cross-ref project_kv_quant_ampere_verdict (capacity-not-speed) + project_mtp_verify_attention_fix. Mark long-ctx vs short.` },
  { key:'weight-stream', title:'Weight-stream reduction — the dominant decode cost (EXCLUDING lm_head)',
    focus:`The int4 body weights streamed every token dominate. Per-layer / mixed bit-allocation (push insensitive layers <4bit, keep sensitive higher); sub-4-bit weight attempts (W3/W2 — note no Ampere kernel, accuracy); MoE active-set reduction at decode (the cudagraph phantom-expert masking; expert pruning / caching hot experts; smaller top_k); the dequant-GEMV efficiency at small-M (Marlin blocks_per_sm=D9, thread_n shrink, tile); mem-OC (peak-raise, validated); weight prefetch / L2 (mostly can't — weights ≫ L2). EXCLUDE lm_head. Read ${F}/csrc/quantization/marlin + csrc/moe. Cross-ref kernel-directions D2/D9, project_cudagraph_decode_bandwidth (phantom-expert), project_autotune_gpu_oc (mem-OC), int8-path (tapped-out).` },
  { key:'gdn-hybrid-decode', title:'GatedDeltaNet / Mamba2 hybrid recurrent decode',
    focus:`The 75%-of-layers GDN linear-attn decode path. The recurrent state is bandwidth-bound (launch-tuning DEAD, bf16 state already default). What's left: state COMPRESSION / sub-bf16 state (note accumulation-error NO-GO history — but is there a bounded variant?); fusing the GDN decode sub-ops (conv1d + gating + recurrent) to cut launches/round-trips; selective_state_update / fused_recurrent kernel efficiency; chunk-size at decode; the causal_conv1d state-update. Read ${F}/vllm/model_executor/layers/.../gdn + mamba + ${F}/csrc (if any). Cross-ref the ROADMAP Tier-B (GDN DEAD by measurement) + feedback_qwen35_hybrid_kernels — but the user wants the full menu so list even the dead-by-measurement ones, marked.` },
  { key:'system-scheduling-overlap', title:'System / scheduling / overlap / host-overhead at decode',
    focus:`Non-kernel decode throughput. Prefill-decode OVERLAP (chunked-prefill mixing keeps the GPU/DRAM busy during decode bubbles); continuous-batching efficiency; async scheduling (default-on); multi-step / scheduler overhead; the per-step host work (input prep, block-table build, sampling) and whether it gates the GPU at b1 (the cudagraph investigation found decode is FULL-graph + the b1 duty-cycle host-latency cluster); sampling / logits efficiency (the NON-lm_head part — penalties, top-k/p, the [B,vocab] fp32 round-trip D16); CPU-ahead launch / prefetch-next-step. Read ${F}/vllm/v1/worker/gpu_model_runner.py + v1/core/sched. Cross-ref project_cudagraph_decode_bandwidth.` },
  { key:'batching-capacity', title:'Batching & capacity → aggregate decode throughput',
    focus:`Weights are batch-invariant → bigger batch amortizes the weight read across more tokens (the other amortization axis besides spec). Capacity that ENABLES bigger batch: fp8-KV / mamba-state-dtype (free KV/state bytes → more concurrent seqs), prefix-cache reuse, max-num-seqs (per-deploy), expert-parallel for MoE decode, the spec-vs-batch crossover (spec helps per-stream/b1, batch helps aggregate — do they compose or compete?). Read ${F}/vllm/v1/core + config. Cross-ref project_autotune_gpu_oc (max-num-seqs 128 optimum, fp8 unlocks concurrency). Mark per-deployment vs shippable-default.` },
  { key:'novel-crossengine', title:'Novel / long-tail / cross-engine decode tricks (the "what are we not thinking about" lens)',
    focus:`Survey decode-specific tricks from OTHER engines + recent papers that this fork hasn't considered, on Ampere-runnable terms: SGLang (RadixAttention, overlap scheduler, cuda-graph decode), TensorRT-LLM (in-flight batching, XQA decode), lmdeploy/TurboMind (XQA group-pack), exllamav2/v3, the 1Cat (V100) + 2080Ti forks; and papers: layer-skip / early-exit decoding, dynamic-depth, Jacobi/lookahead, activation-aware / cache-aware decoding, KV-prefetch, speculative-cascades, multi-token prediction variants, quantized-activation-cache. USE WebSearch/WebFetch (load via ToolSearch). The user wants long-tail — include speculative ideas, clearly marked. Cite sources.` },
]

phase('Sweep')
const findPrompt = (d) => `You are collecting DECODE optimization DIRECTIONS for the Ampere W4A8 fork. Read real source under ${F} and cross-ref ${MEM}.

${CTX}

YOUR AXIS: ${d.title}
${d.focus}

RULES: NO go/no-go, NO impact-sizing — collect broad, include long-tail/speculative (mark them). EXCLUDE lm_head quant. Anchor to file:line where possible. For each direction set "status" (new / already-verdicted+cite / long-tail-speculative) and "knob" (A/B/C/D). Cross-ref existing verdicts honestly (list-but-mark, don't re-pitch dead as new). 5-10 directions + 2-4 takeaways.`

const enrichPrompt = (d, found) => `Independent reviewer enriching DECODE directions for "${d.title}". Read ${F} + ${MEM} to verify.

DRAFT: ${JSON.stringify(found?.directions ?? []).slice(0,9000)}

For EACH: (1) confirm the mechanism + file:line is real (not filename-inferred). (2) Fix the "status" — if it's actually already-verdicted in memory, mark it + cite (don't let a dead lever masquerade as new); if genuinely new/long-tail, keep. (3) Tighten what_to_measure into a concrete experiment. (4) Keep the knob (A/B/C/D) honest — is it amortize / fewer-bytes / faster-DRAM / system? Keep good, correct weak, add up to 2 missed (esp. long-tail the draft lacked). NO go/no-go. ${CTX}`

const dimResults = await pipeline(
  DIMS,
  (d) => agent(findPrompt(d), { label:`sweep:${d.key}`, phase:'Sweep', schema:SCHEMA }),
  (found,d) => agent(enrichPrompt(d,found), { label:`enrich:${d.key}`, phase:'Enrich', schema:SCHEMA }),
)
const all = dimResults.filter(Boolean)
const allDirs = all.flatMap(r=>(r.directions??[]).map(x=>({...x,_axis:r.axis})))
log(`Sweep+Enrich: ${allDirs.length} decode directions across ${all.length} axes`)

phase('Critic')
const titles = allDirs.map((x,i)=>`${i}. [${x.knob}|${x.status}] ${x.title}`).join('\n').slice(0,11000)
const CRITICS = [
  { key:'missing-axis', q:`What DECODE optimization is ABSENT from the list below (excluding lm_head quant)? Think across: spec-decode variants, KV/attention, weight-stream, GDN/hybrid, scheduling/overlap, batching/capacity, AND cross-cutting (quantized activation cache, draft-model sharing, multi-request fusion, output-token batching, embedding/rope decode cost, the int8 per-token quant op at decode, sampling penalties). Propose concrete missing directions with knob + status.` },
  { key:'first-principles', q:`First-principles on a weight-BW-bound decode: the ONLY ways to go faster are (A) more tokens per weight-sweep, (B) fewer bytes per token, (C) higher achieved/peak DRAM, (D) less GPU-idle. For each knob, name the HIGHEST-leverage decode direction nobody listed, and call out any listed direction that is mis-classified (e.g. a 'bandwidth' idea that's really host-latency, or a per-deployment knob dressed as a kernel lever). Be the skeptic but propose, don't just critique.` },
]
const crit = await parallel(CRITICS.map(l=>()=>agent(
  `Completeness critic for a DECODE direction sweep (Ampere W4A8). ${CTX}\n\nCOLLECTED (titles):\n${titles}\n\nLENS: ${l.q}\n\nSurface genuinely-missing or mis-classified directions. NO go/no-go. EXCLUDE lm_head quant.`,
  { label:`critic:${l.key}`, phase:'Critic', schema:SCHEMA })))
const critDirs = crit.filter(Boolean).flatMap(r=>(r.directions??[]).map(x=>({...x,_axis:'critic:'+r.axis})))
log(`Critic: ${critDirs.length} added`)

phase('Synthesize')
const corpus = JSON.stringify([...allDirs,...critDirs]).slice(0,100000)
const report = await agent(
  `Lead author: consolidate the FULL DECODE OPTIMIZATION DIRECTION MENU for the Ampere W4A8 hybrid+MoE+MTP fork. Markdown. The user wants BREADTH and NO go/no-go, NO impact-sizing — this is a menu to pick from later. lm_head quant is EXCLUDED (covered elsewhere).

${CTX}

ALL DIRECTIONS (sweep+enrich+critic) JSON:
${corpus}

WRITE:
- 4-6 line orientation (the 3-knob+system model; that decode is weight-BW-bound so amortize (A) + fewer-bytes (B) dominate; this is a no-verdict menu).
- DEDUP/MERGE across axes.
- A "Direction index" table: # · title · knob (A-amortize/B-bytes/C-DRAM/D-system) · regime · status (new / already-verdicted / long-tail).
- GROUP by knob (A Amortize · B Fewer-bytes · C Faster-DRAM · D System/scheduling). Within each, one entry per direction: Mechanism · Where (file:line / paper) · Regime · Status (+ cite the memory verdict if already-verdicted) · What to measure.
- A "Already-verdicted (listed for completeness, not re-pitched)" note grouping the dead/known ones with their one-line verdict, so the NEW + long-tail directions stand out.
- A short "Long-tail / speculative (worth a look despite uncertainty)" highlight.
- NO go/no-go, NO impact estimates anywhere. Be technically dense and Ampere-specific. Return ONLY the Markdown.`,
  { label:'synthesize', phase:'Synthesize', effort:'xhigh' })

return { report, axes: all.length, directions: allDirs.length, critic_adds: critDirs.length }
