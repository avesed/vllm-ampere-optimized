export const meta = {
  name: 'runtime-fp8-draft-lmhead',
  description: 'Feasibility + design + impl plan for a load-time-quantized (fp8/int4) MTP DRAFT lm_head on Ampere (verify stays bf16)',
  phases: [
    { title: 'Design',     detail: '7 aspects: load-sharing, runtime-quant mechanics, routing, fp8-vs-int4, accuracy, mem/config/risk, precedent' },
    { title: 'Challenge',  detail: 'verify each against real code; stress the 0%-accept + cudagraph + correctness risks' },
    { title: 'Synthesize', detail: 'feasibility verdict + grounded implementation plan + measurement plan' },
  ],
}

const F = '/home/trevor/vllm-ampere-optimized/vllm'
const MEM = '/home/trevor/.claude/projects/-home-trevor-vllm-ampere-optimized/memory'

const CTX = `DESIGN UNDER EVALUATION (the user's): at model load, IF MTP/speculative-decode is enabled, automatically COPY the MTP draft's lm_head and quantize it (fp8, or int4) into a SEPARATE low-precision head; the MTP draft proposer uses this quantized head, while the base model's VERIFY lm_head stays bf16. Goal = cut the draft-side HBM bytes (each draft token currently reads the full bf16 lm_head ≈2.0GB on 9B / ~2.5GB on 27B; the K draft reads dominate the draft step).

GROUNDED FACTS (already confirmed, build on these):
- The MTP draft projects through ITS OWN head: \`Qwen3_5MTP.self.lm_head\` (a separate \`ParallelLMHead\` at ${F}/vllm/model_executor/models/qwen3_5_mtp.py:381, or tied to mtp embed_tokens if config.tie_word_embeddings). The draft path is \`self.model.compute_logits(hidden).argmax()\` / \`get_top_tokens\` in ${F}/vllm/v1/spec_decode/llm_base_proposer.py:406-407,429 — \`self.model\` IS the MTP module. So quantizing the MTP head does NOT touch the base verify lm_head (a different module). compute_logits = logits_processor(self.lm_head, hidden) at qwen3_5_mtp.py:437.
- fp8 W8A16 Marlin ALREADY EXISTS for Ampere (dequant fp8->fp16, then HMMA): ${F}/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a16_fp8.py + utils/marlin_utils_fp8.py + fp8.py + csrc/quantization/marlin/marlin_int4_fp8_preprocess.cu. The W4A16 int4 Marlin path also exists (the fork's flagship). So NO new GEMM kernel is needed — the work is runtime-quantize + wire.
- HARDWARE: Ampere sm_80/sm_86 has NO fp8 tensor cores (sm_89+). So "fp8 lm_head" = W8A16 (1 byte/wt, dequant->fp16 HMMA, ~half the bf16 bytes = 2x cut). int4 = 4x cut but needs Marlin repack at load. Decode is HBM-bandwidth-bound so byte-cut ≈ draft speedup.
- CORRECTNESS: the base model VERIFIES exactly, so a low-precision DRAFT head can NEVER corrupt output — it only changes which tokens are PROPOSED ⇒ at worst a small accept-len drop. This is the safety argument; state it precisely.

KNOWN RISK to settle: the fork's MTP memory ([${MEM}/project_spec_decode_ampere.md]) says "mtp.fc must stay bf16 AND mtp must be in the quant-ignore (\`re:.*mtp.*\`) or you get silent 0% accept". That ignore is about the OFFLINE ckpt quantizer (compressed-tensors) wrongly quantizing mtp.fc (the input-combine). This design RUNTIME-quantizes mtp.LM_HEAD (the output projection) — a DIFFERENT tensor. Resolve whether runtime-quantizing the output head is safe (it feeds argmax for proposals, verify re-scores exactly) vs whether it risks the same 0%-accept failure mode. Read the actual code + that memory.

Models: Qwen3.5/3.6 9B (hidden 4096, vocab 248320 → lm_head 2.03GB bf16), 27B, 35B-A3B MoE. The fork already ships graft_mtp tooling. Read ${MEM}/project_spec_decode_ampere.md, project_kernel_throughput_directions.md (D12-16 lm_head levers), project_general_ampere_patch_roadmap.md (W8A16-fp8 Marlin auto <sm89).`

const SCHEMA = {
  type:'object', additionalProperties:false, required:['aspect','findings','recommendation'],
  properties:{
    aspect:{type:'string'},
    findings:{ type:'array', items:{ type:'object', additionalProperties:false,
      required:['point','detail','where','risk','what_to_verify'],
      properties:{
        point:{type:'string'},
        detail:{type:'string', description:'concrete mechanism / fact, grounded in code'},
        where:{type:'string', description:'file:line in the fork (vllm/ or csrc/) or the memory file'},
        risk:{type:'string', description:'failure mode / caveat / open question; "none" if clean'},
        what_to_verify:{type:'string', description:'experiment or code-check that resolves it'},
      } } },
    recommendation:{type:'string', description:'this aspect\'s crisp recommendation for the design'},
  },
}

const DIMS = [
  { key:'lmhead-load-sharing', title:'Does mtp.lm_head load its OWN bf16 weights or share/point-at the base lm_head?',
    focus:`Resolve PRECISELY (decides memory delta + the hook point). Read ${F}/vllm/model_executor/models/qwen3_5_mtp.py:378-388 (ParallelLMHead vs tie) + load_weights:439-452 (does it load a distinct mtp.lm_head tensor, or remap/skip so the MTP head ends up pointing at the base model's lm_head?), and how the proposer builds self.model (llm_base_proposer load path). Also check qwen3_6 if present. If SEPARATE bf16 tensor → runtime-quant SAVES ~1GB(9B)/~1.25GB(27B) VRAM on top of the speedup. If SHARED (points at base) → must allocate a SEPARATE fp8 copy = +~1GB, and must NOT mutate the shared tensor (would break verify). State which it is with the code line that proves it.` },
  { key:'runtime-quant-mechanics', title:'Where/how to runtime-quantize the head at load + route it to the existing Ampere kernel',
    focus:`Design the load-time hook: after MTP weights load, gated on (speculative MTP enabled + a flag). To produce fp8: compute an e4m3 scale (per-tensor vs per-output-channel) from the bf16 lm_head, cast to fp8, and wrap it so compute_logits routes through the EXISTING ${F}/vllm/.../compressed_tensors_w8a16_fp8.py / marlin_utils_fp8.py Ampere kernel (study how that scheme's process_weights_after_loading + apply work, and marlin_int4_fp8_preprocess.cu). Same for int4 via the W4A16 Marlin repack (RTN per-group absmax + the gptq/awq marlin repack at load). Identify the cleanest injection point (a wrapper QuantMethod on mtp.lm_head, or swap the ParallelLMHead's linear_method). MUST be cudagraph-capturable (no .tolist()/.item()/dynamic shape at decode — the draft runs under cudagraph per the cudagraph investigation). What does process_weights_after_loading need?` },
  { key:'draft-routing', title:'Route the draft through the quantized head; keep verify bf16 + no double-materialization',
    focus:`Confirm the draft's logits path (llm_base_proposer.py:406-407 get_top_tokens / :407,429 compute_logits→qwen3_5_mtp.py:437 logits_processor(self.lm_head,...)) and that swapping self.lm_head's compute to the quantized kernel is sufficient + isolated from the base verify head (which lives in the target model, a different module — confirm where the VERIFY lm_head is and that it's untouched). Interaction with use_local_argmax_reduction / get_top_tokens (does the quantized path still support local-argmax, avoiding the full-vocab materialization?). Ensure greedy draft argmax over fp8-dequant logits is well-defined + capturable.` },
  { key:'fp8-vs-int4', title:'fp8 (2x, trivial runtime, existing W8A16 kernel) vs int4-RTN (4x, repack-at-load) — recommend',
    focus:`Quantify the byte accounting per accept cycle on 9B / 27B / 35B-A3B: base body read once + verify-lm_head once + K×(mtp-head + draft-lm_head). Show draft-lm_head as a fraction of cycle bytes, and the decode-tok/s delta if it goes bf16→fp8 (÷2) vs bf16→int4 (÷4). Effort/accuracy tradeoff: fp8 e4m3 runtime quant is trivial (scale+cast, no calibration) and the W8A16 kernel exists; int4-RTN needs the Marlin repack at load + group scales (more code) but 2x more byte cut. per-tensor vs per-output-channel fp8 scale impact on draft accept-len (per-channel ≈ near-lossless proposals). Recommend the FIRST-SHIP choice + the higher-ceiling follow-up, and whether vocab-prune stacks.` },
  { key:'accuracy-acceptlen', title:'Correctness guarantee + accept-len cost + the validation plan',
    focus:`State the correctness proof: because the base model re-scores every proposed token EXACTLY with the bf16 verify head, the EMITTED tokens are independent of the draft head precision (modulo sampling RNG / tie-breaks) → output should be byte-identical to a bf16-draft baseline under fixed seed; the ONLY cost is accept-len (fewer proposals accepted). Design the measurement: accept-len fp8-draft vs bf16-draft on diverse content AND Chinese long-CoT (where the effective vocab is large — worst case for a quantized head); GSM8K/MMLU should be UNCHANGED (verify exact); the harness (bench_decode_clean.py, the graft_mtp accept-len check). Estimate how much e4m3 (and per-tensor vs per-channel) plausibly moves accept-len. Net-win condition: draft-step speedup × (accept-len_fp8 / accept-len_bf16) > 1.` },
  { key:'mem-config-risk', title:'VRAM, the flag, graft_mtp/quant-ignore interaction, and the 0%-accept risk',
    focus:`VRAM delta (from the shared-vs-separate finding). The opt-in flag (default-on-when-MTP, or off-by-default? mem-tight 27B-W4A8 is VRAM-constrained per project_e2e_perf_validation). CRITICAL RISK: read ${MEM}/project_spec_decode_ampere.md on "mtp.fc bf16 + re:.*mtp.* quant-ignore or 0% accept" — establish that this design quantizes mtp.LM_HEAD (output proj) NOT mtp.fc (input combine), and reason about whether quantizing the OUTPUT head can also cause a 0%-accept collapse (e.g. if a bad scale makes the draft argmax degenerate). The runtime path also must not collide with the OFFLINE compressed-tensors ignore (which keeps the whole mtp.* bf16 in the ckpt — good, because we quantize at RUNTIME from that bf16). How it ships (fork patch). Add the degenerate-stream quality canary from [[project_1cat_vllm_crossanalysis]] (accept=1.0=collapse) to the validation.` },
  { key:'external-precedent', title:'Prior art: reduced-precision / quantized draft / speculator heads',
    focus:`Survey whether other engines / papers ship a QUANTIZED-DRAFT-HEAD-ONLY (or low-precision speculator) design: SGLang EAGLE, TensorRT-LLM speculative, Medusa head quantization, EAGLE-2/3, the "draft can be lower precision than target" idea, fp8/int4 draft heads, self-speculative with quantized heads. Use WebSearch/WebFetch (load via ToolSearch). Does anyone runtime-quantize the draft head specifically because the target verifies exactly? Validate or complicate the user's design with precedent + any known accept-len numbers.` },
]

phase('Design')
const findPrompt = (d) => `You are designing a SHIPPABLE feature for the Ampere vLLM fork. Read real code under ${F}.

${CTX}

YOUR ASPECT: ${d.title}
${d.focus}

RULES: ground every claim in a real file:line. Be concrete and implementation-oriented (this becomes an impl plan, not a survey). Flag risks honestly — especially the 0%-accept and cudagraph-capturability failure modes. This is a feasibility+design task: give a clear recommendation. 4-8 findings + a crisp recommendation.`

const challengePrompt = (d, found) => `Independent reviewer hardening the design aspect "${d.title}" for the runtime-fp8-draft-lmhead feature. Read ${F} to verify.

DRAFT: ${JSON.stringify(found).slice(0,9000)}

For EACH finding: (1) confirm the code fact at the cited file:line (correct if wrong). (2) Stress the failure modes: would this break cudagraph capture (any .tolist()/.item()/dynamic shape)? Could it trigger the 0%-accept collapse? Does it actually leave the base VERIFY head untouched? (3) Sanity-check the byte math + the Ampere-has-no-fp8-TC constraint (fp8 = W8A16 dequant path, not native fp8 GEMM). (4) Sharpen what_to_verify into a runnable check. Keep good, correct weak, drop unsupported, add up to 2 missed. Return the full augmented aspect. ${CTX}`

const results = await pipeline(
  DIMS,
  (d) => agent(findPrompt(d), { label:`design:${d.key}`, phase:'Design', schema:SCHEMA }),
  (found,d) => agent(challengePrompt(d,found), { label:`challenge:${d.key}`, phase:'Challenge', schema:SCHEMA }),
)
const all = results.filter(Boolean)
log(`Design+Challenge: ${all.length}/${DIMS.length} aspects, ${all.flatMap(r=>r.findings??[]).length} findings`)

phase('Synthesize')
const corpus = JSON.stringify(all).slice(0,100000)
const report = await agent(
  `Lead author: write the DESIGN DOC for "runtime-quantized MTP draft lm_head (fp8/int4) on Ampere" for the fork maintainer. Markdown. This is a feasibility+design task — give a clear VERDICT and a concrete implementation plan.

${CTX}

ALL DESIGN FINDINGS (7 aspects, find+challenge) JSON:
${corpus}

WRITE:
1. **Verdict** (4-6 lines): is the design sound + shippable? the headline (separate mtp.lm_head module + existing Ampere W8A16-fp8 kernel + verify-exact correctness ⇒ near-kernel-free, output-lossless, only accept-len at risk). State the expected decode win range (byte math) and the ONE biggest risk.
2. **Resolved mechanics** — mtp.lm_head shared-vs-separate (with the proof line + the VRAM consequence); the load-time quant hook; how it routes to the existing kernel; how the draft path picks it up; verify stays bf16.
3. **fp8 vs int4 recommendation** — the byte/effort/accuracy table (9B/27B/35B), first-ship choice + higher-ceiling follow-up, scale granularity, vocab-prune stacking.
4. **Correctness & accept-len** — the exactness proof (output independent of draft precision), the accept-len cost, the validation plan incl. the degenerate-stream canary + Chinese long-CoT worst case.
5. **Risks & mitigations** — the 0%-accept interaction (mtp.lm_head vs mtp.fc), cudagraph capturability, VRAM on mem-tight 27B, the flag/default.
6. **Implementation plan** — ordered steps with the exact files to touch (qwen3_5_mtp.py, the proposer, the quant scheme wrapper, the flag in config/arg_utils), what reuses existing infra, and the smallest first PR (fp8, single model, accept-len A/B).
7. **Measurement plan** — the A/B that proves net-win (draft-step tok/s × accept-len ratio), the metrics, the harness.
Anchor to file:line. Be decisive and concrete. Return ONLY the Markdown.`,
  { label:'synthesize', phase:'Synthesize', effort:'xhigh' })

return { report, aspects: all.length, findings: all.flatMap(r=>r.findings??[]).length }
