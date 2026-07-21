export const meta = {
  name: 'dflash-on-ampere-w4a8-feasibility',
  description: 'Feasibility of deploying off-the-shelf z-lab DFlash (block-diffusion) draft heads with W4A8 Qwen targets on 2x3090 Ampere — does it obsolete the MTP stack?',
  phases: [
    { title: 'Investigate', detail: '7 dimensions: the heads, the method, fork compat, Ampere compute, W4A8 interaction, vs-MTP-stack, deploy plan' },
    { title: 'Challenge',   detail: 'verify claims vs real fork code + hardware constraints; separate published-on-bf16 from Ampere-W4A8 reality' },
    { title: 'Synthesize',  detail: 'feasibility verdict + integration plan + what survives of the MTP roadmap' },
  ],
}

const F = '/home/trevor/vllm-ampere-optimized/vllm'
const MEM = '/home/trevor/.claude/projects/-home-trevor-vllm-ampere-optimized/memory'

const CTX = `EVALUATE: deploying OFF-THE-SHELF DFlash draft heads (z-lab, "Block Diffusion for Flash Speculative Decoding", arXiv:2602.06036, LMSYS blog 2026-06-15) with the maintainer's W4A8-quantized Qwen targets on **2×3090 Ampere sm_86, NO NVLink**. DFlash = a lightweight BLOCK-DIFFUSION drafter that produces a whole block of draft tokens in ONE parallel forward; claimed mean accept-len ~7.8 (proposes ~15/step), up to 2.5× faster than EAGLE-3, 6.17× on Qwen3-8B. Every draft layer gets full target context (accept-len scales with depth, unlike EAGLE-3).

FINISHED HF HEADS for the maintainer's EXACT models exist (z-lab org): \`z-lab/Qwen3.6-27B-DFlash\`, \`z-lab/Qwen3.5-35B-A3B-DFlash\`, \`z-lab/Qwen3-8B-DFlash-b16\`, \`z-lab/Qwen3.5-4B-DFlash\`, \`z-lab/Qwen3-Coder-Next-DFlash\`; GGUF \`spiritbuun/Qwen3.6-27B-DFlash-GGUF\`; community vLLM fork \`github.com/AEON-7/vllm-dflash\`; deploy guide ravchat.com/deploy-dflash-speculative-decoding-vllm; project z-lab.ai/projects/dflash.

THE FORK already has DFlash support: \`${F}/vllm/v1/spec_decode/dflash.py\` (DFlashProposer, parallel single-pass, context-vs-query buffer split for cudagraph), \`${F}/vllm/model_executor/models/qwen3_dflash.py\` (DFlashQwen3ForCausalLM, registry "DFlashDraftModel"), \`${F}/vllm/config/speculative.py\` (method="dflash" at :282/:707, "DFlash needs a non-causal-capable backend like FLASH_ATTN" at :114, only_one_forward_pass). Base = vLLM v0.23.0.

MAINTAINER CONTEXT: serves W4A8 (int4 wt + int8 act) Qwen3.6-27B + Qwen3.5-35B-A3B on 2×3090 no-NVLink (tp2; mem-tight, --max-num-seqs 32). Current MTP = +25% decode, accept ~2.2, patch-A fwd_kvcache verify (project_mtp_verify_attention_fix). Has DEEP diffusion-LM expertise (dev-dllm branch = DiffusionGemma int8-act, diffusion=prefill-like-compute-bound where int8 HELPS — project_dllm_int8_quant, project_ar_to_diffusion_conversion). Ampere has NO fp8 TC. Read ${MEM} project_spec_decode_ampere, project_decode_roadmap_prioritized, project_dllm_int8_quant, project_e2e_perf_validation.

This is a feasibility task — render a clear verdict + plan, but be SKEPTICAL of published numbers (they're on bf16/fp16 targets + likely NVLink/newer-vLLM; the maintainer's reality is W4A8 + 2×3090-no-NVLink + v0.23.0).`

const SCHEMA = {
  type:'object', additionalProperties:false, required:['dimension','findings','recommendation'],
  properties:{
    dimension:{type:'string'},
    findings:{ type:'array', items:{ type:'object', additionalProperties:false,
      required:['point','detail','source','risk','what_to_verify'],
      properties:{
        point:{type:'string'},
        detail:{type:'string', description:'concrete fact, grounded in the HF card / paper / fork code'},
        source:{type:'string', description:'HF url / arXiv / file:line in the fork / memory file'},
        risk:{type:'string', description:'gap between published claim and the maintainer Ampere-W4A8-noNVLink reality; "none" if clean'},
        what_to_verify:{type:'string', description:'the experiment / code-check that resolves it'},
      } } },
    recommendation:{type:'string'},
  },
}

const DIMS = [
  { key:'the-heads', title:'The z-lab DFlash heads: architecture, size, precision, serve config, claimed numbers',
    focus:`WebFetch the HF cards: huggingface.co/z-lab/Qwen3.6-27B-DFlash, /z-lab/Qwen3.5-35B-A3B-DFlash, /z-lab/Qwen3-8B-DFlash-b16, the GGUF huggingface.co/spiritbuun/Qwen3.6-27B-DFlash-GGUF, and github.com/AEON-7/vllm-dflash. Extract: the drafter's parameter count + dtype (bf16? the "-b16" suffix?), the EXACT vLLM serve command + speculative config, the REQUIRED vLLM version (does it match v0.23.0 or need newer?), which TARGET model version it pairs with (Qwen3.6-27B base — does it work against a QUANTIZED W4A8 target?), per-model+per-workload claimed accept-len/speedup, license, num_speculative_tokens (block size ~15?), and any note about quantization / Ampere / older GPUs.` },
  { key:'the-method', title:'DFlash block-diffusion drafter: how it actually computes (NFE, the compute profile)',
    focus:`WebFetch arxiv.org/html/2602.06036v1 + lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2 + z-lab.ai/projects/dflash + ravchat.com/deploy-dflash-speculative-decoding-vllm. Pin down: how the block-diffusion draft runs — is it ONE forward or N denoising steps (NFE) per draft block? the compute cost of the drafter forward (it's diffusion = prefill-like/compute-bound, NOT a cheap GEMV); how "every layer gets full context" works; why accept-len ~7.8 and how it degrades by workload (prose ~2.0, code ~5.5); the non-causal cross-attention; whether the drafter is lossless (verify-exact). Connect to the maintainer's diffusion-LM knowledge (project_dllm_int8_quant: diffusion denoising is compute-bound where int8 helps).` },
  { key:'fork-compat', title:'Fork v0.23.0 compatibility: dflash.py + qwen3_dflash.py + the non-causal backend',
    focus:`Read ${F}/vllm/v1/spec_decode/dflash.py (full), ${F}/vllm/model_executor/models/qwen3_dflash.py, ${F}/vllm/v1/spec_decode/utils.py (copy_and_expand_dflash_inputs_kernel), ${F}/vllm/config/speculative.py (:55,114,282,697-755,1071-1074). Does the fork's DFlash code match what the z-lab heads expect (config keys, the DFlashQwen3ForCausalLM interface, num_speculative_tokens)? The "non-causal-capable backend like FLASH_ATTN" requirement (:114) — does it CONFLICT with patch-A's fwd_kvcache verify route + the fp16-PV path (project_mtp_verify_attention_fix, project_pv_subfp16_ampere)? Does DFlash's verify use a different attention path than MTP's? cudagraph capturability (the context/query buffer split at dflash.py:40-60 is built for it). What, if anything, needs porting from a newer vLLM.` },
  { key:'ampere-compute', title:'Can the block-diffusion drafter run efficiently on Ampere (no fp8 TC) + VRAM on mem-tight 27B',
    focus:`The drafter is a block-diffusion model = compute-heavy prefill-like forward (possibly multi-NFE), unlike MTP's single cheap layer. On Ampere sm_86 (no fp8/fp4 TC, fp16/int8-IMMA only), is the drafter forward cheap ENOUGH relative to the W4A8 verify so the amortization still wins? Estimate: drafter params/dtype × NFE vs the verify forward. Could the DFlash drafter ITSELF be int8/W4A8-accelerated on Ampere (the maintainer's Marlin/dllm-int8 edge — diffusion compute is exactly where int8 pays, project_dllm_int8_quant)? VRAM: the drafter head size added to the mem-tight 27B-W4A8 tp2 config (--max-num-seqs 32, project_e2e_perf_validation) — does it fit? does it shrink max-num-seqs? Quantize the drafter?` },
  { key:'w4a8-target', title:'W4A8 quantized target × DFlash: accept-len hold-up + output exactness',
    focus:`The z-lab heads were trained/measured against a bf16/fp16 TARGET. The maintainer's target is W4A8-quantized (int4 wt + int8 act) → its output distribution differs slightly from bf16. Output is EXACT regardless (verify re-scores with the W4A8 target — speculative sampling is unbiased for any draft), so quality is safe; the open question is ACCEPT-LEN: does a draft trained on the bf16 target's distribution still get ~7.8 accepted by the W4A8 target, or does the quant-induced distribution shift erode it? Reason about it + design the measurement. Also: is there a W4A8/AWQ DFlash head, or only bf16-target heads? Does the GGUF version imply a quantized-target-compatible head?` },
  { key:'vs-mtp-stack', title:'Does DFlash obsolete the MTP optimization roadmap? What survives.',
    focus:`Map DFlash against the prioritized roadmap (${MEM}/project_decode_roadmap_prioritized.md): the MTP serving-path bundle B2 (R04 probabilistic-draft, R08 adaptive-K, R11 FULL-capture, R17 local-argmax) and the stronger-draft-head bundle B3 (R03 distillation, R15 multi-layer, R16=DFlash itself). If DFlash lands at accept-len ~7.8 it SUBSUMES most of these (they nudge MTP's ~2.2). State HONESTLY which become redundant vs which STILL apply with DFlash: patch-A verify attention (still needed — DFlash still verifies q=1+K... or q=1+15?), the verify lm_head quant (R01, still applies to the target's verify head), mem-OC, the non-MTP B-knob weight-byte levers, the cudagraph FULL-graph. Does DFlash's bigger block (15 vs K=2) change the verify-attention cost / the patch-A win / the long-ctx cliff? Re-rank: is "deploy DFlash" now the #1 decode move, ahead of R04?` },
  { key:'deploy-validate', title:'Concrete deploy + validation plan on 2×3090 W4A8',
    focus:`The smallest experiment to KNOW if DFlash works for the maintainer: pull z-lab/Qwen3.6-27B-DFlash (or the 8B first as a fast smoke) + serve against the W4A8 target on 2×3090 tp2 (or the AEON-7/vllm-dflash fork as reference). The serve command (from the HF card + ravchat guide), the speculative config, the non-causal backend flag, tp2-no-NVLink considerations. MEASURE: accept-len (vs the claimed ~7.8 and vs MTP's ~2.2) on prose/code/zh-long-CoT, decode tok/s (T(N)-T(1), not streaming-TTFT per feedback_decode_bench_tpot_artifact), quality (GSM8K/MMLU + the accept=1.0=collapse canary from project_1cat_vllm_crossanalysis), VRAM, tp2 scaling. Define the GO signal + the fallback if accept-len collapses on W4A8. Sandbox-test gate per the maintainer's rules.` },
]

phase('Investigate')
const findPrompt = (d) => `You are assessing the feasibility of off-the-shelf DFlash (block-diffusion spec-decode) on the maintainer's Ampere W4A8 fork. Use WebFetch (load via ToolSearch) for HF cards / arXiv / blogs, and Read the fork code under ${F}.

${CTX}

YOUR DIMENSION: ${d.title}
${d.focus}

RULES: ground every claim in a real HF-card / arXiv / file:line / memory source. Be SKEPTICAL — separate "published on bf16 target + NVLink + newer vLLM" from "Ampere W4A8 + 2×3090-no-NVLink + v0.23.0 reality". Mark risks where the gap is real. This is feasibility → give a clear recommendation. 4-8 findings + a crisp recommendation.`

const challengePrompt = (d, found) => `Independent verifier hardening the DFlash-feasibility dimension "${d.title}". Read ${F} + WebFetch the cited sources to confirm.

DRAFT: ${JSON.stringify(found).slice(0,9000)}

For EACH finding: (1) confirm the fact at the cited source (HF card / arXiv / file:line) — correct if wrong. (2) Stress the Ampere-W4A8-noNVLink-v0.23.0 gap: does the published number assume bf16 target / NVLink / a newer vLLM? Does the non-causal backend break patch-A? Is the drafter compute-cheap on Ampere? Does it fit the mem-tight 27B? (3) Sharpen what_to_verify into a runnable check. Keep good, correct weak, drop unsupported, add up to 2 missed. ${CTX}`

const results = await pipeline(
  DIMS,
  (d) => agent(findPrompt(d), { label:`dflash:${d.key}`, phase:'Investigate', schema:SCHEMA }),
  (found,d) => agent(challengePrompt(d,found), { label:`challenge:${d.key}`, phase:'Challenge', schema:SCHEMA }),
)
const all = results.filter(Boolean)
log(`Investigate+Challenge: ${all.length}/${DIMS.length} dims, ${all.flatMap(r=>r.findings??[]).length} findings`)

phase('Synthesize')
const corpus = JSON.stringify(all).slice(0,105000)
const report = await agent(
  `Lead author: write the FEASIBILITY VERDICT + INTEGRATION PLAN for deploying off-the-shelf z-lab DFlash (block-diffusion spec-decode) on the maintainer's Ampere W4A8 fork. Markdown, decisive.

${CTX}

ALL FINDINGS (7 dims, find+challenge) JSON:
${corpus}

WRITE:
1. **Verdict** (5-7 lines): is it worth trying NOW, and how strong is the "obsoletes the MTP stack" case? The headline (finished heads exist for the EXACT models; fork has DFlash support; claimed accept-len ~7.8 vs MTP ~2.2) AND the honest gaps (block-diffusion drafter compute on no-fp8-TC Ampere; accept-len on a W4A8 target vs bf16; non-causal backend vs patch-A; v0.23.0-vs-required-version; VRAM on mem-tight 27B). The single biggest unknown that the first experiment must resolve.
2. **What DFlash is + why it's two-axis** — block diffusion, single parallel forward (overhead) + every-layer-full-context accept-len ~7.8 (accept), vs MTP.
3. **The heads + fork compat** — which z-lab head for which target, dtype/size, the serve config, the v0.23.0 gap (what needs porting), the non-causal-backend × patch-A interaction.
4. **Ampere-W4A8 reality check** — the drafter compute cost on no-fp8 Ampere (and whether the maintainer's int8/Marlin/dllm edge can accelerate the diffusion drafter), accept-len risk on a W4A8 target, VRAM on tp2 mem-tight 27B.
5. **Does it obsolete the MTP roadmap?** — explicit: which roadmap items (R03/R04/R08/R11/R15/R17, B2/B3) become REDUNDANT, which SURVIVE (patch-A verify, R01 lm_head verify-head quant, mem-OC, the B-knob weight levers), and whether "deploy DFlash" should become the #1 decode move ahead of R04.
6. **Deploy + validation plan** — the smallest experiment (8B smoke → 27B W4A8 on 2×3090), the serve command, the measurements (accept-len/tok/s/quality/VRAM + the collapse canary), the GO signal + fallback. Sandbox-test gated.
7. **Open risks / unknowns** — ranked.
Anchor to sources. Return ONLY the Markdown.`,
  { label:'synthesize', phase:'Synthesize', effort:'xhigh' })

return { report, dims: all.length, findings: all.flatMap(r=>r.findings??[]).length }
