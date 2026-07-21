export const meta = {
  name: 'dspark-deepseek-specdecode-analysis',
  description: 'Analyze DeepSeek DSpark (semi-parallel spec-decode = DFlash backbone + Markov head) for the Ampere W4A8 fork: usable? does it change the DFlash/MTP decision?',
  phases: [
    { title: 'Investigate', detail: '5 dims: the paper mechanism, DeepSpec code+training, vs DFlash/MTP, Ampere-W4A8-vLLM portability, strategic verdict' },
    { title: 'Challenge',   detail: 'verify claims vs sources + the maintainer reality (no finished Qwen3.5/3.6 head, 2x3090, v0.23.0, W4A8)' },
    { title: 'Synthesize',  detail: 'verdict + how it changes the DFlash/MTP roadmap decision' },
  ],
}

const F = '/home/trevor/vllm-ampere-optimized/vllm'
const MEM = '/home/trevor/.claude/projects/-home-trevor-vllm-ampere-optimized/memory'
const DDOC = '/home/trevor/vllm-ampere-optimized/docs/RESEARCH-dflash-feasibility.md'

const CTX = `ANALYZE: DeepSeek **DSpark** (released 2026-06-27, arXiv:2606.19348), a "semi-parallel" speculative-decoding framework, for the maintainer's Ampere W4A8 Qwen fork. KEY ARCHITECTURE (from the announcement): DSpark = a **heavy PARALLEL backbone (which IS DFlash in their setup) producing base logits for every block position, PLUS a lightweight SEQUENTIAL head (default: Markov, rank-256 factorized) that adds a prefix-dependent bias before sampling each token** — the sequential head conditions on the immediately-preceding token, fixing the "multi-modal collision / rapid acceptance decay" of pure-parallel drafting (DFlash). Also has a **confidence head** (per-position verify-probability → ADAPTIVE verification, "selectively verify promising guesses") + Sequential Temperature Scaling. Draft module reuses frozen target embeddings + output head; trained via total-variation loss. Shipped config = DSpark-5 (5-token block + Markov head).

NUMBERS: vs DFlash accept-len +16-18%, vs EAGLE-3 +27-31% (Qwen3-4B/8B/14B offline); **60-85% faster per-user than MTP-1** on DeepSeek-V4-Flash (57-78% Pro). Confidence-sweep: chat accept 45.7%→95.7%, math 76.9%→92.5%. OPEN-SOURCE = **DeepSpec** (github.com/deepseek-ai/DeepSpec, MIT, full-stack TRAIN+eval; supports DSpark/DFlash/EAGLE3; Qwen3 + Gemma4 configs; train.sh→eval.sh on 9 datasets; **target cache ~38TB for Qwen3-4B**, "1 node 8 GPUs"). Production heads = DeepSeek-V4-Pro/Flash-DSpark only. Hardware NOT specified (no Ampere mention); vLLM/SGLang integration NOT mentioned (only a "minimal inference example"); quantization NOT addressed.

MAINTAINER REALITY: serves **W4A8 (int4 wt + int8 act) Qwen3.5/3.6 27B + 35B-A3B** on **2×3090 sm_86 NO NVLink**, tp2, vLLM **v0.23.0** fork, mem-tight. Current = MTP (+25-67%, accept ~2.2-3.0) + patch-A fwd_kvcache verify. JUST analyzed DFlash ([[project_dflash_feasibility]], full doc ${DDOC}): finished z-lab DFlash heads exist BUT the one 2×3090+int4 datapoint = **accept-len 2.00 (tie w/ MTP)**, likely a causal-SWA integration gap + W4A8 distribution shift. The fork already vendored DFlash (PR#40898: dflash.py, qwen3_dflash.py). DSpark's Markov head is LITERALLY the fix for DFlash's accept decay = the maintainer's exact problem. CRITICAL CONSTRAINTS: no off-the-shelf head for Qwen3.5/3.6 27B/35B (DeepSpec benches Qwen3-4B/8B/14B); 2×3090 CANNOT train (38TB cache + 8 GPU); Ampere has no fp8 TC; v0.23.0 + W4A8. Read ${MEM} project_dflash_feasibility, project_spec_decode_ampere, project_decode_roadmap_prioritized, project_dllm_int8_quant.

Be SKEPTICAL: separate "published on V4/Hopper-Blackwell + bf16 + 8-GPU-trained heads" from the maintainer's "Ampere W4A8 + 2×3090 + v0.23.0 + no-finished-head + can't-train" reality.`

const SCHEMA = {
  type:'object', additionalProperties:false, required:['dimension','findings','recommendation'],
  properties:{
    dimension:{type:'string'},
    findings:{ type:'array', items:{ type:'object', additionalProperties:false,
      required:['point','detail','source','risk','what_to_verify'],
      properties:{
        point:{type:'string'}, detail:{type:'string'}, source:{type:'string', description:'arXiv / DeepSpec file / HF / file:line in fork / memory'},
        risk:{type:'string', description:'gap vs the maintainer Ampere-W4A8-2x3090-v0.23.0-no-head reality; "none" if clean'},
        what_to_verify:{type:'string'} } } },
    recommendation:{type:'string'},
  },
}

const DIMS = [
  { key:'mechanism', title:'DSpark mechanism (the paper): semi-parallel = DFlash backbone + Markov head + confidence head',
    focus:`WebFetch arxiv.org/abs/2606.19348 (+ /html/2606.19348v1) + z-lab/DeepSeek blog. Pin down: HOW the Markov rank-256 sequential head adds a prefix-dependent bias to the parallel DFlash logits (the math), WHY this fixes DFlash's multi-modal-collision / accept-decay (the exact failure the maintainer hit = fouvy accept-len 2.00), the confidence head + adaptive verification (how it decides which of the 5-block tokens to verify), Sequential Temperature Scaling, the total-variation training objective. The accept-len numbers vs DFlash (+16-18%) / EAGLE-3 / MTP-1 and on WHICH targets/precision/hardware. Is output lossless (verify-exact)?` },
  { key:'deepspec-train', title:'DeepSpec codebase + can the maintainer train/get a head for Qwen3.5/3.6',
    focus:`WebFetch github.com/deepseek-ai/DeepSpec (README, config/, scripts/train, the inference/ example). What's in it (DSpark/DFlash/EAGLE3 impls); the Qwen3 config — does it cover Qwen3.5/3.6 27B/35B-A3B or only Qwen3-4B/8B/14B? The training pipeline cost (the ~38TB target cache for Qwen3-4B → extrapolate for 27B/35B; the 8-GPU node) — is training a DSpark head FEASIBLE on 2×3090 (almost certainly NO — quantify why). Is there a FINISHED DSpark head for any model the maintainer serves? The inference example — is it vLLM/SGLang or custom? License (MIT). The honest answer: can the maintainer USE DSpark at all without training, today?` },
  { key:'vs-dflash-mtp', title:'DSpark vs DFlash vs the maintainer MTP — head-to-head + roadmap mapping',
    focus:`DSpark beats DFlash +16-18% accept AND fixes the decay (the Markov head) — so architecturally it's "the better DFlash" and directly addresses the maintainer's fouvy-2.00 collapse. BUT: DFlash has FINISHED z-lab heads (Qwen3.5-35B-A3B, Qwen3.6-27B) + is ALREADY vendored in the fork (PR#40898); DSpark has NEITHER for the maintainer's models. Map onto the roadmap (${MEM}/project_decode_roadmap_prioritized.md: R16=DFlash, R04=probabilistic-draft #1, R03=distill, MTP baseline). Which is more DEPLOYABLE NOW for the maintainer? Does DSpark validate that DFlash's decay is real+known (→ informs whether to pursue DFlash at all)? Could DFlash's z-lab head + a hand-added Markov head approximate DSpark cheaply?` },
  { key:'ampere-vllm-port', title:'Ampere W4A8 + vLLM v0.23.0 portability of DSpark',
    focus:`The fork has DFlash (dflash.py) but NOT DSpark's Markov+confidence head. Read ${F}/vllm/v1/spec_decode/dflash.py + qwen3_dflash.py + config/speculative.py to gauge: how much NEW vLLM code would DSpark need (the Markov sequential head + the confidence-head adaptive verifier on top of the existing DFlash proposer)? The Markov head is lightweight (rank-256, conditions on prev token) → cheap on Ampere — but is the ADAPTIVE-verification (variable accepted count) cudagraph-compatible + compatible with patch-A (fwd_kvcache q=1+K verify)? Same DFlash caveats inherited: patch-A for the q=block verify, non-causal backend, fp8-KV conflict (#41559), W4A8 distribution shift on the frozen-reused target head/embeddings. Does DSpark's confidence head help or hurt the W4A8 accept-len (it's calibrated on bf16)? Quantization story (none published).` },
  { key:'strategic-verdict', title:'Strategic: does DSpark change the maintainer\'s spec-decode plan?',
    focus:`Honest strategic read. Given DSpark = the better-DFlash that fixes the exact decay the maintainer hit, but is NOT deployable for them (no finished Qwen3.5/3.6 head, no vLLM integration, can't train on 2×3090, no Ampere/W4A8 story): does it change the roadmap decision? Options: (a) wait for z-lab/community to ship a DSpark head + vLLM integration for Qwen-class, then consume it (like DFlash); (b) the whole parallel-draft-head family (DFlash/DSpark) is a Hopper/Blackwell + finished-head + 8-GPU-train game the maintainer can only CONSUME not produce → stick with MTP+patch-A + R04 until a finished head lands; (c) DSpark confirms DFlash's decay is the real blocker → de-prioritize the DFlash spike. What's the watch-list (z-lab DSpark heads for Qwen3.5/3.6? a vLLM DSpark PR?) + the single recommended action.` },
]

phase('Investigate')
const findPrompt = (d) => `You are analyzing DeepSeek DSpark for the maintainer's Ampere W4A8 fork. Use WebFetch (load via ToolSearch) for arXiv/DeepSpec/HF, and Read the fork code under ${F} + the DFlash doc ${DDOC}.

${CTX}

YOUR DIMENSION: ${d.title}
${d.focus}

RULES: ground every claim in arXiv:2606.19348 / the DeepSpec repo / HF / file:line / memory. Be SKEPTICAL of published numbers (V4/Hopper-Blackwell/bf16/8-GPU-trained vs the maintainer's Ampere-W4A8-2x3090-v0.23.0-no-head reality). Clear recommendation. 4-8 findings.`

const challengePrompt = (d, found) => `Independent verifier hardening the DSpark dimension "${d.title}". WebFetch the cited sources + Read ${F} to confirm.

DRAFT: ${JSON.stringify(found).slice(0,9000)}

For EACH: (1) confirm at the cited source. (2) Stress the maintainer-reality gap: no finished Qwen3.5/3.6 head, can't train (38TB/8GPU) on 2×3090, v0.23.0 has DFlash-not-DSpark, W4A8 distribution shift, Ampere no fp8. (3) Don't conflate DSpark with DFlash (DSpark = DFlash backbone + Markov+confidence head). (4) Sharpen what_to_verify. Keep good, correct weak, add up to 2 missed. ${CTX}`

const results = await pipeline(
  DIMS,
  (d) => agent(findPrompt(d), { label:`dspark:${d.key}`, phase:'Investigate', schema:SCHEMA }),
  (found,d) => agent(challengePrompt(d,found), { label:`challenge:${d.key}`, phase:'Challenge', schema:SCHEMA }),
)
const all = results.filter(Boolean)
log(`Investigate+Challenge: ${all.length}/${DIMS.length} dims, ${all.flatMap(r=>r.findings??[]).length} findings`)

phase('Synthesize')
const corpus = JSON.stringify(all).slice(0,100000)
const report = await agent(
  `Lead author: write the ANALYSIS of DeepSeek DSpark for the maintainer's Ampere W4A8 fork. Markdown, decisive.

${CTX}

ALL FINDINGS (5 dims, find+challenge) JSON:
${corpus}

WRITE:
1. **Verdict** (5-7 lines): what DSpark is (DFlash backbone + Markov head + confidence head), whether it's relevant + usable for the maintainer NOW, and the one-sentence bottom line (architecturally the better-DFlash that fixes the maintainer's exact decay, BUT no finished Qwen3.5/3.6 head + no vLLM integration + can't-train → consume-later not build-now).
2. **What DSpark is + why it fixes DFlash** — the semi-parallel mechanism, the Markov head's prefix-bias addressing multi-modal collision, the confidence-head adaptive verification.
3. **Can the maintainer use it?** — DeepSpec (train-your-own but 38TB/8GPU = infeasible on 2×3090), no finished Qwen3.5/3.6 head, vLLM integration status, what would be needed.
4. **DSpark vs DFlash vs MTP** — the head-to-head + which is deployable now; does DSpark validate DFlash's decay is the real blocker.
5. **Ampere W4A8 portability** — new vLLM code needed, the inherited DFlash caveats (patch-A, non-causal, fp8-KV, W4A8 shift), the Markov+confidence head on Ampere, adaptive-verify × cudagraph.
6. **Does it change the roadmap?** — the strategic call (wait-and-consume vs the family-is-Hopper-only vs de-prioritize-DFlash) + the watch-list + the single recommended action (likely: R04+MTP stays the plan; add DSpark/DFlash-for-Qwen3.5 finished-head + vLLM-DSpark-PR to the watch-list).
Anchor to sources. Return ONLY the Markdown.`,
  { label:'synthesize', phase:'Synthesize', effort:'xhigh' })

return { report, dims: all.length, findings: all.flatMap(r=>r.findings??[]).length }
