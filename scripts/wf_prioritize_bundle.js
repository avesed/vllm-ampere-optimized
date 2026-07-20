export const meta = {
  name: 'prioritize-and-bundle-directions',
  description: 'Rank the kernel-throughput + decode-optimization direction menus and find which can be done together (bundles)',
  phases: [
    { title: 'Ingest',     detail: 'extract both menus → one deduped, status-tagged, actionable list' },
    { title: 'Assess',     detail: '3-judge scoring panel + bundle-finder + sleeper-adversary (parallel)' },
    { title: 'Synthesize', detail: 'ranked tiers + bundles + execution sequence' },
  ],
}

const DOCS = '/home/trevor/vllm-ampere-optimized/docs'
const KDOC = `${DOCS}/RESEARCH-kernel-throughput-directions.md`
const DDOC = `${DOCS}/RESEARCH-decode-optimization-menu.md`
const F = '/home/trevor/vllm-ampere-optimized/vllm'

const CTX = `Project = \`vllm-ampere-optimized\` (vLLM v0.23.0, Ampere sm_80/sm_86), serving Qwen3.5/3.6 27B & 35B-A3B HYBRID (GatedDeltaNet + few hd256 full-attn) + MoE + MTP, W4A8. DECODE = HBM-weight-bandwidth-bound (the real target). PROJECT PRIORITIES (from the maintainer): squeeze-every-bit (a shippable +1-2% is worth it if correct), STABILITY-FIRST (correctness-gated, complete-not-quick), prefer SHIPPABLE-GENERAL sm80/86 fork value over per-deployment config, and decode throughput is the prize. The maintainer wants to BUILD next, so ranking must surface what's worth doing + what to do TOGETHER.

THE TWO MENUS TO RANK:
- Kernel-throughput menu: ${KDOC} (D1-D21: fp16-accum HMMA family, norm→int8 fusion, lm_head/sampling, MoE small-M, etc.)
- Decode menu: ${DDOC} (A amortize / B fewer-bytes / C faster-DRAM / D system; ~40 directions tagged new/already-verdicted/long-tail).
They OVERLAP — dedup across them (e.g. kernel-D9 blocks_per_sm ↔ decode-C4; kernel-D5/D7 norm→int8 fusion ↔ decode-D9/D10; kernel-D2 M-shaped Marlin ↔ decode-C4; kernel-D12-16 lm_head ↔ decode-excluded-lm_head). ADJACENT findings to fold in where they BUNDLE (not as separate menu items): the MoE phantom-expert masking + "decode already FULL-graph + DRAM-SOL" (${DOCS}/RESEARCH-cudagraph-decode-bandwidth.md), the runtime/shared lm_head quant (${DOCS}/RESEARCH-mtp-draft-lmhead-quant.md), and the 1Cat STEAL items — quality-canary, cudagraph-stale-buffer (${DOCS}/RESEARCH-1cat-vllm-crossanalysis.md).

GOAL: (1) RANK the ACTIONABLE directions (skip already-shipped; mark dead/no-kernel as parked), (2) find BUNDLES (do-together) by shared infra/file/kernel, multiplicative stacking, shared measurement probe, or shared prerequisite, (3) propose an execution SEQUENCE honoring prereqs (e.g. the DRAM-SOL gate probe must precede the C-knob occupancy levers; training-needed heads (DFlash/multi-layer) gate A5/A6).`

phase('Ingest')
const EXTRACT_SCHEMA = { type:'object', additionalProperties:false, required:['directions'],
  properties:{ directions:{ type:'array', items:{ type:'object', additionalProperties:false,
    required:['id','title','category','mechanism','status'],
    properties:{ id:{type:'string',description:'stable id, e.g. K-D5 or DEC-A6'}, title:{type:'string'},
      category:{type:'string',description:'knob A/B/C/D or kernel-layer'}, mechanism:{type:'string',description:'1-2 line'},
      status:{type:'string',description:'new | long-tail | already-verdicted/shipped | dead-no-kernel'} } } } } }

const extracted = await parallel([
  () => agent(`Extract EVERY distinct optimization direction from ${KDOC} into the schema. id prefix "K-" (e.g. K-D5). category = its kernel layer. status from the doc's novelty/verdict tags. Read the whole doc.`,
    { label:'extract:kernel', phase:'Ingest', schema:EXTRACT_SCHEMA }),
  () => agent(`Extract EVERY distinct optimization direction from ${DDOC} into the schema. id prefix "DEC-" (e.g. DEC-A6). category = its knob (A/B/C/D). status from the doc's tag (new/already-verdicted/long-tail). Read the whole doc.`,
    { label:'extract:decode', phase:'Ingest', schema:EXTRACT_SCHEMA }),
])
const rawList = extracted.filter(Boolean).flatMap(r=>r.directions ?? [])

const MERGE_SCHEMA = { type:'object', additionalProperties:false, required:['directions'],
  properties:{ directions:{ type:'array', items:{ type:'object', additionalProperties:false,
    required:['id','title','knob','mechanism','status','sources','actionable'],
    properties:{ id:{type:'string'}, title:{type:'string'}, knob:{type:'string',description:'A-amortize|B-bytes|C-DRAM|D-system|prefill (its dominant axis)'},
      mechanism:{type:'string'}, status:{type:'string'}, sources:{type:'array',items:{type:'string'},description:'the K-/DEC- ids merged into this'},
      actionable:{type:'boolean',description:'true if worth-doing-and-not-yet-done (new/long-tail/to-build); false if already-shipped or dead-no-kernel'} } } } } }

const merged = await agent(
  `Here are extracted directions from the two menus (JSON):\n${JSON.stringify(rawList).slice(0,40000)}\n\nMERGE + DEDUP into ONE canonical list. Combine cross-menu duplicates into a single entry (keep one id, list the merged ids in "sources"): K-D9↔DEC-C4 (blocks_per_sm), K-D5/D7↔DEC-D9/D10 (norm/GDN fusion), K-D2↔DEC-C4 (M-shaped Marlin), K-D12-16↔the lm_head cluster (note: decode menu EXCLUDES lm_head, but the kernel menu has it — keep ONE lm_head cluster entry). Assign each a dominant "knob" (A/B/C/D/prefill). Set actionable=false for already-shipped (MTP+patch-A, mem-OC, fp16-PV...) and dead-no-kernel (W3/W2, GDN launch-tuning, int8-GEMM-tapped). Keep the canonical, deduped, actionable-flagged list. ${CTX}`,
  { label:'merge-dedup', phase:'Ingest', schema:MERGE_SCHEMA })

const unified = (merged.directions ?? []).filter(Boolean)
const actionable = unified.filter(d=>d.actionable)
const unifiedStr = JSON.stringify(unified.map(d=>({id:d.id,knob:d.knob,title:d.title,mechanism:d.mechanism,status:d.status,actionable:d.actionable}))).slice(0,30000)
log(`Ingest: ${unified.length} canonical directions (${actionable.length} actionable) from ${rawList.length} raw`)

// ============================================================ ASSESS (parallel)
phase('Assess')
const SCORE_SCHEMA = { type:'object', additionalProperties:false, required:['judge','scores'],
  properties:{ judge:{type:'string'}, scores:{ type:'array', items:{ type:'object', additionalProperties:false,
    required:['id','leverage','effort_cheapness','confidence','generality','tier','one_line'],
    properties:{ id:{type:'string'},
      leverage:{type:'integer',description:'1-5: plausible impact on decode/throughput (5=biggest)'},
      effort_cheapness:{type:'integer',description:'1-5: 5=cheap/fast (config/python, days), 1=heavy (new kernel / needs training data)'},
      confidence:{type:'integer',description:'1-5: 5=very likely to work + correctness-safe, 1=speculative/risky'},
      generality:{type:'integer',description:'1-5: 5=ships general sm80/86 fork patch, 1=per-deployment-only'},
      tier:{type:'integer',description:'this judge\'s suggested tier 1(do-now)..4(park)'},
      one_line:{type:'string'} } } } } }

const RUBRIC = `Score EACH actionable direction in the unified list on 4 axes (integers 1-5): leverage (impact on bandwidth-bound decode), effort_cheapness (5=cheap config/python, 1=new-kernel-or-needs-training-data), confidence (5=will-work+correctness-safe, 1=speculative), generality (5=shippable-general-sm80/86, 1=per-deploy-only). Then a tier 1(do-now)..4(park). You MAY open ${KDOC} / ${DDOC} to check a direction's detail. Score only actionable=true entries; for actionable=false give tier and a one_line note ("shipped"/"dead") with low scores. UNIFIED LIST:\n${unifiedStr}`

const assess = await parallel([
  () => agent(`You are the SHIP-FAST PRAGMATIST judge (weight low-effort + high-confidence + shippable; reward what lands this week). ${RUBRIC}\n${CTX}`,
    { label:'judge:pragmatist', phase:'Assess', schema:SCORE_SCHEMA }),
  () => agent(`You are the MAX-LEVERAGE judge (weight impact on the decode bandwidth wall + ceiling, even at higher effort; reward the big swings). ${RUBRIC}\n${CTX}`,
    { label:'judge:leverage', phase:'Assess', schema:SCORE_SCHEMA }),
  () => agent(`You are the CORRECTNESS/STABILITY SKEPTIC judge (weight confidence + correctness-safety + generality; penalize training-needed / accuracy-risky / per-deploy). ${RUBRIC}\n${CTX}`,
    { label:'judge:skeptic', phase:'Assess', schema:SCORE_SCHEMA }),
  () => agent(`Find BUNDLES — sets of directions worth doing TOGETHER. A bundle is justified by ONE of: shared infrastructure (same kernel/file/quant-scheme/recipe), multiplicative STACKING (combined gain > sum, e.g. all shrink the draft, or all shrink weight bytes), a shared MEASUREMENT probe (one ncu/nsys run resolves several), or a shared PREREQUISITE (same training run / same kernel enabler). Read ${KDOC}/${DDOC} for detail. For each bundle: name, the direction ids, why-together (which justification), the shared artifact, and the combined rationale. Aim for 5-9 bundles covering the actionable set; note any direction that is a strict prereq of others. UNIFIED LIST:\n${unifiedStr}\n${CTX}`,
    { label:'bundle-finder', phase:'Assess',
      schema:{ type:'object', additionalProperties:false, required:['bundles'], properties:{ bundles:{ type:'array', items:{ type:'object', additionalProperties:false,
        required:['name','direction_ids','justification','shared_artifact','rationale'],
        properties:{ name:{type:'string'}, direction_ids:{type:'array',items:{type:'string'}}, justification:{type:'string',description:'shared-infra|stacking|shared-probe|shared-prereq'}, shared_artifact:{type:'string'}, rationale:{type:'string'} } } } } } }),
  () => agent(`You are the SLEEPER ADVERSARY. Looking at the unified list, flag: (a) directions that will likely be UNDER-ranked but are sleepers (cheap + safe + real), (b) directions that will likely be OVER-ranked (hyped but blocked/dead/needs-training), (c) any BUNDLE that looks superficial (the "shared" infra isn't actually shared), (d) missing PREREQUISITES (a direction that can't start before a measurement/kernel/training). Read the docs as needed. UNIFIED LIST:\n${unifiedStr}\n${CTX}`,
    { label:'sleeper-adversary', phase:'Assess',
      schema:{ type:'object', additionalProperties:false, required:['notes'], properties:{ notes:{ type:'array', items:{ type:'object', additionalProperties:false,
        required:['target','type','claim'], properties:{ target:{type:'string',description:'direction id or bundle name'}, type:{type:'string',description:'under-ranked|over-ranked|superficial-bundle|missing-prereq'}, claim:{type:'string'} } } } } } }),
])
const judges = assess.slice(0,3).filter(Boolean)
const bundles = assess[3]?.bundles ?? []
const sleeper = assess[4]?.notes ?? []
log(`Assess: ${judges.length} judges scored; ${bundles.length} bundles; ${sleeper.length} sleeper notes`)

// ============================================================ SYNTHESIZE
phase('Synthesize')
const scoresStr = JSON.stringify(judges.map(j=>({judge:j.judge, scores:j.scores}))).slice(0,55000)
const report = await agent(
  `Lead author: produce the PRIORITIZED ROADMAP + BUNDLES for the two direction menus (kernel-throughput D1-D21 + decode A/B/C/D). Markdown, for the maintainer who wants to BUILD next. ${CTX}

CANONICAL DEDUPED LIST: ${unifiedStr}
3-JUDGE PANEL SCORES (pragmatist / max-leverage / skeptic, axes 1-5 + tier): ${scoresStr}
BUNDLES (do-together): ${JSON.stringify(bundles).slice(0,16000)}
SLEEPER ADVERSARY notes: ${JSON.stringify(sleeper).slice(0,9000)}

WRITE:
1. **How to read this** (3-4 lines): the rubric, that ranking is across BOTH deduped menus, decode-BW-bound lens.
2. **Ranked tier table** — every actionable direction: id · title · knob · mean(leverage/effort/confidence/generality) · judge-tier-spread · one-line why. Sort into **Tier 1 (do-now: cheap+confident+real)**, **Tier 2 (high-leverage, more effort/uncertainty)**, **Tier 3 (long-tail/speculative)**, **Parked (shipped or dead — name them briefly)**. Reconcile judge disagreement explicitly where they split; honor the sleeper-adversary's under/over-ranked flags (state when you moved something).
3. **Bundles — "do these together"** — the 5-9 bundles, each: the directions, the shared artifact/justification, why the combination beats doing them separately, and which single measurement/PR kicks it off. Drop or fix any bundle the adversary flagged as superficial.
4. **Execution sequence** — a concrete ordering honoring prerequisites: what measurement(s) to run FIRST (e.g. the DRAM-SOL gate + the draft-step byte census gate several at once), then the Tier-1 bundle(s), then the gated/training-needed Tier-2. Make it a short numbered plan a person could start Monday.
5. **The single highest-ROI first move** — name it and why.
Be decisive and concrete; cite ids. Return ONLY the Markdown.`,
  { label:'synthesize', phase:'Synthesize', effort:'xhigh' })

return { report, canonical: unified.length, actionable: actionable.length, bundles: bundles.length }
