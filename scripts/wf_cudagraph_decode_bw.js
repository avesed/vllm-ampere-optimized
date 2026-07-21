export const meta = {
  name: 'sm80-86-cudagraph-decode-bandwidth',
  description: 'Investigate SHIPPABLE sm80/sm86 cudagraph-machinery changes that improve DECODE bandwidth efficiency (collect directions, no go/no-go)',
  phases: [
    { title: 'Investigate', detail: '7 dimensions of cudagraph×bandwidth×decode, grounded in the fork source' },
    { title: 'Challenge',   detail: 'verify mechanism + file:line, separate real headroom from already-DRAM-SOL' },
    { title: 'Critic',      detail: '2 lenses: missing cudagraph-bandwidth angle / first-principles roofline' },
    { title: 'Synthesize',  detail: 'direction menu: shippable cudagraph decode-bandwidth levers' },
  ],
}

const AMPERE = '/home/trevor/vllm-ampere-optimized'
const MEM    = '/home/trevor/.claude/projects/-home-trevor-vllm-ampere-optimized/memory'

const CTX = `Fork = \`vllm-ampere-optimized\` (vendored vLLM v0.23.0) for Ampere sm_80(A100)/sm_86(3090/A40/A6000/A10), source at ${AMPERE}/vllm. Models = hybrid GatedDeltaNet/Mamba2 + a few full-attn hd256 layers + modern MoE + MTP spec-decode, W4A8 (int4 weight + int8 act). **DECODE is HBM-WEIGHT-BANDWIDTH-BOUND** (the int4 weights streamed every token dominate; attention is a small slice). **HYPOTHESIS UNDER TEST (the user's): cudagraph-related BANDWIDTH-EFFICIENCY improvements can speed decode on sm_80/sm_86.**

cudagraph machinery (anchor here): vllm/v1/cudagraph_dispatcher.py; vllm/compilation/cuda_graph.py + breakable_cudagraph.py; vllm/v1/worker/gpu_model_runner.py (capture loop + pad-to-captured-size, e.g. \`pad_attn = cudagraph_mode==FULL\` at ~:4157/:5807); vllm/v1/worker/gpu/cudagraph_utils.py + vllm/v1/worker/gpu/spec_decode/autoregressive/cudagraph_utils.py; vllm/v1/worker/gpu/model_states/mamba_hybrid.py (hybrid recurrent state under capture). v0.23 names: \`cudagraph_capture_sizes\`, \`max_cudagraph_capture_size\`, \`cudagraph_mode\` ∈ {FULL, PIECEWISE, FULL_AND_PIECEWISE=default, FULL_DECODE_ONLY}; CLI \`-cc\` (mode) / \`-O\` (level).

ENGAGE (do NOT merely re-derive) these PRIOR fork conclusions, in ${MEM}/project_autotune_gpu_oc.md: enforce-eager measured **−39% aggregate / −74% single-stream** → cudagraph is ESSENTIAL; and "**capture-size LIST = startup-time/VRAM/per-bucket-latency, NOT throughput**" + "v1 FULL_AND_PIECEWISE already optimal; engine AUTO-DOWNGRADES to PIECEWISE for hybrid-GDN/mamba + spec-decode". Those are PER-DEPLOYMENT-CONFIG conclusions.

**SCOPE for THIS run:** find SHIPPABLE, GENERAL (sm_80+sm_86; fork patch / kernel / dispatcher change / better DEFAULT machinery / tooling) cudagraph-MACHINERY levers that raise DECODE BANDWIDTH EFFICIENCY — e.g. cutting padding-waste bytes, maximizing FULL-graph coverage for hybrid+MTP (host-bubble between pieces = idle DRAM), persistent-buffer layout/coalescing/L2 residency, frozen occupancy/launch-config. EXCLUDE per-deployment capture-size tuning (out of scope + already concluded not-throughput). It is a VALID and welcome outcome to conclude a sub-area is already DRAM-SOL with NO headroom — say so and cite the metric that proves it. COLLECT directions; render NO go/no-go verdict (later phase).`

const SCHEMA = {
  type:'object', additionalProperties:false, required:['dimension','directions','takeaways'],
  properties:{
    dimension:{type:'string'},
    directions:{ type:'array', items:{ type:'object', additionalProperties:false,
      required:['title','mechanism','where','bandwidth_link','shippable_as','regime','novelty','what_to_measure','headroom_risk'],
      properties:{
        title:{type:'string'},
        mechanism:{type:'string', description:'the concrete cudagraph-machinery change'},
        where:{type:'string', description:'file:line in vllm/ (or external kernel/PR) anchoring it'},
        bandwidth_link:{type:'string', description:'EXACTLY how it raises decode bandwidth efficiency — fewer wasted bytes / less DRAM-idle host-bubble / better coalescing-L2 / higher occupancy on the bandwidth-bound kernel'},
        shippable_as:{type:'string', description:'patch / kernel / dispatcher change / better-default / tooling — and why it is GENERAL sm80+86 not per-deployment'},
        regime:{type:'string', description:'b1 single-stream / batched decode / hybrid+MTP / which model class'},
        novelty:{type:'string', description:'new | nuance-of-known | re-examination-of-autotune-conclusion'},
        what_to_measure:{type:'string', description:'ncu/nsys metric or A-B that proves real headroom vs already-DRAM-SOL'},
        headroom_risk:{type:'string', description:'honest: is there plausibly real bandwidth headroom here, or is decode likely already DRAM-SOL so this nets ~0?'},
      } } },
    takeaways:{ type:'array', items:{type:'string'} },
  },
}

const DIMS = [
  { key:'padding-waste', title:'Decode batch padding to nearest captured size = wasted KV/weight bytes',
    focus:`At decode, vLLM pads num_input_tokens / batch up to the nearest captured cudagraph size before replay (gpu_model_runner.py pad logic ~:4157/:5807, the capture-size set in cudagraph_dispatcher/cuda_graph.py). A batch of 5 padded to captured 8 streams weights+KV for 8 rows = wasted DRAM traffic on a bandwidth-bound kernel. QUANTIFY the average pad-waste fraction across a realistic decode batch distribution; ask whether the SHIPPABLE DEFAULT capture-size *ladder* (denser at low batch) cuts pad-waste generally, or whether a runtime "compute only the real rows, pad only the graph shape (not the GEMM/attn work)" is possible. Distinguish weight-stream waste (weights are batch-independent → padding a GEMV adds ~0 weight bytes!) from KV/activation waste (∝ rows). Be precise about WHICH bytes padding actually wastes for a weight-bandwidth-bound decode — this may shrink the lever a lot.` },
  { key:'full-vs-piecewise', title:'FULL vs PIECEWISE coverage for hybrid GDN/Mamba + MTP — host-bubble = idle DRAM',
    focus:`The engine auto-downgrades hybrid-GDN/mamba + spec-decode to PIECEWISE (cudagraph_dispatcher.py, mamba_hybrid.py, spec_decode cudagraph_utils.py). PIECEWISE has graph-breaks → host-side gaps between captured pieces where SMs/DRAM idle (no bytes moving). At bandwidth-bound decode, keeping DRAM continuously fed (FULL graph back-to-back) is the bandwidth-efficiency angle. WHY exactly does hybrid+spec force PIECEWISE (in-place mamba state update? dynamic spec shapes? a CPU op mid-graph)? What SHIPPABLE change (make the mamba state-update capturable, route verify to a captured path, FULL_DECODE_ONLY for the decode phase) would extend FULL coverage and remove host-bubbles? Cross-ref the 2080Ti/1Cat "PIECEWISE-for-hybrid" findings + the user's mtp_verify fix (does patch-A's fwd_kvcache route stay capturable?).` },
  { key:'buffer-layout-pool', title:'Persistent cudagraph buffer layout / coalescing / L2 residency',
    focus:`Captured graphs replay against fixed persistent input buffers (input_ids, positions, slot_mapping, block_table, seq_lens, query_start_loc — allocated once in gpu_model_runner capture setup). Their dtype/layout/alignment affects coalescing + L2 residency of the index/metadata reads the bandwidth-bound decode kernels do every step. Are block_table / slot_mapping gathers (paged attention + KV) laid out for coalesced 128B access? Could a better persistent-buffer layout or pinning hot metadata in L2 cut the non-weight DRAM traffic at decode? Also the cudagraph memory POOL — does graph capture's mempool placement hurt KV-cache locality? (Relate to 1Cat's stale-persistent-buffer root-cause, but here for SPEED not correctness.)` },
  { key:'frozen-launch-config', title:'Capture freezes grid/occupancy/launch-config of the bandwidth-bound decode kernels',
    focus:`cudagraph capture bakes the launch config (grid, block, dynamic-smem, occupancy) of each decode kernel — Marlin W4A8 decode GEMV/8-row tile, GDN fused_recurrent, paged-attn — at capture time, per captured size. If a kernel's launched grid under-saturates DRAM (too few CTAs for the bandwidth-bound regime, or blocks_per_sm=1 per the kernel-directions D9 finding), capture LOCKS that suboptimal config for every replay. Is there a SHIPPABLE per-captured-size or per-arch (sm80 vs sm86) launch-config that better saturates DRAM at decode, baked at capture? Connect to kernel-directions D9 (blocks_per_sm) + persistent-kernel/grid-stride interplay with graph capture. Does capturing prevent the kernel from adapting its grid to the real (unpadded) row count?` },
  { key:'overlap-async-stream', title:'cudagraph × async-scheduling / stream-overlap / input-prep hiding',
    focus:`Async scheduling is default-on (autotune memory). At decode the per-step host work (sampler, input prep, block-table build) and the GPU graph replay can overlap; the question is whether the cudagraph boundary serializes work that could overlap to hide latency and keep DRAM busy. Does the FULL graph include or exclude sampling/logits (the [B,vocab] traffic — kernel-directions D15/D16)? Is there a multi-stream / prefetch-next-step opportunity under capture that keeps the weight stream flowing across step boundaries? Is the decode step gapped by un-captured host ops (DRAM idle between steps)? SHIPPABLE: capture more of the step / overlap input-prep / prefetch.` },
  { key:'external-cross-engine', title:'External + cross-engine cudagraph-for-bandwidth-bound-decode tricks (Ampere)',
    focus:`Survey how OTHER engines handle cudagraph for bandwidth-bound Ampere decode: SGLang (cuda_graph_bs ladder, padding policy, graph capture for MoE/hybrid, the "capture + replay" decode path), TensorRT-LLM (CUDA graph + in-flight batching), lmdeploy/TurboMind, the 1Cat (V100, FULL-graph staleness discipline) + weicj 2080Ti (PIECEWISE-for-hybrid default) forks, and RECENT vLLM cudagraph PRs (FULL_DECODE_ONLY, piecewise for spec-decode, hybrid capture, mamba capture). Use WebSearch/WebFetch (load via ToolSearch). Find concrete cudagraph mechanics that reduce decode wasted-bandwidth / host-bubbles that this fork lacks. Cite repos/PRs.` },
  { key:'measurement-grounding', title:'Does the hypothesis even hold? Measure DRAM-SOL vs cudagraph-attributable idle at decode',
    focus:`The decisive grounding. Decode is weight-bandwidth-bound — so the FIRST question is whether cudagraph is leaving ANY bandwidth on the table or decode is already at DRAM SOL (in which case the whole hypothesis nets ~0). Design the measurement that settles it: nsys timeline of a single decode step (is there host-bubble / DRAM-idle between kernels or between steps?), ncu dram__throughput.avg.pct_of_peak_sustained_elapsed on the decode GEMV/attn (already ~90%+ = SOL, or gap?), the pad-waste fraction, FULL-vs-PIECEWISE A/B on the hybrid model (does PIECEWISE show measurable inter-piece DRAM-idle?). Use the fork harness (bench_decode_clean.py, prof_decode_batchsweep.py, torch_prof_phase). State the metric that would CONFIRM real cudagraph-bandwidth headroom vs DEBUNK the hypothesis. This dimension is allowed to conclude "no headroom".` },
]

phase('Investigate')
const findPrompt = (d) => `You are a GPU performance engineer collecting cudagraph×decode-bandwidth optimization DIRECTIONS for the Ampere fork. Read real source under ${AMPERE}/vllm.

${CTX}

YOUR DIMENSION: ${d.title}
${d.focus}

RULES: anchor to real file:line in ${AMPERE}/vllm. Be precise about the BANDWIDTH mechanism — decode is WEIGHT-bandwidth-bound, so be honest about whether a lever touches the dominant weight stream or only the smaller KV/metadata/activation traffic (a lever that only saves metadata bytes is small; SAY SO). Mark "headroom_risk" honestly — much of this may already be DRAM-SOL. Prefer SHIPPABLE general machinery over per-deployment config. No go/no-go. 4-8 directions + 2-4 takeaways.`

const challengePrompt = (d, found) => `Independent reviewer hardening cudagraph-decode-bandwidth directions for "${d.title}". Read ${AMPERE}/vllm to verify.

DRAFT: ${JSON.stringify(found?.directions ?? []).slice(0,9000)}

For EACH: (1) confirm the cudagraph mechanism is really at the cited file:line (not inferred). (2) STRESS-TEST the bandwidth_link — for a WEIGHT-bandwidth-bound decode, does padding/coverage/layout actually move the dominant weight bytes, or only metadata/KV? Downgrade or kill levers whose "bandwidth win" is only tiny metadata traffic. (3) Sharpen "headroom_risk": is this plausibly real or already DRAM-SOL? (4) Tighten what_to_measure into a runnable ncu/nsys/A-B. Keep good, correct weak, drop unsupported, add up to 2 missed. ${CTX}`

const dimResults = await pipeline(
  DIMS,
  (d) => agent(findPrompt(d), { label:`find:${d.key}`, phase:'Investigate', schema:SCHEMA }),
  (found,d) => agent(challengePrompt(d,found), { label:`challenge:${d.key}`, phase:'Challenge', schema:SCHEMA }),
)
const all = dimResults.filter(Boolean)
const allDirs = all.flatMap(r=>(r.directions??[]).map(x=>({...x,_dim:r.dimension})))
log(`Investigate+Challenge: ${allDirs.length} directions across ${all.length} dims`)

phase('Critic')
const titles = allDirs.map((x,i)=>`${i}. ${x.title} [${x.novelty}] headroom:${x.headroom_risk?.slice(0,40)}`).join('\n').slice(0,10000)
const CRITICS = [
  { key:'missing-angle', q:`What cudagraph×decode-bandwidth angle is ABSENT? Consider: capture of the sampling/logits step, KV-cache block-size×coalescing, CUDA graph conditional nodes (12.4+) for dynamic spec shapes, graph instantiation/update cost, multi-graph stream parallelism, mempool↔KV fragmentation, warmup/replay-cache. Propose concrete shippable directions.` },
  { key:'first-principles', q:`First-principles: decode = weight-BW-bound GEMV wall on Ampere. cudagraph removes LAUNCH overhead (host), not bytes. Be the skeptic: which collected directions are REAL bandwidth levers vs which are launch/host-latency levers mislabeled as bandwidth (still useful at b1 where launch dominates, but be honest)? And conversely, what is the ONE highest-leverage cudagraph change for a bandwidth-bound decode that nobody listed? Separate "fills DRAM-idle host-bubbles" (real) from "saves bytes" (rare for cudagraph).` },
]
const crit = await parallel(CRITICS.map(l=>()=>agent(
  `Completeness critic for a cudagraph×decode-bandwidth direction sweep on the Ampere fork. ${CTX}\n\nCOLLECTED (titles):\n${titles}\n\nLENS: ${l.q}\n\nSurface only genuinely-missing or genuinely-mislabeled directions. No go/no-go.`,
  { label:`critic:${l.key}`, phase:'Critic', schema:SCHEMA })))
const critDirs = crit.filter(Boolean).flatMap(r=>(r.directions??[]).map(x=>({...x,_dim:'critic:'+r.dimension})))
log(`Critic: ${critDirs.length} added`)

phase('Synthesize')
const corpus = JSON.stringify([...allDirs,...critDirs]).slice(0,95000)
const report = await agent(
  `Lead author: consolidate a DIRECTION MENU for "shippable sm_80/sm_86 cudagraph-machinery changes that improve DECODE bandwidth efficiency" on the Ampere fork. ${CTX}

ALL DIRECTIONS (find+challenge+critic) JSON:
${corpus}

WRITE THE REPORT (Markdown, no go/no-go — direction menu):
- 4-6 line orientation, INCLUDING the honest headline: does the hypothesis hold? (decode is weight-BW-bound; cudagraph removes host-launch overhead not bytes — so separate REAL bandwidth levers (fill DRAM-idle host-bubbles in PIECEWISE; cut KV/metadata waste) from host-latency levers that only help at b1).
- DEDUP/MERGE across dimensions.
- A "Direction index" table: # · title · bandwidth-mechanism-class (DRAM-idle-bubble / wasted-bytes / occupancy / host-latency-mislabeled) · regime · headroom (real / likely-DRAM-SOL / b1-only) · shippable-as.
- GROUP by mechanism class. Per direction: Mechanism · Where (file:line) · Bandwidth link (honest: weight vs metadata/KV bytes) · Regime · Shippable-as · What to measure · Headroom risk.
- A prominent "Likely already DRAM-SOL / hypothesis-limits" subsection — where cudagraph is NOT leaving bandwidth on the table and the lever nets ~0 (engage the autotune "capture-size = not throughput" conclusion honestly).
- "Suggested first probes" — the 2-3 cheapest measurements that decide whether this whole area has a target (esp. the nsys host-bubble + ncu dram-SOL check at decode).
Be technically dense, skeptical, Ampere-specific. Return ONLY the Markdown.`,
  { label:'synthesize', phase:'Synthesize', effort:'xhigh' })

return { report, dims: all.length, directions: allDirs.length, critic_adds: critDirs.length }
