export const meta = {
  name: '1cat-vllm-crossanalysis',
  description: 'Cross-analyze the 1CatAI/1Cat-vLLM Volta sm_70 fork vs the user\'s Ampere fork: what to steal/port, what is V100-specific, claims to verify',
  phases: [
    { title: 'Analyze', detail: '7 dimensions, deep-read the cloned sm_70 fork + compare to the Ampere fork' },
    { title: 'Verify',  detail: 'independent claim-check: downgrade overclaims, sharpen portability' },
    { title: 'Synthesize', detail: 'cross-analysis doc: TAKE / portable-to-Ampere / V100-specific / debunk' },
  ],
}

const CLONE  = '/tmp/claude-1000/-home-trevor-vllm-ampere-optimized/7ce736dd-d616-425f-994a-37d245d8f1f9/scratchpad/1cat'
const AMPERE = '/home/trevor/vllm-ampere-optimized'
const MEM    = '/home/trevor/.claude/projects/-home-trevor-vllm-ampere-optimized/memory'

const PROFILE = `COMPARISON BASELINE = the user's OWN fork \`vllm-ampere-optimized\` at ${AMPERE}: vendored vLLM v0.23.0 for Ampere sm_80(A100)/sm_86(3090/A40/A6000/A10). Flagship = W4A8 (int4 weight + int8 DYNAMIC activation) via Marlin (patches 0001/0002 un-gate int8-act Marlin + an 8-row int8 decode tile). Also has: MTP spec-decode + a verify-attention fwd_kvcache long-ctx fix (patch A), fp16-accum PV (patch 0007, GeForce-GA10x runtime-gated), int8-QK that was BUILT then REMOVED (net-negative), W4A8 MoE via Marlin moe_wna16 (patch 0005). Serves the SAME Qwen3.5/3.6 27B & 35B-A3B HYBRID models (GatedDeltaNet linear-attn + a few full-attn hd256 layers); the user ships the Avesed/Qwen3.6 AWQ/int4 quants. Test rig = 2×3090 NO-NVLink (PCIe). The user keeps cross-fork analyses; their prior memory verdicts are in ${MEM} (read the relevant project_*.md to CITE the user's existing conclusions, e.g. MXFP4 0-save on Ampere, TurboQuant=capacity-not-speed, FlashQLA=Hopper-only NO-GO, int8-GEMM tapped-out, KV-quant=capacity-first).`

const VFACTS = `V100 HARDWARE = Volta sm_70: 1st-gen tensor cores = FP16-input/FP32-accum HMMA (hmma.884 m8n8k4-class) ONLY — NO int8/int4 IMMA (Turing sm_75+), NO bf16 (sm_80+), NO FP8 tensor cores (sm_89+). HAS NVLink (unlike the user's 2×3090). 16/32GB HBM2. IMPLICATIONS to apply: on V100 any 'AWQ 4-bit' GEMM must dequant int4→fp16 and run FP16 tensor cores (≈ the Ampere W4A16 path, NOT the user's W4A8-int8 flagship which REQUIRES int8 IMMA); bf16-trained Qwen MUST run in fp16 (overflow/accuracy risk); 'FP8' on Volta is storage/emulation, never TC; upstream vLLM DROPPED sm_70 so they re-enabled the whole build. Keep these constraints front-of-mind — they decide what is V100-specific vs portable up to Ampere.`

const SCHEMA = {
  type:'object', additionalProperties:false, required:['dimension','findings','takeaways'],
  properties:{
    dimension:{type:'string'},
    findings:{ type:'array', items:{ type:'object', additionalProperties:false,
      required:['topic','what_they_built','where','maturity','vs_ampere','portable_to_ampere','v100_specific','claim_flag'],
      properties:{
        topic:{type:'string'},
        what_they_built:{type:'string', description:'concrete mechanism / technique'},
        where:{type:'string', description:'file path(s) in the 1cat clone, or the doc/log, anchoring the claim'},
        maturity:{type:'string', description:'shipped-default / opt-in / experimental / benchmark-only / claimed-unverified'},
        vs_ampere:{type:'string', description:'how it compares to the user\'s Ampere fork approach — same / different / they-have-more / user-has-more'},
        portable_to_ampere:{type:'string', description:'could the user adopt it (code or methodology)? what would change for sm80/86? or N/A'},
        v100_specific:{type:'string', description:'what is Volta-only (no int8 IMMA / no bf16 / NVLink) and would not or need not port'},
        claim_flag:{type:'string', description:'any overclaim / unverified perf / or a DEBUNK or CONFIRM of one of the user\'s prior memory verdicts; else "none"'},
      } } },
    takeaways:{ type:'array', items:{type:'string'}, description:'crisp TAKE / PORTABLE / V100-ONLY / DEBUNK bullets' },
  },
}

const DIMS = [
  { key:'overview-claims', title:'Overview, maturity, validated-vs-experimental, headline claims',
    focus:`Read ${CLONE}/README.md, RELEASE.md, AGENTS.md, CLAUDE.md, the two top-level audit logs (SM70_FLASH_V100_QUALITY_EXPERIMENT_LOG_*, SM70_MTP_OUTPUT_QUALITY_AUDIT_*), and the GitHub release notes if cited. Establish: project maturity (451★, v1.2.1), what is SHIPPED-DEFAULT vs OPT-IN vs EXPERIMENTAL vs BENCHMARK-ONLY, exact model coverage (Qwen3.6-27B/35B-A3B/3.5-122B AWQ), hardware envelope (4×/2×V100 32GB, NVLink, TP4, 256K ctx default), and every HEADLINE PERF/QUALITY CLAIM. VERIFY each headline claim against an actual file/log/benchmark in the repo — flag any that are asserted without evidence. Note their dev process (they use Claude Code: AGENTS.md/CLAUDE.md).` },
  { key:'sm70-marlin', title:'sm70 Marlin GEMM zoo (dense + MoE): AWQ on Volta without int8 IMMA',
    focus:`Deep-read ${CLONE}/csrc/quantization/marlin/sm70_*.{cu,cuh} and ${CLONE}/csrc/moe/marlin_moe_wna16/sm70_*. Determine the ACTUAL MMA used (hmma.884 fp16 m8n8k4? dp4a CUDA-core? — there is NO int8 IMMA on sm_70), the format zoo (u4 / u4b8 / u8 / u8b128 / fp8 / mxfp4 / nvfp4), split-K, the awq/gptq repack, the dispatch. Compare to the user's Marlin patches (${AMPERE}/patches/0001,0002 + csrc). KEY questions: (1) their 'AWQ 4-bit' is necessarily W4A16-fp16-accum, NOT the user's W4A8-int8 — confirm. (2) They ship mxfp4/nvfp4 Marlin — does that DEBUNK or CONFIRM the user's memory verdict 'MXFP4 0-save on Ampere / NVFP4 footnote'? (3) Is anything (split-K skinny-GEMM, repack, a format) PORTABLE up to Ampere where the user might benefit?` },
  { key:'flash-attn-v100', title:'FLASH_ATTN_V100 attention backend (XQA decode, D=256 paged-prefix low-smem)',
    focus:`Deep-read ${CLONE}/vllm/v1/attention/backends/flash_attn_v100.py + any csrc it calls (grep for flash_v100 / XQA / sm70 attention kernels) + the FLASH_V100 quality log. Map: decode path, prefill path, the 'guarded XQA decode', the 'D=256 paged-prefix low-smem fast path', the SM70 compile-graph, how the backend is REGISTERED/selected. THIS IS THE MOST RELEVANT DIMENSION — the user is building their OWN Ampere attention backend (read ${AMPERE} flashampere/int8qk backend + ${MEM}/project_flashampere_backend.md, project_standalone_ampere_attn_backend.md, project_famp_own_kernel_scope.md, project_mtp_verify_attention_fix.md). 1Cat ships XQA decode on EVEN OLDER hardware and the user just researched un-gating FI XQA for MTP-verify. What is the design lesson / portable code? How does their D=256 (= Qwen hd256) handling compare to the user's? Does their XQA-on-Volta inform the user's XQA-verify plan?` },
  { key:'sm70-turbomind', title:'Vendored TurboMind (lmdeploy) sm70 kernels + the FlashQLA question',
    focus:`Explore ${CLONE}/csrc/sm70_turbomind/ — what TurboMind/lmdeploy components they VENDORED (GEMM? attention? kv-cache? allocator?), for what, and how it is wired into the vLLM runtime. The user's int8 roadmap (${MEM}/project_int8_path_roadmap_ampere.md) only CONSIDERED QServe/TurboMind — this is an ACTUAL integration to learn from. ALSO resolve the FlashQLA question: there are benchmarks/*flashqla* files — do they actually RUN a FlashQLA-style kernel on V100, or benchmark/reject it? The user's memory (${MEM}/project_2080ti_fork_crossanalysis.md) DEBUNKED FlashQLA as 'Hopper-only / Ampere-excluded / prefill-only NO-GO'. Read benchmark_sm70_flashqla_*.py + any flashqla source. CONFIRM or COMPLICATE that verdict with what 1Cat actually does on sm_70.` },
  { key:'mtp-specdecode', title:'MTP / speculative decode on V100 + quality audit',
    focus:`Read ${CLONE}/SM70_MTP_OUTPUT_QUALITY_AUDIT_20260616.md, the MTP env-flag handling (grep SM70 MTP, the 1.2.1 commit '[Bugfix] Respect zero-valued SM70 MTP env flags'), benchmark_sm70_* MTP/decode files, and how MTP is gated (README: 'long-context public profiles default to NO MTP'). Compare to the user's MTP work: ${MEM}/project_spec_decode_ampere.md + project_mtp_verify_attention_fix.md (the user found MTP is net-NEGATIVE beyond ~10k ctx UNTIL their verify-attention fwd_kvcache fix; K=2 sweet). DOES 1Cat's 'no-MTP at long-context default' CONFIRM the user's long-ctx MTP cliff finding (i.e. they hit the same wall and just disabled it rather than fixing the verify-attention kernel)? Accept-len, quality, env flags, why opt-in.` },
  { key:'gdn-fp8-kv-turboquant', title:'GDN hybrid on Volta + FP8 + KV-quant + TurboQuant',
    focus:`Read benchmark_sm70_gdn_exactness.py / benchmark_sm70_gdn_prefill_compare.py + grep for GatedDeltaNet/mamba/causal_conv1d handling on sm_70; the FP8 paths (benchmark_sm70_fp8_*, sm70_marlin_fp8_gemm.cu, '[Core] Fix SM70 FP8 compact allreduce') — FP8 on a GPU with NO fp8 TC = storage/emulation, characterize exactly what they do; and benchmark_sm70_turboquant_quality.py + any TurboQuant/KV-quant code. Compare to the user's verdicts: ${MEM}/project_kv_quant_ampere_verdict.md (KV-quant=capacity-not-speed, TurboQuant NO-GO), and the hybrid GDN handling (${MEM}/feedback_qwen35_hybrid_kernels.md needs fla+causal-conv1d). CONFIRM/DEBUNK: is their TurboQuant a speed or capacity play? How do they run GDN (fla? custom sm70 kernel?) on Volta?` },
  { key:'build-fp16world', title:'CUDA 12.8 sm_70 build/packaging + the fp16-only-world accuracy lesson',
    focus:`Read the build/packaging story (README Quick Start, RELEASE.md, setup.py/CMakeLists for sm_70 arch enabling, the wheel-bundling commits '[Build] Bundle Flash-V100 / upstream FA2 in SM70 wheel', how they re-enabled an arch upstream DROPPED) and compare to the user's ${MEM}/project_fork_build_from_source.md. THEN the bigger synthesis lever: V100 has NO bf16, so they run bf16-trained Qwen ENTIRELY in fp16 — read their attention/GEMM exactness benchmarks (benchmark_sm70_attention_exactness.py, benchmark_sm70_turbomind_exactness.py, the quality logs) for how they handle fp16 overflow/accuracy on long-CoT. This is DIRECT EVIDENCE for the user's NEW fp16-accum levers (today's kernel-directions doc D1 Marlin use_fp16_accum + D4 fp16-QK + shipped fp16-PV patch 0007, all GeForce-fp16-accum): a whole fork that lives in fp16-accumulate land. What does their accuracy experience say about the user's fp16-accum-HMMA accuracy gate?` },
]

phase('Analyze')
const findPrompt = (d) => `You are doing a rigorous CROSS-ANALYSIS of the 1CatAI/1Cat-vLLM fork (Volta sm_70 / Tesla V100) for an expert who maintains a SIBLING Ampere fork. Full clone is at ${CLONE} (read real code with Read/Grep/Glob). Do NOT render go/no-go for the user's roadmap — produce factual, sourced findings + sharp takeaways.

${VFACTS}

${PROFILE}

YOUR DIMENSION: ${d.title}
${d.focus}

RULES: anchor every claim to a real file path in ${CLONE} (or the user's ${AMPERE}/${MEM} when comparing). Mark maturity HONESTLY (shipped-default vs opt-in vs experimental vs benchmark-only vs claimed-unverified) — this fork has many experimental/benchmark files; do not present a benchmark script as a shipped feature. For every technique, explicitly answer: is it PORTABLE up to Ampere (code or just methodology, what changes), or V100-SPECIFIC (and why — usually no-int8-IMMA / no-bf16 / NVLink). Where it bears on one of the user's PRIOR memory verdicts, CONFIRM or DEBUNK it with evidence. Be concrete and skeptical. 5-9 findings + 2-5 takeaways.`

const verifyPrompt = (d, found) => `You are an independent verifier hardening a cross-analysis of the 1Cat-vLLM (sm_70) fork. Clone at ${CLONE}; the user's fork/memory at ${AMPERE} / ${MEM}.

DRAFT FINDINGS for "${d.title}":
${JSON.stringify(found?.findings ?? []).slice(0, 9000)}
DRAFT TAKEAWAYS: ${JSON.stringify(found?.takeaways ?? [])}

For EACH finding: (1) re-open the cited file in ${CLONE} and CONFIRM the mechanism is really there (not inferred from a filename) — downgrade maturity to 'benchmark-only' or 'claimed-unverified' if the code/log doesn't back it, or correct it. (2) Sanity-check the V100-hardware reasoning (no int8 IMMA / no bf16 / no fp8 TC / NVLink) — fix any wrong claim that something uses int8 tensor cores on Volta, etc. (3) Tighten 'portable_to_ampere' to be specific and honest. (4) Verify any CONFIRM/DEBUNK of the user's prior verdict by reading the cited ${MEM} file. Keep good findings, correct weak ones, drop unsupported ones, add up to 2 the draft missed. Prefix each takeaway with [VERIFIED]/[CORRECTED]/[ADDED]. ${VFACTS}`

const dimResults = await pipeline(
  DIMS,
  (d) => agent(findPrompt(d), { label:`find:${d.key}`, phase:'Analyze', schema:SCHEMA }),
  (found, d) => agent(verifyPrompt(d, found), { label:`verify:${d.key}`, phase:'Verify', schema:SCHEMA }),
)

const all = dimResults.filter(Boolean)
const corpus = JSON.stringify(all).slice(0, 110000)
log(`Analyze+Verify done: ${all.length}/${DIMS.length} dimensions, ${all.flatMap(r=>r.findings??[]).length} findings`)

phase('Synthesize')
const report = await agent(
  `You are the lead author writing a CROSS-ANALYSIS of the 1CatAI/1Cat-vLLM fork (Volta sm_70 / Tesla V100, 451★, v1.2.1) for an expert who maintains the SIBLING Ampere fork \`vllm-ampere-optimized\`. Output a single dense Markdown report.

${VFACTS}
${PROFILE}

ALL VERIFIED FINDINGS (7 dimensions) as JSON:
${corpus}

WRITE THE REPORT — model it on the user's existing cross-fork analyses (TAKE / DEBUNK framing). Sections:
1. **What it is** (4-6 lines): maturity, model coverage, hardware, the one-sentence thesis (a serious, validated sm_70 fork that ports nearly every modern lever — Marlin/FlashAttn/MTP/GDN/TurboMind — DOWN to Volta, serving the SAME Qwen3.5/3.6 AWQ models the user quantizes).
2. **Architecture map** — a table: lever × {1Cat's sm_70 approach, the user's Ampere approach, relationship}.
3. **STEAL / portable up to Ampere** — the concrete things (code or methodology) the user could adopt, each with the 1Cat file ref + what would change for sm80/86 + why it's worth it. Rank by value.
4. **V100-SPECIFIC — won't/needn't port** — what is Volta-only (no int8 IMMA → their Marlin is fp16-accum W4A16 not the user's W4A8-int8; no bf16; NVLink TP) and the inverse: what the USER has that 1Cat structurally CANNOT (W4A8 int8-act, int8 IMMA prefill win).
5. **DEBUNK / CONFIRM of the user's prior verdicts** — for each (MXFP4/NVFP4 Marlin, TurboQuant, FlashQLA, MTP long-ctx cliff, fp16-accum accuracy), state what 1Cat's actual code shows and whether it confirms or complicates the user's memory verdict. Quote the user's verdict.
6. **The fp16-only-world lesson** — 1Cat lives entirely in fp16-accumulate land (no bf16); what their accuracy/exactness experience says about the user's NEW fp16-accum-HMMA levers (today's D1/D4 + shipped fp16-PV).
7. **Claims to take with salt** — overclaims / unverified perf / benchmark-only items dressed as features.
8. **Notable divergences & dev process** — anything surprising (they use Claude Code; 256K default; the 122B model; etc.).
Anchor to 1Cat file paths throughout. Be specific and skeptical. Return ONLY the Markdown.`,
  { label:'synthesize', phase:'Synthesize', effort:'xhigh' }
)

return { report, dims: all.length, findings: all.flatMap(r=>r.findings??[]).length }
