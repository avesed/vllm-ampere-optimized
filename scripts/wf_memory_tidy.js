export const meta = {
  name: 'memory-index-tidy',
  description: 'Compress the bloated MEMORY.md index hooks to <=200 chars each WITHOUT losing info (verify each fact is in its topic file; migrate any index-only fact into the file first)',
  phases: [
    { title: 'Parse',    detail: 'read MEMORY.md → ordered entry list' },
    { title: 'Compress', detail: 'per entry: verify file-covers-hook, migrate if missing, emit tight hook' },
  ],
}

const MEM = '/home/trevor/.claude/projects/-home-trevor-vllm-ampere-optimized/memory'

// ---------------------------------------------------------------- PARSE
phase('Parse')
const parsed = await agent(
  `Read ${MEM}/MEMORY.md. It is an index file: each non-empty line is \`- [Title](file.md) — <hook text>\`. Return EVERY entry IN ORDER as {idx (0-based position among non-empty entry lines), file (the .md filename in the link), title (text in [ ]), hook (everything after the first " — ")}. Do not skip, merge, reorder, or edit anything. Ignore blank separator lines (do not emit them as entries).`,
  { label: 'parse-index', phase: 'Parse',
    schema: { type:'object', additionalProperties:false, required:['entries'],
      properties:{ entries:{ type:'array', items:{ type:'object', additionalProperties:false,
        required:['idx','file','title','hook'],
        properties:{ idx:{type:'integer'}, file:{type:'string'}, title:{type:'string'}, hook:{type:'string'} } } } } } }
)
const entries = (parsed.entries ?? []).slice().sort((a,b)=>a.idx-b.idx)
log(`Parsed ${entries.length} index entries`)

// ---------------------------------------------------------------- COMPRESS (batched by file, no conflicts)
phase('Compress')
const BATCH = 5
const batches = []
for (let i=0;i<entries.length;i+=BATCH) batches.push(entries.slice(i,i+BATCH))

const compressOne = (batch, bi) => agent(
  `You are tidying an engineer's persistent memory INDEX (MEMORY.md) for the vllm-ampere-optimized fork. The index has bloated: many hooks are 300-3600 chars. Your job: compress each assigned hook to a tight one-liner WITHOUT losing information, by first guaranteeing the dropped detail lives in the topic file.

For EACH entry below, do this:
1. Read the topic file ${MEM}/<file>.
2. Compare the current (long) hook against the file. Identify any LOAD-BEARING fact in the hook that is NOT already in the file — load-bearing = a verdict, a current status (SHIPPED/NO-GO/DONE/branch/commit), a make-or-break gotcha, a specific number/path/flag the engineer would later rely on.
3. If (and ONLY if) such a fact is missing from the file, append it to the file minimally: add a short bullet under a clearly-labelled \`## Index-migrated facts\` section at the end (create the section once if absent). Do NOT rewrite or restructure existing file content. Do NOT mutate the file if it already covers everything (avoid churn).
4. Produce a COMPRESSED replacement line of the form \`- [<title>](<file>) — <hook>\` where the new hook is <=200 chars TOTAL line length if at all possible (hard ceiling 240 only for entries that genuinely cannot survive shorter). Keep title and filename EXACTLY as given. The hook must preserve: the core claim/verdict + current status (e.g. SHIPPED / NO-GO / DONE / in-progress + critical branch) + at most ONE make-or-break gotcha. DROP into-the-file-only: measurement tables, commit hashes, line numbers, multi-step recipes, perf deltas. Keep at most one \`[[cross-link]]\` if it is load-bearing and fits. Keep it information-dense and specific (this is recall-relevance text), just short.

Return one result per entry: {idx, new_line (the full compressed markdown line), migrated (empty string if you changed nothing, else a <=120-char note of exactly what you appended to which file)}.

ENTRIES (JSON):
${JSON.stringify(batch.map(e=>({idx:e.idx,file:e.file,title:e.title,hook:e.hook})))}`,
  { label: `compress:batch${bi}`, phase: 'Compress',
    schema: { type:'object', additionalProperties:false, required:['results'],
      properties:{ results:{ type:'array', items:{ type:'object', additionalProperties:false,
        required:['idx','new_line','migrated'],
        properties:{ idx:{type:'integer'}, new_line:{type:'string'}, migrated:{type:'string'} } } } } } }
)

const out = await parallel(batches.map((b,bi)=>()=>compressOne(b,bi)))
const results = out.filter(Boolean).flatMap(r=>r.results ?? [])

// stitch back in original order
const byIdx = new Map(results.map(r=>[r.idx, r]))
const missing = entries.filter(e=>!byIdx.has(e.idx)).map(e=>e.idx)
const newLines = entries.map(e => byIdx.get(e.idx)?.new_line ?? `- [${e.title}](${e.file}) — ${e.hook}`)
const migrations = results.filter(r=>r.migrated && r.migrated.trim()).map(r=>({idx:r.idx, note:r.migrated}))
const overLimit = newLines.map((l,i)=>({i,len:l.length})).filter(x=>x.len>240)

log(`Compressed ${results.length}/${entries.length}; ${migrations.length} files got migrated facts; ${overLimit.length} lines still >240`)

return {
  body: newLines.join('\n'),
  stats: { entries: entries.length, compressed: results.length, missing, migrations, overLimit },
}
