# famp_marlin — serve recipe + correctness gate (Ampere sm80/sm86, 2x RTX 3090)

`FampMarlinKernel` mirrors vLLM's stock `MarlinLinearKernel` EXACTLY but routes the marlin ops to
`torch.ops.famp_marlin.*` (the vendored kernel). A `vllm.general_plugins` entry point inserts it
BEFORE stock Marlin so it is selected for W4A8-int8 / W4A16 layers. This is a maintenance/packaging
win (the fork drops most marlin patches down to the vendored kernel + at most 1 config line), NOT a
perf change — famp is expected to be at PARITY with stock.

## P4 verdict (one line)
No NEW source patch is needed to deliver int8 act-type to FampMarlinKernel (it reads `c.act_type` from
its config; `VLLM_MARLIN_INPUT_DTYPE` reaches the worker via normal env propagation). ONE existing fork
line must be WIDENED: `compressed_tensors_wNa16.py:115` `if kernel_type is MarlinLinearKernel:` ->
`if kernel_type in (MarlinLinearKernel, FampMarlinKernel):` — required ONLY for the
W4A16-ckpt + `VLLM_MARLIN_INPUT_DTYPE=int8` path (native-W4A8 / auto_gptq / awq paths need zero edits).

---

## 0. Build the kernel (once, on the GPU box, normal env with nvcc)
    cd /home/trevor/vllm-ampere-optimized
    python -m flashampere.marlin.build            # -> flashampere/marlin/build/famp_marlin.so
    # expect: FAMP_MARLIN_BUILT ops_registered: ['marlin_gemm','gptq_marlin_repack','awq_marlin_repack']

> ARCH WARNING: the .so is single-arch (`-gencode sm_<cc>`). `torch.ops.load_library` REGISTERS a
> wrong-arch .so successfully — the mismatch only surfaces as a CUDA error at kernel LAUNCH (mid-serve).
> You MUST build on the SAME SM the workers serve on (both 3090s are sm_86, so this is a no-op here).

## 1. Pre-serve correctness gate (BIT-EXACT vs stock _C marlin) — MUST pass before serving
    python -m flashampere.marlin.test_kernel_equiv
    # expect:
    #   EQUIV_OK W4A8-int8 g32  M=17 ...
    #   EQUIV_OK W4A16    bf16  M=17 ...
    #   EQUIV_OK W4A8-int8 g128 M=17 ...
    #   EQUIV_OK W4A16 GPTQ-sym M=17 ...
    #   EQUIV_OK ALL
    #   ALL EQUIVALENCE CHECKS PASSED — FampMarlinKernel is bit-exact vs stock MarlinLinearKernel

> WHAT THIS GATE DOES *NOT* COVER — run §1b too: test_kernel_equiv runs apply_weights EAGERLY (under
> torch.no_grad), so it does NOT exercise the COMPILED forward. famp_marlin must register a meta/fake
> for marlin_gemm (build.py:_register_fakes, byte-identical to stock _C) or Inductor (VLLM_COMPILE, the
> default V1 serve mode) raises "no meta/abstract implementation" or graph-breaks to eager — silently
> killing compile/cudagraph (the whole ownership/perf goal). The eager gate passes either way; only the
> compiled serve in §1b/§3 catches a missing fake.

## 1b. Compiled-path smoke (the fake-impl gate the eager test misses) — MUST pass before serving
Serve any W4A8/W4A16 ckpt with the DEFAULT compilation (do NOT pass --enforce-eager), send one request,
and confirm there is NO "Unable to find meta/abstract" / "graph break" on famp_marlin in serve.log:
    grep -iE "famp_marlin.*(meta|abstract|fake|graph break)" serve.log   # expect: no hits
    grep -iE "could not find|no fake impl|abstract impl" serve.log        # expect: no famp_marlin hits
If hits appear, _register_fakes did not run (check that get_famp_marlin() fast path loaded the .so).

## 2. Make the plugin discoverable (sanitized-worker-safe: .pth + manual dist-info, NOT PYTHONPATH)
vLLM spawns TP workers with a SANITIZED env (no PATH/PYTHONPATH). So make `flashampere` importable via
a site-packages `.pth` (NOT PYTHONPATH) and register the entry point via a manual `.dist-info`.

Site-packages of the SERVE interpreter (`/usr/local/lib/python3.12/dist-packages`):

  a) `flashampere.pth` (one line):
         /home/trevor/vllm-ampere-optimized
  b) `famp_marlin-0.0.0.dist-info/entry_points.txt`:
         [vllm.general_plugins]
         famp_marlin = flashampere.marlin.plugin:register_fampmarlin
  c) `famp_marlin-0.0.0.dist-info/METADATA` (minimal):
         Metadata-Version: 2.1
         Name: famp-marlin
         Version: 0.0.0

Verify discovery (NOT in PYTHONPATH — relies on the .pth):
    python -c "from importlib.metadata import entry_points; \
      print([e.name for e in entry_points(group='vllm.general_plugins')])"
    # must include 'famp_marlin'

(If `flashampere` is already on a `.pth` for the attention/fused_silu plugins, reuse it — add only the
`famp_marlin` dist-info entry-point.)

> VLLM_PLUGINS GOTCHA: if `VLLM_PLUGINS` is SET (not None), vLLM loads ONLY the named plugins. If you
> allowlist the flashampere attention / fused_silu plugins there, you MUST also include `famp_marlin`,
> else this plugin is silently skipped -> stock Marlin used, ownership goal missed with no error. Leave
> `VLLM_PLUGINS` unset to load all discovered plugins (the default).

## 3. Serve W4A8 (27B on 2x3090)

Native W4A8-int8 ckpt (act_type hardcoded int8 in the scheme — no env needed):
    vllm serve <W4A8-int8-ckpt> -tp 2 \
      --max-num-seqs 32 \
      ... (shm 8g + --ipc=host if containerized; expandable_segments per E2E notes)

W4A16 ckpt served as int8-act W4A8 (needs the env + the wNa16.py:115 widen patch — see P4):
    VLLM_MARLIN_INPUT_DTYPE=int8 \
    vllm serve <W4A16-ckpt> -tp 2 --max-num-seqs 32 ...
    # or the CLI alias:  --marlin-input-dtype int8

Opt-out: `FAMP_MARLIN=0` (where env is readable, e.g. main process) OR — the standard escape hatch,
worker-safe — `VLLM_DISABLED_KERNELS=FampMarlinKernel`. Either makes the plugin no-op / not insert, so
choose_mp_linear_kernel falls through to stock MarlinLinearKernel (which stays in the list as fallback).
NOTE: `VLLM_DISABLED_KERNELS=MarlinLinearKernel` does NOT disable famp (distinct __name__) — it disables
the stock fallback while famp keeps serving; use `FampMarlinKernel` to turn famp OFF.
Do NOT pass `--linear-backend` (any non-`auto` value filters famp out — `_LINEAR_BACKEND_KERNEL_MAP`
does not list FampMarlinKernel; default is `auto`, so this only bites if you set the flag).

## 4. CONFIRM famp is actually used (not stock _C) — independent markers

a) Plugin insert log (every worker, at load_general_plugins time):
       grep -i "famp_marlin: FampMarlinKernel inserted" serve.log
   # "FampMarlinKernel inserted at _POSSIBLE_KERNELS[CUDA][N] (before Marlin/AllSpark)"

b) Scheme selection log: the chosen kernel name is logged once per layer-class:
       grep -i "Using FampMarlinKernel" serve.log     # FampMarlin, NOT Marlin
   (If 'Using MarlinLinearKernel' appears instead -> famp inserted AFTER stock or import failed;
    check for "famp_marlin: registration skipped" / ".so load failed" warnings.)

   HARD FAIL — treat this as famp NOT running (silently fell back to stock):
       grep -i "famp_marlin: .so load failed, reverting insertion" serve.log   # must be EMPTY
   This warning = the .so was missing or wrong-arch (see §0), so the insert was reverted and stock
   Marlin is serving. The A/B harness MUST fail the famp leg if this line is present.

c) Op-load marker (out of band):
       python -c "import flashampere.marlin.build as b; b.get_famp_marlin(); import torch; \
         print(hasattr(torch.ops,'famp_marlin'), hasattr(torch.ops.famp_marlin,'marlin_gemm'))"
       # True True

## 5. Correctness: GSM8K (must match stock W4A8 within noise)
Serve as above, then run the repo GSM8K harness via the OpenAI API (project rule: API not offline
LLM()):
    bash scripts/run_api_bench.sh   # or scripts/api_bench_client.py against the served endpoint
    # GSM8K target: match the published W4A8 number for the model (e.g. int4 96.8%).
A pass on `test_kernel_equiv` (bit-exact) PREDICTS GSM8K parity; GSM8K is the end-to-end confirmation
that selection + the wNa16 override + the int8-act path are all wired correctly under real serving.

## 6. Throughput (optional A/B, famp vs stock)
A: `FAMP_MARLIN=1` (famp).  B: `FAMP_MARLIN=0` (stock _C).  Same ckpt/flags.
Report decode/prefill/batched in tok/s (decode=1000/TPOT, prefill=input_len/TTFT). Expectation:
PARITY (same kernel sources). Any deviation = a wiring bug; investigate before shipping.
