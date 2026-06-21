# RESEARCH — Autotuner + GPU-Silicon-OC tool for vllm-ampere-optimized

> **Scope**: design/feasibility doc for adding an automated **tuning tool** to the fork —
> a `measure → classify-bottleneck → prescribe → verify` loop (à la `jungledesh/profile`)
> **extended to tune the GPU silicon itself** (memory/core clock offset + power limit), as a
> shippable Ampere-line artifact distinct from the W4A8/Marlin patches.
> **Generation**: 5 research lenses (OC mechanism on headless Linux · correctness gating ·
> autotuner architecture · OC ROI & SKU coverage · 24/7 stability & shippability) +
> adversarial verdicts on the 6 riskiest claims (2 CONFIRMED/PARTIAL, 3 REFUTED, 1 PARTIAL).
> **Date**: 2026-06-19.
> **⚠️ Honesty note**: knowledge cutoff is 2026-01; the silicon-OC half rests on driver/NVML
> behavior (R555+/R570+ `nvmlDeviceSetClockOffsets`) and field OC reports verified by live
> search. The headline mem-OC numbers (+7-12% decode) are **projected from roofline, NOT yet
> measured on this rig** — §7 is the experiment that decides it. Every claim that an
> adversarial verdict refuted or down-weighted has been corrected below; do not re-inflate.

---

## 1. Executive summary + recommendation

**Recommendation: GO-WITH-GUARDRAILS.**

Ship the **autotuner tool** plus the **two safe silicon tiers (power-limit, locked-clocks)** ON
by default across the whole Ampere line. Ship **memory-overclock (the headline +7-12% decode
lever) only as a default-OFF, consumer-sm_86-only, correctness-gated + thermally-gated, host-root
opt-in.** Do **not** ship it on by default and do **not** present it as a line-wide win.

**The single most important reason**: the high-value lever (GDDR6X mem-OC) sits on memory that has
**no array ECC**, so an unstable overclock emits *plausible-but-wrong tokens with zero hardware
signal* — the exact corrupt-then-coherent failure mode this project has already burned sessions
chasing (shm/ckpt). The adversarial round **refuted** the three claims that would have made mem-OC
a clean GO:
- **Golden-drift alone is NOT a sufficient correctness gate** (CLAIM 3 REFUTED) — it only covers
  the cells/paths the probe touches; must pair with a memtest + GEMM self-check + soak.
- **Mem-OC does NOT generalize across the Ampere line** (CLAIM 4 REFUTED) — it is a consumer-GeForce
  (3090/3080) **+ workstation A6000/A5000** lever (their GA102 vBIOS leaves offsets unlocked on
  **headless Linux** — corrected from the earlier "Windows-Afterburner-only" framing); **A100 / A40 /
  A10 / A2 are clock-locked**. The portable cross-*line* levers are still just {power-limit,
  locked-clocks}. And the **EDR bandwidth-knee gate is GDDR6X-only** (3090/3080) — GDDR6 parts
  (A6000/A5000) gate on ECC counters / golden-drift instead.
- **Sustained 3090 mem-OC is NOT thermally stable by default** (CLAIM 6 REFUTED) — GDDR6X
  mem-junction heat-soaks toward its ~110 °C throttle (bandwidth reduction starts ~92-95 °C),
  invisible to Linux `nvidia-smi`, and OC drifts unstable over weeks.

What **survived**: mem-OC *can* be applied headless on a 3090 with root via documented NVML
(CLAIM 1 CONFIRMED); the physics is right and it is the biggest lever found in the whole project
(CLAIM 2 PARTIAL — real but **sub-proportional**, ~+6-8% not a guaranteed 1:1, and unmeasured for
decode); power-limit is a genuinely safe always-on perf/watt lever (CLAIM 5 PARTIAL — favorable
but a few % loss, not literally free).

So the **tool is shippable and worth building**; the **autotuner + power-limit + locked-clocks are
a clean GO**; **mem-OC is a guardrailed opt-in whose payoff (~+6-8% decode) must be proven on the
rig in §7 to exceed the gating engineering cost before it is enabled anywhere.**

**Packaging conclusion.** The autotuner does **NOT** ship in the fork `scripts/` or image — three
independently-sufficient reasons: (1) it needs **sudo** (host-root NVML SET / BAR0 `/dev/mem`); (2) in a
container it needs **`--privileged` + `/dev/mem` + `SYS_ADMIN`**; (3) it **will not run in the shipped
unprivileged serving deployment**. It splits into **HALF-A** (no-privilege vLLM-flag tuner) → the vLLM
tuning ecosystem (contribute to `jungledesh/profile` / `vllm auto_tune`) and **HALF-B** (host-root
silicon-OC supervisor) → a separate opt-in host-side repo (static binary + `systemd` unit). The fork
keeps only the `benchmarks/` diagnostic harness + this doc + a pointer (see §5/§6).

---

## 2. Physics hook & why it maps onto this project

The fork established empirically that on Ampere **vLLM decode is weight/state memory-BANDWIDTH-bound**
(no tile/launch/config change moves it; only "fewer bytes" — W4A8 int4 weights, int8/fp8 KV, bf16
state — help) and **prefill is compute-bound and near-optimal** (W4A8 Marlin GEMM ~68% IMMA, tapped
out). The standard rig is Qwen3.5-9B-W4A8 single-card (tp1) on a 3090: **~87 decode tok/s, ~5753
prefill tok/s**.

The hook: decode tok/s ≈ effective-bandwidth ÷ (weight bytes + KV bytes). W4A8 attacks the
**numerator's denominator** (fewer bytes). **Mem-OC attacks the bandwidth itself** — it raises the
roofline. Prefill being compute-bound, its complement is **core-clock OC + power-limit headroom**.

**Realistic expected gain range per lever** (hedged by the verdicts):

| Lever | Nominal effect | Realistic decode/prefill gain | Verdict hedge |
|---|---|---|---|
| **Mem-OC** (3090, ~+1000 MHz offset, 19.5→21 Gbps) | +7-8% nominal BW; up to ~+12-13% only with a larger lottery-dependent offset | **~+6-8% decode tok/s** (sub-proportional: real decode hits only 60-80% of the BW roofline; KV reads/attention/launch don't all scale with mem-clock) | CLAIM 2 PARTIAL: "+12% optimistic, +7-8% solid"; **no published 3090 decode-vs-mem-OC measurement exists → §7 must measure it** |
| **Core-OC** (+100-180 MHz GPC offset) | higher boost clock | **modest prefill gain only** — Marlin already ~68% IMMA, prefill compute-tapped | smaller win; core errors tend to crash (less silent than mem) |
| **Power-limit** (cap, NOT raise) | constrains budget on the factory V/F curve | **NOT a speedup** — a perf/watt + thermal/aging play: FP16/TF32 hold ~95% at ~270-280 W vs ~350 W stock (a ~20-30% power cut); ~10-20% cut for <5% loss is the most solid band | CLAIM 5 PARTIAL: gaming/FP32 loses ~10% at -30%; not literally free |
| **Locked-clocks** (pin within stock) | removes boost jitter | reproducible/stable TPOT; on GeForce `-lgc` can reach the *top of the boost V/F table* (above advertised boost) but **cannot exceed it** | safe; portable across A100/A40/A10/A6000 where offsets are not |

**The GDDR6X EDC/EDR sweet-spot caveat (load-bearing).** GDDR6X's link protection is **EDR =
Error-Detection-and-Replay**: CRC on each bus transfer, *retransmit* on failure. It **corrects
nothing** and protects **only the controller↔DRAM link**, not stored cells, not compute cores.
Consequence — there is a **throughput knee**: as you raise mem-clock, you get real gains, then EDR
replays start *silently eating effective bandwidth* (tok/s rolls over **before** any crash), then
just past that, uncorrected cell/compute flips produce **silent wrong tokens**. So the tuner must
**gate on MEASURED decode tok/s, never on the applied clock offset** — clock-reported gain ≠
effective-bandwidth gained — and stop at the knee. The knee is a *soft* ceiling, not a guaranteed
safe band below corruption; the correctness gate is still the real line of defense. A direct
**achieved-bandwidth + integrity measurement** (the BW+verify kernel, §4) makes the knee *observable* — but it is a
**proxy, not a safety gate**: EDR only covers CRC'd *bus* errors, so no-ECC *cell/retention* flips can
corrupt output with **zero bandwidth penalty** (adversarial verdict: PARTIAL). Correctness stays the gate. **The EDR knee is GDDR6X-only** (3090/3080-class):
plain GDDR6 (A6000/A5000/A40/A10/A2, and Turing) has no auto-replay — it CRC/EDC-flags but hard-errors
past the limit rather than rolling off, so those parts gate on **ECC counters / golden-drift**, not BW
rollover.

---

## 3. OC mechanism on Ampere — per-SKU lever table

Three layers of "tuning the silicon", with very different headless behavior:
1. **Power-limit + locked-clocks** — NVML/`nvidia-smi`, fully headless (no X), root. Pin/cap
   *within* the factory V/F curve. Safe, portable, the primary serving knob.
2. **True clock OFFSET above stock** — the actual overclock. Headless path = NVML
   `nvmlDeviceSetClockOffsets` (R555.85+; supersedes deprecated
   `nvmlDeviceSetGpcClkVfOffset`/`SetMemClkVfOffset`). CLI wrappers `nvidia_oc` / LACT do this with
   **no X dependency**. GeForce-tier only in practice.
3. **Legacy Coolbits + `nvidia-settings`** — same offsets but **requires a running X server** (or a
   dummy/xvfb X with fake EDID). Worst fit for a headless serving box — avoid; prefer NVML.

**Per-SKU lever matrix** (✓ supported / ✗ rejected-or-no-op / ~ unreliable):

| SKU (arch) | Power-limit | Locked-clocks | Core OFFSET | Mem OFFSET | Mechanism / notes |
|---|---|---|---|---|---|
| **RTX 3090** (GA102, GeForce sm_86) | ✓ | ✓ | ✓ | ✓ | `nvmlDeviceSetClockOffsets` (R555.85+) or deprecated VF-offset; field-proven +150 core / +1700 MT/s VRAM (~+850 MHz mem) via LACT headless |
| **A40** (GA102, server sm_86) | ✓ | ✓ | ✗ | ✗ | datacenter-locked vBIOS; **no offset path** (CLAIM 4) — power-limit + locked/application clocks only |
| **A6000** (GA102, workstation sm_86, GDDR6, ECC-toggleable) | ✓ | ✓ | ✓ | ✓ | mem-offset is a **REAL headless-Linux** NVML/LACT lever — field-verified +150 core / **+2000 MT/s VRAM** (LACT, Fedora-42 headless, Level1Techs); workstation GA102 vBIOS leaves offsets **UNLOCKED** (the earlier "Windows-Afterburner-only" framing was wrong). GDDR6 → **no EDR knee**; gate on ECC counters (ECC on) or golden-token drift (ECC off), NOT BW rollover. `gputemps` **supports** A6000 (junction-abort works). **Safest mem-OC target** (see ECC note below) |
| **A5000** (GA102, workstation sm_86, GDDR6, ECC-toggleable) | ✓ | ✓ | ✓ | ✓ | same GA102 offset-unlocked headless-Linux path as A6000; toggleable ECC; `gputemps` supports it. GDDR6 → no EDR knee → gate via ECC/golden-token. 24GB sibling — RECOMMENDABLE |
| **A4000** (GA104, workstation sm_86, GDDR6, ECC finicky) | ✓ | ✓ | ~ | ~ | GA104 → offset path *likely* works but **UNVERIFIED** in field reports; ECC toggle forum-finicky; **NOT in `gputemps` list → no thermal-abort path**; 140W single-slot → least headroom. LEAST attractive — leave experimental |
| **A10** (GA102, server sm_86) | ✓ | ✓ | ✗ | ✗ | passively-cooled server card, locked vBIOS; offsets not exposed |
| **A100** (GA100, sm_80) | ✓ | ✓ (+ `nvidia-smi -ac` application clocks) | ✗ | ✗ | clock-LOCKED — offset API returns **Not-Supported** (NVIDIA staff confirm "A100 doesn't support overclocking"); HBM2e mem-clock **fixed at 1215 MHz** (no mem-OC, no 2nd P-state) + **true HBM2e ECC**. Only Tier-0/1 **anti-throttle pin** (SM gain caps at stock **1410 MHz**; ~0 on a healthy card, reclaims perf only if throttling) + NVML/DCGM ECC counters as ground truth — **not** OC |
| **A30 / A40 / A10 / A2** (GA100/GA10x, datacenter) | ✓ | ✓ | ✗ | ✗ | all clock-LOCKED like A100 (offset Not-Supported, pin-only); all have **ECC** (A30 HBM2; A40/A10/A2 GDDR6, **not** GDDR6X) → Tier-0/1 + ECC counters only; consumer mem-OC does not transfer |

**API / flag / gate detail:**
- **Power-limit**: NVML `nvmlDeviceSetPowerManagementLimit` (limit in **milliwatts**, Kepler+, root,
  headless) / `nvidia-smi -i <id> -pl <watts>` (+ `-pm 1` persistence mode). Works on all Ampere
  incl. A100. **Not persistent** across reboot/driver-reload.
- **Locked-clocks**: `nvmlDeviceSetGpuLockedClocks` (Volta+) / `nvmlDeviceSetMemoryLockedClocks`
  (**Ampere+**); `nvidia-smi -lgc min,max` / `-lmc min,max` (reset `-rgc`/`-rmc`). Headless. On
  GeForce `-lgc` reaches the top of the boost V/F table; datacenter clamps to the stock list. A mem
  lock may "take effect next time the GPU is initialized" on some paths.
- **Clock offset (the OC)**: `nvmlDeviceSetClockOffsets`/`GetClockOffsets` — NVML **v555 / driver
  R555.85** (≥555.42 required); `nvmlDeviceGetMinMaxClockOfPState` reads bounds but **reported
  offset 0 before R570** → require **R555+, prefer R570+**. Only **P0** (the busy/compute pstate a
  serving vLLM runs in) is meaningfully overclockable. Open-kernel-module snaps offsets to **15 MHz**
  multiples → round tuner steps to 15 MHz. **Root**, **no X**. **Not persistent.**
- **Driver-version gate**: pre-R555 → only deprecated VF-offset calls or the X-only
  `nvidia-settings` path. **Confirm the fork image's host driver is R555+ (ideally R570+)** before
  building the silicon layer on the new API.
- **Persistence**: every power/lock/offset setting is **wiped on reboot or driver reload** — this is
  a **safety PRO** (built-in dead-man's-switch: any crash/Xid/reboot auto-reverts to stock) but a
  deploy CON. Re-apply via a **root systemd oneshot/unit ordered `Before=` the vLLM container**
  (Puget's `nv-power-limit.service` pattern), or run `lactd`. Enable `nvidia-persistenced`.
- **Units gotcha**: a memory **"MT/s" / transfer-rate** offset is **~2× the underlying GDDR clock**
  offset. `+1700 MT/s` on a 3090 ≈ `+850 MHz` mem clock. `nvidia_oc --mem-offset 850` is
  clock-domain; LACT/coolbits use the doubled transfer-rate number — the tool **must** disambiguate.
- **Container vs host**: clocks/power are **global physical-GPU / driver-session state**, never
  per-container virtualized. The **NVIDIA Container Toolkit injects no OC feature** — it is only a
  device-node + driver-lib injector (`compute`/`utility` caps, `/dev/nvidia*`, libs) and **never
  injects `/dev/mem`**, so the `gputemps` BAR0-MMIO read cannot run via the Toolkit. The SET
  capability lives in the **driver/NVML**, which is privileged: an **unprivileged serving container
  cannot set** clocks/power — **root alone is not enough** (Docker drops `CAP_SYS_ADMIN` →
  `NVML_ERROR_NO_PERMISSION`; nvidia-docker #495). A **root container + `--cap-add=SYS_ADMIN` (or
  `--privileged`) CAN set** them, **but the change is global** (hits the host + every co-resident
  workload) and **does NOT revert when that container restarts** (only reboot / driver-reload /
  explicit reset does) → it **breaks the reboot dead-man's-switch + watchdog isolation**. Clean
  design = **a host root daemon (or a dedicated privileged sidecar that is NOT the serving container)
  applies clocks; the unprivileged serving container runs only the correctness probe**.

**ECC-on workstation mem-OC profile (A6000/A5000).** Unlike the no-ECC 3090, the A6000/A5000 can run
with **ECC enabled** (`nvidia-smi -e 1`), turning silent corruption into a machine-readable signal: SBEs
corrected + counted, DBEs detected (DBE → app termination + page retirement), polled via
`nvmlDeviceGetMemoryErrorCounter` / `nvidia-smi -q -d ECC` / DCGM. This is a *useful additional* abort
signal but **NOT a complete gate**: it is **DRAM-array ECC, not documented bus/link ECC**, and mem-OC
instability frequently shows up as bus/signal-integrity transmission errors the array ECC may miss; only
SBE is corrected (DBE = crash); and the SBE-rise→reliability-knee relationship is uncharacterized. So
keep golden-token-drift + BW+verify + junction-temp gating and **never treat "no ECC counter rise" as
proof of correctness**. Cost: **~6.25% VRAM** (1/16 inline GDDR6) + a real bandwidth tax that partly
offsets the mem-OC gain (magnitude unverified; NVIDIA calls it "minor latency overhead"). Offer two
profiles: **ECC-ON "safe/gated"** vs **ECC-OFF "max-headroom"** (the latter falls back to 3090-style
probe-only gating).

---

### 3.1 OC-tool backend — LACT & peers vs roll-own NVML (decided: roll-own)

A focused adversarial deep-dive answered the §8 "ship-as-wrapper vs native" question.
**Decision: roll-own `nvidia-ml-py`/pynvml for the safety-critical tune loop; document LACT only as
an optional manual GUI + boot-persistence helper** (thin backend interface, NVML default, optional
LACT adapter — never route the safety loop through the daemon).

| Tool | Ampere NVIDIA | mem-offset | core-offset | power-limit | headless (no X) | automatable | license | viable backend? |
|---|---|---|---|---|---|---|---|---|
| **raw pynvml / NVML** | native | ✓ | ✓ | ✓ | ✓ | ✓ (lib calls) | NVIDIA + BSD wrapper | **yes — DEFAULT** |
| **LACT** v0.9 | ✓ (NVML, drv ≥555/565) | ✓ | ✓ (per-pstate) | ✓ | ✓ (`lactd` daemon) | ~ (JSON socket; CLI limited) | MIT | yes — optional adapter |
| **nvidia_oc** (Rust CLI) | ✓ (NVML) | ✓ | ✓ | ✓ | ✓ | ✓ (CLI) | free | yes — alt |
| GreenWithEnvy (gwe) | ✓ | ✓ | ✓ | ✓ | ✗ (GTK+Xext+Coolbits) | ✗ (GUI) | GPL-3 | no (archived/EOL) |
| nvidia-settings+Coolbits | native | ✓ | ✓ | ✗ | ✗ (needs live X) | ~ | proprietary | no (headless dead-end) |
| TuxClocker | ✓ | ✓ | ✓ | ✓ | ✗ (Coolbits/Xorg) | ✗ (CLI "future") | GPL-3 | no |
| nvidia-pstated | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ (daemon) | MIT | no (idle P-state daemon, not OC) |

**Adversarial verdicts on the tool question:**
- "LACT can do headless NVIDIA Ampere offset + power-limit OC" → **CONFIRMED**: `lactd` is a no-X
  systemd daemon; field-verified ~+1700 MT/s VRAM on a 3090 and +2000 on an A6000 via LACT.
- "LACT is mature enough to be *the* production backend" → **PARTIAL**: NVIDIA support is the
  AMD-first project's newer half (added v0.7.0, late 2024; per-pstate offsets v0.7.1). It inherits
  the **same documented NVML bug where setting a mem-clock offset breaks `SetMemoryLockedClocks`**
  (the exact offset+lock combo this loop uses — see the §3 dev-forum ref), needed a suspend/resume
  quirk for clean 3090 persistence, and its v0.9 VF-curve editor self-flags "zero guarantees."
- "LACT is the *only* headless option" → **REFUTED**: `nvidia_oc` and raw NVML are equally headless;
  LACT is only the most complete *named daemon/GUI* tool that also happens to run headless.

**The one decisive reason for roll-own**: every tool here bottoms out on the **identical NVML calls**,
so a dependency adds **zero NVIDIA capability** — while forcing the safety-critical
`step → set → verify → back-off` loop through an async daemon with **5 s auto-revert** semantics, when
the no-ECC GDDR6X gate needs a **synchronous, instant** revert it fully owns. Owning the NVML call
also lets the gate encode the **mem-offset unit footgun explicitly** (NVML clock-domain value = **½**
the GDDR transfer-rate number LACT/coolbits expose → getting it wrong over/under-steps the riskiest
knob 2×; §3 "Units gotcha"). Roll-own keeps the tool **pure-Python** (pynvml + driver already present
for vLLM), host-side root, in the **HALF-B host-side repo** (static binary + `systemd` unit) — **not** a
fork `scripts/` exe.
**Packaging = hybrid**: a thin backend interface (`set_core_offset` / `set_mem_offset` /
`set_power_limit` / `read_back`) with **NVML default** and an **optional LACT adapter**; recommend
LACT to users as the interactive GUI + the `/etc/lact/config.yaml` boot-persistence path, never as the
tune-loop driver.

---

## 4. Correctness gating (the hard part)

GDDR6X has no ECC; consumer cards expose **no SBE/DBE counters**, so `nvidia-smi -q` ECC fields are
blank. The published GPU-SDC literature is decisive: **~99% of GPU silent data corruptions are NOT
NaN/Inf** (~1% special values; ~51% nullify-to-zero; ~48% corrupt-but-plausible) and **<40% are
single-bit** — so NaN/Inf guards, perplexity-spike heuristics, and Xid/ECC counters catch only a
tiny slice. **Exact-match golden comparison is required, not a tolerance band.**

**The verdict that shaped this section: CLAIM 3 REFUTED.** A golden-output drift probe is
**NECESSARY and a fast, reliable REJECT gate, but NOT SUFFICIENT on its own.** It only exercises the
VRAM cells/address-wires and compute paths the probe's fixed prompts touch; it can miss flips in
unexercised cells and in compute units with low per-fault corruption rates. NVIDIA's own DCGM treats
memory-pattern testing as a **separate** diagnostic from compute stress.

**Recommended design (layered gate):**

1. **Golden capture (the oracle).** Run vLLM with **`VLLM_BATCH_INVARIANT=1`** (Ampere sm80+;
   tested archs include Qwen3 → matches the rig). This swaps in batch-invariant
   RMSNorm/matmul/attention kernels so two clean runs are **bitwise identical** — without it, temp=0
   still yields many distinct completions (a known demo: 1000 runs → 80 unique) and you'd be forced
   into a tolerance band that hides SDC. Capture a fixed (model, prompt-set, seed, temp=0) golden as
   **exact token-id sequences** (+ optionally bitwise logits). **Self-verify determinism first**:
   capture **twice** at stock clocks, require identical; if not, abort — the engine isn't
   deterministic on this build/backend and drift can't be trusted. (Docs do not *contractually*
   guarantee cross-run bitwise identity → re-verify on every vLLM/kernel/driver update.)
2. **Fast per-step REJECT gate (~30-90 s).** After each clock bump: (a) check NVML throttle-reason +
   any new Xid in dmesg → **instant reject**; (b) re-run the golden set, require **EXACT token-id
   match** → any mismatch = reject + back off one 15-MHz step; (c) free NaN/Inf scan (catches the ~1%
   special-value SDCs). This is enough to **REJECT** a bad clock fast — **never** enough to
   **PROMOTE** one.
3. **BW+verify pass (merged bandwidth + cell-memtest).** One bandwidth-SATURATING kernel that also
   verifies integrity: write a recompute-on-read pattern (LFSR/index-derived + moving-inversions, no
   golden buffer → memory-frugal on a tight 27B card) at peak BW, read-back-compare with an atomic
   mismatch-count + checksum, time write/read separately → `{read_GB_s, write_GB_s, mismatch_count,
   first_bad_addr, bitflip_hist}`. Off-the-shelf = **memtest_vulkan** (reports R/W GB/s *and*
   classifies single- vs multi-bit errors per pass); `cuda_memtest` has the best patterns but emits no
   GB/s. This **folds the old separate achieved-BW (BabelStream) and `cuda_memtest` steps into one**
   and **closes the zero-BW-penalty blind spot** (a no-ECC cell flip the EDR knee can't see surfaces as
   `mismatch_count>0`). Still add a short **gpu-burn** soak (compute+thermal); **do not** trust
   gpu-burn's GPU-to-GPU self-compare as oracle. It **tightens but does not close** the proxy gap
   (misses contention-only/transient SDC + flip-then-flip-back under live traffic) → golden-drift stays
   the irreducible compute-path oracle. (On ECC cards this step is moot — read NVML/DCGM SBE/DBE
   counters instead.)
4. **Thermal-steady-state + soak + continuous canary (to PROMOTE).** Re-run the golden check **after
   a sustained-load heat-soak**, not cold — a clock that passes a cold 30-90 s probe can SDC once
   GDDR6X junction heat-soaks (and that junction temp is **invisible to Linux `nvidia-smi`/NVML**).
   Require a **multi-hour soak under representative traffic** with periodic in-band golden canaries
   before promoting; in production keep a **low-rate golden canary** running and **auto-revert to
   stock on first mismatch**. Bake in a fixed safety margin (1-2 steps) below the last-passing clock.

**Does golden-drift alone suffice?** **No** (CLAIM 3). It must pair with the BW+verify pass (§4 step 3)
per accepted step and a soak before promotion.

**Tolerance**: exact token-id (or bitwise logit) match — **zero** tolerance under batch-invariance;
any nonzero delta = corruption.

**Latency**: per-step gate ~30-90 s (reject-only); promotion gate = hours. Run `VLLM_BATCH_INVARIANT`
as a **brief separate gate pass**, not in the hot serving path (it carries an unspecified perf hit).

**False-negative residual**: **irreducible without ECC.** Bounded only by probe coverage + canary
rate + safety margin. The honest posture is **conservative offsets + continuous in-band canaries in
production**, not a one-time validation.

**Per-OC-step ordered gate (host root, ~15 MHz-rounded steps)** — the concrete sequence each step runs,
fast sensitive proxies first, the real safety gate last:
1. **Set** offset → 2. **readback-verify** (`nvmlDeviceGetClockOffsets`) + NVML throttle-reason / new-Xid
(instant reject on either) → 3. **BW+verify** (one bandwidth-saturating write / read-back-compare
kernel, median of 3-5 runs, §7): `read_GB_s` no rise / regress vs prior step = **EDR knee**, stop
climbing; **`mismatch_count>0` = instant reject** (a no-ECC flip with zero BW penalty) → 4. **hotspot
temp** (GDDR6X junction via BAR0-MMIO `gputemps`, §6): **hard-abort + zero-offset if junction ≥ 95 °C**
→ 5. **correctness probe** (exact golden token-id under `VLLM_BATCH_INVARIANT=1`): any mismatch = reject
+ back off one step → 6. **decode tok/s** (`vllm_verify.py`, the real objective). **Promote only if** BW
rose AND golden matched AND temp OK; final clock = min(first golden fail, EDR knee) − 1-2 step margin.
Steps 3-4 catch most bad clocks in seconds but are **proxies** (the BW signal is not foolproof — §2);
step 5 is the real gate.

**Search & back-off — adaptive coarse-up / fine-down state machine.** All offsets in **clock-domain
MHz** (NOT MT/s), quantised to the KMD **15 MHz tick**: `COARSE = 7 ticks (105 MHz)`, `FINE = 2 ticks
(30 MHz)`, `MARGIN = 1 tick (15 MHz)`. **Objective: maximise measured decode tok/s subject to ZERO
*sustained* errors** — the optimum is the **EDR knee** (BW rollover), which sits *below* corruption
onset, so we target the knee, never the corruption edge.

```
INIT      capture golden oracle ×2 (self-determinism) + baseline read_GB_s/junction @ offset 0; require clean
CLIMB     step +COARSE; on read_GB_s slope-rollover *trend* drop to FINE BEFORE the next jump
          (never blind-jump the knee→corruption band); cap a near-knee step to the estimated gap.
          STOP on FIRST of { EDR knee (read_GB_s rollover) | mismatch_count>0 | golden FAIL | junction≥95°C }
          → record clk_hi; on any CORRUPT reset offset to last-good immediately.
FINE_DOWN step −FINE until a SUSTAINED-window PASS (not single-shot) — linear descent, NO bisection
          (bisection commits hard to a noisy verdict on an intermittent oracle).
MARGIN    clk_pass − layered_guard_band ; re-gate sustained.
HEAT_SOAK soak to steady-state junction (production p99 duty/ambient); re-run HOT golden+mismatch over a
          window; hot fail → −FINE & re-soak (floor 0, bounded iters → else fall to stock+alert); add a
          fixed thermal derate beyond measured steady-state. Promote only after a multi-hour clean run.
PROMOTE   apply final_clk; start a production-pattern golden CANARY with auto-revert→0 + PERMANENT ratchet-down.
SERVE     periodic re-search; any (Xid | throttle | golden | junction) → zero-offset + alert + re-search,
          capped below the last field-failed clock, exponential backoff.
```

`final_clk = min(knee, first_golden_fail, first_mismatch) − layered_guard_band`, **never** bounded by
the corruption edge. **Coincident-gap branch:** if `corruption_onset − knee ≤ 2·FINE`, abandon
knee-chasing and take `knee − a larger %-margin` (the upside past the knee is a sliver; the tail risk is
catastrophic).

**Margin is NOT safety.** `−1 tick` (15 MHz) is mere *hysteresis* — it sits inside every
noise/drift/intermittency envelope. The accepted clock instead carries a **layered guard band** =
thermal-guard (≥1 FINE, from the measured cold→hot shift) + coverage-guard (≥1 FINE) + the 1-tick
hysteresis. The **load-bearing safety mechanism is the continuous golden canary + auto-revert +
permanent ratchet-down** (SERVE) — not the static margin. Every accept-decision at **every** state
(CLIMB / FINE_DOWN / HEAT_SOAK), not just PROMOTE, must clear **zero-mismatch AND exact-golden across N
runs / M GB / T s**. Knee detection uses a **robust max** (trimmed/top-K median + slope-trend), not a
raw running-max, rejects out-of-variance probes, and quiesces/pins the GPU during each probe. Rotate
marching/checkerboard **and actual-quantised-weight** patterns through BW+verify (full-resident-VRAM
sweep) so cell coverage isn't pattern-blind.

---

## 5. Autotuner architecture

**Loop**: `measure → classify-bottleneck → prescribe → verify-delta`, the proven shape of
`jungledesh/profile` (Rust CLI: fresh-computed roofline ceiling, dual NVML + `/metrics` polling in
2 s **idle-filtered** windows — "that is where waste lives" — rules R1-R5, $/1M-tok cost model). The
fork's **novel half is silicon tuning**; Profile explicitly never touches clocks.

**Bottleneck classification — use capacity-based DCGM PROF fields, NOT time-based NVML utilization.**
`nvmlDeviceGetUtilizationRates` is "percent of *time* ≥1 kernel ran" — a single thread reads 100%
with all SMs idle; useless for compute-vs-bandwidth. Use:

| Signal | Field | Reads | Implies |
|---|---|---|---|
| Memory active | `DCGM_FI_PROF_DRAM_ACTIVE` (1005) | cycles the mem interface is busy | **bandwidth-bound (decode)** when high + TENSOR low |
| Tensor active | `TENSOR_ACTIVE` (1004) / `SM_ACTIVE` (1002) | cycles HMMA / SM warps active | **compute-bound (prefill)** when high |
| Interconnect | `PCIE_TX/RX`, `NVLINK_TX/RX` (1009-1012) | bytes | **comm-bound** when high + SM low → **DETECT & REPORT ONLY** (TP/PP/topology is per-deployment, out of fork scope) |

Cross-check with vLLM `/metrics`: TPOT → decode tok/s, TTFT → prefill tok/s, KV-cache usage,
prefix-hit, queue depth → roofline placement.

**Prescription table:**

| Bottleneck | Prescription | Lever class |
|---|---|---|
| DRAM_ACTIVE high (decode) | step **MEM offset** up (gate per §4); + existing fewer-bytes levers (W4A8, `kv_cache_dtype=fp8`) | **silicon (new)** + flags |
| TENSOR_ACTIVE high (prefill) | step **CORE/GPC offset** up + raise power cap so it isn't power-throttled | **silicon (new)** |
| KV usage ≥88% / OOM (R2/R4) | OOM-walkdown on `gpu_memory_utilization`; lower `--max-num-seqs`; `kv_cache_dtype=fp8` for long-ctx | flags |
| Under-batching / saturation (R1/R5) | raise/lower `--max-num-seqs` vs measured queue depth + GPU efficiency | flags |
| Comm-bound | **report only** (per-deployment, out of scope) | — |

**Reuse of prior art**: borrow `jungledesh/profile`'s roofline + idle-filtered window machinery;
vLLM `benchmarks/auto_tune.sh`'s OOM-walkdown for the capacity knob (start at 0.95, not 0.98 — issue
#21410); GuideLLM/`vllm-bench` as load generator; Optuna TPE as the flag hill-climber. The
**bottleneck-classifier + correctness-gate + silicon layer are net-new.**

**Integration with this fork's `benchmarks/` harness** (already present, reuse as the
perf-measure + probe substrate):
- `benchmarks/vllm_verify.py` → single-stream decode / batch-16 decode / prefill tok/s (the exact
  decode-bandwidth + prefill-compute deltas the loop needs). **Note**: it currently hardcodes
  `tensor_parallel_size=2` — the §7 rig is tp1, so the tool must parametrize TP (or use a tp1 path).
- `benchmarks/prof_decode_batchsweep.py` + `analyze_torch_prof.py` → kernel-bucket shares to
  cross-validate compute-vs-comm.
- `benchmarks/bench_marlin_gemm_imma.py` → ncu IMMA occupancy (≥65% = saturated decision rule).

**Packaging shape (NOT a fork `scripts/` directory).** This loop ships as **two external tools plus the
fork's existing harness — no executable autotuner under fork `scripts/`**:
- **HALF-A — vLLM-flag tuner** (no privilege: NVML-read + `/metrics` + roofline → prescribe
  `max-num-seqs` / `max-num-batched-tokens` / `gpu-memory-utilization` / `--enable-prefix-caching` /
  `--tensor-parallel-size` / `kv_cache_dtype=fp8`). This is **per-deployment config, out of fork scope**,
  and belongs in the vLLM tuning ecosystem. Recommended home: contribute the Ampere parts (hardware
  catalog + quant-byte-aware roofline + DCGM-PROF bottleneck classifier) **upstream to
  `jungledesh/profile`** (Apache-2.0; its roadmap slots v5 DCGM-classifier / v6 quant-sensitivity / v8
  safe auto-apply already match) — best-effort (single maintainer, no plugin API, Rust), gated on a
  confirming issue first. **Fallback:** a thin external Python wrapper on `vllm bench serve` + `/metrics`
  reusing the fork's `benchmarks/` (NOT an in-worker `general_plugins` plugin — wrong layer for an
  out-of-process measure-and-restart tuner; NOT a fork `scripts/` exe).
- **HALF-B — silicon-OC supervisor** (`nvmlDeviceSetClockOffsets` / `SetPowerManagementLimit` /
  `SetGpuLockedClocks` behind the §4 gate; readback-verify offsets via `nvmlDeviceGetClockOffsets`; BAR0
  `gputemps` junction abort; the §9 two-cadence supervisor): a **separate host-side repo** — static
  binary + `systemd` unit ordered `Before=` the vLLM container, host-root, opt-in, monitor-only default,
  independently versioned. **Never in any vLLM image.** Internally Python (pynvml) is fine; ship as a
  self-contained binary so it carries no fork dependency.
- **Persisted per-GPU profile** keyed by **GPU UUID** (silicon lottery — never a fork default):
  `~/.config/ampere-autotune/<gpu-uuid>.json` = `{arch, driver, stock_mem_mhz, max_stable_mem_offset,
  max_stable_gpc_offset, power_limit, validated_temp_c, decode_gain, prefill_gain, vllm_flags}`;
  re-validate on temp delta or driver change. Owned by **HALF-B (host root)**.
- The **fork itself keeps only** the `benchmarks/` harness (perf-delta + token-id golden under
  `VLLM_BATCH_INVARIANT=1`; **parametrize the `tensor_parallel_size=2` hardcode in `vllm_verify.py`** so
  external tuners can drive tp1 rigs) + this design doc + a README/ROADMAP pointer to the two tools.
- **Search staging** (inside HALF-B): clocks first via **bisection + correctness gate**, then HALF-A
  flags via TPE — separately, not a joint multi-objective space.

### 5.1 Runtime-tuning boundary (what can change WHILE serving, no restart)

vLLM's eight perf-critical engine flags (`max-num-seqs`, `max-num-batched-tokens`,
`gpu-memory-utilization`, TP/PP, `kv-cache-dtype`, `block-size`, `max-model-len`,
`enable-prefix-caching`) are bound at `EngineArgs.create_engine_config()` and have **no runtime
hot-reload** — the KV-block count is frozen by one-time startup `gpu-memory-utilization` profiling and
the CUDA graphs are captured against those values; TP/PP fix the worker process group. So **HALF-A
(Profile-style) is necessarily offline-iterative** (measure → recommend → **restart** → re-measure);
"online" flag changes mean **canary/blue-green relaunch, not live mutation**.

- **Tunable WHILE serving (no restart):** silicon clocks/power (HALF-B, NVML — global *driver* state,
  not engine state); admission/concurrency throttling **below** the startup `max-num-seqs` ceiling;
  KV-aware routing / autoscaling across replicas; and narrow vLLM admin ops that **do not touch the
  perf-flag set** (dynamic LoRA add/unload, `/sleep`+`/wake_up`, `/update_weights`, and the **MoE-only**
  `/scale_elastic_ep` live EP/DP rescaling, shipped vLLM 0.17.0, single-node/TP=1/Ray-coupled).
- **NOT without restart:** raising `max-num-seqs` / `max-num-batched-tokens` / `gpu-memory-utilization`,
  TP/PP, `kv-cache-dtype`, `block-size`, `max-model-len`, `enable-prefix-caching`.
- **Already dynamic (not a knob we set):** vLLM's continuous-batching scheduler self-adapts occupancy
  every iteration — admit/retire/preempt up to the startup ceilings. HALF-A only sets the **bound**; you
  can throttle admission below it live, never raise it without a restart.

**The only continuously in-place *perf* knob is silicon (HALF-B).** A true "online tuner" for us is thus
a 3-tier system: (1) **silicon** = HALF-B continuous supervisor (§9, derate-only + canary + auto-revert);
(2) **control-plane** shim (admission/concurrency + KV-aware routing + drained canary relaunch to apply
*new* HALF-A flags with zero dropped traffic) — this is **orchestration, lean on existing stacks**
(production-stack / AIBrix / llm-d / Dynamo), not flag hot-reload; (3) **free in-engine admin ops** above
(LoRA, sleep/wake, watch-not-build Elastic-EP). Keep HALF-A itself offline; do **not** advertise
in-process hot-reload of engine flags.

### 5.2 In-engine adaptive runtime optimization (fork patch)

§5.1 is the **stock-vLLM** boundary (external, restart-bound, coarse ~2s windows). **As a fork we can do
better IN-ENGINE** — close the loop inside the scheduler step with ground-truth internal state
(per-request batch composition, KV free-list, per-request spec accept-length — the accept signal
`num_accepted = len(generated)-1` is already computed every step in `update_from_output`,
`scheduler.py` ~L1418) at a **5-20 ms decode-step cadence**, ~2-3 orders finer than a `/metrics` scraper
— but only for **one class of knob**:
- **Class A (mutable in-engine, re-read every `schedule()` step):** spec-decode **K** (down free = fewer
  draft forwards; up only to the startup-**captured cudagraph ceiling**), chunked-prefill token budget
  `max_num_scheduled_tokens` (down free, up to startup max), `long_prefill_token_threshold` (zero
  plumbing), `max_num_seqs` (down trivial, up bounded by the **frozen KV blocks**), policy / pause /
  prefix-cache / watermark.
- **Class B (genuinely immutable live, even in a fork):** KV-pool size / `gpu_memory_utilization`,
  cudagraph capture sizes, TP/PP process groups, `kv_cache_dtype` / `block_size` layout — each needs a
  stop-the-world re-profile + re-alloc + re-capture (vLLM routes capacity change through the heavyweight
  EEP `RECONFIGURE` path, `core.py:843`, never a per-step hook). `max_model_len` is immutable only
  **upward** (a live `update_max_model_len` RPC can shrink it). **For Class B the fork move is a smarter
  STARTUP profile, not a runtime knob.**

**Honest bar:** adaptive beats the optimal *static* flag **only under workload variability**
(bursty / mixed prefill-decode / variable per-step difficulty); on bandwidth-bound **steady** decode,
static optimum ≈ adaptive. So a lever must both **attack the bandwidth wall** and **gain from
variability**. Ranked for this fork (W4A8 + MTP + hybrid/MoE):
1. **Adaptive spec-decode K — goodput-driven, per-step.** The only lever that modulates **MTP**, our
   measured #1 bandwidth-wall breaker (static K=2 = **+25%, 126→158 tok/s** W4A8-9B tp2). Drive on
   **GOODPUT** (accepted tok/s net of verify), **NOT accept-rate** (PR #26504's data: accept-rate flat
   0.86-0.91 while optimal K flips → miscalibrated), with a hard **K=0 floor so it provably never
   regresses below static**. Ship **uniform-per-step K** only (per-request varying K falls out of the
   captured `uniform_decode` cudagraph / pads to max-K — open upstream gap, #28015). **Honest go/no-go:**
   static K=2 already captures the 25%; the *adaptive premium over best-static-K + the existing
   auto-disable* is an **UNMEASURED ~5-15%**, likely small on steady traffic, real only under
   variability — validate on the W4A8-9B tp2 MTP rig (`--shm-size=8g`) under **bursty** load (steady
   single-stream falsely shows static==adaptive).
2. **Batch-saturation K→0 gate** (RFC #41821; CPU-side counts, zero GPU overhead) — a superset of the
   static disable-by-batch-size≈32 cliff; incremental win = the smooth edge, not a step-change.
3. **Decode-aware `long_prefill_token_threshold`** — protects decode tok/s during prefill bursts;
   cheapest patch, ~0 on pure streams.
4. **Down-safe `max_num_seqs` / watermark guards** — stability/SLO under KV pressure (e.g. 27B-W4A8
   mamba-cache regime), not steady-throughput drivers.

**Scope win:** an in-engine adaptive controller is a **shippable fork patch** that generalizes across the
Ampere line (sm_80+sm_86, portable Python) and serves our MTP/hybrid/MoE target — **strictly more
on-scope than the shelved external autotuner** (which was per-deployment + privileged). Ship the
**thinnest form**: a goodput-driven proposer via the existing `custom_class` proposer seam (loads a class
by path — **no core patch**), feeding it the scheduler's per-step accept counters; respect the real
guards (`mamba_cache_mode='all'`+spec gate, async-scheduling/PP gate). It overlaps active upstream (PR
#26504 DynamicProposer, RFC #41821) → keep it thin so it can later defer to upstream (minimal rebase).

**UPDATE (decision):** the in-engine adaptive **RUNTIME tuner is NOT pursued**. Rationale: the adaptive
spec-K premium is unmeasured (~5-15%), materializes only under bursty load, overlaps active upstream
(PR #26504 / RFC #41821), and a live control loop adds a correctness-gating cost for a speculative gain.
**PIVOT:** the fork advantage is realized instead as a passive **fine-grained measurement PROBE** for
**OFFLINE** tuning + the `benchmarks/` diagnostic harness (§5.3) — not a controller. (The probe does
**not** "prove" the runtime tuner unnecessary — workload shape on a test rig is a deployment property;
the tuner is shelved on its own merits.)

### 5.3 Fine-grained in-engine probe (the fork's actual lever)

The runtime tuner is dropped; the fork's real edge is **observability no external tool can match** —
**but be honest about what stock already gives.** Stock vLLM V1 already ships a full Prometheus set
(TTFT/ITL/e2e histograms, `kv_cache_usage`, preemptions, prefix-cache, and **per-position spec counters**
`spec_decode_num_accepted_tokens_per_pos`), OTLP request tracing (`--collect-detailed-traces`, one span
per *finished* request), the torch profiler, and `SpecDecodingLogging.log()` which already prints mean
accept-length + the full per-position vector + goodput each interval. So **per-position acceptance is NOT
net-new** — only its *cadence and joinability* are. Stock collapses everything into fixed Histogram
buckets / most-recent Gauges / monotonic Counters over a **10 s** in-engine window
(`VLLM_LOG_STATS_INTERVAL` default — 2 s was only an assumed external scrape; the gap is *larger* than
first stated).

**Net-new the probe adds:** (a) a **per-step** trace at decode cadence (5-20 ms), ~2-3 orders finer than
the 10 s window; (b) per-request, **un-bucketed** detail; (c) the **same-step join** of accept-length ↔
batch size (`num_running_reqs`) so the **accept-len-vs-batch crossover** (= the `disable-by-batch-size`
threshold) comes from **ONE mixed-load run**, not N sweeps; (d) per-phase prefill/decode timing **without**
the heavy torch profiler.

**Seam:** a passive `StatLoggerBase` subclass via the `vllm.stat_logger_plugins` entry point (or the
in-process `stat_loggers=` arg) — **zero core patch**, env-gated (`VLLM_STEP_PROBE=1`). Taps the per-step
`SchedulerStats` + `IterationStats` + `SpecDecodingStats` vLLM already materializes on CPU each step
(`v1/metrics/loggers.py`, `stats.py`, `spec_decode/metrics.py`).

**Cudagraph-safety (load-bearing):** `record()` runs in the frontend/engine-core process **outside** the
captured forward graph; every field is already a CPU int/float/list (the sampler's `.tolist()` is in the
model runner, not here). **Hard rules:** NO `.item()`/`.tolist()`/`.cpu()` anywhere reachable from the
forward; bounded ring buffer; JSONL flush off the hot path; **validate under FULL cudagraph** (the int8qk
`.tolist()` bug was masked by `enforce_eager`); target **<1% overhead** on the 126-158 tok/s W4A8-9B path.

**Taps, ranked by decision-value:**
- **A (ship first) — spec/MTP:** per-step per-position accept + goodput **joined to co-step batch size**.
  Yields the `disable-by-batch-size` threshold (GOTCHA3≈32) + static-K confirmation from one mixed run,
  and the `mtp.fc`-quant **0%-accept bug shows as a flat-zero vector instantly**.
- **B — decode roofline:** `perf_stats` bytes/flops per step → GB/s and TF/s. **Needs a small ENGINE FIX
  for hybrid-GDN** — add a Mamba/GDN `ComponentMetrics` and correct `num_kv_layers` to full-attn-only;
  the stock analytic model has **zero mamba terms and overcounts KV ~3× on the 3:1 hybrid 27B**. Ship
  decode-only + label "estimate" until the GDN fix lands.
- **C — KV/mamba pressure:** `kv_cache_usage` + preemptions per step **plus a NEW mamba-state-cache
  occupancy counter** (`kv_cache_usage` tracks only the 8 full-attn layers; the **mamba state cache is
  what actually OOMs**). Pins `max-num-seqs` at the per-step usage cliff instead of bisecting via
  OOM-or-not serving runs.
- **D (once, go/no-go) — burstiness:** `num_waiting` + coarse hit-rate, captured once.

**Drop as noise:** continuous prefix-hit-rate streaming, fine queue-depth time series, full per-request
ITL streaming, and any **new GPU-side** counter (reintroduces a sync). Per-request per-step token split
needs a thin EngineCore tap (`SchedulerOutput.num_scheduled_tokens`), not the `record()` seam — defer.

**Consumers / harness:** ships as a **pluggable logger** (entry-point + in-process arg), NOT a core
patch — *except* the two `perf.py` correctness fixes (GDN `ComponentMetrics`, mamba-state counter) that
gate TAP B/C trust and are small genuine fork patches. It **improves, not replaces** the harness: makes
`prof_decode_batchsweep.py`'s synthetic fixed-batch sweep optional (per-step join gives accept-len-vs-batch
+ bytes-vs-batch from one mixed run); acts as a cheap serving-realistic tier **before**
`analyze_torch_prof.py`/`bench_marlin_gemm_imma.py` (only pay kineto/ncu when the cheap tier says "go").
Add `benchmarks/analyze_step_probe.py` to parse the JSONL into the same is-this-lever-worth-it verdicts.
**Strictly offline diagnostics — never a control input.**

**Honest caveat:** for the #1 use (spec-K, disable-by-batch threshold, 0%-accept bug) stock
`SpecDecodingLogging.log()` **already suffices** for a one-time offline characterization — the fork's
measured K=2 (9B) / K=3 (27B) numbers came from exactly those stock logs. So for pure one-time
characterization, **just run the existing bench** — a probe is not required. The probe earns its keep
**only** where finer cadence/joins genuinely help: accept-len-vs-batch and KV-pressure-vs-step curves
from single mixed runs, and the **hybrid mamba-state pressure stock literally cannot see**. Maintenance
cost: `SchedulerStats`/`IterationStats` are explicitly **non-stable** vLLM interfaces (rebase risk).

---

## 6. Risk / 24/7 stability / shippability

**Tiered risk ladder** (this is how the feature ships):

| Tier | Lever | Default | Scope | Why |
|---|---|---|---|---|
| **0** | **Power-limit** (`-pl` cap) | **SHIP ON** | **all Ampere incl. A100** | stays on factory V/F curve → **cannot** silently corrupt; *lowers* mem-junction temp → improves 24/7 stability/aging; ~95% perf at ~270-280 W. Reverts on reboot. |
| **1** | **Locked-clocks** (`-lgc`/`-lmc` within stock) | **SHIP** (safe) | all Ampere | removes boost jitter → stable TPOT + reproducible benches; no corruption risk (within curve) |
| **2** | **Core OFFSET** (GPC) | **OPT-IN, gated** | **consumer sm_86 only** | core errors tend to crash/hang (less *silent* than mem); still needs root/host + correctness gate; helps prefill only (compute-tapped → small win) |
| **3** | **Mem OFFSET** (VRAM) | **OPT-IN, default OFF, HIGH-CAUTION** | **GeForce/3090 only** (CLAIM 4) | the +6-8% decode hook — but no-ECC silent corruption + EDR auto-downclock + thermal heat-soak. MUST gate on **measured decode tok/s** + correctness probe + mem-junction ceiling + soak. |

**Host/root/container reality**: clocks/power are **global physical-GPU/driver-session state**, and the
**NVIDIA Container Toolkit adds no OC path** (no `/dev/mem`, no SET capability — it only injects device
nodes + driver libs). The **unprivileged serving container cannot** set them (root alone fails; needs
`--cap-add=SYS_ADMIN`/`--privileged`); a **privileged root container *can*, but the effect is global
and survives that container's restart with no auto-revert**, forfeiting the reboot/driver-reload
dead-man's-switch + watchdog isolation. So the SET path **never lives in the serving container** — it
runs as a **host systemd unit ordered `Before=` the vLLM container** (the default, and the only place
the BAR0 `gputemps` read can live), or a **dedicated privileged sidecar / k8s privileged DaemonSet**
where a host daemon is disallowed. CUDA-graph capture is
**independent of clock state** (a graph records the kernel-launch DAG, not frequency) — so setting
locked-clocks once at boot before vLLM starts is clean, and runtime clock changes don't invalidate
captured graphs; only *measure* during steady REPLAY, not capture.

**Production guardrails**: auto-revert watchdog (NVML throttle reasons + Xid + mem-temp + probe
mismatch → instant `-rgc -rmc`/zero-offset → stock + alert); reboot/driver-reload free
dead-man's-switch; continuous low-rate golden canary; conservative safety margin; **periodic
re-validation** (a cold-validated profile drifts unstable over a 24/7 window as GDDR6X heat-soaks —
CLAIM 6). The continuous-serve loop is **fully specified in §9** (two-cadence supervisor +
observability): **monitor-only by default**, FREE-knobs + offset-derate-only on live traffic, with all
offset *UP* gated to drained maintenance windows.

**Thermal reality (CLAIM 6 REFUTED — sustained 3090 mem-OC is NOT stable by default)**: GDDR6X
mem-junction hits ~104-110 °C even at **stock** clocks under sustained load; bandwidth reduction
begins ~92-95 °C; OC adds heat and pushes deeper into the throttle band, **cancelling the OC gain**;
a 3090 at 95 °C VRAM went unstable after ~1 month. The mem-junction sensor is **not** exposed by
`nvidia-smi`/NVML or even **DCGM** (consumer field 140 `DCGM_FI_DEV_MEMORY_TEMP` returns 0/blank on a
GeForce 3090) — **but it IS readable headless** via direct **BAR0-MMIO** register reads (the
`gddr6`/`gputemps` tool family, GA102-verified on the 3090; needs root + `iomem=relaxed` boot param +
Secure Boot off; offsets are reverse-engineered → sanity-check vs NVML core at idle). So a **hard
thermal-abort gate IS feasible** (out-of-band ~1 Hz poll → **revert mem-offset at junction ≥ 95 °C**,
warn ≥ 90 °C) — this is the correction to CLAIM 6's "invisible to Linux". → 24/7 mem-OC needs (a)
thermal-pad replacement (~-20-25 °C) and/or an
aggressive power cap to hold junction ≤~90-95 °C, (b) out-of-band mem-temp monitoring with a hard
abort, (c) steady-state-measured gating, (d) periodic re-validation. **Document this; do not assume
thermal stability.**

**Scope-fit / shippability (NOT shipped in the fork image or `scripts/`)**: the project mandate is
**general Ampere, shippable artifacts, NOT per-deployment config** — so the autotuner is **not a fork
deliverable**. **HALF-B** (silicon-OC) is excluded on **privilege** (host root / `--privileged` +
`/dev/mem` + `SYS_ADMIN`; cannot run in the shipped unprivileged serving container) *and* on per-card
silicon-lottery output → a **separate opt-in host-side tool**. **HALF-A** (flag tuner) is excluded on
**scope** (per-deployment flag-tuning) → it joins the **vLLM tuning ecosystem** (`jungledesh/profile` /
`vllm benchmarks/auto_tune`) as an upstream/ecosystem contribution, thin external wrapper as fallback.
**The fork ships only the diagnostic `benchmarks/` harness + this research doc + a pointer to the two
external tools.** The **per-card profile JSON is an output, never a shipped default** (no default offset,
no copying another card's profile — silicon lottery). Tiers 0/1 (power-limit / locked-clocks) generalize
across the whole line, but as global GPU-driver state they are exercised **by the external HALF-B
supervisor**, not by anything in the fork image. **The datacenter Ampere (A100/A30/A40/A10/A2) get
Tier-0/1 ONLY** — all are clock-LOCKED (offset API Not-Supported; A100 NVIDIA-staff-confirmed), so they
drop both offset tiers *and the entire no-ECC safety stack* (golden-knee / BAR0-MMIO junction abort don't
apply): they have **real ECC** (A100/A30 HBM2e/HBM2, A40/A10/A2 GDDR6) → Tier-0 power-limit + Tier-1
lock-to-max as a pure **anti-throttle** play, with **NVML/DCGM SBE/DBE counters** as ground-truth health.
Tier-1 pinning yields a real gain only when the card was throttling; ~0 on a thermally/power-healthy one.

**Turing / Pascal generalization (out of declared scope).** The autotuner is hardware-generic GeForce
machinery (NVML clock-offsets Maxwell-onward + coolbits/nvidia-settings), so its power-limit /
locked-clocks / core-offset / mem-offset levers **RUN** on consumer Turing (2080 Ti, TU102 sm_75) and
Pascal "for free." But this is **OUT OF the fork's Ampere sm_80+sm_86 scope**, and not shipped/validated
there: (1) the W4A8/Marlin value-add does not transfer — **Marlin GEMM produces GARBAGE on sm_75** (vLLM
#33461, closed not-planned); (2) the EDR bandwidth-knee gate has **no signal on GDDR6** (EDR is
GDDR6X-exclusive; GDDR6 hard-crashes past the limit) → must re-gate on golden-token drift + Xid/crash;
(3) `gputemps`/BAR0-MMIO has **no TU102 support** and the 2080 Ti mem-temp sensor is often absent → no
thermal hard-abort. Corollary: **all GDDR6 Ampere parts (A6000/A5000/A40/A10/A2) also lack the EDR knee**
and use the ECC/golden-token gate instead — the EDR-knee gate is **GDDR6X-only** (3090/3080-class). Treat
Turing/Pascal as "**runs-but-unvalidated, opt-in, gates differ**"; do **not** list the 2080 Ti in the
shipped matrix.

**Hardware/warranty realism**: power-limit and locked-clocks are benign (never void warranty / damage).
Offset OC can void warranty and stress GDDR6X under sustained server load → opt-in with a clear
warning.

---

## 7. Phased experiment plan

Start with the **SAFE, highest-value, decision-gating** step. This is the experiment that converts
the projected +7-12% into a measured number and decides whether Tier-3 is worth shipping at all.

### Pre-conditions / risks (USER APPROVAL REQUIRED before Phase 1)
- **Needs host root** on the sandbox HOST (`trevor@192.168.100.1`), not the workspace container —
  clocks are global host state. The Coder workspace container **cannot** do this.
- **Confirm host driver is R555.85+ (ideally R570+)** so `nvmlDeviceSetClockOffsets` exists; else
  fall back to deprecated VF-offset or abort.
- **Affects other sandbox work**: changing the GPU clock is global; any concurrent job on that 3090
  sees the OC (and risks corruption). Run when the rig is otherwise idle.
- **Start small + back off**: step in **+15 MHz-rounded** increments (e.g. +100/+200 MT/s VRAM),
  correctness-gate **every** step, revert immediately on any mismatch/Xid/throttle. GDDR6X no-ECC →
  an over-aggressive step can corrupt; read the **GDDR6X junction temp directly** via BAR0-MMIO
  (`gputemps`, §6) for a hard abort at ≥ 95 °C, and soak before trusting any value.
- **One-time host prep for the sensors**: build `gputemps` (BAR0-MMIO mem-temp) + the **BW+verify
  kernel** (or memtest_vulkan); the mem-temp read needs **root + `iomem=relaxed` kernel boot param + Secure Boot off**
  (BAR0 MMIO via `/dev/mem`). The MMIO offsets are GA102 reverse-engineered (driver-fragile) → sanity-
  check `gputemps` core temp against NVML at idle before trusting the junction reading.
- **Single card, tp1** (the standard rig) — avoids the no-NVLink TP / 64 MB-shm confounds entirely.

### Phase 0 — plumbing (no OC)
Fix the rig: parametrize `vllm_verify.py` for **tp1** (it hardcodes tp=2). Stand up
`VLLM_BATCH_INVARIANT=1`, **self-verify determinism** (capture golden twice → require identical;
abort if not). Establish the stock-clock baseline: **~87 decode tok/s, ~5753 prefill tok/s**.
Capture the golden token-id set (mix: short + long-ctx + a high-coverage weight-sweeping prompt set).
Build the two host instruments — **`gputemps`** (BAR0-MMIO GDDR6X junction reader; `gcc gputemps.c -o
gputemps -O3 -lnvidia-ml -lpci`) and the **BW+verify kernel** (or off-the-shelf **memtest_vulkan**) —
and do the one-time host prep (`iomem=relaxed` boot param, Secure Boot off, root). Capture the
**stock-clock bandwidth + thermal baseline**: median `read_GB_s` **and its run-to-run stdev** (this
calibrates the EDR-knee threshold), confirm **`mismatch_count==0` at stock**, + GDDR6X junction temp
under sustained load. No OC in Phase 0 → low risk.

### Phase 1 — correctness-gated MEM-OC sweep (the decision experiment)
On the sandbox 3090, Qwen3.5-9B-W4A8, tp1:
1. **Set** the next mem offset per the §4 adaptive schedule (host root, NVML, **clock-domain MHz — not
   MT/s**, 15 MHz ticks): **CLIMB +COARSE (105 MHz / 7 ticks)** until the `read_GB_s` slope rolls over,
   then **descend −FINE (30 MHz / 2 ticks)**; **readback-verify** (`nvmlDeviceGetClockOffsets`) + check
   NVML throttle-reason / new Xid.
2. **BW+verify** (the merged kernel, median of 3-5): `read_GB_s` fails to rise >~1.5-2% vs the prior
   step (or regresses past ~3% ≈ 2× the Phase-0 noise floor) → **EDR knee**, stop; **`mismatch_count>0`
   → instant reject** (a no-ECC cell flip with zero BW penalty).
3. **Hotspot temp** (`gputemps`, BAR0-MMIO): **hard-abort + zero-offset if junction ≥ 95 °C**.
4. **Correctness probe** (exact token-id vs golden under `VLLM_BATCH_INVARIANT=1`, over a SUSTAINED
   window — not single-shot): any mismatch = reject + descend one FINE step.
5. If all pass: **measure decode tok/s** (`vllm_verify.py` single-stream + batch-16). The accepted clock
   = min(EDR knee, first golden fail, first mismatch) − the **§4 layered guard band** (thermal +
   coverage + 1-tick hysteresis) — **not** the corruption edge, and **not** just "1 tick".
6. **Quantify the real decode tok/s gain** at the max stable clock vs the ~87 tok/s baseline.
7. **Heat-soak re-check** (CLAIM 6): re-run the golden probe **and** read the real `gputemps` junction
   after a sustained-load warmup — a clock that passed cold can SDC / throttle once GDDR6X heat-soaks.

**Decision gate**: if measured gain is **>~5% and clears the §4 gate at steady state**, Tier-3 is
worth shipping as a guarded opt-in. If it flattens at the EDR knee first or drifts unstable on soak,
**Tier-3 is a NO-GO** and the tool ships Tiers 0/1 + the autotuner only.

### Phase 2 — CORE-OC + power-limit for prefill
Step GPC offset +100…+180 MHz **with** a raised power cap (so the higher clock isn't
power-throttled); correctness-gate; measure **prefill tok/s** vs ~5753. Expect a **modest** win
(prefill is compute-tapped).

### Phase 3 — power-limit perf/watt sweep (safe, ships regardless)
Sweep `-pl` down from stock; measure decode + prefill tok/s vs watts; find the **knee where tok/s
flattens** (expect ~270-300 W on a 3090, ~95% perf). This is the Tier-0 default and is the right
always-on serving lever independent of whether mem-OC ships.

---

## 8. Open questions / what would change the verdict

- **The decision number**: real measured decode tok/s delta from a correctness-passing mem-OC on the
  Qwen3.5-9B-W4A8 3090 rig. The whole Tier-3 case rests on this being **>~5% at steady state** and
  beating the EDR knee. *No published 3090 decode-vs-mem-OC measurement exists* → Phase 1 decides it.
  **A flat / sub-knee / soak-unstable result flips Tier-3 to NO-GO.**
- **Knee vs corruption gap**: does EDR give a comfortable safe band *below* the first golden-drift
  failure, or do they nearly coincide (making the gate the only defense)?
- **Driver version on the fork image/host**: R555+/R570+ for `nvmlDeviceSetClockOffsets`, or fall
  back to deprecated VF-offset? Open-kernel-module vs proprietary (open module historically lacked
  the clock path; snaps to 15 MHz).
- **`VLLM_BATCH_INVARIANT=1` cost & perturbation**: per-pass overhead on the rig, and does the
  deterministic kernel set *change the optimal clock* (i.e. does it perturb the very
  decode-bandwidth behavior being tuned)?
- **Coverage gap**: can a high-coverage weight-sweeping golden prompt set provably exercise most of a
  W4A8 model's weights/experts in bounded time, to shrink the irreducible false-negative residual?
- **Mem-junction temp on headless Linux** — **RESOLVED (§6)**: readable headless via **BAR0-MMIO**
  (`gddr6`/`gputemps`, GA102/3090-verified) — **not** NVML/DCGM (consumer field 140 blank). Thermal gate
  uses it directly (95 °C hard-abort). Residual: MMIO offsets are driver-fragile → sanity-check vs NVML
  core, and re-verify the offset after any driver bump.
- **W4A8 SDC susceptibility**: does int4-dequant + int8-IMMA change SDC surface vs bf16 (fewer VRAM
  bytes = fewer cell-exposure events, but tensor-core-heavy = more compute-SDC) — affects whether
  mem-OC or core-OC is the riskier knob.
- **Does Tier-1 already capture most of it?**: locking mem at stock-MAX (safe) may capture most of a
  jittering 3090's decode bandwidth headroom, making Tier-3 largely unnecessary.
- **Ship-as-wrapper vs native** — **RESOLVED (§3.1)**: roll-own `nvidia-ml-py` for the safety loop
  (LACT/`nvidia_oc` add zero capability over the same NVML calls, and a daemon's 5 s auto-revert
  fights the synchronous gate), with an optional LACT adapter + LACT documented as the manual /
  boot-persistence helper.

---

## 9. Continuous runtime supervisor + observability

The offline characterizer (§7) **finds** the safe operating point; the online supervisor **holds** it
as ambient / workload / aging drift, and gives margin back the instant the silicon is unhappy — **never
the reverse on live traffic**. A HOST-root daemon (or privileged sidecar, §3.1/§6), never the serving
container; the container exposes only vLLM `/metrics` and the in-band golden/tok-s probe. Per §5.1,
**silicon is the only continuously in-place *perf* knob** — engine flags are restart-bound, so this
supervisor tunes clocks/power, not vLLM flags.

### 9.1 Two-cadence controller
- **Fast watchdog** — 250 ms (4 Hz), single-threaded, no alloc/network, `SCHED_RR`. **SUBTRACT-ONLY**:
  it can revert/derate, **never raise** a knob (its dumbness *is* the safety property). Signals: NVML
  clock-event-reason bitmask (HW_SLOWDOWN / SW+HW_THERMAL / HW_POWER_BRAKE), NVML **Xid** event-set
  (corruption class 48/63/64/79/92-95 → hard trip; Xid 79 fell-off-bus → permanent quarantine), GDDR6X
  junction via BAR0 `gputemps`, ECC volatile DBE/SBE (ECC SKUs only), latched golden-probe verdict.
  **Hard trip** (Xid-corruption / golden mismatch / DBE / junction ≥ 95 °C) → offsets to 0 +
  ratchet-down + alert, zero-debounce. **Soft trip** (thermal bit / junction WARN / SBE spike) → derate
  one FINE (30 MHz), debounced. **The watchdog ACTS before it logs** — revert is one NVML write (µs);
  logging never gates it. Detect→revert < ~300 ms.
- **Slow controller** — 5 s sample → 30-60 s rate-limited decision tick. Reads vLLM `/metrics`
  (TPOT→decode tok/s objective, TTFT→prefill, KV%, queue, prefix-hit) + the shared NVML/junction
  snapshot, all EWMA. Holds the point with **FREE knobs only**; never raises an offset on live traffic.
  One controller computes `max_safe_clock(junction)` **then** hill-climbs under that cap (thermal
  strictly dominant → no two-loop oscillation). Objective deltas are load-conditioned (credited only
  when running-requests / KV% are stable) and probe-excluded.

### 9.2 Asymmetric knob policy

| Knob | Lane | Live-traffic rule |
|---|---|---|
| Power-limit (T0) | **FREE both ways** | hill-climb within [min, factory-default]; cannot corrupt |
| Locked/app clocks (T1) | **FREE both ways** | select within published [VFmin, VFmax]; factory-blessed, not OC. **Thermal restore re-raises HERE, not offsets** |
| Core offset (T2) | **DERATE-only online** | ratchet DOWN anytime; UP forbidden on live traffic |
| Mem offset (T3) | **DERATE-only online** | ratchet DOWN anytime; UP forbidden on live traffic |
| Any offset **UP** | **FULL-GATE-only** | drain (admission-control-confirmed 503/hold) → full offline gate → reopen. Single-card / TP-group always drains; only replica-parallel rolls |

**Invariant: on live traffic, offsets are monotone-non-increasing within a serving epoch.** There is
**no** live-traffic upward-offset path — no "below-ceiling re-gated restore", no live RESEARCH state, no
inferred-trough self-promote.

### 9.3 Thermal-adaptive derate + re-promotion
Junction is the live proxy for the temperature-dependent stability boundary. Step-function derate with a
deadband + **asymmetric hysteresis** (release gap ≫ trip gap): WARN_LO hold / WARN_HI −1 FINE / approach
−2 FINE / ABORT (≥ 95 °C) → 0. Debounce short going down, long (minutes) going up. **Thermal RESTORE
acts on the FREE lane** (re-raise the locked-clock/power-limit lowered for heat) — it **never** re-raises
an offset on live traffic. Offset re-promotion is maintenance-window / drained-gate only; a HARD-TRIP
arms exponential backoff (1 h, doubling) + a ratchet-ceiling drop, capped → quarantine.

### 9.4 Modes (default leans conservative)
- **offline-characterize** — drained; the **only** mode that discovers a new *higher* offset.
- **online monitor-only [DEFAULT]** — watchdog + perf reporting; FREE knobs may be held but no offset
  motion beyond derate. Reverts/derates active.
- **online safe-adapt** — adds FREE-knob hill-climb (perf/watt) + offset DOWN-only derate; degrades
  cleanly to FREE-only on A100/A40/A10.
- **aggressive-gated** — self-schedules drains for gated promotion; **NOT shipped in v1** (trough-detector
  TOCTOU; drain must be admission-control-enforced).

Per the adversarial review, the **production default is monitor-only**, all offset tuning gated to
drained maintenance windows; safe-adapt is opt-in.

### 9.5 Production-safety verdict
On no-ECC GDDR6X a marginal OC flips **intermittent, data/address/temperature-dependent** bits with **no
Xid/ECC/throttle signal**. The golden canary is a **coverage/health probe, NOT a corruption
interceptor**: at ~0.2 Hz beside 150+ tok/s served, ~750+ real tokens go un-inspected per probe gap, and
a flip in a tile the fixed golden prompt never touches reads `canary=OK` forever. The "180 ms revert" is
**detect→revert, not corruption-onset→revert** (which can be ~100 s ≈ ~15,000 silent-wrong tokens).
**Safety therefore lives in the cold-gated guard band (§4/§7), not in any runtime catch** — which is
exactly why the default is monitor-only and every offset-UP is drained-gated. The only honest inspector
of *real* outputs is an optional, throughput-costed, opt-in **redundant-recompute** (recompute a fraction
of decode tokens at stock / on a second replica, compare token-ids).

### 9.6 Observability
Two independent paths so a tick flood never delays an event; **all string formatting off the watchdog
thread**.

**Per-tick line** (5 s aggregate of fast polls; absent signals render `-`): `ts | gpu | state |
mem_off(MHz+MT/s) | core_off | sm_clk | mem_clk | read_GBs | bw_pct | dec_toks | dec_pct | pf_toks |
vram_jc | core_C | pwr_W/cap | throttle | ecc(s/d/mm) | knee | guard | canary`. Example:

```
2026-06-21T14:32:07.412Z gpu0 SERVE mem_off=+1100MHz(+2200MT/s) core_off=+90 sm_clk=1965 mem_clk=9751MHz read=842GB/s bw=+11.4% dec=171.3tok/s +9.8% pf=2480tok/s vram_jc=88C core=71C pwr=312/350W thr=none ecc=noecc/mm0 knee=BELOW guard=60MHz canary=OK
```

The JSONL mirror carries `kind:"tick"`, `schema_version`, `epoch_ms`, `uuid`, raw numbers, and a sample
block `{window_s, n, agg, probe_in_window}`. **NVML offset = ½ the MT/s number — log both.**

**EVENT lines** (never coalesced/dropped; every derate/revert self-explains trigger + value + threshold +
new safe clock):

```
THERMAL_DERATE  SERVE→DERATE gpu0 trigger=vram_junction value=94C thr=>=92C action=mem_off +1100→+800MHz dec 171→164tok/s reason="1C under hard-abort" cooldown=120s
WATCHDOG_REVERT SERVE→REVERT gpu0 trigger=golden_canary value=token_mismatch mm=1 pos[88] exp=785 got=791 action=mem_off +1100→0(STOCK) latency=180ms SEVERITY=CRITICAL
PROMOTE         CHARACTERIZE→SERVE gpu0 mem_off=+1100 core_off=+90 guard=60MHz baseline 156.0→live 171.3tok/s (+9.8%)
```

Other events: KNEE_DETECTED, STEP_ACCEPTED/REJECTED, RATCHET_DOWN, RESTORE (FREE-lane only), RECOMMEND
(deploy-flag, never auto-applied), WATCHDOG_REVERT(Xid). Severity → journald priority + TTY color.

**Formats:** colored TTY (auto-plain when piped / `NO_COLOR`); rotating **JSONL** audit (authoritative,
100 MB / 7 d gzip); optional **Prometheus** exporter reusing DCGM gauge names
(`DCGM_FI_DEV_SM_CLOCK`/`_MEM_CLOCK`/`_GPU_TEMP`/`_MEMORY_TEMP`, gpu+UUID labels) plus `octuner_*` series
(offsets, state-set, decode_toks, vram_junction_celsius, guard_band_mhz, reverts_total / derates_total /
steps_*); optional `octunerctl top` TUI. Verbosity: quiet (events + 1 tick/min) / normal / debug (raw
polls + classifier internals).

**Watching the host daemon:** stdout → journald with structured fields → `journalctl -u octuner -f`,
`-p warning` (derates/reverts/Xid), `EVT=WATCHDOG_REVERT` native filter; `octunerctl watch`/`--follow`
attaches the event-bus socket for a low-latency colored stream (multiple read-only viewers);
`octunerctl status [--json]` prints a cached snapshot (no GPU poll, safe to spam).

**Per-signal source + cadence:** NVML clocks/power/core-temp/throttle/Xid/ECC = watchdog ~1 Hz (Xid
event-driven, instant-revert); junction via BAR0 `gputemps` = ~1 Hz (95 °C abort / 92 °C derate);
**BW+verify kernel = per-step in CHARACTERIZE, maintenance-only in SERVE** (it bandwidth-saturates and
contends with decode — demoted from the live path); vLLM `/metrics` = scrape 5-10 s (objective, *not* a
safety gate); golden canary = ~0.2 Hz in SERVE + dense per characterize step; DCGM PROF = optional,
datacenter only. The objective EWMA drops any `probe_in_window` sample.

---

## Sources / references

**NVIDIA docs / NVML / driver**
- https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceCommands.html
- https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html
- https://docs.nvidia.com/deploy/nvml-api/change-log.html
- https://docs.nvidia.com/deploy/nvml-api/structnvmlClockOffset__v1__t.html
- https://docs.nvidia.com/deploy/nvidia-smi/
- https://docs.nvidia.com/deploy/nvidia-smi/index.html
- https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html
- https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html
- https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/diag-targeted-stress-plugin.html
- https://docs.nvidia.com/datacenter/dcgm/2.4/user-guide/diag-cuda-mats.html
- https://archive.docs.nvidia.com/datacenter/dcgm/2.4/user-guide/diag-cuda-mats.html
- https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf
- https://man.archlinux.org/man/nvidia-smi.1.en

**NVIDIA dev-forum (clock offset / locked-clocks / mem-junction temp)**
- https://forums.developer.nvidia.com/t/nvmldevicegetminmaxclockofpstate-nvmldevicesetclockoffsets-issues/318332
- https://forums.developer.nvidia.com/t/applying-memory-clock-offset-breaks-memory-clock-locking/281722
- https://forums.developer.nvidia.com/t/sudo-nvidia-smi-lgc-lmc/284479
- https://forums.developer.nvidia.com/t/set-core-mem-clock-offset/239428
- https://forums.developer.nvidia.com/t/nvidia-a100-overclock-on-linux/226469
- https://forums.developer.nvidia.com/t/how-to-force-lock-sm-and-memory-clocks-on-rtx-5090-headless-linux/348794
- https://forums.developer.nvidia.com/t/request-gpu-memory-junction-temperature-via-nvidia-smi-or-nvml-api/168346
- https://forums.developer.nvidia.com/t/cant-overclock-memory-using-intel-integrated-as-display-dummy-xorg-entry-needed/51857
- https://forums.developer.nvidia.com/t/nvmldeviceresetmemorylockedclocks/185293
- https://forums.developer.nvidia.com/t/nvidia-smi-pl-safe-throttling-limit/338324
- https://forums.developer.nvidia.com/t/an-idle-vllm-process-consistently-pins-the-nvidia-gb10-gpu-at-max-graphics-clock/365325

**Headless OC tools / kernel module**
- https://github.com/Dreaming-Codes/nvidia_oc
- https://lib.rs/crates/nvidia_oc
- https://github.com/ilya-zlobintsev/LACT/releases
- https://github.com/sasha0552/nvidia-pstated
- https://github.com/NVIDIA/open-gpu-kernel-modules/discussions/236
- https://github.com/BeanGreen247/Linux_NVIDIA_GPU_Overclocking_Guide/blob/master/README.md
- https://github.com/Tresorio/nvidia-oc
- http://blog.zencoffee.org/2021/05/nvidia-overclocking-headless/
- https://wiki.archlinux.org/title/NVIDIA/Tips_and_tricks

**OC ROI / bandwidth / thermals (reputable OC + ML-serving)**
- https://forum.level1techs.com/t/some-gpu-5090-4090-3090-a600-idle-power-consumption-headless-on-linux-fedora-42-and-some-undervolt-overclock-info/237064
- https://itigic.com/gddr6x-memory-why-it-achieves-more-speed-and-overclock/
- https://www.techpowerup.com/review/nvidia-geforce-rtx-3080-founders-edition/39.html
- https://www.signalintegrityjournal.com/articles/1057-gbs-and-beyond-with-single-ended-io-in-high-performance-graphics-memory
- https://www.notebookcheck.net/Poor-memory-OC-scaling-on-the-GeForce-RTX-3080-might-be-a-blessing-in-disguise-with-closer-than-expected-performance-to-the-GeForce-RTX-3090.493458.0.html
- https://www.tweaktown.com/articles/10076/msi-geforce-rtx-3090-ti-suprim-overclocking-oc/index.html
- https://presenc.ai/research/local-llm-tokens-per-second-benchmarks-2026
- https://medium.com/@arjunravi726/why-llm-inference-is-memory-bound-not-compute-bound-ba59c48739e0
- https://hashrate.no/gpus/3090/ETC
- https://www.tomshardware.com/how-to/overclock-graphics-card-gpu
- https://www.tomshardware.com/news/hwinfo64-adds-gddr6x-temp-monitoring-rtx30series
- https://www.igorslab.de/en/gddr6x-am-limit-ueber-100-grad-bei-der-geforce-rtx-3080-fe-im-chip-gemessen-2/
- https://www.notebookcheck.net/NVIDIA-GeForce-RTX-3090-mod-can-reduce-VRAM-temperatures-by-up-to-25-C.528004.0.html
- https://www.formulamod.net/blogs/new/vram-overheating-monitor-fix-gpu-memory-temperature
- https://www.overclock.net/threads/3090-vram-hitting-108-degree-celsius-is-it-safe.1777211/
- https://www.overclock.net/threads/possibly-widespread-cooling-issues-on-rtx-3090-fe.1775245/page-5
- https://www.overclock.net/threads/official-nvidia-rtx-3090-owners-club.1753930/page-405
- https://www.overclock.net/threads/official-nvidia-rtx-3090-owners-club.1753930/page-821

**Power-limit / undervolt perf-watt**
- https://blog.qwertyforce.dev/posts/optimal_power_limit
- https://www.pugetsystems.com/labs/hpc/quad-rtx3090-gpu-power-limiting-with-systemd-and-nvidia-smi-1983/
- https://www.pugetsystems.com/labs/hpc/nvidia-gpu-power-limit-vs-performance-2296/
- https://www.techspot.com/news/94153-rtx-3090-ti-set-300w-rtx-3080-ti.html
- https://linuxconfig.org/how-to-set-nvidia-power-limit-on-ubuntu
- https://xhinker.medium.com/make-gpu-power-limits-persistent-across-reboots-3a35eb123494
- https://www.microway.com/hpc-tech-tips/nvidia-smi_control-your-gpus/

**Workstation/datacenter OC coverage (CLAIM 4) + A100 no-OC**
- https://www.microway.com/hardware/dgx-a100-review-throughput-and-hardware-summary/
- https://www.igorslab.de/en/nvidia-rtx-a5000-and-rtx-a6000-overclocking-quadro-overclocking-as-benefit-or-miss-it/
- https://wccftech.com/nvidia-ampere-rtx-workstations-gpus-capable-of-overclocking-msi-afterburner-beta/
- https://videocardz.com/newz/nvidia-rtx-ampere-workstation-cards-can-now-be-overclocked-with-msi-afterburner
- https://forums.overclockers.co.uk/threads/msi-afterburner-unlocks-overclocking-for-rtx-a5000-a6000.18934276/

**Correctness / SDC / ECC / determinism**
- https://arxiv.org/html/2605.04213
- https://arxiv.org/abs/2605.04213
- https://arxiv.org/pdf/2102.11245
- https://arxiv.org/pdf/2511.17826
- https://arxiv.org/pdf/0910.0505
- https://www.opencompute.org/documents/sdc-in-ai-ocp-whitepaper-final-pdf
- https://dl.acm.org/doi/10.1145/3690825
- https://www.chiplog.io/p/the-uncomfortable-truth-behind-deploying
- https://support.google.com/cloud/answer/10759085?hl=en
- https://docs.vllm.ai/en/latest/features/batch_invariance/
- https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- https://docs.vllm.ai/en/stable/design/cuda_graphs/
- https://linustechtips.com/topic/1515829-gddr6x-still-have-ecc-capabilities/
- https://www.wevolver.com/article/gddr6-vs-gddr6x-a-comprehensive-technical-comparison-for-digital-design-hardware-engineers
- https://www.notebookcheck.net/Nvidia-GeForce-RTX-5090-departs-from-RTX-3090-Ti-and-RTX-4090-flagship-tradition-drops-VRAM-ECC-for-pro-workloads.958141.0.html
- https://forums.anandtech.com/threads/how-do-you-test-vram-overclocking-error-correction.2253826/

**Memtest / stress tools**
- http://wili.cc/blog/gpu-burn.html
- https://github.com/wilicc/gpu-burn
- https://github.com/ComputationalRadiationPhysics/cuda_memtest
- https://github.com/ihaque/memtestG80
- https://www.memtest86.com/tech_individual-test-descr.html
- https://memtest.org/readme
- https://forums.guru3d.com/threads/announcing-memtest_vulkan-opensource-video-memory-stability-test.444817/

**Mem-junction temp (BAR0-MMIO) + bandwidth-stability instrument**
- https://github.com/ThomasBaruzier/gddr6-core-junction-vram-temps
- https://github.com/olealgoritme/gddr6
- https://github.com/UoB-HPC/BabelStream
- https://github.com/NVIDIA/nvbandwidth

**Practitioner OC-stability + autotuner prior art**
- https://foldingathome.org/faqs/rules-policies/best-practices/overclocking/
- https://medium.com/bitcoin-mining-dispatch/optimizing-mining-firmware-undervolting-overclocking-and-finding-the-efficiency-sweet-spot-f84ea246f128
- https://github.com/jungledesh/profile
- https://github.com/vllm-project/vllm/blob/main/benchmarks/auto_tune/README.md
- https://github.com/openshift-psap/auto-tuning-vllm
- https://github.com/vllm-project/vllm/issues/21410
- https://github.com/vllm-project/guidellm
- https://arthurchiao.art/blog/understanding-gpu-performance/
