# instruments/ — native host helpers (built on the host, not in any image)

Two small native tools the HALF-B gate needs. Both are **designed-only stubs** here; build
them on the host. Neither ships in a vLLM image.

## gputemps/ — GDDR6X memory-junction temperature (BAR0-MMIO)

Reads the GDDR6X **junction** temp that `nvidia-smi`/NVML/DCGM do **not** expose on GeForce.
GA10x-verified (3090/A6000-class). Build: `gcc gputemps.c -o gputemps -O3 -lnvidia-ml -lpci`.

**One-time host prep (the "doctor" preflight checks these):**
- `iomem=relaxed` kernel boot param (BAR0 MMIO via `/dev/mem`) — needs reboot.
- Secure Boot **off** (else MMIO is blocked).
- run as **root**.
- sanity-check `gputemps` idle core temp against NVML before trusting the junction reading —
  the MMIO offsets are **GA102 reverse-engineered and driver-fragile** (TU102/Turing not
  supported → no thermal-abort path there).

Used as the **hard thermal-abort** gate: revert mem-offset at junction ≥ 95 °C, warn ≥ 90 °C.

## bw_verify/ — bandwidth + integrity in one pass

A bandwidth-saturating CUDA kernel that ALSO verifies data (write pattern → read-back →
checksum), reporting `read_GB_s` (the **EDR knee** signal, GDDR6X only) AND `mismatch_count`
(catches a no-ECC cell flip with **zero** bandwidth penalty). Build: `nvcc`. Memory-frugal
(LFSR/index-derived pattern, no golden buffer). **Demoted to characterize-only in SERVE** —
it saturates bandwidth and contends with decode.

> The real correctness gate is still the inference **golden token-id** compare under
> `VLLM_BATCH_INVARIANT=1` (covers the weight-layout / int4-dequant / IMMA compute path that
> a raw-cell memtest cannot). `bw_verify` is a fast coverage/health probe, not the gate.
