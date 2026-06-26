"""HALF-A: fold MEASURED hardware ceilings (actual bandwidth + actual compute) into the tuner.

Spec peaks lie: a 3090's 936 GB/s spec is really ~838 achievable (~89.5%), ~888 with mem-OC. And
single-stream decode time splits into a BANDWIDTH part (streaming weights, ∝ 1/bw) and a FIXED part
(compute/launch/overhead that does NOT scale with bandwidth):

    TPOT(bw) = bw_coef / bw + fixed_t        tok/s = 1 / TPOT

Knowing the SPLIT is the actionable bit: it says exactly how much a fewer-bytes lever (mem-OC,
fp8/int4 KV, a bigger/faster card) can move decode — and when you're already compute/overhead-bound
so it CAN'T. The bw number comes from the bw_verify instrument (a plain CUDA read kernel, NO
privilege), so this lives in HALF-A. All pure; validated against the real 9B-w4a8 measurement
(838 GB/s→85 tok/s, 888→88.2 → 64% bandwidth-bound, compute ceiling ~239 tok/s, +3.8% from mem-OC).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

from .analytical import _linreg


@dataclass
class DecodeModel:
    """Single-stream decode split. bandwidth-time(per tok) = bw_coef / bw_gbs; fixed_t = the rest."""
    bw_coef: float       # GB  (so bw_coef/bw_gbs = seconds/token)
    fixed_t: float       # seconds/token that do NOT scale with bandwidth (compute + launch overhead)

    def tpot_s(self, bw_gbs: float) -> float:
        return self.bw_coef / bw_gbs + self.fixed_t

    def toks(self, bw_gbs: float) -> float:
        t = self.tpot_s(bw_gbs)
        return (1.0 / t) if t > 0 else 0.0

    def bw_bound_frac(self, bw_gbs: float) -> float:
        """0..1: how bandwidth-bound decode is at this bw. ~1 = pure weight-streaming (mem-OC /
        fewer-bytes help most); ->0 = compute/overhead-bound (those levers do ~nothing)."""
        t = self.tpot_s(bw_gbs)
        return (self.bw_coef / bw_gbs) / t if t > 0 else 0.0

    def compute_ceiling_toks(self) -> float:
        """tok/s even at INFINITE bandwidth (the fixed/compute wall) — the hard cap mem-OC can't pass."""
        return (1.0 / self.fixed_t) if self.fixed_t > 0 else float("inf")

    def to_dict(self) -> dict:
        return asdict(self)


def fit_decode_two_points(points: List[Tuple[float, float]]) -> Optional[DecodeModel]:
    """[(bw_gbs, toks), ...] at >=2 DISTINCT bandwidths (e.g. stock clock + a mem-OC clock) ->
    solve 1/toks = bw_coef*(1/bw) + fixed_t. This is the model-internal-free path."""
    if len({round(b, 1) for b, _ in points}) < 2 or any(t <= 0 for _, t in points):
        return None
    slope, intercept = _linreg([1.0 / b for b, _ in points], [1.0 / t for _, t in points])
    if slope <= 0 or intercept < 0:                       # non-physical fit (e.g. noise) -> reject
        return None
    return DecodeModel(bw_coef=slope, fixed_t=intercept)


def decode_from_one_point(toks: float, bw_gbs: float, weight_bytes: float) -> Optional[DecodeModel]:
    """ONE measured (toks @ bw) + the model's per-token WEIGHT bytes (from config) -> the split.
    bandwidth-time = weight_bytes/bw; fixed = 1/toks - that. Returns None if the bytes estimate is
    inconsistent (weight-read alone exceeds the measured TPOT)."""
    if toks <= 0 or bw_gbs <= 0 or weight_bytes <= 0:
        return None
    bw_t = weight_bytes / (bw_gbs * 1e9)
    fixed = 1.0 / toks - bw_t
    if fixed < 0:
        return None
    return DecodeModel(bw_coef=weight_bytes / 1e9, fixed_t=fixed)


def memoc_decode_gain_pct(model: DecodeModel, cur_bw: float, new_bw: float) -> float:
    """Predicted decode tok/s change (%) from raising achievable bandwidth cur->new (e.g. a HALF-B
    mem-OC). Sub-proportional to the bw gain by exactly the bw-bound fraction."""
    c, n = model.toks(cur_bw), model.toks(new_bw)
    return (n / c - 1.0) * 100.0 if c > 0 else 0.0


def ridge_batch(tflops: float, bw_gbs: float, bytes_per_param: float) -> float:
    """Decode COMPUTE<->BANDWIDTH crossover BATCH — the actual-compute lever for max-num-seqs.
    Below it decode is weight-bandwidth-bound (more batch amortizes the weight read -> more
    AGGREGATE tok/s); ABOVE it decode is compute-bound (more max-num-seqs adds prefill/latency,
    NOT aggregate decode throughput). B* = bytes_per_param * FLOPs / (2 * bytes/s). Use the
    COMPUTE-dtype TFLOPs (int8 for W4A8) and the WEIGHT-storage bytes/param (0.5 int4 / 1 int8 /
    2 fp16). VALIDATED: 3090 ~284 INT8 TOPS, 838 GB/s, int4 weights -> ~85 ≈ the empirical
    max-num-seqs<=82. So actual compute+bandwidth PREDICT the max-num-seqs ceiling a priori."""
    bps = bw_gbs * 1e9
    return (bytes_per_param * tflops) / (2.0 * bps) if bps > 0 else 0.0


def max_num_seqs_from_ridge(ridge: float, capacity_wall: int) -> Tuple[int, str]:
    """Recommended max-num-seqs ~ min(compute ridge, KV-capacity wall). Above the ridge decode is
    compute-bound (more concurrency adds no aggregate decode tok/s); above the wall it OOMs/preempts.
    Combines the actual-compute factor (ridge) with the capacity model (wall)."""
    rec = int(max(1, min(ridge, capacity_wall)))
    bound = "compute-ridge" if ridge <= capacity_wall else "KV-capacity"
    return rec, bound


def ridge_from_prefill(prefill_toks: float, bw_gbs: float, params_b: float,
                       bytes_per_param: float) -> Optional[float]:
    """ACCURATE max-num-seqs ridge from the REAL serving kernel. Prefill is compute-bound and runs
    the SAME Marlin GEMM as batched decode, so the per-token compute time ≈ 1/prefill_toks (measured,
    not a synthetic GEMM). Ridge = weight-read time (params*bytes/bw) * prefill_toks = the batch where
    the amortized weight read equals the compute. VALIDATED: ~15.4k tok/s prefill, 838 GB/s, 9B int4
    -> ~83 ≈ the empirical max-num-seqs<=82 (the synthetic torch._int_mm gave a too-low 18.6)."""
    if prefill_toks <= 0 or bw_gbs <= 0 or params_b <= 0:
        return None
    weight_read_s = (params_b * 1e9 * bytes_per_param) / (bw_gbs * 1e9)
    return weight_read_s * prefill_toks


def prefill_ceiling_toks(tflops: float, params_b: float) -> float:
    """Compute-bound prefill ceiling (tok/s) = FLOPs / (2 * params). Bounds the empirical fit and
    sizes chunked-prefill / max-num-batched-tokens (prefill is the compute-bound phase)."""
    return tflops / (2.0 * params_b * 1e9) if params_b > 0 else 0.0


def measure_tflops(dtype: str = "fp16", n: int = 4096, iters: int = 50) -> Optional[float]:  # pragma: no cover - GPU
    """Achievable GEMM TFLOP/s via a timed torch matmul (optional; None if torch absent). Use int8
    for the W4A8 decode-compute estimate. This is the ACTUAL achievable, not the spec peak."""
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "int8": torch.int8}.get(dtype, torch.float16)
    dev = "cuda"
    if dt == torch.int8:
        a = torch.randint(-8, 8, (n, n), dtype=torch.int8, device=dev)
        b = torch.randint(-8, 8, (n, n), dtype=torch.int8, device=dev)
        op = lambda: torch._int_mm(a, b)  # noqa: E731
    else:
        a = torch.randn(n, n, dtype=dt, device=dev)
        b = torch.randn(n, n, dtype=dt, device=dev)
        op = lambda: a @ b  # noqa: E731
    for _ in range(5):
        op()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        op()
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    return (2.0 * n ** 3) / (ms / 1000.0) / 1e12  # 2*n^3 FLOPs per matmul -> TFLOP/s


@dataclass
class MeasuredHw:
    """Hardware factors MEASURED at the actual operating point (NOT spec). bw + compute are read
    LIVE because the clock is load-dependent: CUDA runs in P2 (mem downclocked ~2.6%), then thermal
    throttle / power cap / a mem-OC offset all move it further. sm_mhz/mem_mhz are the clocks the
    bench actually ran at — proof these are the under-load values, not the catalog peak."""
    bw_gbs: Optional[float]
    tflops: Optional[float]
    sm_mhz: Optional[int]
    mem_mhz: Optional[int]
    compute_dtype: str = "int8"

    def ridge(self, bytes_per_param: float) -> Optional[float]:
        if not self.bw_gbs or not self.tflops:
            return None
        return ridge_batch(self.tflops * 1e12, self.bw_gbs, bytes_per_param)

    def to_dict(self) -> dict:
        return asdict(self)


def read_clocks_mhz(cuda_visible: Optional[str] = None) -> Tuple[Optional[int], Optional[int]]:  # pragma: no cover - GPU
    """The CURRENT sm/mem clocks via nvidia-smi (the under-load operating point, not spec)."""
    import subprocess
    sel = ["-i", cuda_visible] if cuda_visible else []
    try:
        out = subprocess.run(["nvidia-smi", *sel, "--query-gpu=clocks.sm,clocks.mem",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        sm, mem = out.stdout.strip().splitlines()[0].split(",")
        return int(sm), int(mem)
    except Exception:
        return None, None


def measure_hw_factors(bw_bin: str, *, compute_dtype: str = "int8", cuda_visible: Optional[str] = None,
                       image: Optional[str] = None, bw_size_gib: int = 8, bw_iters: int = 200,
                       gemm_n: int = 4096, gemm_iters: int = 80) -> MeasuredHw:  # pragma: no cover - GPU
    """Measure BOTH factors LIVE at the current operating point + record the clocks they ran at.
    Compute: via the SERVING IMAGE's torch when ``image`` is given (the autotune venv has no torch),
    else in-process. Run on a WARM card (sustained thermal throttle lowers both → slight over-
    estimate, keep a margin). Re-measure if conditions change (OC applied, thermal state)."""
    tflops = measure_tflops(dtype=compute_dtype, n=gemm_n, iters=gemm_iters)   # in-process if torch present
    if tflops is None and image:                          # else borrow the serving image's torch
        tflops = measure_tflops_via_image(image, cuda_visible=cuda_visible, dtype=compute_dtype,
                                          n=gemm_n, iters=gemm_iters)
    bw = measure_bw_gbs(bw_bin, size_gib=bw_size_gib, iters=bw_iters, cuda_visible=cuda_visible)
    sm, mem = read_clocks_mhz(cuda_visible)
    return MeasuredHw(bw_gbs=bw, tflops=tflops, sm_mhz=sm, mem_mhz=mem, compute_dtype=compute_dtype)


# GEMM bench run INSIDE the vLLM serving image (it ships torch — the lightweight autotune venv
# doesn't, by scope). Same idea as running bw_verify: reuse the image's torch+CUDA at the LIVE clock.
_GEMM_BENCH = (
    "import torch,json,time\n"
    "n={n};it={it}\n"
    "dt=torch.int8 if '{dt}'=='int8' else (torch.bfloat16 if '{dt}'=='bf16' else torch.float16)\n"
    "if dt==torch.int8:\n"
    " a=torch.randint(-8,8,(n,n),dtype=dt,device='cuda');b=torch.randint(-8,8,(n,n),dtype=dt,device='cuda')\n"
    " op=lambda: torch._int_mm(a,b)\n"
    "else:\n"
    " a=torch.randn(n,n,dtype=dt,device='cuda');b=torch.randn(n,n,dtype=dt,device='cuda')\n"
    " op=lambda: a@b\n"
    "for _ in range(10): op()\n"
    "torch.cuda.synchronize();t0=time.time()\n"
    "for _ in range(it): op()\n"
    "torch.cuda.synchronize();ms=(time.time()-t0)/it*1000\n"
    "print(json.dumps({{'tflops':2*n**3/(ms/1000)/1e12}}))\n"
)


def _gemm_script(dtype: str, n: int, iters: int) -> str:
    return _GEMM_BENCH.format(n=n, it=iters, dt=dtype)


def measure_tflops_via_image(image: str, *, cuda_visible: Optional[str] = None, dtype: str = "int8",
                             n: int = 8192, iters: int = 80) -> Optional[float]:  # pragma: no cover - docker+GPU
    """Achievable GEMM TFLOP/s by running the bench in the SERVING IMAGE's torch (no torch dep in
    the autotune venv). Runs on the host GPU at its LIVE clock (mem-OC/throttle included).

    CAVEAT: this is torch._int_mm — a generic int8 GEMM, NOT the W4A8 Marlin kernel vLLM serves
    with, so it UNDER-estimates the real serving compute (measured 62 TOPS vs ~142 dense peak) and
    yields a conservative (low) ridge. For an accurate ridge use the SERVER's measured prefill
    throughput (compute-bound, the real kernel) — see TODO. Treat this as a lower bound / sanity.
    """
    import json
    import subprocess
    # the host docker is wired for --runtime=nvidia + NVIDIA_VISIBLE_DEVICES (not --gpus); the
    # image's interpreter is python3 (no `python`). Match the proven serving-container pattern.
    sel = ["-e", f"NVIDIA_VISIBLE_DEVICES={cuda_visible}"] if cuda_visible else []
    cmd = ["docker", "run", "--rm", "--runtime=nvidia", *sel, "--entrypoint", "python3", image,
           "-c", _gemm_script(dtype, n, iters)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in reversed(p.stdout.strip().splitlines()):
        if line.strip().startswith("{"):
            try:
                return float(json.loads(line)["tflops"])
            except (ValueError, KeyError):
                continue
    return None


def measure_bw_gbs(bw_bin: str, *, size_gib: int = 8, iters: int = 200,
                   cuda_visible: Optional[str] = None) -> Optional[float]:  # pragma: no cover - GPU
    """Run the bw_verify CUDA kernel (NO privilege) -> achievable read GB/s, or None on failure."""
    import json
    import os
    import subprocess
    env = dict(os.environ)
    if cuda_visible:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible
    try:
        p = subprocess.run([bw_bin, str(size_gib), str(iters)], env=env,
                           capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in reversed(p.stdout.strip().splitlines()):
        if line.strip().startswith("{"):
            try:
                return float(json.loads(line)["read_GB_s"])
            except (ValueError, KeyError):
                continue
    return None


def render(model: Optional[DecodeModel], cur_bw: float, *, memoc_bw: Optional[float] = None) -> str:
    if model is None:
        return ("hw-factors: need 2 decode points at distinct bandwidths (stock + a mem-OC clock) "
                "OR one point + the model's per-token weight bytes.")
    frac = model.bw_bound_frac(cur_bw)
    lines = [
        "ampere-autotune — measured HW factors (actual bandwidth + compute in the decode model)",
        f"  at {cur_bw:.0f} GB/s: {model.toks(cur_bw):.0f} tok/s "
        f"(TPOT {model.tpot_s(cur_bw) * 1000:.1f} ms) = {frac:.0%} bandwidth-bound + "
        f"{1 - frac:.0%} fixed-compute",
        f"  fixed-compute ceiling ~{model.compute_ceiling_toks():.0f} tok/s (mem-OC/fewer-bytes can NEVER beat this)",
    ]
    if frac < 0.4:
        lines.append("  -> COMPUTE/overhead-bound: mem-OC, fp8/int4-KV won't move decode much; "
                     "spec-decode (MTP) / a faster card is the lever.")
    else:
        lines.append("  -> BANDWIDTH-bound: a fewer-bytes lever pays. "
                     + (f"mem-OC {cur_bw:.0f}->{memoc_bw:.0f} GB/s -> "
                        f"{memoc_decode_gain_pct(model, cur_bw, memoc_bw):+.1f}% decode."
                        if memoc_bw else "run bw_verify at a mem-OC clock to size the gain."))
    return "\n".join(lines)
