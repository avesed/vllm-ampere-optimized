"""HALF-B multi-GPU orchestration.

Two multi-GPU facts the single-card search must NOT ignore:
  1. SILICON LOTTERY is per-card — each GPU gets its OWN characterize + offset (never one offset
     for the box). characterize_each loops search.characterize per UUID.
  2. THERMAL COUPLING — characterizing card A with card B idle OVERESTIMATES A's safe offset: in
     real serving BOTH cards are hot, share chassis airflow, and run hotter -> a lower stable
     clock. So after the per-card knees, we RE-VALIDATE with concurrent_soak: every card at its
     candidate offset, ALL soaking SIMULTANEOUSLY (the real both-cards-hot case). Any card that
     mismatches/hangs under concurrent heat is backed off. Per-card-isolated numbers are an
     UPPER BOUND; the concurrent soak is the deploy gate.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from . import gate as _gate
from . import search as _search
from . import silicon


def characterize_each(uuids: List[str], make_measure_fn: Callable, *, gate_family: str,
                      max_offset_mhz: int = 1500, th=None, samples: int = _search.DEFAULT_SAMPLES
                      ) -> Dict[str, _search.SearchResult]:
    """Per-GPU independent characterize. make_measure_fn(uuid) -> measure_fn(offset)->Measurement
    for that card. Returns {uuid: SearchResult} (each card's own EDR knee / accepted offset)."""
    return {u: _search.characterize(make_measure_fn(u), gate_family=gate_family,
                                    max_offset_mhz=max_offset_mhz, th=th, samples=samples)
            for u in uuids}


@dataclass
class SoakResult:
    uuid: str
    offset_mhz: int
    max_read_gbs: Optional[float]
    total_mismatch: int
    ok: bool
    note: str = ""


def soak_verdict(uuid: str, offset: int, samples: List[tuple]) -> SoakResult:
    """Pure: fold a card's concurrent-soak samples [(read_gbs, mismatch), ...] into a verdict.
    Any crash (mismatch<0) or any positive mismatch over the soak = FAIL (back this card off)."""
    crashed = any(mm is not None and mm < 0 for _, mm in samples)
    total_mm = sum(mm for _, mm in samples if mm and mm > 0)
    reads = [r for r, _ in samples if r]
    mx = max(reads) if reads else None
    if crashed:
        return SoakResult(uuid, offset, mx, total_mm, False, "bw_verify crash/hang under concurrent soak")
    if total_mm > 0:
        return SoakResult(uuid, offset, mx, total_mm, False, f"mismatch={total_mm} under concurrent soak")
    if not samples:
        return SoakResult(uuid, offset, mx, 0, False, "no samples (soak did not run)")
    return SoakResult(uuid, offset, mx, 0, True, "")


def soak_failures(results: Dict[str, SoakResult]) -> List[SoakResult]:
    """Pure: the cards that did NOT pass the concurrent soak (need a back-off + re-soak)."""
    return [r for r in results.values() if not r.ok]


def _soak_one(uuid: str, offset: int, *, deadline: float, stop: "threading.Event",
              set_fn, free_fn, bw_fn, junction_fn=None, junction_abort_c: float = 95.0,
              reporter=None) -> SoakResult:  # pragma: no cover - GPU-bound
    rc, _ = set_fn(uuid, offset)
    if rc != 0:
        stop.set()
        return SoakResult(uuid, offset, None, 0, False, f"set offset rc={rc}")
    size = silicon.fill_gib(free_fn(uuid))           # FILL this card's VRAM (back chips)
    samples: List[tuple] = []
    running_mm = 0
    while time.time() < deadline and not stop.is_set():   # cross-card fail-fast (C5)
        read, mm = bw_fn(uuid, size)
        samples.append((read, mm))
        jt = junction_fn(uuid) if junction_fn else None
        if mm and mm > 0:
            running_mm += mm
        if reporter is not None:                      # live progress for the TUI
            left = max(0, int(deadline - time.time()))
            reporter.update(uuid, phase=f"soak {left}s", offset=offset, read=read,
                            mismatch=running_mm, temp=jt, status="" if running_mm == 0 else "FAIL")
        if jt is not None and jt >= junction_abort_c:
            stop.set()
            return SoakResult(uuid, offset, (read or None), 0, False,
                              f"junction {jt:.0f}C >= {junction_abort_c:.0f}C")
        if (mm is not None and mm < 0) or (mm and mm > 0):
            stop.set()                               # one card's corruption stops the whole box
            break
    return soak_verdict(uuid, offset, samples)


def concurrent_soak(offset_by_uuid: Dict[str, int], *, duration_s: int = 480, iters: int = 200,
                    bw_bin: str = _gate.DEFAULT_BW_BIN, junction_fn=None, junction_abort_c: float = 95.0,
                    reporter=None, set_fn=None, free_fn=None, bw_fn=None
                    ) -> Dict[str, SoakResult]:  # pragma: no cover - GPU
    """Apply ALL offsets + soak ALL gpus SIMULTANEOUSLY for duration_s (the realistic both-cards-
    hot thermal case the per-card characterize misses). Each card fills its OWN VRAM. ANY card's
    corruption/junction-abort sets a shared stop so the whole box leaves the bad state at once
    (C5). ALWAYS reverts every offset to 0 on exit (C10). Returns {uuid: SoakResult}."""
    set_fn = set_fn or silicon.set_mem_offset_mhz
    free_fn = free_fn or silicon.mem_free_bytes
    bw_fn = bw_fn or (lambda u, g: _gate.run_bw_verify(u, g, iters, bw_bin))
    uuids = list(offset_by_uuid)
    stop = threading.Event()
    deadline = time.time() + duration_s
    try:
        with ThreadPoolExecutor(max_workers=max(1, len(uuids))) as ex:
            futs = {u: ex.submit(_soak_one, u, offset_by_uuid[u], deadline=deadline, stop=stop,
                                 set_fn=set_fn, free_fn=free_fn, bw_fn=bw_fn, reporter=reporter,
                                 junction_fn=junction_fn, junction_abort_c=junction_abort_c)
                    for u in uuids}
            return {u: f.result() for u, f in futs.items()}
    finally:
        for u in uuids:                              # guaranteed revert, even on exception/Ctrl-C
            try:
                silicon.reset_mem_offset(u)
            except Exception:
                pass
