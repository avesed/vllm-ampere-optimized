"""HALF-B mem-OC: pure parts (no GPU) — VRAM-fill sizing, bw parse, measurement assembly,
multi-GPU per-card characterize + concurrent-soak verdicts."""
from ampere_autotune.half_b import silicon, orchestrate
from ampere_autotune.half_b.gate import (
    parse_bw_verify, assemble, evaluate, Thresholds, Measurement,
    PASS, REJECT, ABORT, GATE_GOLDEN_EDR_JUNCTION,
)

_MIB = 1024 * 1024
_GIB = 1024 ** 3


# ---- VRAM fill sizing (the back-chip fix) ----------------------------------------------------

def test_fill_gib_covers_top_chip_on_24gb_card():
    # 24126 MiB free (real gpu1) -> 23 GiB, which reaches the 22-24GB top back-side chip.
    # (0.7 headroom would round to 22 and leave that chip cold = the bug the user caught.)
    assert silicon.fill_gib(24126 * _MIB) == 23


def test_fill_gib_leaves_context_headroom():
    assert silicon.fill_gib(8 * _GIB) == 7            # 8 - 0.5 -> 7 (never the full 8)


def test_fill_gib_unknown_or_tiny_free_is_zero():
    assert silicon.fill_gib(None) == 0
    assert silicon.fill_gib(0) == 0
    assert silicon.fill_gib(200 * _MIB) == 0          # smaller than the headroom


def test_mtps_offset_footgun_is_half():
    assert silicon.mtps_to_clock_offset_mhz(2000) == 1000   # +2000 MT/s == +1000 MHz NVML offset
    assert silicon.clock_offset_mhz_to_mtps(1000) == 2000


# ---- bw_verify parse + measurement assembly --------------------------------------------------

def test_parse_bw_verify_reads_last_json_line():
    out = "noise\n{\"read_GB_s\": 888.0, \"write_GB_s\": 700.0, \"mismatch_count\": 0}\n"
    assert parse_bw_verify(out) == (888.0, 0)


def test_parse_bw_verify_crash_returns_sentinel():
    assert parse_bw_verify("CUDA error: out of memory\n") == (None, -1)
    assert parse_bw_verify("") == (None, -1)


def test_assemble_clean_is_pass():
    m = assemble(500, (878.0, 0))
    v = evaluate(m, Measurement(250, True, 0, 860.0), Thresholds(), GATE_GOLDEN_EDR_JUNCTION)
    assert m.golden_ok and m.read_gbs == 878.0 and v.status == PASS


def test_assemble_mismatch_is_reject():
    assert evaluate(assemble(500, (800.0, 7)), None, Thresholds(), GATE_GOLDEN_EDR_JUNCTION).status == REJECT


def test_assemble_bw_crash_is_abort():
    m = assemble(750, (None, -1))                     # bw_verify hung/crashed -> hard unsafe
    assert m.xid and evaluate(m, None, Thresholds(), GATE_GOLDEN_EDR_JUNCTION).status == ABORT


def test_assemble_set_failure_is_abort():
    m = assemble(750, (800.0, 0), set_rc=1)           # couldn't even apply the offset
    assert m.xid and evaluate(m, None, Thresholds(), GATE_GOLDEN_EDR_JUNCTION).status == ABORT


def test_assemble_golden_fail_is_reject():
    assert evaluate(assemble(500, (880.0, 0), golden_ok=False), None,
                    Thresholds(), GATE_GOLDEN_EDR_JUNCTION).status == REJECT


# ---- multi-GPU: per-card characterize + concurrent-soak verdict ------------------------------

def test_characterize_each_runs_per_card_independently():
    # card A is a strong sample (rises far), card B knees early -> different accepted offsets.
    def make(uuid):
        ceil = {"A": 1500, "B": 300}[uuid]
        def measure(off):
            # clean + bandwidth rises until the card's own ceiling, then flat (KNEE)
            read = 800.0 + min(off, ceil) * 0.04
            return Measurement(off, golden_ok=True, mismatch_count=0, read_gbs=read)
        return measure
    res = orchestrate.characterize_each(["A", "B"], make, gate_family=GATE_GOLDEN_EDR_JUNCTION,
                                        max_offset_mhz=900, samples=1)
    assert set(res) == {"A", "B"}
    assert res["B"].boundary_offset_mhz < res["A"].boundary_offset_mhz   # B knees earlier


def test_soak_verdict_clean_passes():
    r = orchestrate.soak_verdict("A", 1000, [(888.0, 0), (887.0, 0), (888.0, 0)])
    assert r.ok and r.total_mismatch == 0 and r.max_read_gbs == 888.0


def test_soak_verdict_mismatch_under_concurrent_heat_fails():
    r = orchestrate.soak_verdict("A", 1000, [(888.0, 0), (886.0, 4)])
    assert not r.ok and "mismatch" in r.note and r.total_mismatch == 4


def test_soak_verdict_crash_fails():
    assert not orchestrate.soak_verdict("A", 1000, [(888.0, 0), (None, -1)]).ok


def test_soak_verdict_no_samples_fails():
    assert not orchestrate.soak_verdict("A", 1000, []).ok


def test_soak_failures_lists_only_failed_cards():
    results = {
        "A": orchestrate.SoakResult("A", 1000, 888.0, 0, True),
        "B": orchestrate.SoakResult("B", 1000, 880.0, 3, False, "mismatch"),
    }
    fails = orchestrate.soak_failures(results)
    assert [r.uuid for r in fails] == ["B"]


# ---- review fixes: revert UUID resolution (CRITICAL) + golden tri-state ----------------------

import types


def test_revert_all_resolves_real_uuid_from_driver_state_dict():
    # CRITICAL bug: matrix GpuCapabilities has no .uuid attr (UUID is driver_state["uuid"]); the old
    # code fell back to g.index -> nvmlGetHandleByUUID("0") -> always failed -> revert no-op'd.
    seen = []
    mtx = types.SimpleNamespace(gpus=[
        types.SimpleNamespace(index=0, driver_state={"uuid": "GPU-aaa"}),
        types.SimpleNamespace(index=1, driver_state={"uuid": "GPU-bbb"}),
    ])
    rc = silicon.revert_all(mtx, reset_fn=lambda u: seen.append(u) or 0)
    assert seen == ["GPU-aaa", "GPU-bbb"] and rc == 0     # real UUIDs, not "0"/"1"


def test_revert_all_skips_cards_without_uuid():
    seen = []
    mtx = types.SimpleNamespace(gpus=[types.SimpleNamespace(index=0, driver_state={"uuid": None})])
    silicon.revert_all(mtx, reset_fn=lambda u: seen.append(u) or 0)
    assert seen == []                                     # never call GetHandleByUUID(None/"0")


def test_evaluate_golden_none_falls_back_not_rejects():
    # golden NOT run (None) must NOT REJECT (would reject everything); rely on mismatch instead.
    rising = evaluate(Measurement(300, golden_ok=None, mismatch_count=0, read_gbs=905.0),
                      Measurement(200, True, 0, 900.0), Thresholds(), GATE_GOLDEN_EDR_JUNCTION)
    assert rising.status == PASS
    corrupt = evaluate(Measurement(300, golden_ok=None, mismatch_count=5, read_gbs=905.0),
                       None, Thresholds(), GATE_GOLDEN_EDR_JUNCTION)
    assert corrupt.status == REJECT                       # mismatch still gates


def test_combine_golden_tristate():
    from ampere_autotune.half_b.search import _combine_golden
    assert _combine_golden([None, None]) is None          # not run -> never fabricate True
    assert _combine_golden([True, None]) is True
    assert _combine_golden([True, False]) is False         # any mismatch -> False (worst-of-N)
