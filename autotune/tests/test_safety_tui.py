"""HALF-B launch consent gates + the live TUI render — pure (no GPU, no real TTY)."""
import io

from ampere_autotune.half_b import safety, tui


def _null():
    return io.StringIO()


# ---- hardware-damage warning -----------------------------------------------------------------

def test_damage_warning_text_is_loud():
    w = safety.OC_DAMAGE_WARNING.upper()
    assert "HARDWARE" in w and ("DAMAGE" in w or "WARRANTY" in w) and "RISK" in w


def test_damage_warning_force_skips_prompt():
    assert safety.confirm_oc_damage_warning(force=True, stream=_null()) is True


def test_damage_warning_requires_explicit_yes():
    assert safety.confirm_oc_damage_warning(reader=lambda _: "yes", stream=_null()) is True
    assert safety.confirm_oc_damage_warning(reader=lambda _: "y", stream=_null()) is False   # only 'yes'
    assert safety.confirm_oc_damage_warning(reader=lambda _: "no", stream=_null()) is False


def test_damage_warning_eof_aborts():
    def boom(_):
        raise EOFError
    assert safety.confirm_oc_damage_warning(reader=boom, stream=_null()) is False


# ---- card-in-use consent ---------------------------------------------------------------------

def test_vram_in_use_verdict():
    assert safety.vram_in_use(0, 0)[0] is False
    assert safety.vram_in_use(50, 0)[0] is False          # below the 200 MiB idle threshold
    busy, msg = safety.vram_in_use(22000, 1)
    assert busy is True and "Another application is using this card" in msg


def test_confirm_vram_clear_card_passes_silently():
    # no prompt at all on a clear card (reader that would fail proves it isn't called)
    assert safety.confirm_vram_in_use(0, 0, reader=lambda _: (_ for _ in ()).throw(AssertionError),
                                      stream=_null()) is True


def test_confirm_vram_in_use_prompts_and_respects_answer():
    assert safety.confirm_vram_in_use(22000, 1, reader=lambda _: "yes", stream=_null()) is True
    assert safety.confirm_vram_in_use(22000, 1, reader=lambda _: "no", stream=_null()) is False
    assert safety.confirm_vram_in_use(22000, 1, force=True, stream=_null()) is True


# ---- live TUI render -------------------------------------------------------------------------

def test_render_frame_formats_cells():
    out = tui.render_frame([{"gpu": "GPU-abc", "phase": "CLIMB", "offset": 1000, "read": 888.4,
                             "mismatch": 0, "temp": 42, "status": "PASS"}], footer="boundary +1000")
    assert "+1000" in out and "888" in out and "42C" in out and "PASS" in out
    assert "boundary +1000" in out and "GPU-abc" in out


def test_render_frame_none_is_dash():
    out = tui.render_frame([{"gpu": "g", "read": None, "offset": None}])
    assert " -" in out


def test_reporter_update_builds_rows():
    seen = []
    r = tui.Reporter(use_rich=False, sink=seen.append)
    r.update("GPU-a", phase="BASE", offset=0, read=838.0, mismatch=0)
    r.update("GPU-a", phase="CLIMB", offset=1000, read=888.0, mismatch=0, status="PASS")
    assert len(r.rowlist()) == 1 and r.rowlist()[0]["offset"] == 1000
    assert "888" in r.frame()
