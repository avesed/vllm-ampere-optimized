"""Profile store — pure I/O, no GPU. Round-trip, the half-MT/s footgun, re-validation."""
from ampere_autotune.half_b.profile_store import (
    Profile, save, load, needs_revalidation, profile_path,
)


def _mk(uuid="GPU-abc", driver="565.57", off=900, temp=85.0):
    return Profile(
        uuid=uuid, arch="8.6", driver=driver, stock_mem_mhz=9751,
        max_stable_mem_offset_mhz=off, max_stable_gpc_offset_mhz=90,
        power_limit_w=350, validated_temp_c=temp, decode_gain_pct=8.0,
        prefill_gain_pct=2.0, vllm_flags={"max_num_seqs": 32},
    )


def test_mtps_is_double_the_clock_offset():
    p = _mk(off=900)
    assert p.max_stable_mem_offset_mtps == 1800
    assert p.to_dict()["max_stable_mem_offset_mtps"] == 1800


def test_roundtrip(tmp_path):
    p = _mk()
    path = save(p, base_dir=str(tmp_path))
    assert path == profile_path(p.uuid, str(tmp_path))
    back = load(p.uuid, base_dir=str(tmp_path))
    assert back is not None
    assert back.max_stable_mem_offset_mhz == p.max_stable_mem_offset_mhz
    assert back.vllm_flags == {"max_num_seqs": 32}
    assert back.max_stable_mem_offset_mtps == 1800  # derived field reconstructs


def test_load_absent_returns_none(tmp_path):
    assert load("GPU-nope", base_dir=str(tmp_path)) is None


def test_revalidate_on_uuid_mismatch():
    p = _mk(uuid="GPU-abc")
    why = needs_revalidation(p, uuid="GPU-other", driver="565.57", current_temp_c=85.0)
    assert why and "UUID mismatch" in why


def test_revalidate_on_driver_change():
    p = _mk(driver="565.57")
    why = needs_revalidation(p, uuid="GPU-abc", driver="570.10", current_temp_c=85.0)
    assert why and "driver changed" in why


def test_revalidate_on_temp_drift():
    p = _mk(temp=85.0)
    assert needs_revalidation(p, uuid="GPU-abc", driver="565.57", current_temp_c=98.0)
    assert needs_revalidation(p, uuid="GPU-abc", driver="565.57", current_temp_c=86.0) is None


def test_no_revalidate_when_stable():
    p = _mk()
    assert needs_revalidation(p, uuid="GPU-abc", driver="565.57", current_temp_c=85.0) is None
