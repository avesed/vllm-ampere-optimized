"""Preflight unit tests — pure logic, no GPU, no root, no pynvml required."""
from ampere_autotune.preflight import sku, oc_perm, matrix
from ampere_autotune.preflight import driver_state as ds
from ampere_autotune.preflight import telemetry_perm as tel


# ---- SKU classification + tier ceiling / gate family (pure helpers) ----

def test_3090_is_gddr6x_tier3_golden_edr():
    cls = sku._classify_sku("NVIDIA GeForce RTX 3090")
    mem = sku._classify_mem("NVIDIA GeForce RTX 3090", cls)
    ceiling, gate = sku._ceiling_and_gate(mem, cls, sku.OFFSET_SUPPORTED, "NVIDIA GeForce RTX 3090")
    assert cls == sku.SKU_GEFORCE
    assert mem == sku.MEM_GDDR6X
    assert ceiling == "T3"
    assert gate == sku.GATE_GOLDEN_EDR_JUNCTION


def test_a6000_is_gddr6_ecc_golden_no_edr_knee():
    cls = sku._classify_sku("NVIDIA RTX A6000")
    mem = sku._classify_mem("NVIDIA RTX A6000", cls)
    ceiling, gate = sku._ceiling_and_gate(mem, cls, sku.OFFSET_SUPPORTED, "NVIDIA RTX A6000")
    assert cls == sku.SKU_WORKSTATION
    assert mem == sku.MEM_GDDR6           # NOT GDDR6X
    assert ceiling == "T3"
    assert gate == sku.GATE_ECC_GOLDEN    # no EDR knee on plain GDDR6


def test_a4000_is_experimental():
    cls = sku._classify_sku("NVIDIA RTX A4000")
    mem = sku._classify_mem("NVIDIA RTX A4000", cls)
    ceiling, _ = sku._ceiling_and_gate(mem, cls, sku.OFFSET_SUPPORTED, "NVIDIA RTX A4000")
    assert ceiling == "T3-EXPERIMENTAL"


def test_a100_is_locked_t1_only():
    cls = sku._classify_sku("NVIDIA A100-SXM4-80GB")
    mem = sku._classify_mem("NVIDIA A100-SXM4-80GB", cls)
    # datacenter offset API returns NOT_SUPPORTED
    ceiling, gate = sku._ceiling_and_gate(mem, cls, sku.OFFSET_NOT_SUPPORTED, "NVIDIA A100-SXM4-80GB")
    assert cls == sku.SKU_DATACENTER
    assert mem == sku.MEM_HBM
    assert ceiling == "T1"
    assert gate == sku.GATE_ECC


def test_a40_datacenter_gddr6_locked():
    cls = sku._classify_sku("NVIDIA A40")
    mem = sku._classify_mem("NVIDIA A40", cls)
    ceiling, _ = sku._ceiling_and_gate(mem, cls, sku.OFFSET_NOT_SUPPORTED, "NVIDIA A40")
    assert cls == sku.SKU_DATACENTER
    assert ceiling == "T1"


# ---- driver tuple gate ----

def test_driver_floor():
    assert oc_perm._driver_tuple("565.57") >= (555, 85)
    assert oc_perm._driver_tuple("550.120") < (555, 85)
    assert oc_perm._driver_tuple("garbage") == (0, 0)


def test_mtps_offset_half_footgun():
    from ampere_autotune.half_b import silicon
    # NVML clock-domain offset = HALF the GDDR MT/s number
    assert silicon.mtps_to_clock_offset_mhz(2000) == 1000
    assert silicon.clock_offset_mhz_to_mtps(1000) == 2000


# ---- decision matrix: unprivileged context refuses HALF-B ----

def _mk(priv, write_cap, driver_ok=True, junction=tel.JUNCTION_READABLE,
        offset=sku.OFFSET_SUPPORTED, mem=sku.MEM_GDDR6X, gate=sku.GATE_GOLDEN_EDR_JUNCTION,
        ceiling="T3", ecc=ds.ECC_OFF, persist=ds.PERSIST_ENABLED):
    s = sku.SkuResult(0, "RTX 3090", "GEFORCE", "8.6", mem, sku.SKU_GEFORCE, offset, ceiling, gate)
    t = tel.TelemetryResult(tel.READ_OK, junction, "", tel.DCGM_BLANK)
    o = oc_perm.OcPermResult(True, True, False, True, priv, "565.57", driver_ok, write_cap, "")
    d = ds.DriverState("GPU-xyz", persist, ecc, ecc)
    return s, t, o, d


def test_host_root_unlocks_t3():
    s, t, o, d = _mk(oc_perm.HOST_ROOT_PRIVILEGED, oc_perm.WRITE_LIKELY)
    half_b, max_tier, _, decision = matrix._decide(s, t, o, d)
    assert half_b is True
    assert max_tier == "T3"
    assert "unlocked" in decision


def test_unprivileged_container_refuses_half_b():
    s, t, o, d = _mk(oc_perm.UNPRIVILEGED_CONTAINER, oc_perm.WRITE_NO_PERMISSION)
    half_b, max_tier, refusals, _ = matrix._decide(s, t, o, d)
    assert half_b is False
    assert max_tier == "T1"
    assert any("REFUSED" in r for r in refusals)


def test_gddr6x_t3_refused_without_junction_sensor():
    s, t, o, d = _mk(oc_perm.HOST_ROOT_PRIVILEGED, oc_perm.WRITE_LIKELY,
                     junction=tel.MISSING_IOMEM_RELAXED)
    half_b, max_tier, refusals, _ = matrix._decide(s, t, o, d)
    assert half_b is True          # T0/1/2 still ok
    assert max_tier == "T2"        # T3 capped: no thermal abort sensor
    assert any("T3 mem-OC REFUSED" in r for r in refusals)


def test_persistence_disabled_warns():
    s, t, o, d = _mk(oc_perm.HOST_ROOT_PRIVILEGED, oc_perm.WRITE_LIKELY, persist=ds.PERSIST_DISABLED)
    _, _, refusals, _ = matrix._decide(s, t, o, d)
    assert any("persistence" in r.lower() for r in refusals)
