"""Probe 4 — driver / persistence / ECC / UUID. All read-only."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

from . import _nvml

PERSIST_ENABLED = "ENABLED"
PERSIST_DISABLED = "DISABLED"
PERSIST_UNKNOWN = "UNKNOWN"

ECC_ON = "ECC_ON"
ECC_OFF = "ECC_OFF"
ECC_UNKNOWN = "ECC_UNKNOWN"


@dataclass
class DriverState:
    uuid: Optional[str]
    persistence: str
    ecc_current: str
    ecc_pending: str

    def to_dict(self) -> dict:
        return asdict(self)


def _ecc(v) -> str:
    if v is None:
        return ECC_UNKNOWN
    return ECC_ON if int(v) == 1 else ECC_OFF


def probe_driver_state(sess: "_nvml.Session", idx: int) -> DriverState:
    h = sess.handle(idx)
    if h is None:
        return DriverState(None, PERSIST_UNKNOWN, ECC_UNKNOWN, ECC_UNKNOWN)

    uuid_c = _nvml.call("nvmlDeviceGetUUID", h)
    uuid = uuid_c.value.decode() if (uuid_c.ok and isinstance(uuid_c.value, bytes)) else (uuid_c.value if uuid_c.ok else None)

    pm = _nvml.call("nvmlDeviceGetPersistenceMode", h)
    persistence = PERSIST_UNKNOWN
    if pm.ok:
        persistence = PERSIST_ENABLED if int(pm.value) == 1 else PERSIST_DISABLED

    ecc = _nvml.call("nvmlDeviceGetEccMode", h)  # returns (current, pending)
    if ecc.ok and isinstance(ecc.value, (tuple, list)) and len(ecc.value) == 2:
        ecc_cur, ecc_pend = _ecc(ecc.value[0]), _ecc(ecc.value[1])
    else:
        ecc_cur = ecc_pend = ECC_UNKNOWN

    return DriverState(uuid, persistence, ecc_cur, ecc_pend)
