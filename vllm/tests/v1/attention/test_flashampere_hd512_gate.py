# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""flashampere must not claim head_size > 256 without the kernel that can serve it.

famp extends past FA's 256 ceiling only through its vendored fp16-PV FlashInfer prefill. If that
kernel is missing (stock FlashInfer) and famp still claims the layer, the call falls through to
super() = FA2, which raises "FlashAttention forward only supports head dimension at most 256" and
kills the engine core on the first request -- observed on gemma-4 (hd512 global layers) against a
stock vLLM 0.28 image. Declining instead lets backend selection pick TRITON, which supports hd512.
"""

import pytest

from vllm.v1.attention.backends.flashampere import capability
from vllm.v1.attention.backends.flashampere.backend import FlashAmpereBackend


@pytest.fixture
def caps_without_kernel(monkeypatch):
    stub = capability.detect(
        cc_major=8,
        device_name="NVIDIA GeForce RTX 3090",
        env={"VLLM_FLASHAMPERE": "1"},
        has_flashinfer=True,
        has_sage=False,
        has_fp16pv_kernel=False,
    )
    monkeypatch.setattr(capability, "caps", lambda: stub)
    return stub


@pytest.fixture
def caps_with_kernel(monkeypatch):
    stub = capability.detect(
        cc_major=8,
        device_name="NVIDIA GeForce RTX 3090",
        env={"VLLM_FLASHAMPERE": "1"},
        has_flashinfer=True,
        has_sage=False,
        has_fp16pv_kernel=True,
    )
    monkeypatch.setattr(capability, "caps", lambda: stub)
    return stub


def test_hd512_declined_without_the_fp16pv_kernel(caps_without_kernel):
    assert FlashAmpereBackend.supports_head_size(512) is False
    assert FlashAmpereBackend.supports_head_size(320) is False


def test_hd512_claimed_when_the_kernel_is_present(caps_with_kernel):
    assert FlashAmpereBackend.supports_head_size(512) is True


def test_head_sizes_up_to_256_are_unaffected(caps_without_kernel):
    # These have a working sink (FA2 handles <=256), so the kernel probe must not gate them.
    for hd in (64, 96, 128, 256):
        assert FlashAmpereBackend.supports_head_size(hd) is True
    assert FlashAmpereBackend.supports_head_size(100) is False  # still must be a multiple of 8
