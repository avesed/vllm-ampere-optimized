# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""flashampere must not claim head_size > 256.

Three reasons, all measured on a 3090 against FlashInfer 0.6.16:
  - famp's vendored 0.6.12 prefill.cuh returns cos~0.25/nan at hd512 there (no VO-split);
  - stock beats famp's XQA at hd512 decode (0.68-0.86x over 90 points);
  - there is no safe sink anyway -- super() is FA2, which raises "head dimension at most 256"
    and kills the engine core on the first request (observed on gemma-4 against a stock 0.28 image).
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


def test_hd512_declined_even_with_the_kernel_present(caps_with_kernel):
    # The fp16-PV probe must not re-open the large-head path: on 0.6.16 that kernel is wrong at
    # hd512, so a True probe would put garbage into production rather than a crash.
    assert FlashAmpereBackend.supports_head_size(512) is False


def test_head_sizes_up_to_256_are_unaffected(caps_without_kernel):
    # These have a working sink (FA2 handles <=256), so the kernel probe must not gate them.
    for hd in (64, 96, 128, 256):
        assert FlashAmpereBackend.supports_head_size(hd) is True
    assert FlashAmpereBackend.supports_head_size(100) is False  # still must be a multiple of 8
