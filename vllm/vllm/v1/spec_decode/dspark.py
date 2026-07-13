# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config import VllmConfig
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.utils import copy_and_expand_dspark_inputs_kernel


class DSparkProposer(DFlashProposer):
    """DeepSpec DSpark next-token-shift readout: anchor hidden is draft slot 0,
    all gamma query slots are sampled. Differs from DFlash only in draft layout +
    sampled-slot set; all KV/precompute/verify plumbing is inherited."""

    _inputs_kernel = staticmethod(copy_and_expand_dspark_inputs_kernel)

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dspark"
        super().__init__(vllm_config, device, runner)

    def _draft_num_query_per_req(self) -> int:
        # DSpark drafts exactly gamma query slots (anchor at slot 0 is sampled).
        return self.num_speculative_tokens
