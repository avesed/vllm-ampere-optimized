"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from flashinfer.jit import env as jit_env
import torch
from flashinfer.jit.utils import filename_safe_dtype_map
from flashinfer.compilation_context import CompilationContext
from flashinfer.jit.core import (
    JitSpec,
    gen_jit_spec,
)
import pathlib
FAMP_CSRC = pathlib.Path(__file__).parent / "csrc"  # famp's vendored XQA kernel

xqa_nvcc_flags = [
    "-DNDEBUG=1",
    "-DBEAM_WIDTH=1",
    "-DUSE_INPUT_KV=0",
    "-DUSE_CUSTOM_BARRIER=1",
]


def gen_xqa_module(
    input_dtype: torch.dtype,
    kv_cache_dtype: torch.dtype,
    page_size: int,
    head_dim: int,
    head_group_ratio: int,
    use_sliding_window: bool,
    output_dtype: torch.dtype,
    q_seq_len: int = 1,
) -> JitSpec:
    if input_dtype == torch.float16:
        flag_input_dtype = ["-DINPUT_FP16=1", "-DDTYPE=__half"]
    elif input_dtype == torch.bfloat16:
        flag_input_dtype = ["-DINPUT_FP16=0", "-DDTYPE=__nv_bfloat16"]
    else:
        raise ValueError(
            f"Invalid dtype: {input_dtype} for XQA, only float16 and bfloat16 input are supported"
        )

    if kv_cache_dtype == torch.float8_e4m3fn:
        flag_kv_cache_dtype = ["-DCACHE_ELEM_ENUM=2"]
    elif kv_cache_dtype == torch.int8:
        flag_kv_cache_dtype = ["-DCACHE_ELEM_ENUM=1"]
    elif kv_cache_dtype == torch.uint8:
        flag_kv_cache_dtype = ["-DCACHE_ELEM_ENUM=3"]
    else:
        flag_kv_cache_dtype = ["-DCACHE_ELEM_ENUM=0"]

    # 256 added for gemma4's hd512 hybrid KV-cache group (physical page_size=256 even at
    # --block-size 64); the kernel's page math is page-count-agnostic (nbPagesPerWarpTile=1).
    if page_size not in [16, 32, 64, 128, 256]:
        raise ValueError(
            f"Invalid page_size: {page_size}, only 16, 32, 64, 128, 256 are supported"
        )
    flag_tokens_per_page = [f"-DTOKENS_PER_PAGE={page_size}"]

    # famp wide-head: 512 (gemma4 full-attn) is supported via the headElemsQK/headElems split in
    # mha.h (gemm0 runs full-512 QK, gemm1 covers the 512 V/output in nbVChunks=2 passes of 256).
    if head_dim % 16 != 0 or head_dim < 16 or (head_dim > 256 and head_dim != 512):
        raise ValueError(
            f"Invalid head_dim: {head_dim}, must be divisible by 16 and in [16, 256] or == 512"
        )
    flag_head_dim = [f"-DHEAD_ELEMS={head_dim}"]

    flag_head_group_ratio = [f"-DHEAD_GRP_SIZE={head_group_ratio}"]

    if use_sliding_window:
        flag_sliding_window = ["-DSLIDING_WINDOW=1"]
    else:
        flag_sliding_window = ["-DSLIDING_WINDOW=0"]

    if output_dtype == torch.float8_e4m3fn:
        flag_low_prec_output = ["-DLOW_PREC_OUTPUT=1"]
    else:
        flag_low_prec_output = ["-DLOW_PREC_OUTPUT=0"]

    if q_seq_len > 1:
        use_spec_dec = True
        if q_seq_len * head_group_ratio <= 32:
            flag_spec_dec = ["-DSPEC_DEC=1", f"-DSPEC_Q_SEQ_LEN={q_seq_len}"]
        else:
            flag_spec_dec = ["-DSPEC_DEC=1"]
    else:
        flag_spec_dec = ["-DSPEC_DEC=0"]
        use_spec_dec = False

    compilation_context = CompilationContext()
    nvcc_flags = compilation_context.get_nvcc_flags_list(
        supported_major_versions=[8, 9, 10, 11, 12]
    )
    sm_nvcc_flags = nvcc_flags

    flag_mla_wrapper = ["-DMLA_WRAPPER=0"]

    sources = [
        FAMP_CSRC / "xqa/mha.cu",
        FAMP_CSRC / "xqa/xqa_wrapper.cu",
        FAMP_CSRC / "flashinfer_xqa_binding.cu",
    ]

    # Ampere-only owned kernel: the sm90 (Hopper GMMA/TMA) and sm120 (MLA) paths are pruned from
    # the vendored csrc; USE_SM90_MHA is always 0 here. (Re-add mha_sm90.cu + the flag to widen.)
    flag_sm90_mha = ["-DUSE_SM90_MHA=0"]

    return gen_jit_spec(
        f"xqa_input_{filename_safe_dtype_map[input_dtype]}_kv_cache_{filename_safe_dtype_map[kv_cache_dtype]}_output_{filename_safe_dtype_map[output_dtype]}_page_size_{page_size}_head_dim_{head_dim}_head_group_ratio_{head_group_ratio}_use_sliding_window_{use_sliding_window}_use_spec_dec_{use_spec_dec}_spec_q_seq_len_{q_seq_len}",
        sources,
        extra_cuda_cflags=xqa_nvcc_flags
        + sm_nvcc_flags
        + flag_tokens_per_page
        + flag_head_dim
        + flag_input_dtype
        + flag_kv_cache_dtype
        + flag_head_group_ratio
        + flag_sliding_window
        + flag_low_prec_output
        + flag_spec_dec
        + flag_mla_wrapper
        + flag_sm90_mha,
        extra_ldflags=["-lcuda"],  # Add CUDA Driver API library
    )

