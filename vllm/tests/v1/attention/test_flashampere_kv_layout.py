# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for flashampere's paged-KV unpack (`_split_kv`). No GPU needed.

vLLM 0.26 packed K and V into the content dim of the page (upstream #44455): the cache is
(num_blocks, num_kv_heads, block_size, 2*head_size) and K/V come out as interleaved strided
views, not the contiguous slabs a leading-dim split used to give. Every famp kernel reads KV
through `_split_kv`, so these assert that one function against the shape the backend actually
publishes -- if upstream moves the layout again, this fails here instead of in a kernel.
"""

import torch

from vllm.v1.attention.backends.flashampere.backend import FlashAmpereBackend
from vllm.v1.attention.backends.flashampere.kernels import _split_kv

NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE = 3, 16, 2, 64


def _cache(num_kv_heads=NUM_KV_HEADS, head_size=HEAD_SIZE):
    shape = FlashAmpereBackend.get_kv_cache_shape(
        NUM_BLOCKS, BLOCK_SIZE, num_kv_heads, head_size
    )
    return torch.arange(
        int(torch.tensor(shape).prod()), dtype=torch.float32
    ).reshape(shape)


def test_split_kv_shape_follows_the_backend():
    k, v = _split_kv(_cache(), HEAD_SIZE)
    expected = (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    assert k.shape == v.shape == expected, (k.shape, v.shape, expected)


def test_head_size_dim_stays_contiguous():
    # The kernels index a whole head's data as one contiguous run; the other dims may stride.
    k, v = _split_kv(_cache(), HEAD_SIZE)
    assert k.stride(-1) == 1 and v.stride(-1) == 1, (k.stride(), v.stride())


def test_content_matches_an_explicit_unpack():
    kv = _cache()
    k, v = _split_kv(kv, HEAD_SIZE)
    # (B, H, N, 2D) -> K is the first head_size of the content dim, V the second.
    assert torch.equal(k, kv[..., :HEAD_SIZE].permute(0, 2, 1, 3))
    assert torch.equal(v, kv[..., HEAD_SIZE:].permute(0, 2, 1, 3))


def test_gather_by_block_then_flatten():
    # What _gather_one does: index_select the blocks, then flatten (block, token) -> token.
    k, _ = _split_kv(_cache(), HEAD_SIZE)
    blocks = torch.tensor([0, 2])
    flat = k.index_select(0, blocks).reshape(len(blocks) * BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    assert torch.equal(flat[:BLOCK_SIZE], k[0])
    assert torch.equal(flat[BLOCK_SIZE:], k[2])


def test_single_kv_head_strides_are_canonicalized():
    # num_kv_heads=1 under TP leaves a degenerate size-1 dim; famp mirrors upstream's fix.
    k, v = _split_kv(_cache(num_kv_heads=1), HEAD_SIZE)
    assert k.shape == v.shape == (NUM_BLOCKS, BLOCK_SIZE, 1, HEAD_SIZE)
    assert k.stride(-1) == 1 and v.stride(-1) == 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
