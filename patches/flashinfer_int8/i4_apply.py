#!/usr/bin/env python3
"""I-4a scale plumbing: expose per-token q_scale/k_scale to the int8 fa2 prefill kernels.

Apply AFTER i1_apply.py, BEFORE i4_compute_qk.py. Idempotent on a pristine package.

What it does (no numerics — pure plumbing):
  (P1) modules.py  gen_single_prefill_module / gen_batch_prefill_module fa2 branches:
       when dtype_q == torch.int8, APPEND maybe_q_scale, maybe_k_scale (float*) to
       additional_tensor_names/dtypes (AFTER the existing names -> stable positional order).
       generate_additional_params then auto-emits:
         Params decl:   float* maybe_q_scale; float* maybe_k_scale;
         func params:   , Optional<ffi::Tensor> maybe_q_scale, ... maybe_k_scale
         setter:        params.maybe_q_scale = maybe_q_scale ? ...data_ptr() : nullptr;  (nullptr-tolerant)
  (P2) prefill.py  fa2 run calls (single run / ragged_run / paged_run):
       when q.dtype == torch.int8, forward scale_q, scale_k into the positional C++ call
       right after the maybe_*_cache_sf tensors (== the position of the new additional tensors),
       before the logits_soft_cap scalar. f16/other dtypes: no extra args (module has no such
       params), so we pass them ONLY for int8.

The compute_qk side (reading params.maybe_q_scale[q_idx] * params.maybe_k_scale[kv_idx]) is in
i4_compute_qk.py; the SFINAE has_maybe_q_scale_field guard keeps f16 modules (which lack the
fields) compiling.
"""
import os, flashinfer
FI = os.path.dirname(flashinfer.__file__)


def patch_file(rel, edits, tag):
    p = os.path.join(FI, rel)
    s = open(p).read()
    orig = s
    for a, b, required in edits:
        if a not in s:
            if required:
                raise SystemExit(f"FAILED {tag}: anchor not found in {rel}:\n---\n{a[:200]}\n---")
            else:
                continue
        # idempotency: skip if already applied (b present, a absent of the marker)
        s = s.replace(a, b, 1)
    if s != orig:
        if not os.path.exists(p + ".i4orig"):
            open(p + ".i4orig", "w").write(orig)
        open(p, "w").write(s)
    print(f"  {tag}: {'applied' if s != orig else 'noop (already applied?)'}")


# =====================================================================================
# P1 — modules.py: append maybe_q_scale/maybe_k_scale to int8 fa2 additional tensors.
# =====================================================================================
# single_prefill fa2 branch. Anchor the exact list + dtypes, append a post-list int8 hook.
SINGLE_OLD = (
    '    if backend == "fa2":\n'
    '        assert not fp8_enabled, "fp8 tensor core is not supported in fa2 backend"\n'
    '        additional_tensor_names = [\n'
    '            "maybe_custom_mask",\n'
    '            "maybe_alibi_slopes",\n'
    '            "maybe_k_cache_sf",\n'
    '            "maybe_v_cache_sf",\n'
    '        ]\n'
    '        additional_tensor_dtypes = ["uint8_t", "float", "uint8_t", "uint8_t"]\n'
)
SINGLE_NEW = (
    '    if backend == "fa2":\n'
    '        assert not fp8_enabled, "fp8 tensor core is not supported in fa2 backend"\n'
    '        additional_tensor_names = [\n'
    '            "maybe_custom_mask",\n'
    '            "maybe_alibi_slopes",\n'
    '            "maybe_k_cache_sf",\n'
    '            "maybe_v_cache_sf",\n'
    '        ]\n'
    '        additional_tensor_dtypes = ["uint8_t", "float", "uint8_t", "uint8_t"]\n'
    '        if dtype_q == torch.int8:  # I-4a: per-token int8-QK dequant scales\n'
    '            # I-5 audit: the int8 IMMA QK read assumes the k128B smem swizzle. For\n'
    '            # head_dim_vo==64 the KV smem uses the k64B swizzle + a different token-row\n'
    '            # layout, which the int8 path does not implement -> would be silently wrong.\n'
    '            # Guard it loudly (head_dim 64 is not in any deployed int8-QK path: hd128/hd256).\n'
    '            assert head_dim_vo != 64, (\n'
    '                "int8-QK prefill does not support head_dim_vo==64 (k64B smem swizzle "\n'
    '                "unimplemented); use head_dim 128/256 or the f16 path."\n'
    '            )\n'
    '            additional_tensor_names += ["maybe_q_scale", "maybe_k_scale"]\n'
    '            additional_tensor_dtypes += ["float", "float"]\n'
)

# batch_prefill fa2 branch.
BATCH_OLD = (
    '        additional_tensor_dtypes = [\n'
    '            "uint8_t",\n'
    '            "int32_t",\n'
    '            "float",\n'
    '            "uint32_t",\n'
    '            "uint16_t",\n'
    '            "uint16_t",\n'
    '            "uint8_t",\n'
    '            "uint8_t",\n'
    '        ]  # NOTE(Zihao): int32_t should follow dtype_idx\n'
)
BATCH_NEW = (
    '        additional_tensor_dtypes = [\n'
    '            "uint8_t",\n'
    '            "int32_t",\n'
    '            "float",\n'
    '            "uint32_t",\n'
    '            "uint16_t",\n'
    '            "uint16_t",\n'
    '            "uint8_t",\n'
    '            "uint8_t",\n'
    '        ]  # NOTE(Zihao): int32_t should follow dtype_idx\n'
    '        if dtype_q == torch.int8:  # I-4a: per-token int8-QK dequant scales\n'
    '            # I-5 audit: head_dim_vo==64 uses the k64B KV smem swizzle (unimplemented in the\n'
    '            # int8 IMMA read) -> guard loudly. Deployed int8-QK paths are hd128/hd256 only.\n'
    '            assert head_dim_vo != 64, (\n'
    '                "int8-QK prefill does not support head_dim_vo==64 (k64B smem swizzle "\n'
    '                "unimplemented); use head_dim 128/256 or the f16 path."\n'
    '            )\n'
    '            additional_tensor_names += ["maybe_q_scale", "maybe_k_scale"]\n'
    '            additional_tensor_dtypes += ["float", "float"]\n'
    '            # I-5: per-request k-scale offset (kv-TOKEN prefix sum) for MULTI-request batch.\n'
    '            additional_tensor_names += ["maybe_kv_scale_indptr"]\n'
    '            additional_tensor_dtypes += ["int32_t"]\n'
)

patch_file("jit/attention/modules.py",
           [(SINGLE_OLD, SINGLE_NEW, True), (BATCH_OLD, BATCH_NEW, True)],
           "P1 modules.py int8 scale tensors")

# =====================================================================================
# P2 — prefill.py: forward scale_q/scale_k into the fa2 C++ run calls for int8 q.
# =====================================================================================
# (a) single run_single_prefill fa2 else-branch.
SP_OLD = (
    "        else:\n"
    "            run_func(\n"
    "                q,\n"
    "                k,\n"
    "                v,\n"
    "                tmp,\n"
    "                o,\n"
    "                maybe_lse,\n"
    "                mask_mode,\n"
    "                layout,\n"
    "                window_left,\n"
    "                maybe_packed_custom_mask,\n"
    "                maybe_alibi_slopes,\n"
    "                maybe_k_cache_sf,\n"
    "                maybe_v_cache_sf,\n"
    "                logits_soft_cap,\n"
    "                sm_scale,\n"
    "                1.0 / rope_scale,  # rope_rcp_scale\n"
    "                1.0 / rope_theta,  # rope_rcp_theta\n"
    "            )\n"
)
SP_NEW = (
    "        else:\n"
    "            if q.dtype == torch.int8:  # I-4a: per-token int8-QK dequant scales\n"
    "                run_func(\n"
    "                    q,\n"
    "                    k,\n"
    "                    v,\n"
    "                    tmp,\n"
    "                    o,\n"
    "                    maybe_lse,\n"
    "                    mask_mode,\n"
    "                    layout,\n"
    "                    window_left,\n"
    "                    maybe_packed_custom_mask,\n"
    "                    maybe_alibi_slopes,\n"
    "                    maybe_k_cache_sf,\n"
    "                    maybe_v_cache_sf,\n"
    "                    scale_q,\n"
    "                    scale_k,\n"
    "                    logits_soft_cap,\n"
    "                    sm_scale,\n"
    "                    1.0 / rope_scale,  # rope_rcp_scale\n"
    "                    1.0 / rope_theta,  # rope_rcp_theta\n"
    "                )\n"
    "            else:\n"
    "                run_func(\n"
    "                    q,\n"
    "                    k,\n"
    "                    v,\n"
    "                    tmp,\n"
    "                    o,\n"
    "                    maybe_lse,\n"
    "                    mask_mode,\n"
    "                    layout,\n"
    "                    window_left,\n"
    "                    maybe_packed_custom_mask,\n"
    "                    maybe_alibi_slopes,\n"
    "                    maybe_k_cache_sf,\n"
    "                    maybe_v_cache_sf,\n"
    "                    logits_soft_cap,\n"
    "                    sm_scale,\n"
    "                    1.0 / rope_scale,  # rope_rcp_scale\n"
    "                    1.0 / rope_theta,  # rope_rcp_theta\n"
    "                )\n"
)

# (b) ragged_run fa2 branch.
RG_OLD = (
    '        if backend == "fa2":\n'
    "            ragged_run_func(\n"
    "                float_workspace_buffer,\n"
    "                int_workspace_buffer,\n"
    "                plan_info_vec,\n"
    "                q,\n"
    "                k,\n"
    "                v,\n"
    "                qo_indptr,\n"
    "                kv_indptr,\n"
    "                o,\n"
    "                maybe_lse,\n"
    "                mask_mode,\n"
    "                layout,\n"
    "                window_left,\n"
    "                enable_pdl,\n"
    "                maybe_custom_mask,\n"
    "                maybe_mask_indptr,\n"
    "                maybe_alibi_slopes,\n"
    "                maybe_prefix_len_ptr,\n"
    "                maybe_token_pos_in_items_ptr,\n"
    "                maybe_max_item_len_ptr,\n"
    "                maybe_k_cache_sf,\n"
    "                maybe_v_cache_sf,\n"
    "                logits_soft_cap,\n"
    "                sm_scale,\n"
    "                1.0 / rope_scale,  # rope_rcp_scale\n"
    "                1.0 / rope_theta,  # rope_rcp_theta,\n"
    "                token_pos_in_items_len,\n"
    "            )\n"
)
RG_NEW = (
    '        if backend == "fa2":\n'
    "            if q.dtype == torch.int8:  # I-4a: per-token int8-QK dequant scales\n"
    "                ragged_run_func(\n"
    "                    float_workspace_buffer,\n"
    "                    int_workspace_buffer,\n"
    "                    plan_info_vec,\n"
    "                    q,\n"
    "                    k,\n"
    "                    v,\n"
    "                    qo_indptr,\n"
    "                    kv_indptr,\n"
    "                    o,\n"
    "                    maybe_lse,\n"
    "                    mask_mode,\n"
    "                    layout,\n"
    "                    window_left,\n"
    "                    enable_pdl,\n"
    "                    maybe_custom_mask,\n"
    "                    maybe_mask_indptr,\n"
    "                    maybe_alibi_slopes,\n"
    "                    maybe_prefix_len_ptr,\n"
    "                    maybe_token_pos_in_items_ptr,\n"
    "                    maybe_max_item_len_ptr,\n"
    "                    maybe_k_cache_sf,\n"
    "                    maybe_v_cache_sf,\n"
    "                    scale_q,\n"
    "                    scale_k,\n"
    "                    scale_kv_indptr,\n"
    "                    logits_soft_cap,\n"
    "                    sm_scale,\n"
    "                    1.0 / rope_scale,  # rope_rcp_scale\n"
    "                    1.0 / rope_theta,  # rope_rcp_theta,\n"
    "                    token_pos_in_items_len,\n"
    "                )\n"
    "            else:\n"
    "                ragged_run_func(\n"
    "                    float_workspace_buffer,\n"
    "                    int_workspace_buffer,\n"
    "                    plan_info_vec,\n"
    "                    q,\n"
    "                    k,\n"
    "                    v,\n"
    "                    qo_indptr,\n"
    "                    kv_indptr,\n"
    "                    o,\n"
    "                    maybe_lse,\n"
    "                    mask_mode,\n"
    "                    layout,\n"
    "                    window_left,\n"
    "                    enable_pdl,\n"
    "                    maybe_custom_mask,\n"
    "                    maybe_mask_indptr,\n"
    "                    maybe_alibi_slopes,\n"
    "                    maybe_prefix_len_ptr,\n"
    "                    maybe_token_pos_in_items_ptr,\n"
    "                    maybe_max_item_len_ptr,\n"
    "                    maybe_k_cache_sf,\n"
    "                    maybe_v_cache_sf,\n"
    "                    logits_soft_cap,\n"
    "                    sm_scale,\n"
    "                    1.0 / rope_scale,  # rope_rcp_scale\n"
    "                    1.0 / rope_theta,  # rope_rcp_theta,\n"
    "                    token_pos_in_items_len,\n"
    "                )\n"
)

# (c) paged_run fa2 branch.
PG_OLD = (
    '        elif backend == "fa2":\n'
    "            assert not is_float8(q)\n"
    "            paged_run_func(\n"
    "                float_workspace_buffer,\n"
    "                int_workspace_buffer,\n"
    "                plan_info_vec,\n"
    "                q,\n"
    "                paged_k_cache,\n"
    "                paged_v_cache,\n"
    "                qo_indptr,\n"
    "                paged_kv_indptr,\n"
    "                paged_kv_indices,\n"
    "                paged_kv_last_page_len,\n"
    "                o,\n"
    "                maybe_lse,\n"
    "                mask_mode,\n"
    "                layout,\n"
    "                window_left,\n"
    "                enable_pdl,\n"
    "                maybe_custom_mask,\n"
    "                maybe_mask_indptr,\n"
    "                maybe_alibi_slopes,\n"
    "                maybe_prefix_len_ptr,\n"
    "                maybe_token_pos_in_items_ptr,\n"
    "                maybe_max_item_len_ptr,\n"
    "                key_block_scales,\n"
    "                value_block_scales,\n"
    "                logits_soft_cap,\n"
    "                sm_scale,\n"
    "                1.0 / rope_scale,  # rope_rcp_scale\n"
    "                1.0 / rope_theta,  # rope_rcp_theta\n"
    "                token_pos_in_items_len,\n"
    "            )\n"
)
PG_NEW = (
    '        elif backend == "fa2":\n'
    "            assert not is_float8(q)\n"
    "            if q.dtype == torch.int8:  # I-4a: per-token int8-QK dequant scales\n"
    "                paged_run_func(\n"
    "                    float_workspace_buffer,\n"
    "                    int_workspace_buffer,\n"
    "                    plan_info_vec,\n"
    "                    q,\n"
    "                    paged_k_cache,\n"
    "                    paged_v_cache,\n"
    "                    qo_indptr,\n"
    "                    paged_kv_indptr,\n"
    "                    paged_kv_indices,\n"
    "                    paged_kv_last_page_len,\n"
    "                    o,\n"
    "                    maybe_lse,\n"
    "                    mask_mode,\n"
    "                    layout,\n"
    "                    window_left,\n"
    "                    enable_pdl,\n"
    "                    maybe_custom_mask,\n"
    "                    maybe_mask_indptr,\n"
    "                    maybe_alibi_slopes,\n"
    "                    maybe_prefix_len_ptr,\n"
    "                    maybe_token_pos_in_items_ptr,\n"
    "                    maybe_max_item_len_ptr,\n"
    "                    key_block_scales,\n"
    "                    value_block_scales,\n"
    "                    scale_q,\n"
    "                    scale_k,\n"
    "                    scale_kv_indptr,\n"
    "                    logits_soft_cap,\n"
    "                    sm_scale,\n"
    "                    1.0 / rope_scale,  # rope_rcp_scale\n"
    "                    1.0 / rope_theta,  # rope_rcp_theta\n"
    "                    token_pos_in_items_len,\n"
    "                )\n"
    "            else:\n"
    "                paged_run_func(\n"
    "                    float_workspace_buffer,\n"
    "                    int_workspace_buffer,\n"
    "                    plan_info_vec,\n"
    "                    q,\n"
    "                    paged_k_cache,\n"
    "                    paged_v_cache,\n"
    "                    qo_indptr,\n"
    "                    paged_kv_indptr,\n"
    "                    paged_kv_indices,\n"
    "                    paged_kv_last_page_len,\n"
    "                    o,\n"
    "                    maybe_lse,\n"
    "                    mask_mode,\n"
    "                    layout,\n"
    "                    window_left,\n"
    "                    enable_pdl,\n"
    "                    maybe_custom_mask,\n"
    "                    maybe_mask_indptr,\n"
    "                    maybe_alibi_slopes,\n"
    "                    maybe_prefix_len_ptr,\n"
    "                    maybe_token_pos_in_items_ptr,\n"
    "                    maybe_max_item_len_ptr,\n"
    "                    key_block_scales,\n"
    "                    value_block_scales,\n"
    "                    logits_soft_cap,\n"
    "                    sm_scale,\n"
    "                    1.0 / rope_scale,  # rope_rcp_scale\n"
    "                    1.0 / rope_theta,  # rope_rcp_theta\n"
    "                    token_pos_in_items_len,\n"
    "                )\n"
)

patch_file("prefill.py",
           [(SP_OLD, SP_NEW, True), (RG_OLD, RG_NEW, True), (PG_OLD, PG_NEW, True)],
           "P2 prefill.py fa2 int8 scale forwarding")

# (d) I-3: BatchPrefillWithPagedKVCacheWrapper.run — let int8 q extract per-token scale_q/scale_k
#     from *args (q passes scales positionally: run(q, kv, scale_q_tensor, scale_k_tensor)).
#     The fp8 path already does this for float8; mirror it for int8 so paged_run gets the tensors.
WR_OLD = (
    "                # Extract FP8 scale tensors from *args if q is FP8\n"
    "                fp8_scale_q = None\n"
    "                fp8_scale_k = None\n"
    "                fp8_scale_v = None\n"
    "                if is_float8(q) and len(args) >= 3:\n"
    "                    fp8_scale_q = args[0]\n"
    "                    fp8_scale_k = args[1]\n"
    "                    fp8_scale_v = args[2]\n"
)
WR_NEW = (
    "                # Extract FP8 scale tensors from *args if q is FP8\n"
    "                fp8_scale_q = None\n"
    "                fp8_scale_k = None\n"
    "                fp8_scale_v = None\n"
    "                if is_float8(q) and len(args) >= 3:\n"
    "                    fp8_scale_q = args[0]\n"
    "                    fp8_scale_k = args[1]\n"
    "                    fp8_scale_v = args[2]\n"
    "                elif q.dtype == torch.int8 and len(args) >= 2:  # I-3: per-token int8-QK scales\n"
    "                    fp8_scale_q = args[0]\n"
    "                    fp8_scale_k = args[1]\n"
)
patch_file("prefill.py", [(WR_OLD, WR_NEW, True)], "P3 paged wrapper int8 scale extract")

# (e) I-3: BatchPrefillWithRaggedKVCacheWrapper.run — extend run_args with per-token scale
#     tensors for int8 q (the fp8 path already does run_args.extend(args)).
RWR_OLD = (
    "            # For FP8, append scale tensors\n"
    "            if is_float8(q):\n"
    "                run_args.extend(list(args))  # scale_q, scale_k, scale_v\n"
)
RWR_NEW = (
    "            # For FP8, append scale tensors\n"
    "            if is_float8(q):\n"
    "                run_args.extend(list(args))  # scale_q, scale_k, scale_v\n"
    "            elif q.dtype == torch.int8:  # I-3: per-token int8-QK scales (scale_q, scale_k)\n"
    "                run_args.extend(list(args))\n"
)
patch_file("prefill.py", [(RWR_OLD, RWR_NEW, True)], "P4 ragged wrapper int8 scale extend")

# (f) I-3: ragged wrapper out-dtype fix. Stock forces bf16 output for ALL 1-byte q (fp8 path),
#     but our int8-QK module is compiled with o_data_type=float16 (DTypeO=half). Writing half
#     into a bf16 buffer = bit-reinterpret = garbage. For int8 q, honor _cached_o_data_type.
ODT_OLD = (
    "        if out is None:\n"
    "            # when input dtype is fp8, we need to use bf16 output\n"
    "            out_dtype = torch.bfloat16 if q.dtype.itemsize == 1 else q.dtype\n"
)
ODT_NEW = (
    "        if out is None:\n"
    "            # when input dtype is fp8, we need to use bf16 output\n"
    "            # I-3: int8-QK module is compiled with o_data_type=float16 -> honor it (not bf16).\n"
    "            if q.dtype == torch.int8:\n"
    "                out_dtype = self._cached_o_data_type or torch.float16\n"
    "            else:\n"
    "                out_dtype = torch.bfloat16 if q.dtype.itemsize == 1 else q.dtype\n"
)
patch_file("prefill.py", [(ODT_OLD, ODT_NEW, True)], "P5 ragged wrapper int8 out-dtype")

# =====================================================================================
# I-5 — per-request k-scale offset for MULTI-request batched int8 prefill.
#
# The kernel reads maybe_k_scale offset by maybe_kv_scale_indptr[request_idx] (the kv-TOKEN
# prefix sum), exactly mirroring the q-scale offset by q_indptr[request_idx]. modules.py (above)
# already appended the int32_t* maybe_kv_scale_indptr additional tensor to int8 BATCH modules,
# and i4_compute_qk.py's get_k_dequant_scale already does the offset (nullptr-tolerant). Here we
# thread one optional tensor `scale_kv_indptr` from each batch wrapper's run() to the C++ op:
#   (P6) add `scale_kv_indptr=None` trailing param to paged_run / ragged_run (both occurrences:
#        get_batch_prefill_module + get_batch_prefill_jit_module). The int8 C++ forwarding (P2)
#        already passes `scale_kv_indptr` right after scale_k (so it maps to maybe_kv_scale_indptr).
#   (P8) paged wrapper run(): derive the kv-TOKEN prefix sum from the planned page indptr +
#        last_page_len + page_size, pass scale_kv_indptr=... by KEYWORD (register_custom_op is a
#        no-op passthrough here, so kwargs work and don't perturb the positional run_args).
#   (P9) ragged wrapper run(): self._kv_indptr_buf IS the kv-token prefix sum -> pass it directly.
# For single-request / non-int8 callers scale_kv_indptr stays None -> nullptr -> offset 0 (the
# prior behavior is bit-identical), so this is purely additive.
# =====================================================================================

# (P6) trailing `scale_kv_indptr` param on paged_run + ragged_run signatures (2 defs each).
P6_PAGED_SIG_OLD = (
    "        skip_softmax_threshold_scale_factor: Optional[float] = None,\n"
    "        uses_shared_paged_kv_idx: bool = True,\n"
    "    ) -> None:\n"
)
P6_PAGED_SIG_NEW = (
    "        skip_softmax_threshold_scale_factor: Optional[float] = None,\n"
    "        uses_shared_paged_kv_idx: bool = True,\n"
    "        scale_kv_indptr: Optional[torch.Tensor] = None,  # I-5: per-request k-scale offset\n"
    "    ) -> None:\n"
)
P6_RAGGED_SIG_OLD = (
    "        scale_q: Optional[torch.Tensor] = None,\n"
    "        scale_k: Optional[torch.Tensor] = None,\n"
    "        scale_v: Optional[torch.Tensor] = None,\n"
    "    ) -> None:\n"
)
P6_RAGGED_SIG_NEW = (
    "        scale_q: Optional[torch.Tensor] = None,\n"
    "        scale_k: Optional[torch.Tensor] = None,\n"
    "        scale_v: Optional[torch.Tensor] = None,\n"
    "        scale_kv_indptr: Optional[torch.Tensor] = None,  # I-5: per-request k-scale offset\n"
    "    ) -> None:\n"
)
# replace_all: there are 2 paged_run defs (module + jit_module) and 2 ragged_run defs; the jit_module
# variant's int8 C++ call references scale_kv_indptr too -> needs the param. patch_file does .replace
# once per call, so call patch with the same edit twice via count guard.
def patch_all(rel, a, b, tag):
    p = os.path.join(FI, rel); s = open(p).read(); orig = s
    n = s.count(a)
    if n == 0:
        raise SystemExit(f"FAILED {tag}: anchor not found in {rel}")
    s = s.replace(a, b)
    if s != orig:
        if not os.path.exists(p + ".i4orig"):
            open(p + ".i4orig", "w").write(orig)
        open(p, "w").write(s)
    print(f"  {tag}: {n} site(s)")

patch_all("prefill.py", P6_PAGED_SIG_OLD, P6_PAGED_SIG_NEW, "P6 paged_run scale_kv_indptr param")
patch_all("prefill.py", P6_RAGGED_SIG_OLD, P6_RAGGED_SIG_NEW, "P6 ragged_run scale_kv_indptr param")

# (P8) paged wrapper run(): compute kv-token prefix sum + pass scale_kv_indptr by keyword.
P8_OLD = (
    '            assert self._cached_module is not None, "cached module is not initialized"\n'
    "            self._cached_module.paged_run(*run_args)\n"
)
P8_NEW = (
    '            assert self._cached_module is not None, "cached module is not initialized"\n'
    "            # I-5: per-request k-scale offset (kv-TOKEN prefix sum) for multi-request int8 batch.\n"
    "            # Only the standard get_batch_prefill_module path declares scale_kv_indptr; the\n"
    "            # user-supplied _jit_module passthrough does not, so guard the keyword on it.\n"
    "            if self._jit_module is None:\n"
    "                paged_scale_kv_indptr = None\n"
    "                if q.dtype == torch.int8 and self._batch_size is not None and self._batch_size > 1:\n"
    "                    _pi = self._paged_kv_indptr_buf[: self._batch_size + 1].to(torch.int64)\n"
    "                    _lpl = self._paged_kv_last_page_len_buf[: self._batch_size].to(torch.int64)\n"
    "                    _pages = _pi[1:] - _pi[:-1]\n"
    "                    _kvlen = (_pages - 1) * page_size + _lpl\n"
    "                    paged_scale_kv_indptr = torch.zeros(\n"
    "                        self._batch_size + 1, dtype=torch.int32, device=q.device\n"
    "                    )\n"
    "                    paged_scale_kv_indptr[1:] = torch.cumsum(_kvlen, 0).to(torch.int32)\n"
    "                self._cached_module.paged_run(*run_args, scale_kv_indptr=paged_scale_kv_indptr)\n"
    "            else:\n"
    "                self._cached_module.paged_run(*run_args)\n"
)
patch_file("prefill.py", [(P8_OLD, P8_NEW, True)], "P8 paged wrapper kv_scale_indptr")

# (P9) ragged wrapper run(): self._kv_indptr_buf is already the kv-token prefix sum.
P9_OLD = (
    '        assert self._cached_module is not None, "cached module is not initialized"\n'
    "        self._cached_module.ragged_run(*run_args)\n"
)
P9_NEW = (
    '        assert self._cached_module is not None, "cached module is not initialized"\n'
    "        # I-5: per-request k-scale offset. For ragged, kv_indptr IS the kv-token prefix sum.\n"
    "        # Guard the keyword on the standard module path (user _jit_module lacks the param).\n"
    "        if self._jit_module is None:\n"
    "            ragged_scale_kv_indptr = (\n"
    "                self._kv_indptr_buf if q.dtype == torch.int8 else None\n"
    "            )\n"
    "            self._cached_module.ragged_run(*run_args, scale_kv_indptr=ragged_scale_kv_indptr)\n"
    "        else:\n"
    "            self._cached_module.ragged_run(*run_args)\n"
)
patch_file("prefill.py", [(P9_OLD, P9_NEW, True)], "P9 ragged wrapper kv_scale_indptr")

print("I4_APPLY_DONE")
