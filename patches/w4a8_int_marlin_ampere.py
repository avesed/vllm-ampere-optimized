"""Enable vLLM's W4A8-INT8 (int4 weight + int8 dynamic-per-token activation) path on
Ampere (RTX 3090 / A100, sm_8x) by routing it to the **Marlin** kernel.

Background
----------
vLLM ships two dedicated W4A8 kernels — CutlassW4A8LinearKernel and MacheteLinearKernel —
both of which require compute capability 9.0 (Hopper). Machete uses `wgmma`, an sm_90-only
instruction, so it cannot be recompiled for Ampere. The **Marlin** kernel, however, already
supports an 8-bit activation path on its CUDA side (`is_a_8bit`); it was only gated off at
the Python layer by a config bug: `create_weights` passed `act_type=params_dtype` (bf16)
instead of `torch.int8`, so Marlin never entered the int8 branch and the layer fell through
to weight-only WNA16 -> "Failed to find a kernel that can implement the WNA16 linear layer".

This script applies the fix (upstream vllm#38064 / PR #38066) as 3 pure-Python edits — no
CUDA recompile. It auto-detects the installed vLLM, backs each file up to `.bak`, and is
idempotent (re-running on an already-patched tree is a no-op).

Usage
-----
    python w4a8_int_marlin_ampere.py                 # auto-detect installed vllm
    python w4a8_int_marlin_ampere.py /path/to/site-packages/vllm
    python w4a8_int_marlin_ampere.py --revert        # restore the .bak files

Verify after patching: load a W4A8 (`int-quantized`, int4 wt + int8 dynamic act) model and
look for `Using MarlinLinearKernel for CompressedTensorsW4A8Int` in the vLLM log.
"""
import os
import sys


def find_vllm():
    """Locate the installed vllm package dir without importing torch/cuda."""
    import importlib.util
    spec = importlib.util.find_spec("vllm")
    if spec and spec.origin:
        return os.path.dirname(spec.origin)
    raise SystemExit("[FAIL] could not locate an installed `vllm`; pass its path explicitly")


def patch(path, old, new, label, sentinel):
    with open(path, encoding="utf-8") as f:
        s = f.read()
    if sentinel in s:
        print(f"[skip] {label}: already patched")
        return
    if old not in s:
        raise SystemExit(f"[FAIL] {label}: anchor string not found (vLLM version skew) in {path}")
    if s.count(old) != 1:
        raise SystemExit(f"[FAIL] {label}: {s.count(old)} matches of anchor, need exactly 1")
    bak = path + ".bak"
    if not os.path.exists(bak):
        with open(bak, "w", encoding="utf-8") as f:
            f.write(s)
    with open(path, "w", encoding="utf-8") as f:
        f.write(s.replace(old, new))
    print(f"[ok]   {label}")


def revert(files):
    for p in files:
        bak = p + ".bak"
        if os.path.exists(bak):
            with open(bak, encoding="utf-8") as f:
                s = f.read()
            with open(p, "w", encoding="utf-8") as f:
                f.write(s)
            os.remove(bak)
            print(f"[revert] {os.path.basename(p)}")
        else:
            print(f"[revert] no .bak for {os.path.basename(p)} (skipped)")


def main():
    argv = [a for a in sys.argv[1:] if a]
    do_revert = "--revert" in argv
    argv = [a for a in argv if a != "--revert"]
    vllm_dir = argv[0] if argv else find_vllm()
    if not os.path.isdir(vllm_dir):
        raise SystemExit(f"[FAIL] not a directory: {vllm_dir}")

    SCH = os.path.join(vllm_dir, "model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_int.py")
    MP = os.path.join(vllm_dir, "model_executor/kernels/linear/mixed_precision/marlin.py")
    MU = os.path.join(vllm_dir, "model_executor/layers/quantization/utils/marlin_utils.py")
    for p in (SCH, MP, MU):
        if not os.path.exists(p):
            raise SystemExit(f"[FAIL] expected file missing (vLLM layout changed?): {p}")

    print(f"[info] vllm at {vllm_dir}")
    if do_revert:
        revert([SCH, MP, MU])
        print("REVERT_DONE")
        return

    # 1) scheme: act_type bf16 -> int8 (the #38064 root cause). Also normalize group_size=-1
    #    (per-channel) so the Marlin path gets a valid group size.
    patch(SCH,
"""            weight_type=self.quant_type,
            act_type=params_dtype,
            group_size=effective_group_size,""",
"""            weight_type=self.quant_type,
            act_type=torch.int8,
            group_size=-1 if self.group_size == -1 else effective_group_size,""",
          "scheme: act_type=torch.int8", "act_type=torch.int8")

    # 2a) marlin: allow signed int4 weight in the 8-bit-activation assert
    patch(MP,
'''        if is_a_8bit:
            assert c.weight_type == scalar_types.uint4b8, (
                "W8A8 is not supported by marlin kernel."
            )''',
'''        if is_a_8bit:
            assert c.weight_type in (scalar_types.uint4b8, scalar_types.int4), (
                "W4A8-INT8 marlin only supports uint4b8 or int4 weights."
            )''',
          "marlin: int4 allowed in 8-bit-act assert", "W4A8-INT8 marlin only supports")

    # 2b) marlin: pack signed int4 -> uint4b8 layout (add 8, 8 nibbles per int32) before repack
    patch(MP,
'''        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)''',
'''        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            if c.weight_type == scalar_types.int4:
                w = x.data
                assert w.shape[1] % 8 == 0, (
                    f"int4 marlin: in dim {w.shape[1]} must be a multiple of 8"
                )
                w_u4 = (w.to(torch.int32) + 8) & 0xF
                w_u4 = w_u4.reshape(w.shape[0], w.shape[1] // 8, 8)
                shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=w.device)
                packed = (w_u4 << shifts[None, None, :]).sum(dim=2).to(torch.int32)
                x.data = packed.T.contiguous()
            else:
                permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)''',
          "marlin: pack signed int4 -> uint4b8", "int4 marlin: in dim")

    # 2c) marlin: pass the effective weight type (int4 -> uint4b8) to the kernel call
    patch(MP,
'''            workspace=self.workspace,
            wtype=c.weight_type,''',
'''            workspace=self.workspace,
            wtype=(
                scalar_types.uint4b8
                if c.weight_type == scalar_types.int4
                else c.weight_type
            ),''',
          "marlin: effective wtype uint4b8", "if c.weight_type == scalar_types.int4\n                else c.weight_type")

    # 3) marlin_utils: add int4 to the supported quant types list
    patch(MU,
"        res = [scalar_types.uint4b8, scalar_types.uint8b128]",
"        res = [scalar_types.uint4b8, scalar_types.uint8b128, scalar_types.int4]",
          "marlin_utils: int4 in supported types", "scalar_types.uint8b128, scalar_types.int4")

    print("PATCH_DONE")


if __name__ == "__main__":
    main()
