"""Regenerate the W4A8-INT8 Marlin Ampere patch against a vLLM source checkout.

Applies the 5 edits IN-PLACE to a vLLM source tree, so you can then `git diff` to
produce patches/0001-marlin-w4a8-int8-ampere.patch. Run from anywhere:

    git clone --depth 1 --branch <tag> https://github.com/vllm-project/vllm.git vllm
    python regenerate.py vllm
    cd vllm && git diff > ../0001-marlin-w4a8-int8-ampere.patch

Exits non-zero (listing the misses) if an anchor no longer matches the tree — that
IS the version-skew signal; refresh the anchors against the new tag and rerun.
"""
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "vllm"
V = os.path.join(ROOT, "vllm")
SCH = os.path.join(V, "model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a8_int.py")
MP = os.path.join(V, "model_executor/kernels/linear/mixed_precision/marlin.py")
MU = os.path.join(V, "model_executor/layers/quantization/utils/marlin_utils.py")

EDITS = [
    (SCH,
"""            weight_type=self.quant_type,
            act_type=params_dtype,
            group_size=effective_group_size,""",
"""            weight_type=self.quant_type,
            act_type=torch.int8,
            group_size=-1 if self.group_size == -1 else effective_group_size,""",
     "scheme: act_type=torch.int8"),

    (MP,
'''        if is_a_8bit:
            assert c.weight_type == scalar_types.uint4b8, (
                "W8A8 is not supported by marlin kernel."
            )''',
'''        if is_a_8bit:
            assert c.weight_type in (scalar_types.uint4b8, scalar_types.int4), (
                "W4A8-INT8 marlin only supports uint4b8 or int4 weights."
            )''',
     "marlin: int4 allowed in 8-bit-act assert"),

    (MP,
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
     "marlin: pack signed int4 -> uint4b8"),

    (MP,
'''            workspace=self.workspace,
            wtype=c.weight_type,''',
'''            workspace=self.workspace,
            wtype=(
                scalar_types.uint4b8
                if c.weight_type == scalar_types.int4
                else c.weight_type
            ),''',
     "marlin: effective wtype uint4b8"),

    (MU,
"        res = [scalar_types.uint4b8, scalar_types.uint8b128]",
"        res = [scalar_types.uint4b8, scalar_types.uint8b128, scalar_types.int4]",
     "marlin_utils: int4 in supported types"),
]


def main():
    misses = []
    # group edits per file so multiple edits to the same file accumulate
    files = {}
    for path, old, new, label in EDITS:
        if not os.path.exists(path):
            misses.append(f"{label}: FILE MISSING {path}")
            continue
        files.setdefault(path, []).append((old, new, label))
    for path, edits in files.items():
        s = open(path, encoding="utf-8").read()
        for old, new, label in edits:
            n = s.count(old)
            if n != 1:
                misses.append(f"{label}: {n} matches (need 1) in {os.path.relpath(path, ROOT)}")
                continue
            s = s.replace(old, new)
            print(f"[ok]   {label}")
        open(path, "w", encoding="utf-8").write(s)
    if misses:
        print("\n[VERSION SKEW] anchors no longer match — refresh these:")
        for m in misses:
            print("  - " + m)
        sys.exit(1)
    print("PATCH_APPLIED_CLEAN")


if __name__ == "__main__":
    main()
