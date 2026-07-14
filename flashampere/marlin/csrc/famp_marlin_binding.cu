// famp vendored Marlin — op SCHEMAS (m.def). The IMPLs come from the vendored marlin.cu /
// gptq_marlin_repack.cu / awq_marlin_repack.cu, which register under TORCH_LIBRARY_IMPL_EXPAND(
// TORCH_EXTENSION_NAME, CUDA, ...). When built as the cpp_extension named "famp_marlin",
// TORCH_EXTENSION_NAME == famp_marlin, so def+impl land in the same library -> torch.ops.famp_marlin.*
#include <torch/library.h>
#include "core/registration.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  m.def(
      "marlin_gemm(Tensor a, Tensor? c_or_none, Tensor b_q_weight, "
      "Tensor? b_bias_or_none, Tensor b_scales, "
      "Tensor? a_scales, Tensor? global_scale, Tensor? b_zeros_or_none, "
      "Tensor? g_idx_or_none, Tensor? perm_or_none, Tensor workspace, int b_type_id, "
      "SymInt size_m, SymInt size_n, SymInt size_k, bool is_k_full, "
      "bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) -> Tensor");
  m.def(
      "gptq_marlin_repack(Tensor b_q_weight, Tensor perm, "
      "SymInt size_k, SymInt size_n, int num_bits, bool is_a_8bit) -> Tensor");
  m.def(
      "awq_marlin_repack(Tensor b_q_weight, SymInt size_k, "
      "SymInt size_n, int num_bits, bool is_a_8bit) -> Tensor");
}
