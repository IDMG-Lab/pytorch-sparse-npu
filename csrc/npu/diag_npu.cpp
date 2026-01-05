#include "diag_npu.h"
#include "utils.h"
#include "../npu_utils.h"

torch::Tensor non_diag_mask_npu(torch::Tensor row, torch::Tensor col,
                                 int64_t M, int64_t N, int64_t k) {
  TORCH_CHECK_NPU(row);
  TORCH_CHECK_NPU(col);

  auto E = row.size(0);
  auto num_diag = k < 0 ? std::min(M + k, N) : std::min(M, N - k);

  auto row_data = row.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();

  auto mask = torch::zeros({E + num_diag}, row.options().dtype(torch::kBool));
  auto mask_data = mask.data_ptr<bool>();

  if (E == 0)
    return mask;

  EXEC_NPU_CMD(aclnnNoneDiagMask, row, col, N, k, num_diag, E, mask);

  return mask;
}