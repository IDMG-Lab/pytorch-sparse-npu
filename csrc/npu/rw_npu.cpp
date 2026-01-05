#include "rw_npu.h"
#include "utils.h"
#include "../npu_utils.h"

torch::Tensor random_walk_npu(torch::Tensor rowptr, torch::Tensor col,
                               torch::Tensor start, int64_t walk_length) {
  TORCH_CHECK_NPU(rowptr);
  TORCH_CHECK_NPU(col);
  TORCH_CHECK_NPU(start);

  CHECK_INPUT(rowptr.dim() == 1);
  CHECK_INPUT(col.dim() == 1);
  CHECK_INPUT(start.dim() == 1);

  auto rand = torch::rand({walk_length, start.size(0)},
                          start.options().dtype(torch::kFloat));
  auto out = torch::full({walk_length + 1, start.size(0)}, -1, start.options());

  auto num_starts = start.numel();
  EXEC_NPU_CMD(aclnnUniformRandomWalk, rowptr, col, start, rand, walk_length, num_starts, out);

  return out.t().contiguous();
}
