#include "convert_npu.h"
#include "utils.h"
#include "../npu_utils.h"

torch::Tensor ind2ptr_npu(torch::Tensor ind, int64_t M)
{
  TORCH_CHECK_NPU(ind);
  auto out = at::zeros({M + 1}, ind.options().dtype(at::kLong).device(ind.device()));
  if (ind.numel() == 0)
    return out.zero_();
  EXEC_NPU_CMD(aclnnInd2ptr, ind, M, out);
  return out;
}

torch::Tensor ptr2ind_npu(torch::Tensor ptr, int64_t E) {
  TORCH_CHECK_NPU(ptr);

  auto out = torch::empty({E}, ptr.options());
  auto ptr_data = ptr.data_ptr<int64_t>();
  auto out_data = out.data_ptr<int64_t>();
  auto ptr_size = (int64_t)ptr.numel() - 1;
  EXEC_NPU_CMD(aclnnPtr2ind, ptr, E,out);
  return out;
}
