#include "extensions.h"

inline bool is_npu(const at::Tensor& tensor)
{
#ifdef COMPILE_WITH_XLA
    return tensor.device().type() == at::kXLA;
#else
    return tensor.device().type() == at::kPrivateUse1;
#endif
}
#define TORCH_CHECK_NPU(tensor) TORCH_CHECK(is_npu(tensor), #tensor " must be NPU tensor")