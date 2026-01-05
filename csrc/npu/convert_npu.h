#pragma once

#include "../extensions.h"

torch::Tensor ind2ptr_npu(torch::Tensor ind, int64_t M);
torch::Tensor ptr2ind_npu(torch::Tensor ptr, int64_t E);
