#include "kernel_operator.h"

using namespace AscendC;

__aicore__ inline constexpr uint32_t ALIGN32(uint32_t x)
{
  const uint32_t a = 32;
  return ((x + a - 1) / a) * a;
}

template <typename T>
class KernelInd2PtrOneCore
{
private:
  TPipe pipe_;
  TBuf<QuePosition::VECCALC> count_buf_;
  LocalTensor<uint32_t> count_;
  GlobalTensor<T> ind_gm_;
  GlobalTensor<T> ptr_gm_;

  int64_t numel;
  int64_t M;

public:
  __aicore__ inline KernelInd2PtrOneCore() {}

  __aicore__ inline void Init(GM_ADDR indices, GM_ADDR ptr, int64_t numel, int64_t M)
  {
    this->numel = numel;
    this->M = M;
    ind_gm_.SetGlobalBuffer((__gm__ T *)indices, numel);
    ptr_gm_.SetGlobalBuffer((__gm__ T *)ptr, M + 1);
    // 申请一块本地buffer放 count（uint32），并做 32B 对齐
    pipe_.InitBuffer(count_buf_, ALIGN32(M * sizeof(uint32_t)));

    count_ = count_buf_.Get<uint32_t>();
  }

  __aicore__ inline void Process()
  {

    Duplicate<uint32_t>(count_, static_cast<uint32_t>(0), static_cast<int32_t>(M));
    // 2) 直方图统计：count[idx]++(表示同一个idx出现的次数)
    // （单核标量循环，后续再做向量化/多核优化）
    for (uint32_t i = 0; i < numel; ++i)
    {
      uint32_t v = ind_gm_(i); // indices 是 int64,这里的v是row的序号
      if (v >= 0 && v < M)
      {
        uint32_t oldvsize = count_(v);
        count_.SetValue(v, oldvsize + 1);
      }
    }

    // prefix sum -> ptr[0..M]
    int64_t run = 0;
    ptr_gm_.SetValue(0, 0);
    for (uint32_t row = 0; row < M; ++row)
    {
      run += static_cast<int64_t>(count_(row));
      ptr_gm_.SetValue(row + 1, run);
    }
    // 这里 run == numel（如果所有 indices 都在 [0,M)
  }
};

extern "C" __global__ __aicore__ void ind2ptr(GM_ADDR indices, GM_ADDR ptr, GM_ADDR workspace, GM_ADDR tiling)
{
  // 只用 1 个核：host 侧 SetBlockDim(1) 时，这里 block_idx 只有 0
  if (GetBlockIdx() != 0)
    return;
  GET_TILING_DATA(tiling_data, tiling);
  int64_t M = tiling_data.M;
  int64_t numel = tiling_data.indsize;
  KernelInd2PtrOneCore<int64_t> op;
  op.Init(indices, ptr, numel, M);
  op.Process();
}
