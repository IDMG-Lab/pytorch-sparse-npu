#include "kernel_operator.h"


using namespace AscendC;


#define BLOCK 1024
template<typename T1, typename T2> class KernelSetAtomicAdd {
public:
    __aicore__ inline KernelSetAtomicAdd() {}
    __aicore__ inline void Init(GM_ADDR rowptr, GM_ADDR col, GM_ADDR value, GM_ADDR dense, GM_ADDR start, GM_ADDR y, size_t size, size_t len, size_t D)
    {
        rowptrGlobal.SetGlobalBuffer((__gm__ T1*)rowptr);
        colGlobal.SetGlobalBuffer((__gm__ T1*)col);
        startGlobal.SetGlobalBuffer((__gm__ T1*)start);
        valueGlobal.SetGlobalBuffer((__gm__ T2*)value);
        denseGlobal.SetGlobalBuffer((__gm__ T2*)dense);
        yGlobal.SetGlobalBuffer((__gm__ T2*)y);

        size_t L = GetBlockIdx() * len, R = (GetBlockIdx() + 1) * len;
        if (L > size) L = size;
        if (R > size) R = size;
        this->L = L; this->R = R; this->D = D;
        this->rowptr_idx = startGlobal.GetValue(GetBlockIdx());
        this->next_rowptr = rowptrGlobal.GetValue(this->rowptr_idx + 1);

    }
    __aicore__ inline void Process()
    {
        AscendC::SetAtomicAdd<T2>();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        size_t src_offset, dst_offset = this->rowptr_idx * this->D, d;
        if (this->D <= BLOCK)
        {
            for (size_t i = L; i < R; i++)
            {
                while (i == this->next_rowptr)
                {
                    this->rowptr_idx++;
                    dst_offset += this->D;
                    this->next_rowptr = rowptrGlobal.GetValue(this->rowptr_idx + 1);
                }
                T1 col = colGlobal.GetValue(i);
                T2 v = valueGlobal.GetValue(i);
                src_offset = col * this->D;
                // if (dst_offset == 114*this->D) printf("%d %d %d %d %f\n",GetBlockIdx(), col, this->next_rowptr, i, v);

                CopyOut(src_offset, dst_offset, this->D, v);
            }
        }
        else
        {
            for (size_t i = L; i < R; i++)
            {
                while (i == this->next_rowptr)
                {
                    this->rowptr_idx++;
                    dst_offset += this->D;
                    this->next_rowptr = rowptrGlobal.GetValue(this->rowptr_idx + 1);
                }
                T1 col = colGlobal.GetValue(i);
                T2 v = valueGlobal.GetValue(i);
                src_offset = col * this->D;

                for (d = 0; d + BLOCK < this->D; d += BLOCK)
                    CopyOut(src_offset + d, dst_offset + d, BLOCK, v);

                CopyOut(src_offset + d, dst_offset + d, this->D - d, v);
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        
        AscendC::SetAtomicNone();
    }
private:
    __aicore__ inline void CopyIn()
    {}
    __aicore__ inline void Compute()
    {}
    __aicore__ inline void CopyOut(size_t src_offset, size_t dst_offset, size_t len, T2 v)
    {

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);

        AscendC::DataCopy(this->src0Local, denseGlobal[src_offset], len);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        
        AscendC::Muls(this->src0Local, this->src0Local, v, len);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);

        AscendC::DataCopy(yGlobal[dst_offset], this->src0Local, len);
        
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
    }
private:
    size_t L, R, D, rowptr_idx, next_rowptr;
    AscendC::LocalTensor<T2> src0Local = AscendC::LocalTensor<T2>(AscendC::TPosition::VECCALC, 0, BLOCK);
    AscendC::GlobalTensor<T1> rowptrGlobal, colGlobal, startGlobal;
    AscendC::GlobalTensor<T2> valueGlobal, denseGlobal, yGlobal;
};
extern "C" __global__ __aicore__ void spmm_csr(GM_ADDR rowptr, GM_ADDR col, GM_ADDR value, GM_ADDR dense, GM_ADDR start, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    AscendC::InitSocState();
    // TODO: user kernel impl
    KernelSetAtomicAdd<DTYPE_COL, DTYPE_DENSE> op;
    op.Init(rowptr, col, value, dense, start, y, tiling_data.size, tiling_data.len, tiling_data.D);
    op.Process();
}