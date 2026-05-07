#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelNonDiagMask {
public:
    __aicore__ inline KernelNonDiagMask() {}
    __aicore__ inline void Init(GM_ADDR row, GM_ADDR col, GM_ADDR mask,
                                uint32_t start_tgt, uint32_t end_tgt,
                                uint32_t min_idx, uint32_t coreProcessLength,
                                uint32_t tileNum, uint32_t tileDataNum, uint32_t tailDataNum,
                                int64_t M, int64_t N, int64_t k, int64_t num_diag,
                                uint32_t totalLength)
    {
        this->start_tgt = start_tgt;
        this->end_tgt = end_tgt;
        this->min_idx = min_idx;
        this->tileNum = tileNum;
        this->tileDataNum = tileDataNum;
        this->tailDataNum = tailDataNum;
        this->M = M;
        this->N = N;
        this->k = k;
        this->num_diag = num_diag;
        this->totalLength = totalLength;

        // 【修复 1】：计算当前 Core 负责的输出长度
        // DataCopy 需要 32B 对齐
        this->align_32_len = (end_tgt - start_tgt + 31) / 32 * 32;
        // Duplicate 向量指令 (int8) 建议 256B 对齐
        this->align_256_len = (align_32_len + 255) / 256 * 256;

        rowGm.SetGlobalBuffer((__gm__ int64_t*)row + min_idx);
        colGm.SetGlobalBuffer((__gm__ int64_t*)col + min_idx);
        // maskGm 偏移到当前 Core 的起始位置
        maskGm.SetGlobalBuffer((__gm__ int8_t*)mask + start_tgt);

        // 初始化 Pipe 缓冲区
        pipe.InitBuffer(inQueueRow, BUFFER_NUM, tileDataNum * sizeof(int64_t));
        pipe.InitBuffer(inQueueCol, BUFFER_NUM, tileDataNum * sizeof(int64_t));
        // 分配输出掩码的 Local Buffer
        pipe.InitBuffer(maskBuf, this->align_256_len * sizeof(int8_t));
    }

    __aicore__ inline void Process()
    {
        if (this->align_32_len == 0) return;

        // 【修复 2】：在 UB 中申请 Local Tensor，并使用向量指令清零
        LocalTensor<int8_t> maskLocal = maskBuf.Get<int8_t>();
        //Duplicate(maskLocal, (int8_t)0, this->align_256_len);
        // ====================== 核心修改：类型强转+Duplicate清零 ======================
        // 1. 将 int8_t 张量重新解释为 uint16_t（Duplicate支持的类型）
        auto maskLocalCast = maskLocal.ReinterpretCast<uint16_t>();
        // 2. 计算uint16_t元素个数（256字节对齐，必然整除，安全无余数）
        uint32_t fillCount = this->align_256_len / sizeof(uint16_t);
        // 3. 调用支持的向量指令清零（全0内存布局，int8_t/uint16_t完全一致）
        Duplicate(maskLocalCast, (uint16_t)0, fillCount);
        // ==========================================================================

        for (uint32_t i = 0; i < this->tileNum; i++) {
            uint32_t curProcessDataNum = (i == this->tileNum - 1) ? this->tailDataNum : this->tileDataNum;
            CopyIn(i, curProcessDataNum);
            // 将 maskLocal 传给 Compute
            Compute(i, curProcessDataNum, maskLocal);
        }

        // 【修复 3】：全部计算完毕后，通过 VEC->GM 的搬运引擎一次性写回
        // 这里必须用 align_32_len，防止 256B 尾部对齐写穿界覆盖到下一个 Core 的数据！
        DataCopy(maskGm, maskLocal, this->align_32_len);
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t curProcessDataNum)
    {
        LocalTensor<int64_t> rowLocal = inQueueRow.AllocTensor<int64_t>();
        LocalTensor<int64_t> colLocal = inQueueCol.AllocTensor<int64_t>();
        
        uint32_t alignedDataNum = (curProcessDataNum + 3) / 4 * 4;

        DataCopy(rowLocal, rowGm[progress * this->tileDataNum], alignedDataNum);
        DataCopy(colLocal, colGm[progress * this->tileDataNum], alignedDataNum);
        
        inQueueRow.EnQue(rowLocal);
        inQueueCol.EnQue(colLocal);
    }

    __aicore__ inline void Compute(uint32_t progress, uint32_t curProcessDataNum, LocalTensor<int8_t>& maskLocal)
    {
        LocalTensor<int64_t> rowLocal = inQueueRow.DeQue<int64_t>();
        LocalTensor<int64_t> colLocal = inQueueCol.DeQue<int64_t>();

        // 维持同步屏障，确保 VECIN (MTE2) 数据被 Scalar 读取前就绪
        PipeBarrier<PIPE_ALL>();

        uint32_t base_idx = this->min_idx + progress * this->tileDataNum;
        
        for (uint32_t i = 0; i < curProcessDataNum; i++) {
            uint32_t idx = base_idx + i;
            if (idx >= this->totalLength) break;

            int64_t r = rowLocal.GetValue(i);
            int64_t c = colLocal.GetValue(i);
            int64_t target_idx_long = -1; // 初始化为一个无效索引

            if (this->k < 0) {
                if (r + this->k < 0) target_idx_long = (int64_t)idx;
                else if (r + this->k >= this->N) target_idx_long = (int64_t)idx + this->num_diag;
                else if (r + this->k > c) target_idx_long = (int64_t)idx + (r + this->k);
                else if (r + this->k < c) target_idx_long = (int64_t)idx + (r + this->k) + 1; // 显式判断 <
                // 如果 r + k == c，target_idx_long 保持为 -1
            } else {
                if (r + this->k >= this->N) target_idx_long = (int64_t)idx + this->num_diag;
                else if (r + this->k > c) target_idx_long = (int64_t)idx + r;
                else if (r + this->k < c) target_idx_long = (int64_t)idx + r + 1; // 显式判断 <
                // 如果 r + k == c，target_idx_long 保持为 -1
            }

            // 只有当计算出了有效的目标索引时，才进行写入
            if (target_idx_long != -1 && 
                target_idx_long >= (int64_t)this->start_tgt && 
                target_idx_long < (int64_t)this->end_tgt) {
                uint32_t local_offset = (uint32_t)(target_idx_long - this->start_tgt);
                maskLocal.SetValue(local_offset, (int8_t)1);
            }
        }

        inQueueRow.FreeTensor(rowLocal);
        inQueueCol.FreeTensor(colLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueRow, inQueueCol;
    TBuf<QuePosition::VECOUT> maskBuf; // 添加用于输出的 UB 内存池
    GlobalTensor<int64_t> rowGm, colGm;
    GlobalTensor<int8_t> maskGm;

    uint32_t start_tgt, end_tgt, min_idx;
    uint32_t tileNum, tileDataNum, tailDataNum;
    uint32_t align_32_len, align_256_len;
    int64_t M, N, k, num_diag;
    uint32_t totalLength;
};

extern "C" __global__ __aicore__ void non_diag_mask(GM_ADDR row, GM_ADDR col, GM_ADDR mask, GM_ADDR workspace, GM_ADDR tiling)
{
    // ... 此处的代码与你的原版完全一致，无需改动 ...
    GET_TILING_DATA(tiling_data, tiling);
    
    uint32_t blockIdx = GetBlockIdx();
    uint32_t coreNum = tiling_data.coreNum;
    uint32_t outTotalLen = tiling_data.outTotalLen;
    uint32_t totalLength = tiling_data.totalLength;
    uint32_t max_diff = tiling_data.max_diff;
    uint32_t tileDataNum = 4096;
    
    uint32_t outLengthPerCore = (outTotalLen + coreNum - 1) / coreNum;
    outLengthPerCore = (outLengthPerCore + 31) / 32 * 32;
    
    uint32_t start_tgt = blockIdx * outLengthPerCore;
    if (start_tgt >= outTotalLen) return;
    
    uint32_t end_tgt = start_tgt + outLengthPerCore;
    if (end_tgt > outTotalLen) end_tgt = outTotalLen;
    
    uint32_t min_idx = (start_tgt > max_diff) ? (start_tgt - max_diff) : 0;
    min_idx = (min_idx / 8) * 8;
    
    uint32_t max_idx = end_tgt + 32;
    if (max_idx > totalLength) max_idx = totalLength;

    if (min_idx >= max_idx) return;
    uint32_t coreProcessLength = max_idx - min_idx;
    
    uint32_t tileNum = (coreProcessLength + tileDataNum - 1) / tileDataNum;
    uint32_t tailDataNum = coreProcessLength % tileDataNum;
    if (tailDataNum == 0) tailDataNum = tileDataNum;
    
    KernelNonDiagMask op;
    op.Init(row, col, mask,
            start_tgt, end_tgt,
            min_idx, coreProcessLength,
            tileNum, tileDataNum, tailDataNum,
            tiling_data.M, tiling_data.N, tiling_data.k, tiling_data.num_diag, totalLength);
    op.Process();
}