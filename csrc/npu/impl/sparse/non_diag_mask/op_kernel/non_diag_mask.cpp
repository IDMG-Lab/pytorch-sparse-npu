#include "kernel_operator.h"

using namespace AscendC;

// ============================================================
// 配置
// ============================================================

// 不再需要双缓冲。
// 当前实现 CopyIn -> Compute 是串行消费，双缓冲只会额外占 UB。
constexpr int32_t BUFFER_NUM = 1;

// 每次从 GM 搬入 4096 个 int64。
// 每个输入队列占 4096 * 8 = 32 KB。
// row + col 合计约 64 KB UB。
constexpr uint32_t TILE_DATA_NUM = 4096;

// Scalar 访问 GM 时以 64B CacheLine 为基本单位。
// 多核输出区间必须以 64B 为边界切分，避免两个 Core
// 修改同一个 CacheLine。
constexpr uint32_t CACHE_LINE_BYTES = 64;


class KernelNonDiagMask {
public:
    __aicore__ inline KernelNonDiagMask() {}

    __aicore__ inline void Init(
        GM_ADDR row,
        GM_ADDR col,
        GM_ADDR mask,

        uint32_t startTgt,
        uint32_t endTgt,

        uint32_t minIdx,
        uint32_t coreProcessLength,

        uint32_t tileNum,
        uint32_t tailDataNum,

        int64_t M,
        int64_t N,
        int64_t k,
        int64_t numDiag,

        uint32_t totalLength,
        uint32_t outTotalLen)
    {
        this->startTgt = startTgt;
        this->endTgt = endTgt;

        this->minIdx = minIdx;
        this->coreProcessLength = coreProcessLength;

        this->tileNum = tileNum;
        this->tailDataNum = tailDataNum;

        this->M = M;
        this->N = N;
        this->k = k;
        this->numDiag = numDiag;

        this->totalLength = totalLength;
        this->outTotalLen = outTotalLen;

        // ====================================================
        // row / col 使用 uint8_t 视图。
        //
        // 原因：
        // DataCopy 对 int64 搬运要求按照 32B，即 4 个 int64 对齐。
        // 最后一个 tile 如果不足 4 个元素，直接向上对齐可能访问
        // row / col Tensor 尾部之外。
        //
        // 将数据看成 byte，再通过 DataCopyPad 精确搬运
        // curProcessDataNum * sizeof(int64_t) 个字节。
        // ====================================================

        rowGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint8_t*>(row),
            totalLength * sizeof(int64_t));

        colGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint8_t*>(col),
            totalLength * sizeof(int64_t));

        // mask 的单位本身就是 byte。
        maskGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ int8_t*>(mask),
            outTotalLen);

        // ====================================================
        // UB 只保留两个固定大小的输入 Tile。
        //
        // 不再分配：
        //
        //     maskBuf = endTgt - startTgt
        //
        // 因此 UB 使用量与 E / M / N 无关。
        // ====================================================

        pipe.InitBuffer(
            inQueueRow,
            BUFFER_NUM,
            TILE_DATA_NUM * sizeof(int64_t));

        pipe.InitBuffer(
            inQueueCol,
            BUFFER_NUM,
            TILE_DATA_NUM * sizeof(int64_t));
    }


    __aicore__ inline void Process()
    {
        if (coreProcessLength == 0) {
            return;
        }

        for (uint32_t progress = 0; progress < tileNum; ++progress) {

            uint32_t curProcessDataNum =
                (progress == tileNum - 1)
                    ? tailDataNum
                    : TILE_DATA_NUM;

            CopyIn(progress, curProcessDataNum);

            Compute(progress, curProcessDataNum);
        }

        // ====================================================
        // maskGm.SetValue() 属于 Scalar -> GM 写操作。
        //
        // SetValue 首先修改当前 AI Core 的 Data Cache，
        // 所以 Kernel 结束前必须刷新 Data Cache，使修改真正写回 GM。
        //
        // 每个 Core 只写自己独占的 64B 对齐输出区间，因此不会产生
        // 不同 Core 对同一个 CacheLine 的写回覆盖。
        // ====================================================

        DataCacheCleanAndInvalid<
            int8_t,
            CacheLine::ENTIRE_DATA_CACHE,
            DcciDst::CACHELINE_OUT>(maskGm);
    }


private:

    // ========================================================
    // CopyIn
    // ========================================================

    __aicore__ inline void CopyIn(
        uint32_t progress,
        uint32_t curProcessDataNum)
    {
        LocalTensor<uint8_t> rowLocalByte =
            inQueueRow.AllocTensor<uint8_t>();

        LocalTensor<uint8_t> colLocalByte =
            inQueueCol.AllocTensor<uint8_t>();

        // 当前 tile 在原始 row/col 中的起始元素位置。
        uint32_t globalElementOffset =
            this->minIdx + progress * TILE_DATA_NUM;

        // 转换成 byte offset。
        uint64_t globalByteOffset =
            static_cast<uint64_t>(globalElementOffset) *
            sizeof(int64_t);

        // 实际需要搬运的字节数。
        uint32_t copyBytes =
            curProcessDataNum * sizeof(int64_t);

        // ====================================================
        // DataCopyPad 支持非 32B 对齐搬运。
        //
        // blockCount = 1
        // blockLen   = copyBytes，单位 Byte
        // srcStride  = 0
        // dstStride  = 0
        // ====================================================

        DataCopyExtParams copyParams{
            1,
            copyBytes,
            0,
            0,
            0
        };

        // 不主动增加 left/right padding。
        // DataCopyPad 会自动处理 Local 侧的数据块对齐。
        DataCopyPadExtParams<uint8_t> padParams{
            true,
            0,
            0,
            0
        };

        DataCopyPad(
            rowLocalByte,
            rowGm[globalByteOffset],
            copyParams,
            padParams);

        DataCopyPad(
            colLocalByte,
            colGm[globalByteOffset],
            copyParams,
            padParams);

        inQueueRow.EnQue(rowLocalByte);
        inQueueCol.EnQue(colLocalByte);
    }


    // ========================================================
    // Compute
    // ========================================================

    __aicore__ inline void Compute(
        uint32_t progress,
        uint32_t curProcessDataNum)
    {
        LocalTensor<uint8_t> rowLocalByte =
            inQueueRow.DeQue<uint8_t>();

        LocalTensor<uint8_t> colLocalByte =
            inQueueCol.DeQue<uint8_t>();

        // byte Tensor 重新解释成 int64 Tensor。
        LocalTensor<int64_t> rowLocal =
            rowLocalByte.ReinterpretCast<int64_t>();

        LocalTensor<int64_t> colLocal =
            colLocalByte.ReinterpretCast<int64_t>();

        // 等待 MTE2 -> UB 数据就绪。
        PipeBarrier<PIPE_ALL>();

        uint32_t baseIdx =
            this->minIdx + progress * TILE_DATA_NUM;


        for (uint32_t i = 0;
             i < curProcessDataNum;
             ++i) {

            uint32_t idx = baseIdx + i;

            if (idx >= this->totalLength) {
                break;
            }

            int64_t r = rowLocal.GetValue(i);
            int64_t c = colLocal.GetValue(i);

            int64_t targetIdx = -1;


            // =================================================
            // 完全保持 pytorch_sparse non_diag_mask 的
            // CUDA reference 逻辑。
            // =================================================

            if (this->k < 0) {

                if (r + this->k < 0) {

                    targetIdx =
                        static_cast<int64_t>(idx);

                } else if (r + this->k >= this->N) {

                    targetIdx =
                        static_cast<int64_t>(idx)
                        + this->numDiag;

                } else if (r + this->k > c) {

                    targetIdx =
                        static_cast<int64_t>(idx)
                        + r
                        + this->k;

                } else if (r + this->k < c) {

                    targetIdx =
                        static_cast<int64_t>(idx)
                        + r
                        + this->k
                        + 1;
                }

                // r + k == c：
                // 当前 COO 元素本身位于目标对角线上，
                // 不写 mask，保持 0。

            } else {

                if (r + this->k >= this->N) {

                    targetIdx =
                        static_cast<int64_t>(idx)
                        + this->numDiag;

                } else if (r + this->k > c) {

                    targetIdx =
                        static_cast<int64_t>(idx)
                        + r;

                } else if (r + this->k < c) {

                    targetIdx =
                        static_cast<int64_t>(idx)
                        + r
                        + 1;
                }

                // r + k == c：
                // 不写，mask 保持 0。
            }


            // =================================================
            // 当前 Core 按「输出 index 范围」负责写 mask。
            //
            // 只有 targetIdx 落到自己的输出区间才写。
            //
            // 这样：
            //
            // Core 0 -> [start0, end0)
            // Core 1 -> [start1, end1)
            // ...
            //
            // 每个边界都按 64B 对齐。
            // =================================================

            if (targetIdx >=
                    static_cast<int64_t>(this->startTgt) &&
                targetIdx <
                    static_cast<int64_t>(this->endTgt) &&
                targetIdx >= 0 &&
                targetIdx <
                    static_cast<int64_t>(this->outTotalLen)) {

                maskGm.SetValue(
                    static_cast<uint64_t>(targetIdx),
                    static_cast<int8_t>(1));
            }
        }


        inQueueRow.FreeTensor(rowLocalByte);
        inQueueCol.FreeTensor(colLocalByte);
    }


private:

    TPipe pipe;

    // BUFFER_NUM = 1，固定约 64 KB UB。
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueRow;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueCol;

    // row/col 通过 byte 搬运。
    GlobalTensor<uint8_t> rowGm;
    GlobalTensor<uint8_t> colGm;

    GlobalTensor<int8_t> maskGm;


    uint32_t startTgt;
    uint32_t endTgt;

    uint32_t minIdx;
    uint32_t coreProcessLength;

    uint32_t tileNum;
    uint32_t tailDataNum;

    uint32_t totalLength;
    uint32_t outTotalLen;

    int64_t M;
    int64_t N;
    int64_t k;
    int64_t numDiag;
};


// ============================================================
// Kernel entry
// ============================================================

extern "C" __global__ __aicore__
void non_diag_mask(
    GM_ADDR row,
    GM_ADDR col,
    GM_ADDR mask,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);


    uint32_t blockIdx =
        GetBlockIdx();

    uint32_t coreNum =
        tiling_data.coreNum;

    uint32_t outTotalLen =
        tiling_data.outTotalLen;

    uint32_t totalLength =
        tiling_data.totalLength;

    uint32_t maxDiff =
        tiling_data.max_diff;


    // ========================================================
    // 基础保护
    // ========================================================

    if (coreNum == 0 ||
        blockIdx >= coreNum ||
        outTotalLen == 0 ||
        totalLength == 0) {
        return;
    }


    // ========================================================
    // 1. 按输出长度进行多核划分
    //
    // 必须按 64 Byte 对齐，而不是原来的 32 Byte。
    //
    // 原因不是 DataCopy，而是：
    //
    //     maskGm.SetValue()
    //
    // 属于 Scalar GM 写，使用 Data Cache。
    // 一个 CacheLine = 64B。
    //
    // 不允许两个 Core 写同一 CacheLine。
    // ========================================================

    uint32_t outLengthPerCore =
        (outTotalLen + coreNum - 1) /
        coreNum;

    outLengthPerCore =
        (outLengthPerCore +
         CACHE_LINE_BYTES - 1) /
        CACHE_LINE_BYTES *
        CACHE_LINE_BYTES;


    uint32_t startTgt =
        blockIdx * outLengthPerCore;

    if (startTgt >= outTotalLen) {
        return;
    }


    uint32_t endTgt =
        startTgt + outLengthPerCore;

    if (endTgt > outTotalLen) {
        endTgt = outTotalLen;
    }


    // ========================================================
    // 2. 确定当前 Core 需要检查的输入范围
    //
    // 保留你原 kernel 的 max_diff 思路。
    //
    // 对于某个目标输出区间：
    //
    //      [startTgt, endTgt)
    //
    // 只需要扫描可能映射到这个区间的 COO 输入。
    // ========================================================

    uint32_t minIdx =
        (startTgt > maxDiff)
            ? (startTgt - maxDiff)
            : 0;


    // 原代码的：
    //
    //     min_idx = (min_idx / 8) * 8
    //
    // 这里已经不需要。
    //
    // 因为使用 DataCopyPad 做非对齐输入搬运。


    uint64_t maxIdx64 =
        static_cast<uint64_t>(endTgt) + 32;

    if (maxIdx64 >
        static_cast<uint64_t>(totalLength)) {

        maxIdx64 =
            static_cast<uint64_t>(totalLength);
    }

    uint32_t maxIdx =
        static_cast<uint32_t>(maxIdx64);


    if (minIdx >= maxIdx) {
        return;
    }


    uint32_t coreProcessLength =
        maxIdx - minIdx;


    // ========================================================
    // 3. 输入 Tile 划分
    // ========================================================

    uint32_t tileNum =
        (coreProcessLength +
         TILE_DATA_NUM - 1) /
        TILE_DATA_NUM;


    uint32_t tailDataNum =
        coreProcessLength %
        TILE_DATA_NUM;


    if (tailDataNum == 0) {
        tailDataNum =
            TILE_DATA_NUM;
    }


    // ========================================================
    // 4. Kernel
    // ========================================================

    KernelNonDiagMask op;

    op.Init(
        row,
        col,
        mask,

        startTgt,
        endTgt,

        minIdx,
        coreProcessLength,

        tileNum,
        tailDataNum,

        tiling_data.M,
        tiling_data.N,
        tiling_data.k,
        tiling_data.num_diag,

        totalLength,
        outTotalLen);


    op.Process();
}