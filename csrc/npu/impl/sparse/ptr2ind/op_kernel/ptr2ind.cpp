#include "kernel_operator.h"
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_LENGTH = 128;

class KernelPtr2Ind {
public:
    __aicore__ inline KernelPtr2Ind() {}

    __aicore__ inline void Init(GM_ADDR ptr_gm, GM_ADDR ind_gm, GM_ADDR tiling_gm) {
        GET_TILING_DATA(tiling, tiling_gm);

        totalE = tiling.totalE;
        numSegments = tiling.numSegments;
        blockSize = tiling.blockSize;
        tileNum = tiling.tileNum;

        blockIdx = GetBlockIdx();
        blockStart = blockIdx * blockSize;
        blockCount = (blockStart + blockSize > totalE) ? (totalE - blockStart) : blockSize;

        // 完全对齐学长：只传基地址，绝不偏移
        ptrIn.SetGlobalBuffer((__gm__ int64_t *)ptr_gm, numSegments + 1);
        indOut.SetGlobalBuffer((__gm__ int64_t *)ind_gm, totalE);

        pipe.InitBuffer(outQueue, BUFFER_NUM, TILE_LENGTH * sizeof(int64_t));
    }

    __aicore__ inline void Process() {
        for (uint32_t t = 0; t < tileNum; t++) {
            Compute(t);
            CopyOut(t);
        }
    }

private:
    __aicore__ inline void Compute(uint32_t tileIdx) {
        // 旧版兼容写法（和学长风格一致）
        LocalTensor<int64_t> out = outQueue.AllocTensor<int64_t>();
        uint32_t base = tileIdx * TILE_LENGTH;

        for (uint32_t i = 0; i < TILE_LENGTH; i++) {
            uint32_t pos = base + i;
            if (pos >= blockCount) break;

            int64_t val = 0;
            uint32_t gpos = blockStart + pos;

            for (uint32_t seg = 0; seg < numSegments; seg++) {
                int64_t s = ptrIn.GetValue(seg);
                int64_t e = ptrIn.GetValue(seg + 1);
                if (gpos >= s && gpos < e) {
                    val = (int64_t)seg;
                    break;
                }
            }
            out.SetValue(i, val);
        }
        outQueue.EnQue(out);
    }

    __aicore__ inline void CopyOut(uint32_t tileIdx) {
        LocalTensor<int64_t> out = outQueue.DeQue<int64_t>();
        // 偏移放在这里，绝对安全、编译器支持
        DataCopy(indOut[blockStart + tileIdx * TILE_LENGTH], out, TILE_LENGTH * sizeof(int64_t));
        outQueue.FreeTensor(out);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueue;

    GlobalTensor<int64_t> ptrIn;
    GlobalTensor<int64_t> indOut;

    uint32_t totalE;
    uint32_t numSegments;
    uint32_t blockSize;
    uint32_t tileNum;
    uint32_t blockIdx;
    uint32_t blockStart;
    uint32_t blockCount;
};

// 无 workspace，参数完全匹配
extern "C" __global__ __aicore__ void ptr2ind(GM_ADDR ptr, GM_ADDR ind, GM_ADDR workspace, GM_ADDR tiling) {
    KernelPtr2Ind op;
    op.Init(ptr, ind, tiling);
    op.Process();
}