#include "kernel_operator.h"
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_LENGTH = 4096; // 扩大 UB 缓存块，32KB，大幅减少流水线开销

class KernelPtr2Ind {
public:
    __aicore__ inline KernelPtr2Ind() {}

    __aicore__ inline void Init(GM_ADDR ptr_gm, GM_ADDR ind_gm, GM_ADDR tiling_gm) {
        GET_TILING_DATA(tiling, tiling_gm);

        totalE = tiling.totalE;
        numSegments = tiling.numSegments;
        outLengthPerCore = tiling.outLengthPerCore;

        uint32_t blockIdx = GetBlockIdx();
        start_tgt = blockIdx * outLengthPerCore;
        end_tgt = start_tgt + outLengthPerCore;
        if (end_tgt > totalE) {
            end_tgt = totalE;
        }

        // 全局内存绑定
        ptrIn.SetGlobalBuffer((__gm__ int64_t *)ptr_gm, numSegments + 1);
        indOut.SetGlobalBuffer((__gm__ int64_t *)ind_gm, totalE);

        pipe.InitBuffer(outQueue, BUFFER_NUM, TILE_LENGTH * sizeof(int64_t));
    }

    __aicore__ inline void Process() {
        if (start_tgt >= totalE) return; // 当前核没有分配到任务

        uint32_t core_len = end_tgt - start_tgt;

        // =========================================================================
        // 【核心优化 1：二分查找初始节点】O(log N)
        // 快速找到当前 Core 负责的起始边 (start_tgt) 属于哪一个节点 (cur_node)
        // =========================================================================
        uint32_t low = 0;
        uint32_t high = numSegments - 1;
        uint32_t cur_node = 0;

        while (low <= high) {
            uint32_t mid = low + (high - low) / 2;
            int64_t val = ptrIn.GetValue(mid);
            if (val <= (int64_t)start_tgt) {
                cur_node = mid;
                low = mid + 1; // 尝试向右逼近，找到最后一个满足条件的节点
            } else {
                if (mid == 0) break;
                high = mid - 1;
            }
        }

        int64_t node_end = ptrIn.GetValue(cur_node + 1);
        uint32_t processed = 0;

        // =========================================================================
        // 【核心优化 2：O(E) 顺序流水线填充】
        // =========================================================================
        while (processed < core_len) {
            uint32_t tile_len = core_len - processed;
            if (tile_len > TILE_LENGTH) {
                tile_len = TILE_LENGTH;
            }

            LocalTensor<int64_t> out = outQueue.AllocTensor<int64_t>();

            // 1. 在 SRAM(UB) 中极速填充
            for (uint32_t i = 0; i < tile_len; i++) {
                uint32_t gpos = start_tgt + processed + i;
                
                // 如果当前边超出了当前节点的范围，就步进到下一个节点
                // 用 while 处理出度为 0 的孤立节点
                while (gpos >= node_end) {
                    cur_node++;
                    if (cur_node < numSegments) {
                        node_end = ptrIn.GetValue(cur_node + 1);
                    } else {
                        node_end = totalE + 1; // 安全越界保护
                    }
                }
                out.SetValue(i, cur_node);
            }

            outQueue.EnQue(out);

            // 2. 将数据写回 Global Memory
            LocalTensor<int64_t> freeOut = outQueue.DeQue<int64_t>();
            
            // 【核心优化 3：安全内存对齐写回】
            uint32_t aligned_len = (tile_len / 4) * 4; // 计算 32 Byte 对齐的元素个数
            
            if (aligned_len > 0) {
                // 大块数据利用 MTE3 引擎无缝拷贝，直接打满带宽
                DataCopy(indOut[start_tgt + processed], freeOut, aligned_len);
            }
            // 尾部无法对齐的 1~3 个元素，使用标量安全写入，绝不越界污染相邻核！
            for (uint32_t i = aligned_len; i < tile_len; i++) {
                indOut.SetValue(start_tgt + processed + i, freeOut.GetValue(i));
            }

            outQueue.FreeTensor(freeOut);
            processed += tile_len;
        }
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;

    GlobalTensor<int64_t> ptrIn;
    GlobalTensor<int64_t> indOut;

    uint32_t totalE;
    uint32_t numSegments;
    uint32_t outLengthPerCore;
    uint32_t start_tgt;
    uint32_t end_tgt;
};

extern "C" __global__ __aicore__ void ptr2ind(GM_ADDR ptr, GM_ADDR ind, GM_ADDR workspace, GM_ADDR tiling) {
    KernelPtr2Ind op;
    op.Init(ptr, ind, tiling);
    op.Process();
}