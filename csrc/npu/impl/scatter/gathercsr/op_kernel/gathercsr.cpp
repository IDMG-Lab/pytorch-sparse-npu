#include "kernel_operator.h"
using namespace AscendC;
//
// 当使用多核时，对应的xGm
//
#define BUFFER_NUM 2
static __aicore__ inline int32_t AlignDown(int32_t x, int32_t a)
{
    return (x / a) * a;
}
template <typename T>
class GatherCsrKernel
{

private:
    int32_t MAX_EDGES_PER_TILE; // UB里一次处理的元素数（按UB大小自己算
    int32_t N, val1, core_num, core_id;
    int32_t out_size; // 表示对应输出的大小，应该是ptr[-1],表示nnz
    uint32_t blockIdx;
    uint32_t tileLength = 1024;
    const uint32_t DUMP_NUM = 32 / sizeof(T);
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> zGm;

    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueZ;

public:
    __aicore__ inline GatherCsrKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, int32_t N_input)
    {
        core_num = GetBlockNum();
        core_id = GetBlockIdx();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams copyParams{
            (uint16_t)1,
            (uint32_t)(DUMP_NUM * sizeof(T)), // 确认不会溢出,int32_t * DUMP_NUM刚好是32字节对齐
            0, 0, 0};
        blockIdx = GetBlockIdx();
        N = N_input;
        xGm.SetGlobalBuffer((__gm__ T *)x, N);
        yGm.SetGlobalBuffer((__gm__ T *)y, N + 1);
        out_size = yGm(N);
        // if (core_id == 0)
        //     printf("out_size:%d", out_size);
        zGm.SetGlobalBuffer((__gm__ T *)z, out_size);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, DUMP_NUM * sizeof(T));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, DUMP_NUM * sizeof(T));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, DUMP_NUM * sizeof(T));

        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();

        // DataCopyPad(xLocal, xGm[0], copyParams, padParams);
        // DataCopyPad(yLocal, yGm[0], copyParams, padParams);

        // DumpTensor(xLocal, 47, xLocal.GetSize());
        // DumpTensor(yLocal, 48, yLocal.GetSize());
    }
    __aicore__ inline void Process()
    {
        // 表示总共有N节点个数
        // rows_per_core表示每个aicore中需处理的节点个数
        int32_t n_per_core = (N + core_num - 1) / core_num;
        int32_t n_start = core_id * n_per_core;
        int32_t n_end = n_start + n_per_core;
        // if (core_id == 0)
        // {
        //     printf("n_per_core:%d", n_per_core);
        //     printf("n_start:%d", n_start);
        //     printf("n_end:%d", n_end);
        // }
        if (n_start >= N)
            return;
        if (n_end > N)
            n_end = N;
        // printf("n_end:%d", n_end);
        // LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
        for (int32_t n_now = n_start; n_now < n_end; ++n_now)
        {
            // 这里面才是每个index对应的n_now
            int32_t start = yGm(n_now);
            int32_t end = yGm(n_now + 1);
            int32_t edge_cnt = end - start; // 第n_now行对应的元素个数
            if (edge_cnt <= 0)
                continue;

            int32_t src_val = xGm(n_now);
            // printf("src_val:%d", src_val);

            for (int32_t i = 0; i < edge_cnt; i++)
            {
                // zGm.SetValue(start + i, src_val);
                zGm(start + i) = src_val; // ✅ 真标量写，能覆盖非对齐地址
            }
            //     while (done < edge_cnt)
            //     {
            //         // LocalTensor<int32_t> outLocal = ubBuf_.AllocTensor<int32_t>();

            //         int32_t cur = (edge_cnt - done) > MAX_EDGES_PER_TILE
            //                           ? MAX_EDGES_PER_TILE
            //                           : (edge_cnt - done);

            //         // int32: 32B对齐 => calCount 必须是8的倍数
            //         int32_t bulk = AlignDown(cur, 8);
            //         int32_t tail = cur - bulk;

            //         if (bulk > 0)
            //         {
            //             // UB里用 Duplicate 把 src_val 广播填充 bulk 个元素 :contentReference[oaicite:3]{index=3}
            //             Duplicate(outLocal, src_val, bulk);

            //             // 一次 DataCopy 从 UB 搬到 GM（连续搬运） :contentReference[oaicite:4]{index=4}
            //             int32_t gm_offset = start + done;
            //             // DataCopy(outGm[gm_offset], outLocal, bulk);
            //             DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
            //             DataCopyExtParams copyParams{
            //                 (uint16_t)1,
            //                 (uint32_t)(bulk * sizeof(int32_t)),
            //                 0, 0, 0};
            //             DataCopyPad(outGm[gm_offset], outLocal, copyParams);
            //             if (core_id == 0)
            //                 AscendC::DumpTensor(outLocal, 78, outLocal.GetSize());
            //             ubBuf_.FreeTensor(outLocal);
            //         }

            //         // 尾巴不满足32B对齐，直接标量写（很小）
            //         if (tail > 0)
            //         {
            //             int32_t gm_offset = start + done + bulk;
            //             for (int32_t j = 0; j < tail; ++j)
            //             {
            //                 outGm(gm_offset + j) = src_val;
            //             }
            //         }

            //         done += cur;
            //     }
        }
    }
};

extern "C" __global__ __aicore__ void gathercsr(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{

    //  printf("可以定义GatherKernel");
    //  int32_t N = 3;
    GET_TILING_DATA(tiling_data, tiling);
    GatherCsrKernel<int32_t> op;
    // printf("tiling_data'N:%d", tiling_data.N);
    // ----------------------可以正确传入N
    op.Init(x, y, z, tiling_data.N);
    op.Process();
}
