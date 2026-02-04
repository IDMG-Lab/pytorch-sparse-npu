#ifndef BUFFER_NUM
#define BUFFER_NUM 2
#endif

#ifndef MAX_DIM_NUMBER
#define MAX_DIM_NUMBER 4
#endif

using namespace AscendC;
template <typename T>
class KernelFmaxBroadcastLast
{
private:
  TPipe pipe;
  TQue<TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
  TQue<TPosition::VECOUT, BUFFER_NUM> outQueueZ;
  TQue<QuePosition::VECCALC, BUFFER_NUM> calQueueX, calQueueY, calQueueZ;

  GlobalTensor<T> xGm;
  GlobalTensor<T> yGm;
  GlobalTensor<T> zGm;

  uint32_t blockBlocks;
  uint32_t loopCount;
  uint32_t castLength;
  uint32_t tileLength;      // 每个 tile 的有效元素数（最后一维长度）
  uint32_t tileLengthAlign; // 32 元素对齐后的长度
  uint32_t blockIdx;
  uint32_t shape1[4];
  uint32_t shape2[4];
  uint32_t shapez[4];
  uint32_t strideX[4];
  uint32_t strideY[4];
  uint32_t strideZ[4];
  uint32_t blockBaseTile = 0; // 当前 block 处理的首个 tile 序号（以 tile 为单位）
  uint32_t last_tile;         // 如果最后一维大小超过2048，要进行多少次tile
  uint32_t elemsX, elemsY, elemsZ;
  bool islastboard;
  uint32_t islastboardidx;

public:
  __aicore__ inline KernelFmaxBroadcastLast() {}

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                              uint32_t totalBlocks, // 保留但不再使用
                              uint32_t tailNum,
                              uint32_t tailBlocks,
                              uint32_t formerBlocks,
                              uint32_t formerNum,
                              int32_t shapezIn[4],
                              int32_t shape1In[4],
                              int32_t shape2In[4])
  {
    blockIdx = AscendC::GetBlockIdx();

    for (int i = 0; i < MAX_DIM_NUMBER; ++i)
    {
      shapez[i] = shapezIn[i];
      shape1[i] = shape1In[i];
      shape2[i] = shape2In[i];
    }
    // printf("shapez:");
    // for (int i = 0; i < MAX_DIM_NUMBER; i++)
    // {
    //   printf("%d ", shapez[i]);
    // }
    if (shape1[MAX_DIM_NUMBER - 1] == 1 && shape2[MAX_DIM_NUMBER - 1] != 1 || shape1[MAX_DIM_NUMBER - 1] != 1 && shape2[MAX_DIM_NUMBER - 1] == 1)
    {
      islastboard = true;
      if (shape1[MAX_DIM_NUMBER - 1] == 1)
        islastboardidx = 0;
      else
        islastboardidx = 1;
    } // 表示最后维度需要广播
    else
      islastboard = false;
    elemsX = shape1[0] * shape1[1] * shape1[2] * shape1[3];
    elemsY = shape2[0] * shape2[1] * shape2[2] * shape2[3];
    elemsZ = shapez[0] * shapez[1] * shapez[2] * shapez[3];

    // 以最后一维长度作为 tile
    tileLength = shapez[MAX_DIM_NUMBER - 1];
    if (tileLength > 2048)
    {
      tileLength = 2048;
      last_tile = (shapez[MAX_DIM_NUMBER - 1] + 2048 - 1) / 2048;
    }
    else
      last_tile = 1;
    // printf("last_tile:%u", last_tile);

    tileLengthAlign = (tileLength + 32 - 1) / 32 * 32;
    // 如果最后一维长度小于32时怎么办
    uint32_t tileNum = elemsZ / shapez[MAX_DIM_NUMBER - 1];

    // 平均分配 tile 给 block（沿用 former/tail 的分法）
    if (blockIdx < formerNum)
      blockBlocks = formerBlocks;
    else
      blockBlocks = tailBlocks;

    if (blockBlocks == 0)
      return;
    loopCount = blockBlocks;
    // printf("当前aicoreIdx:%d\n", blockIdx);
    // printf("LoopCount%u\n", loopCount);

    // 计算当前 block 的 tile 起点
    uint32_t blocksBefore = (blockIdx < formerNum)
                                ? (blockIdx * formerBlocks)
                                : (formerNum * formerBlocks + (blockIdx - formerNum) * tailBlocks);
    blockBaseTile = blocksBefore; // 表示该aicore对应的数据起始位置

    xGm.SetGlobalBuffer((__gm__ T *)x, elemsX);
    yGm.SetGlobalBuffer((__gm__ T *)y, elemsY);
    zGm.SetGlobalBuffer((__gm__ T *)z, elemsZ);

    pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLengthAlign * sizeof(T));
    pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLengthAlign * sizeof(T));
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLengthAlign * sizeof(T));

    if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)
    {
      // castLength = tileLengthAlign * sizeof(float);
      pipe.InitBuffer(calQueueX, BUFFER_NUM, tileLengthAlign * sizeof(half));
      pipe.InitBuffer(calQueueY, BUFFER_NUM, tileLengthAlign * sizeof(half));
      pipe.InitBuffer(calQueueZ, BUFFER_NUM, tileLengthAlign * sizeof(half));
    }
    if constexpr (std::is_same_v<T, int64_t>)
    {
      pipe.InitBuffer(calQueueX, BUFFER_NUM, tileLengthAlign * sizeof(int32_t));
      pipe.InitBuffer(calQueueY, BUFFER_NUM, tileLengthAlign * sizeof(int32_t));
      pipe.InitBuffer(calQueueZ, BUFFER_NUM, tileLengthAlign * sizeof(int32_t));
    }
    if constexpr (std::is_same_v<T, bfloat16_t>)
    {
      pipe.InitBuffer(calQueueX, BUFFER_NUM, tileLengthAlign * sizeof(float));
      pipe.InitBuffer(calQueueY, BUFFER_NUM, tileLengthAlign * sizeof(float));
      pipe.InitBuffer(calQueueZ, BUFFER_NUM, tileLengthAlign * sizeof(float));
    }
    Process();
  }

  __aicore__ inline void Process()
  {
    for (uint32_t t = 0; t < loopCount; t++)
    {
      // printf("进入aicore的部分的idx:%u", blockIdx);
      // printf("loopCount:%u\n", loopCount);
      // printf("当前的t:%u\n", t);
      uint32_t j = 1;
      uint32_t offset = 0;
      for (j = 1; j <= last_tile; j++)
      {
        if (j == last_tile)
        {
          tileLength = shapez[MAX_DIM_NUMBER - 1] - (last_tile - 1) * 2048;
          offset = 2048 * (j - 1);
        }
        else
        {
          tileLength = 2048;
          offset = 2048 * (j - 1);
        }
        copyin(t, j);
        compute(t, j);
        copyout(t, j);
      }
    }
  }

  __aicore__ inline void copyin(uint32_t tileIndex, uint32_t last_tile_index)
  {
    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();

    // tileIndex → 3D 坐标（最后一维全覆盖）
    uint32_t t = blockBaseTile + tileIndex; // tileIndex是代表该aicore区域的tileid，表示tile‘块的id
    uint32_t idxVec[4];
    for (int i = 2; i >= 0; --i)
    {
      idxVec[i] = t % shapez[i];
      t /= shapez[i];
    }
    idxVec[3] = 0;
    // 映射到 x 的真实坐标: [j0, j1, j2]
    uint32_t j0 = (shape1[0] == 1) ? 0 : idxVec[0];
    uint32_t j1 = (shape1[1] == 1) ? 0 : idxVec[1];
    uint32_t j2 = (shape1[2] == 1) ? 0 : idxVec[2];
    // x 的线性偏移（假设连续存储）
    uint32_t offX = ((j0 * shape1[1] + j1) * shape1[2] + j2) * shape1[3];

    j0 = (shape2[0] == 1) ? 0 : idxVec[0];
    j1 = (shape2[1] == 1) ? 0 : idxVec[1];
    j2 = (shape2[2] == 1) ? 0 : idxVec[2];
    // y 的线性偏移（假设连续存储）
    uint32_t offY = ((j0 * shape2[1] + j1) * shape2[2] + j2) * shape2[3];
    // DataCopy(xLocal, xGm[offX], tileLength);

    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyExtParams copyParamsVec = {
        (uint16_t)1,
        (uint32_t)(tileLength * sizeof(T)),
        0,
        0,
        0};
    DataCopyExtParams copyParamsScalar = {
        (uint16_t)1,
        (uint32_t)(1 * sizeof(T)),
        0,
        0,
        0};
    // DataCopy(xLocal, xGm[offX], tileLength);
    if (islastboard)
    {
      if (islastboardidx == 0)
      {
        DataCopyPad(xLocal, xGm[offX], copyParamsScalar, padParams);
        DataCopyPad(yLocal, yGm[offY + 2048 * (last_tile_index - 1)], copyParamsVec, padParams);
      }
      else
      {
        DataCopyPad(xLocal, xGm[offX + 2048 * (last_tile_index - 1)], copyParamsVec, padParams);
        DataCopyPad(yLocal, yGm[offY], copyParamsScalar, padParams);
      }
    }
    else
    {
      DataCopyPad(xLocal, xGm[offX + 2048 * (last_tile_index - 1)], copyParamsVec, padParams);
      DataCopyPad(yLocal, yGm[offY + 2048 * (last_tile_index - 1)], copyParamsVec, padParams);
    }

    // if (blockIdx == 0 && tileIndex == 0)
    // {
    //   printf("看一下输入的tensor\n");
    //   DumpTensor(xLocal, 10011, xLocal.GetSize());
    //   DumpTensor(yLocal, 10011, yLocal.GetSize());
    // }
    inQueueX.EnQue(xLocal);
    inQueueY.EnQue(yLocal);
  }

  __aicore__ inline void compute(uint32_t tileIndex, uint32_t last_tile_index)
  {

    LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    LocalTensor<T> yLocal = inQueueY.DeQue<T>();
    LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
    if (!islastboard)
    {
      if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)
      {
        LocalTensor<half> calxLocal = calQueueX.AllocTensor<half>();
        LocalTensor<half> calyLocal = calQueueY.AllocTensor<half>();
        LocalTensor<half> calzLocal = calQueueZ.AllocTensor<half>();
        Cast(calxLocal, xLocal, RoundMode::CAST_NONE, tileLength);
        Cast(calyLocal, yLocal, RoundMode::CAST_NONE, tileLength);
        Max(calzLocal, calxLocal, calyLocal, tileLength);
        Cast(zLocal, calzLocal, RoundMode::CAST_TRUNC, tileLength);
        // if (blockIdx == 0)
        // {
        //   printf("Compute中打印calxLocal（输入）的值---------------\n");
        //   DumpTensor(calxLocal, 5, tileLength);
        //   printf("Compute中打印calyLocal（输入）的值---------------\n");
        //   DumpTensor(calyLocal, 5, tileLength);
        //   printf("Compute中打印calzLocal（输出）的值---------------\n");
        //   DumpTensor(calzLocal, 5, tileLength);
        //   printf("Compute中打印zLocal（输出）的值---------------\n");
        //   DumpTensor(zLocal, 5, tileLength);
        // }
        calQueueX.FreeTensor(calxLocal);
        calQueueY.FreeTensor(calyLocal);
        calQueueZ.FreeTensor(calzLocal);
        outQueueZ.EnQue<T>(zLocal);
        inQueueY.FreeTensor(yLocal);
        inQueueX.FreeTensor(xLocal);
      }
      else if constexpr (std::is_same_v<T, int64_t>)
      {
        LocalTensor<int32_t> calxLocal = calQueueX.AllocTensor<int32_t>();
        LocalTensor<int32_t> calyLocal = calQueueY.AllocTensor<int32_t>();
        LocalTensor<int32_t> calzLocal = calQueueZ.AllocTensor<int32_t>();
        Cast(calxLocal, xLocal, RoundMode::CAST_NONE, tileLength);
        Cast(calyLocal, yLocal, RoundMode::CAST_NONE, tileLength);
        Max(calzLocal, calxLocal, calyLocal, tileLength);
        Cast(zLocal, calzLocal, RoundMode::CAST_NONE, tileLength);
        // printf("Compute中打印calxLocal（输入）的值---------------\n");
        // DumpTensor(calzLocal, 5, tileLength);
        // printf("Compute中打印calyLocal（输入）的值---------------\n");
        // DumpTensor(calyLocal, 5, tileLength);
        // printf("Compute中打印calzLocal（输出）的值---------------\n");
        // DumpTensor(calzLocal, 5, tileLength);
        // printf("Compute中打印zLocal（输出）的值---------------\n");
        // DumpTensor(zLocal, 5, tileLength);
        calQueueX.FreeTensor(calxLocal);
        calQueueY.FreeTensor(calyLocal);
        calQueueZ.FreeTensor(calzLocal);
        outQueueZ.EnQue<T>(zLocal);
        inQueueY.FreeTensor(yLocal);
        inQueueX.FreeTensor(xLocal);
      }
      else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, float> || std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t>)
      { // float / half / int16 / int32
        Max(zLocal, xLocal, yLocal, tileLength);
        // if (blockIdx == 0 && tileIndex == 0)
        // {
        //   // DumpTensor(zLocal, 280, tileLength); // 这个打印不出来
        //   DumpTensor(zLocal, 283, zLocal.GetSize());
        // }
        outQueueZ.EnQue<T>(zLocal);
        inQueueY.FreeTensor(yLocal);
        inQueueX.FreeTensor(xLocal);
      }
      else if constexpr (std::is_same_v<T, bfloat16_t>)
      {
        LocalTensor<float> calxLocal = calQueueX.AllocTensor<float>();
        LocalTensor<float> calyLocal = calQueueY.AllocTensor<float>();
        LocalTensor<float> calzLocal = calQueueZ.AllocTensor<float>();
        Cast(calxLocal, xLocal, RoundMode::CAST_NONE, tileLength);
        Cast(calyLocal, yLocal, RoundMode::CAST_NONE, tileLength);
        Max(calzLocal, calxLocal, calyLocal, tileLength);
        // 这里是float==>bfloat16
        Cast(zLocal, calzLocal, RoundMode::CAST_RINT, tileLength);
        calQueueX.FreeTensor(calxLocal);
        calQueueY.FreeTensor(calyLocal);
        calQueueZ.FreeTensor(calzLocal);
        outQueueZ.EnQue<T>(zLocal);
        inQueueY.FreeTensor(yLocal);
        inQueueX.FreeTensor(xLocal);
      }
    }
    else
    {
      if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)
      {
        half calscalar;
        LocalTensor<half> calxLocal = calQueueX.AllocTensor<half>();
        LocalTensor<half> calyLocal = calQueueY.AllocTensor<half>();
        LocalTensor<half> calzLocal = calQueueZ.AllocTensor<half>();
        Cast(calxLocal, xLocal, RoundMode::CAST_NONE, tileLength);
        Cast(calyLocal, yLocal, RoundMode::CAST_NONE, tileLength);
        if (islastboardidx == 0) // 表示xLocal是标量
        {
          calscalar = calxLocal.GetValue(0);
          Maxs(calzLocal, calyLocal, calscalar, tileLength);
        }
        else
        {
          calscalar = calyLocal.GetValue(0);
          Maxs(calzLocal, calxLocal, calscalar, tileLength);
        }
        // Max(calzLocal, calxLocal, calyLocal, tileLength);
        Cast(zLocal, calzLocal, RoundMode::CAST_TRUNC, tileLength);
        // if (blockIdx == 0)
        // {
        //   printf("Compute中打印calxLocal（输入）的值---------------\n");
        //   DumpTensor(calxLocal, 5, tileLength);
        //   printf("Compute中打印calyLocal（输入）的值---------------\n");
        //   DumpTensor(calyLocal, 5, tileLength);
        //   printf("Compute中打印calzLocal（输出）的值---------------\n");
        //   DumpTensor(calzLocal, 5, tileLength);
        //   printf("Compute中打印zLocal（输出）的值---------------\n");
        //   DumpTensor(zLocal, 5, tileLength);
        // }
        calQueueX.FreeTensor(calxLocal);
        calQueueY.FreeTensor(calyLocal);
        calQueueZ.FreeTensor(calzLocal);
        outQueueZ.EnQue<T>(zLocal);
        inQueueY.FreeTensor(yLocal);
        inQueueX.FreeTensor(xLocal);
      }
      else if constexpr (std::is_same_v<T, int64_t>)
      {
        LocalTensor<int32_t> calxLocal = calQueueX.AllocTensor<int32_t>();
        LocalTensor<int32_t> calyLocal = calQueueY.AllocTensor<int32_t>();
        LocalTensor<int32_t> calzLocal = calQueueZ.AllocTensor<int32_t>();
        Cast(calxLocal, xLocal, RoundMode::CAST_NONE, tileLength);
        Cast(calyLocal, yLocal, RoundMode::CAST_NONE, tileLength);
        if (islastboardidx == 0) // 表示xLocal是标量
        {
          int32_t calscalar = calxLocal.GetValue(0);
          Maxs(calzLocal, calyLocal, calscalar, tileLength);
        }
        else
        {
          int32_t calscalar = calyLocal.GetValue(0);
          Maxs(calzLocal, calxLocal, calscalar, tileLength);
        }
        // Max(calzLocal, calxLocal, calyLocal, tileLength);
        Cast(zLocal, calzLocal, RoundMode::CAST_NONE, tileLength);

        calQueueX.FreeTensor(calxLocal);
        calQueueY.FreeTensor(calyLocal);
        calQueueZ.FreeTensor(calzLocal);
        outQueueZ.EnQue<T>(zLocal);
        inQueueY.FreeTensor(yLocal);
        inQueueX.FreeTensor(xLocal);
      }
      else if constexpr (std::is_same_v<T, bfloat16_t>)
      {
        LocalTensor<float> calxLocal = calQueueX.AllocTensor<float>();
        LocalTensor<float> calyLocal = calQueueY.AllocTensor<float>();
        LocalTensor<float> calzLocal = calQueueZ.AllocTensor<float>();
        Cast(calxLocal, xLocal, RoundMode::CAST_NONE, tileLength);
        Cast(calyLocal, yLocal, RoundMode::CAST_NONE, tileLength);
        if (islastboardidx == 0) // 表示xLocal是标量
        {
          float calscalar = calxLocal.GetValue(0);
          Maxs(calzLocal, calyLocal, calscalar, tileLength);
        }
        else
        {
          float calscalar = calyLocal.GetValue(0);
          Maxs(calzLocal, calxLocal, calscalar, tileLength);
        }
        Cast(zLocal, calzLocal, RoundMode::CAST_RINT, tileLength);
        calQueueX.FreeTensor(calxLocal);
        calQueueY.FreeTensor(calyLocal);
        calQueueZ.FreeTensor(calzLocal);
        outQueueZ.EnQue<T>(zLocal);
        inQueueY.FreeTensor(yLocal);
        inQueueX.FreeTensor(xLocal);
      }
      else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, float> || std::is_same_v<T, int16_t> || std::is_same_v<T, int32_t>)
      {                          // float / half / int16 / int32
        if (islastboardidx == 0) // 表示xLocal是标量
        {
          T calscalar = xLocal.GetValue(0);
          Maxs(zLocal, yLocal, calscalar, tileLength);
        }
        else
        {
          T calscalar = yLocal.GetValue(0);
          Maxs(zLocal, xLocal, calscalar, tileLength);
        }

        outQueueZ.EnQue<T>(zLocal);
        inQueueY.FreeTensor(yLocal);
        inQueueX.FreeTensor(xLocal);
      }
    }
  }

  __aicore__ inline void copyout(uint32_t tileIndex, uint32_t last_tile_index)
  {

    LocalTensor<T> zLocal = outQueueZ.DeQue<T>();
    // printf("打印zLocal（输出）的值---------------\n");
    // DumpTensor(zLocal, 5, tileLength);
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyExtParams copyParamsVec = {
        (uint16_t)1,
        (uint32_t)(tileLength * sizeof(T)),
        0,
        0,
        0};
    DataCopyExtParams copyParamsScalar = {
        (uint16_t)1,
        (uint32_t)(1 * sizeof(T)),
        0,
        0,
        0};
    // DataCopy(zGm[(blockBaseTile + tileIndex) * tileLength], zLocal, zLocal.GetSize());
    DataCopyPad(zGm[(blockBaseTile + tileIndex) * shapez[MAX_DIM_NUMBER - 1] + 2048 * (last_tile_index - 1)], zLocal, copyParamsVec);
    // if (blockIdx == 0 && tileIndex == 0)
    // {
    //   // DumpTensor(zGm[0], 301, 10); // 这个打印不出来
    //   DumpTensor(zGm[(blockBaseTile + tileIndex) * tileLength], 302, zLocal.GetSize());
    // }
    outQueueZ.FreeTensor(zLocal);
  }
};