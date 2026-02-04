
#include "register/tilingdata_base.h"

namespace optiling
{
  BEGIN_TILING_DATA_DEF(Ind2ptrTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, indsize);
  TILING_DATA_FIELD_DEF(uint32_t, M); // M表示稀疏矩阵的行数
  END_TILING_DATA_DEF;

  REGISTER_TILING_DATA_CLASS(Ind2ptr, Ind2ptrTilingData)
}
