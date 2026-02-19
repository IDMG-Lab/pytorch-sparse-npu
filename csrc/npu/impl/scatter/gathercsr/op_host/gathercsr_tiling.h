
#include "register/tilingdata_base.h"

namespace optiling
{
  BEGIN_TILING_DATA_DEF(GathercsrTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, size);
  TILING_DATA_FIELD_DEF(int32_t, N);
  END_TILING_DATA_DEF;

  REGISTER_TILING_DATA_CLASS(Gathercsr, GathercsrTilingData)
}
