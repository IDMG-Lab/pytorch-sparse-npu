#ifndef NON_DIAG_MASK_TILING_H
#define NON_DIAG_MASK_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TilingData)
  // 严格的内存对齐: 4 个 32位 占 16 bytes
  TILING_DATA_FIELD_DEF(uint32_t, coreNum);     
  TILING_DATA_FIELD_DEF(uint32_t, totalLength); 
  TILING_DATA_FIELD_DEF(uint32_t, outTotalLen); 
  TILING_DATA_FIELD_DEF(uint32_t, max_diff);    
  // 4 个 64位 占 32 bytes
  TILING_DATA_FIELD_DEF(int64_t, M);
  TILING_DATA_FIELD_DEF(int64_t, N);
  TILING_DATA_FIELD_DEF(int64_t, k);
  TILING_DATA_FIELD_DEF(int64_t, num_diag);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(NonDiagMask, TilingData)
}
#endif // NON_DIAG_MASK_TILING_H