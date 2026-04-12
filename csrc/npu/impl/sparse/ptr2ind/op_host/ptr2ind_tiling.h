#ifndef PTR2IND_TILING_H
#define PTR2IND_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(Ptr2indTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalE);
    TILING_DATA_FIELD_DEF(uint32_t, numSegments);
    TILING_DATA_FIELD_DEF(uint32_t, blockSize);
    TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Ptr2ind, Ptr2indTilingData)

} // namespace optiling

#endif