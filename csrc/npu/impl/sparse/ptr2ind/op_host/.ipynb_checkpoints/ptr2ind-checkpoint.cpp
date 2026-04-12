#include "ptr2ind_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

namespace optiling {

const uint32_t TILE_LENGTH = 128;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Ptr2IndTilingData tiling;

    auto attrs = context->GetAttrs();
    uint32_t totalE = (uint32_t)*attrs->GetInt(0);

    const gert::StorageShape* ptr_shape = context->GetInputShape(0);
    uint32_t numSegments = ptr_shape->GetStorageShape().GetShapeSize() - 1;

    // ====================== 自动选择核数 ======================
    uint32_t BLOCK_DIM = 1; // 小数据直接用 1 核，避免越界
    if (totalE > 128) {
        BLOCK_DIM = 8;
    }

    uint32_t blockSize = (totalE + BLOCK_DIM - 1) / BLOCK_DIM;
    uint32_t tileNum = (blockSize + TILE_LENGTH - 1) / TILE_LENGTH;

    tiling.set_totalE(totalE);
    tiling.set_numSegments(numSegments);
    tiling.set_blockSize(blockSize);
    tiling.set_tileNum(tileNum);

    context->SetBlockDim(BLOCK_DIM);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    auto attrs = context->GetAttrs();
    int64_t E = *attrs->GetInt(0);
    
    gert::Shape* y_shape = context->GetOutputShape(0);
    y_shape->SetDimNum(1);
    y_shape->SetDim(0, E);
    
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_INT64);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Ptr2Ind : public OpDef {
public:
    explicit Ptr2Ind(const char* name) : OpDef(name)
    {
        this->Input("ptr")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
            
        this->Output("ind")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
            
        this->Attr("E").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Ptr2Ind);
} // namespace ops