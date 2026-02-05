
#include "ind2ptr_tiling.h"
#include "register/op_def_registry.h"

namespace optiling
{
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {

        Ind2ptrTilingData tiling;
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const uint32_t *M = attrs->GetAttrPointer<uint32_t>(0);

        tiling.set_M(*M);
        const gert::StorageShape *x1_shape = context->GetInputShape(0);
        int32_t indsize = 1;
        for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
            indsize *= x1_shape->GetStorageShape().GetDim(i);

        tiling.set_indsize(indsize);
        context->SetBlockDim(1);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        return ge::GRAPH_SUCCESS;
    }
}

namespace ge
{
    static ge::graphStatus InferShape(gert::InferShapeContext *context)
    {
        const gert::Shape *x1_shape = context->GetInputShape(0);
        gert::Shape *y_shape = context->GetOutputShape(0);
        *y_shape = *x1_shape;
        return GRAPH_SUCCESS;
    }
    static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
    {
        const auto inputDataType = context->GetInputDataType(0);
        context->SetOutputDataType(0, inputDataType);
        return ge::GRAPH_SUCCESS;
    }
}

namespace ops
{
    class Ind2ptr : public OpDef
    {
    public:
        explicit Ind2ptr(const char *name) : OpDef(name)
        {
            this->Input("indices")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32, ge::DT_INT64})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("ptr")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32, ge::DT_INT64})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("M").Int();

            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

            this->AICore()
                .SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend910b");
        }
    };

    OP_ADD(Ind2ptr);
}
