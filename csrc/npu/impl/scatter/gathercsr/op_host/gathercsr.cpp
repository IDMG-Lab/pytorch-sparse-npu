
#include "gathercsr_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
using namespace std;
namespace optiling
{
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {

        GathercsrTilingData tiling;
        const gert::StorageShape *x0_shape = context->GetInputShape(0);
        const gert::StorageShape *x1_shape = context->GetInputShape(1);
        auto shapesize0 = context->GetInputTensor(0)->GetShapeSize(); // 表示src,大小为N
        auto shapesize1 = context->GetInputTensor(1)->GetShapeSize();

        tiling.set_N(shapesize0);
        // cout << "shapesize0:" << shapesize0 << endl;
        // cout << "shapesize1:" << shapesize1 << endl;

        // 打印outshape
        const gert::StorageShape *out_shape = context->GetOutputShape(0);
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        // auto aivNum = ascendcPlatform.GetCoreNumAiv();
        auto aivNum = 1;
        context->SetBlockDim(aivNum);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
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
    class Gathercsr : public OpDef
    {
    public:
        explicit Gathercsr(const char *name) : OpDef(name)
        {
            this->Input("src")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32, ge::DT_INT64})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("ptr")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32, ge::DT_INT64})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32, ge::DT_INT64})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

            this->AICore()
                .SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend910b");
        }
    };

    OP_ADD(Gathercsr);
}
