#include "non_diag_mask_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    TilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetStorageShape().GetShapeSize();

    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t sysCoreNum = ascendcPlatform.GetCoreNum();
    if (sysCoreNum == 0) sysCoreNum = 24; 
    
    context->SetBlockDim(sysCoreNum);

    // 计算数学属性
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    int64_t M = *attrs->GetInt(0);
    int64_t N = *attrs->GetInt(1);
    int64_t k = *attrs->GetInt(2);
    //int64_t num_diag = (k < 0) ? std::min(M + k, N) : std::min(M, N - k);
    // 【修改点】：增加 std::max(0LL, ...) 防止异常输入导致负数对角线数量
    int64_t num_diag = (k < 0) ? std::max((int64_t)0, std::min(M + k, N)) : std::max((int64_t)0, std::min(M, N - k));
    uint32_t outTotalLen = totalLength + num_diag;

    // 【核心推导】：计算 target_idx 领先于 idx 的最大可能偏移量
    // 用于指导 Kernel 只需要多捞多少个输入数据，避免全量遍历
    int64_t max_diff_val = std::max(num_diag, M + std::max((int64_t)0, k) + 1);

    tiling.set_coreNum(sysCoreNum);
    tiling.set_totalLength(totalLength);
    tiling.set_outTotalLen(outTotalLen);
    tiling.set_max_diff((uint32_t)max_diff_val);
    tiling.set_M(M);
    tiling.set_N(N);
    tiling.set_k(k);
    tiling.set_num_diag(num_diag);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

// ... (以下保留你原本的 ge::InferShape, InferDataType 和 OP_ADD 注册代码，无需修改) ...

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context)
    {
        const gert::Shape *inShape = context->GetInputShape(0);
        int64_t E = inShape->GetShapeSize();
        
        // 获取所有属性：新增 M
        const auto *attrs = context->GetAttrs();
        int64_t M = *attrs->GetInt(0);
        int64_t N = *attrs->GetInt(1);
        int64_t k = *attrs->GetInt(2);
        
        // 【修改点】：同步加上防御性截断
        int64_t num_diag;
        if (k < 0) {
            num_diag = std::max((int64_t)0, std::min(M + k, N));
        } else {
            num_diag = std::max((int64_t)0, std::min(M, N - k));
        }
        int64_t outSize = E + num_diag;
    
        gert::Shape *outShape = context->GetOutputShape(0);
        outShape->SetDim(0, outSize);
        return GRAPH_SUCCESS;
    }

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_INT8);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class NonDiagMask : public OpDef {
public:
    explicit NonDiagMask(const char *name) : OpDef(name)
    {
        this->Input("row").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Input("col").ParamType(REQUIRED).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND});
        this->Output("mask").ParamType(REQUIRED).DataType({ge::DT_INT8}).Format({ge::FORMAT_ND});

        // 修复：新增 M 属性，严格对齐CUDA参数
        this->Attr("M").AttrType(REQUIRED).Int();
        this->Attr("N").AttrType(REQUIRED).Int();
        this->Attr("k").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(NonDiagMask);
}