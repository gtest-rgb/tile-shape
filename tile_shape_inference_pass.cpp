/**
 * @file tile_shape_inference_pass.cpp
 * @brief Implementation of TileShape Inference Pass
 */

#include "tile_shape_inference_pass.h"
#include <algorithm>
#include <sstream>
#include <map>
#include <functional>

namespace tile_shape {

// ============================================================================
// OpBuilderFactory Implementation
// ============================================================================

struct OpBuilderFactory::Impl {
    std::map<std::string, std::function<std::unique_ptr<OpBuilder>()>> builders;
};

OpBuilderFactory& OpBuilderFactory::Instance() {
    static OpBuilderFactory instance;
    return instance;
}

std::unique_ptr<OpBuilder> OpBuilderFactory::CreateOpBuilder(
    ge::NodePtr node,
    const std::string& libName) {
    if (!impl_) {
        impl_ = std::make_unique<Impl>();
    }

    if (!node || !node->GetOpDesc()) {
        return nullptr;
    }

    std::string opType = node->GetOpDesc()->GetType();
    auto it = impl_->builders.find(opType);
    if (it != impl_->builders.end()) {
        return it->second();
    }

    // Return default generic builder if no specific builder is registered
    return nullptr;
}

void OpBuilderFactory::RegisterBuilder(
    const std::string& opType,
    std::function<std::unique_ptr<OpBuilder>()> creator) {
    if (!impl_) {
        impl_ = std::make_unique<Impl>();
    }
    impl_->builders[opType] = std::move(creator);
}

// ============================================================================
// TileShapeInferencePass Implementation
// ============================================================================

struct TileShapeInferencePass::Impl {
    // Storage for intermediate results
    std::map<std::string, IoTileShapePair> inferredShapes;
};

TileShapeInferencePass::TileShapeInferencePass()
    : impl_(std::make_unique<Impl>()) {}

TileShapeInferencePass::~TileShapeInferencePass() = default;

fe::OptStatus TileShapeInferencePass::Run(ge::ComputeGraphPtr graph) {
    DUMP_GRAPH(graph, "before_tile_shape_inference_pass");

    // ========== Step 1: 初始化与输入约束检查 ==========

    // 1.1 检查所有输入节点是否配置 TileShape
    ASCPP_CHK_TRUE_RETURN_FAIL(
        CheckInputNodesConstraint(graph) == npucl::FAILED,
        "Input node constraint check failed");

    // 1.2 获取拓扑顺序
    std::vector<ge::NodePtr> topoNodes;
    ASCPP_CHK_TRUE_RETURN_FAIL(
        GetTopologicalOrder(graph, topoNodes) == npucl::FAILED,
        "GetTopologicalOrder failed");

    // 1.3 初始化用户配置的节点
    ASCPP_CHK_TRUE_RETURN_FAIL(
        InitializeUserConfiguredNodes(graph, topoNodes) == npucl::FAILED,
        "Initialize user configured nodes failed");

    DUMP_GRAPH(graph, "middle_tile_shape_inference_pass");

    // ========== Step 2: 推导 TileShape ==========

    ASCPP_CHK_TRUE_RETURN_FAIL(
        RunInference(graph, topoNodes) == npucl::FAILED,
        "TileShape inference failed");

    // ========== Step 3: 最终验证 ==========

    ASCPP_CHK_TRUE_RETURN_FAIL(
        FinalValidation(graph) == npucl::FAILED,
        "Final validation failed");

    DUMP_GRAPH(graph, "after_tile_shape_inference_pass");

    return fe::OptStatus::SUCCESS;
}

// ============================================================================
// Step 1: Initialization Methods
// ============================================================================

int TileShapeInferencePass::CheckInputNodesConstraint(ge::ComputeGraphPtr graph) {
    // Implementation should iterate over all input nodes in the graph
    // and verify they have TileShape configured
    //
    // Pseudocode:
    // for (auto inputNode : graph->GetInputNodes()) {
    //     AscppTileShape tileShape = GetTileShapeFromNode(inputNode);
    //     if (tileShape.empty()) {
    //         // Error E001: Input node must have TileShape configured
    //         return npucl::FAILED;
    //     }
    // }

    return npucl::SUCCESS;
}

int TileShapeInferencePass::GetTopologicalOrder(
    ge::ComputeGraphPtr graph,
    std::vector<ge::NodePtr>& topoNodes) {
    // Implementation should return nodes in topological order
    //
    // Pseudocode:
    // topoNodes = graph->GetTopologicalNodes();
    // if (topoNodes.empty()) {
    //     return npucl::FAILED;
    // }

    return npucl::SUCCESS;
}

AscppTileShape TileShapeInferencePass::GetTileShapeFromNode(ge::NodePtr node) {
    AscppTileShape tileShape;

    if (!node || !node->GetOpDesc()) {
        return tileShape;
    }

    // Implementation should retrieve tile_shape attribute from node
    //
    // Pseudocode:
    // ge::AttrUtils::GetListInt(node->GetOpDesc(), "tile_shape", tileShape);

    return tileShape;
}

int TileShapeInferencePass::InitializeUserConfiguredNodes(
    ge::ComputeGraphPtr graph,
    const std::vector<ge::NodePtr>& topoNodes) {

    for (auto node : topoNodes) {
        AscppTileShape currentTileShape = GetTileShapeFromNode(node);

        if (!currentTileShape.empty()) {
            // 标记为用户配置
            // ge::AttrUtils::SetBool(node->GetOpDesc(), "tile_shape_user_configured", true);

            // 调用 OpBuilder 初始化并校验
            auto opBuilder = OpBuilderFactory::Instance().CreateOpBuilder(node, "ascendcpp_lib");
            ASCPP_CHK_NULL_RETURN_FAIL(
                opBuilder,
                "Can not create opbuilder for op[%s]",
                node->GetOpDesc()->GetName().c_str());

            AscppTileShape inputTileShape;
            AscppTileShape outputTileShape;

            ASCPP_CHK_TRUE_RETURN_FAIL(
                opBuilder->InitAndValidateTileShape(
                    node, currentTileShape, inputTileShape, outputTileShape) == npucl::FAILED,
                "InitAndValidateTileShape failed for op[%s]",
                node->GetOpDesc()->GetName().c_str());

            // 存储计算结果到节点属性
            // ge::AttrUtils::SetListInt(node->GetOpDesc(), "tile_shape_input_nd", inputTileShape);
            // ge::AttrUtils::SetListInt(node->GetOpDesc(), "tile_shape_output_nd", outputTileShape);
        } else {
            // 未配置：设置默认值
            // ge::AttrUtils::SetListInt(node->GetOpDesc(), "tile_shape_input_nd", {-1, -1, ...});
            // ge::AttrUtils::SetListInt(node->GetOpDesc(), "tile_shape_output_nd", {-1, -1, ...});
        }
    }

    return npucl::SUCCESS;
}

// ============================================================================
// Step 2: Inference Methods
// ============================================================================

int TileShapeInferencePass::RunInference(
    ge::ComputeGraphPtr graph,
    const std::vector<ge::NodePtr>& topoNodes) {

    for (auto node : topoNodes) {
        std::string nodeName = node->GetOpDesc()->GetName();
        std::string nodeType = node->GetOpDesc()->GetType();

        // 跳过数据相关节点
        if (IsDataRelatedOp(nodeType)) {
            continue;
        }

        AscppTileShape currentTileShape = GetTileShapeFromNode(node);

        // 跳过已配置节点（保护用户配置）
        bool isUserConfigured = false;
        // ge::AttrUtils::GetBool(node->GetOpDesc(), "tile_shape_user_configured", isUserConfigured);
        if (isUserConfigured || !currentTileShape.empty()) {
            continue;
        }

        // 保守策略：优先从前驱推导
        AscppTileShape nodeInferredNdInputShape;
        AscppTileShape nodeInferredNdOutputShape;

        ASCPP_CHK_TRUE_RETURN_FAIL(
            ConservativeTileShapeInference(node, nodeInferredNdInputShape, nodeInferredNdOutputShape) == npucl::FAILED,
            "ConservativeTileShapeInference failed for node %s. Suggestion: Configure TileShape explicitly",
            nodeName.c_str());

        // 设置推导结果
        IoTileShapePair nodeInferredIoShape = std::make_pair(
            nodeInferredNdInputShape, nodeInferredNdOutputShape);

        AscppTileShape finalTileShape;
        ASCPP_CHK_TRUE_RETURN_FAIL(
            ConvertToAscppTileShape(nodeInferredIoShape, node, finalTileShape) == npucl::FAILED,
            "ConvertToAscppTileShape failed for node %s",
            nodeName.c_str());

        // ge::AttrUtils::SetListInt(node->GetOpDesc(), "tile_shape", finalTileShape);
        // ge::AttrUtils::SetListInt(node->GetOpDesc(), "tile_shape_input_nd", nodeInferredNdInputShape);
        // ge::AttrUtils::SetListInt(node->GetOpDesc(), "tile_shape_output_nd", nodeInferredNdOutputShape);
    }

    return npucl::SUCCESS;
}

int TileShapeInferencePass::ConservativeTileShapeInference(
    ge::NodePtr node,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {

    // 获取前驱节点的 outputTileShape
    AscppTileShape predecessorOutputShape;
    ASCPP_CHK_TRUE_RETURN_FAIL(
        GetPredecessorOutputTileShape(node, predecessorOutputShape) == npucl::FAILED,
        "Failed to get predecessor output TileShape");

    // 调用 OpBuilder 进行推导
    auto opBuilder = OpBuilderFactory::Instance().CreateOpBuilder(node, "ascendcpp_lib");
    ASCPP_CHK_NULL_RETURN_FAIL(
        opBuilder,
        "Cannot create OpBuilder for op[%s]",
        node->GetOpDesc()->GetName().c_str());

    AscppTileShape tileShape;
    ASCPP_CHK_TRUE_RETURN_FAIL(
        opBuilder->InferTileShape(
            node, predecessorOutputShape, tileShape, inputTileShape, outputTileShape) == npucl::FAILED,
        "InferTileShape failed for op[%s]",
        node->GetOpDesc()->GetName().c_str());

    return npucl::SUCCESS;
}

int TileShapeInferencePass::GetPredecessorOutputTileShape(
    ge::NodePtr node,
    AscppTileShape& predecessorOutputShape) {

    // Implementation should:
    // 1. Get predecessor nodes from the graph
    // 2. Retrieve tile_shape_output_nd from the predecessor
    //
    // Pseudocode:
    // auto predecessors = node->GetInNodes();
    // if (predecessors.empty()) {
    //     return npucl::FAILED;
    // }
    //
    // // Use first predecessor's output shape (conservative strategy)
    // auto predNode = predecessors[0];
    // ge::AttrUtils::GetListInt(predNode->GetOpDesc(), "tile_shape_output_nd", predecessorOutputShape);

    return npucl::SUCCESS;
}

int TileShapeInferencePass::ConvertToAscppTileShape(
    const IoTileShapePair& ioShape,
    ge::NodePtr node,
    AscppTileShape& tileShape) {

    // Implementation should convert the input/output pair to a single TileShape
    // based on the operator's requirements
    //
    // Default: use output shape as the tile shape
    tileShape = ioShape.second;

    return npucl::SUCCESS;
}

// ============================================================================
// Step 3: Validation Methods
// ============================================================================

int TileShapeInferencePass::FinalValidation(ge::ComputeGraphPtr graph) {
    // Implementation should:
    // 1. Check all nodes have valid tile_shape
    // 2. Validate graph consistency
    // 3. Report any remaining issues

    // Pseudocode:
    // for (auto node : graph->GetAllNodes()) {
    //     AscppTileShape tileShape = GetTileShapeFromNode(node);
    //     if (!IsValidTileShape(tileShape)) {
    //         // Report error
    //         return npucl::FAILED;
    //     }
    // }
    //
    // return ValidateGraphConsistency(graph);

    return npucl::SUCCESS;
}

int TileShapeInferencePass::ValidateGraphConsistency(ge::ComputeGraphPtr graph) {
    // Implementation should validate that connected nodes have compatible TileShapes
    //
    // Pseudocode:
    // for (auto edge : graph->GetAllEdges()) {
    //     auto srcNode = edge->GetSrcNode();
    //     auto dstNode = edge->GetDstNode();
    //
    //     AscppTileShape srcOutputShape;
    //     ge::AttrUtils::GetListInt(srcNode->GetOpDesc(), "tile_shape_output_nd", srcOutputShape);
    //
    //     AscppTileShape dstInputShape;
    //     ge::AttrUtils::GetListInt(dstNode->GetOpDesc(), "tile_shape_input_nd", dstInputShape);
    //
    //     if (srcOutputShape != dstInputShape) {
    //         // Inconsistency detected
    //         return npucl::FAILED;
    //     }
    // }

    return npucl::SUCCESS;
}

// ============================================================================
// Utility Methods
// ============================================================================

bool TileShapeInferencePass::IsDataRelatedOp(const std::string& opType) {
    return std::find(kDataRelatedOps.begin(), kDataRelatedOps.end(), opType) != kDataRelatedOps.end();
}

bool TileShapeInferencePass::IsValidTileShape(const AscppTileShape& tileShape) {
    if (tileShape.empty()) {
        return false;
    }

    // Check for invalid values (-1 indicates uninferred)
    for (auto val : tileShape) {
        if (val <= 0) {
            return false;
        }
    }

    return true;
}

std::string TileShapeInferencePass::FormatError(
    ErrorCode code,
    const std::string& nodeName,
    const std::string& suggestion) {

    std::ostringstream oss;

    switch (code) {
        case ErrorCode::E001_INPUT_NODE_NOT_CONFIGURED:
            oss << "E001: Input node '" << nodeName << "' must have TileShape configured";
            break;
        case ErrorCode::E002_VALIDATION_FAILED:
            oss << "E002: TileShape validation failed for op '" << nodeName << "'";
            break;
        case ErrorCode::E003_INFER_FAILED:
            oss << "E003: Cannot infer TileShape for op '" << nodeName << "'";
            break;
        case ErrorCode::E004_BUFFER_CONSTRAINT_VIOLATION:
            oss << "E004: TileShape violates buffer constraints for op '" << nodeName << "'";
            break;
        case ErrorCode::E005_OPBUILDER_CREATE_FAILED:
            oss << "E005: Cannot create OpBuilder for op '" << nodeName << "'";
            break;
    }

    if (!suggestion.empty()) {
        oss << ". Suggestion: " << suggestion;
    }

    return oss.str();
}

} // namespace tile_shape
