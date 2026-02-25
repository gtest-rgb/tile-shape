/**
 * @file op_builders.cpp
 * @brief Implementation of concrete OpBuilder classes
 */

#include "op_builders.h"

namespace tile_shape {

// ============================================================================
// OpBuilder Implementation
// ============================================================================

AscppTileShape OpBuilder::GetTileShapeFromNode() {
    AscppTileShape tileShape;

    if (!node_ || !node_->GetOpDesc()) {
        return tileShape;
    }

    // Implementation should retrieve tile_shape attribute from node
    //
    // Pseudocode:
    // ge::AttrUtils::GetListInt(node_->GetOpDesc(), "tile_shape", tileShape);

    return tileShape;
}

bool OpBuilder::CanTransformTileShape(
    const AscppTileShape& inputShape,
    const AscppTileShape& targetShape,
    TileShapeTransformType& transformType) {

    // 1. Exact match - no transform needed
    if (inputShape == targetShape) {
        transformType = TileShapeTransformType::EXACT_MATCH;
        return true;
    }

    // 2. Calculate total elements
    int64_t inputElements = 1;
    for (auto val : inputShape) {
        inputElements *= val;
    }

    int64_t targetElements = 1;
    for (auto val : targetShape) {
        targetElements *= val;
    }

    // 3. View transform: same total elements, just different dimension layout
    if (inputElements == targetElements) {
        transformType = TileShapeTransformType::VIEW;
        return true;
    }

    // 4. Assemble transform: requires data reorganization
    // Default conservative: don't support Assemble
    transformType = TileShapeTransformType::NONE;
    return false;
}

int OpBuilder::InferUnifiedTileShape(
    const std::vector<AscppTileShape>& inputShapes,
    AscppTileShape& unifiedShape,
    std::string& errorMsg) {

    if (inputShapes.empty()) {
        errorMsg = "No input shapes provided";
        return npucl::FAILED;
    }

    // Default: use first input shape as unified shape
    unifiedShape = inputShapes[0];
    return npucl::SUCCESS;
}

// ============================================================================
// BaseOpBuilder Implementation
// ============================================================================

int BaseOpBuilder::InitAndValidateTileShape() {
    if (!node_ || !node_->GetOpDesc()) {
        return npucl::FAILED;
    }

    // Get TileShape from node
    AscppTileShape userTileShape = GetTileShapeFromNode();

    // If not configured, nothing to validate
    if (userTileShape.empty()) {
        return npucl::SUCCESS;
    }

    AscppTileShape inputTileShape;
    AscppTileShape outputTileShape;

    // Calculate input and output TileShapes based on user configuration
    int ret = CalculateOutputTileShape(userTileShape, outputTileShape);
    if (ret != npucl::SUCCESS) {
        return npucl::FAILED;
    }

    ret = CalculateInputTileShape(outputTileShape, inputTileShape);
    if (ret != npucl::SUCCESS) {
        return npucl::FAILED;
    }

    // Mark as user configured
    // ge::AttrUtils::SetBool(node_->GetOpDesc(), "tile_shape_user_configured", true);

    // Store results to node attributes
    // ge::AttrUtils::SetListInt(node_->GetOpDesc(), "tile_shape_input_nd", inputTileShape);
    // ge::AttrUtils::SetListInt(node_->GetOpDesc(), "tile_shape_output_nd", outputTileShape);

    // Validate constraints
    return ValidateConstraints();
}

int BaseOpBuilder::InferTileShape(
    const AscppTileShape& predecessorOutputShape,
    AscppTileShape& tileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {

    if (!node_ || !node_->GetOpDesc()) {
        return npucl::FAILED;
    }

    // Conservative strategy: use predecessor's output as our input
    inputTileShape = predecessorOutputShape;

    // Calculate our output TileShape
    int ret = CalculateOutputTileShape(inputTileShape, outputTileShape);
    if (ret != npucl::SUCCESS) {
        return npucl::FAILED;
    }

    // Final tile shape is typically the output shape
    tileShape = outputTileShape;

    return ValidateConstraints();
}

int BaseOpBuilder::ValidateConstraints() {
    // Default: no additional constraints
    return npucl::SUCCESS;
}

int BaseOpBuilder::CalculateInputTileShape(
    const AscppTileShape& outputTileShape,
    AscppTileShape& inputTileShape) {
    // Default: input shape equals output shape (for element-wise ops)
    inputTileShape = outputTileShape;
    return npucl::SUCCESS;
}

int BaseOpBuilder::CalculateOutputTileShape(
    const AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {
    // Default: output shape equals input shape (for element-wise ops)
    outputTileShape = inputTileShape;
    return npucl::SUCCESS;
}

int BaseOpBuilder::GetNumDimensions() {
    // Default: 4D (NCHW format)
    return 4;
}

bool BaseOpBuilder::ValidateDimensions(const AscppTileShape& tileShape, int expectedDims) {
    if (static_cast<int>(tileShape.size()) != expectedDims) {
        return false;
    }

    for (auto val : tileShape) {
        if (val <= 0) {
            return false;
        }
    }

    return true;
}

// ============================================================================
// ConvOpBuilder Implementation - Placeholder (requires full implementation)
// ============================================================================

int ConvOpBuilder::InitAndValidateTileShape() {
    return BaseOpBuilder::InitAndValidateTileShape();
}

int ConvOpBuilder::InferTileShape(
    const AscppTileShape& predecessorOutputShape,
    AscppTileShape& tileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {
    return BaseOpBuilder::InferTileShape(
        predecessorOutputShape, tileShape, inputTileShape, outputTileShape);
}

int ConvOpBuilder::ValidateConstraints() {
    return BaseOpBuilder::ValidateConstraints();
}

int ConvOpBuilder::CalculateInputTileShape(
    const AscppTileShape& outputTileShape,
    AscppTileShape& inputTileShape) {
    // Conv-specific: reverse calculation based on stride, pad, dilation
    // Placeholder implementation
    return BaseOpBuilder::CalculateInputTileShape(outputTileShape, inputTileShape);
}

int ConvOpBuilder::CalculateOutputTileShape(
    const AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {
    // Conv-specific: forward calculation based on stride, pad, dilation
    // Placeholder implementation
    return BaseOpBuilder::CalculateOutputTileShape(inputTileShape, outputTileShape);
}

int ConvOpBuilder::GetConvAttributes(
    std::vector<int64_t>& strides,
    std::vector<int64_t>& pads,
    std::vector<int64_t>& dilations,
    std::vector<int64_t>& kernels) {
    // Placeholder: retrieve from node attributes
    return npucl::SUCCESS;
}

int64_t ConvOpBuilder::CalculateOutputSize(
    int64_t inputSize, int64_t kernelSize, int64_t stride,
    int64_t padStart, int64_t padEnd, int64_t dilation) {
    int64_t effectiveKernel = (kernelSize - 1) * dilation + 1;
    return (inputSize + padStart + padEnd - effectiveKernel) / stride + 1;
}

int64_t ConvOpBuilder::CalculateInputSize(
    int64_t outputSize, int64_t kernelSize, int64_t stride,
    int64_t padStart, int64_t padEnd, int64_t dilation) {
    int64_t effectiveKernel = (kernelSize - 1) * dilation + 1;
    return (outputSize - 1) * stride + effectiveKernel - padStart - padEnd;
}

// ============================================================================
// MatmulOpBuilder Implementation - Placeholder
// ============================================================================

int MatmulOpBuilder::InitAndValidateTileShape() {
    return BaseOpBuilder::InitAndValidateTileShape();
}

int MatmulOpBuilder::InferTileShape(
    const AscppTileShape& predecessorOutputShape,
    AscppTileShape& tileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {
    return BaseOpBuilder::InferTileShape(
        predecessorOutputShape, tileShape, inputTileShape, outputTileShape);
}

int MatmulOpBuilder::ValidateConstraints() {
    return BaseOpBuilder::ValidateConstraints();
}

int MatmulOpBuilder::CalculateInputTileShape(
    const AscppTileShape& outputTileShape,
    AscppTileShape& inputTileShape) {
    return BaseOpBuilder::CalculateInputTileShape(outputTileShape, inputTileShape);
}

int MatmulOpBuilder::CalculateOutputTileShape(
    const AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {
    return BaseOpBuilder::CalculateOutputTileShape(inputTileShape, outputTileShape);
}

int MatmulOpBuilder::GetTransposeAttrs(bool& transposeA, bool& transposeB) {
    // Placeholder: retrieve from node attributes
    return npucl::SUCCESS;
}

// ============================================================================
// PoolOpBuilder Implementation - Placeholder
// ============================================================================

int PoolOpBuilder::InitAndValidateTileShape() {
    return BaseOpBuilder::InitAndValidateTileShape();
}

int PoolOpBuilder::InferTileShape(
    const AscppTileShape& predecessorOutputShape,
    AscppTileShape& tileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {
    return BaseOpBuilder::InferTileShape(
        predecessorOutputShape, tileShape, inputTileShape, outputTileShape);
}

int PoolOpBuilder::ValidateConstraints() {
    return BaseOpBuilder::ValidateConstraints();
}

int PoolOpBuilder::CalculateInputTileShape(
    const AscppTileShape& outputTileShape,
    AscppTileShape& inputTileShape) {
    return BaseOpBuilder::CalculateInputTileShape(outputTileShape, inputTileShape);
}

int PoolOpBuilder::CalculateOutputTileShape(
    const AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {
    return BaseOpBuilder::CalculateOutputTileShape(inputTileShape, outputTileShape);
}

int PoolOpBuilder::GetPoolAttributes(
    std::vector<int64_t>& kernelSize,
    std::vector<int64_t>& strides,
    std::vector<int64_t>& pads,
    bool& ceilMode) {
    // Placeholder: retrieve from node attributes
    return npucl::SUCCESS;
}

// ============================================================================
// ElementwiseOpBuilder Implementation
// ============================================================================

int ElementwiseOpBuilder::InitAndValidateTileShape() {
    return BaseOpBuilder::InitAndValidateTileShape();
}

int ElementwiseOpBuilder::InferTileShape(
    const AscppTileShape& predecessorOutputShape,
    AscppTileShape& tileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {
    return BaseOpBuilder::InferTileShape(
        predecessorOutputShape, tileShape, inputTileShape, outputTileShape);
}

int ElementwiseOpBuilder::ValidateConstraints() {
    return BaseOpBuilder::ValidateConstraints();
}

int ElementwiseOpBuilder::InferUnifiedTileShape(
    const std::vector<AscppTileShape>& inputShapes,
    AscppTileShape& unifiedShape,
    std::string& errorMsg) {

    if (inputShapes.empty()) {
        errorMsg = "No input shapes provided";
        return npucl::FAILED;
    }

    if (inputShapes.size() == 1) {
        unifiedShape = inputShapes[0];
        return npucl::SUCCESS;
    }

    // Use broadcast rules to find compatible shape
    unifiedShape = inputShapes[0];
    for (size_t i = 1; i < inputShapes.size(); i++) {
        if (!CanBroadcastShapes(unifiedShape, inputShapes[i])) {
            errorMsg = "Input shapes are not broadcast compatible";
            return npucl::FAILED;
        }
        unifiedShape = GetBroadcastedShape(unifiedShape, inputShapes[i]);
    }

    return npucl::SUCCESS;
}

bool ElementwiseOpBuilder::CanBroadcastShapes(
    const AscppTileShape& shape1,
    const AscppTileShape& shape2) {

    // Broadcast rule: dimensions are compatible if they are equal or one of them is 1
    size_t maxDims = std::max(shape1.size(), shape2.size());

    for (size_t i = 0; i < maxDims; i++) {
        int64_t dim1 = (i < shape1.size()) ? shape1[shape1.size() - 1 - i] : 1;
        int64_t dim2 = (i < shape2.size()) ? shape2[shape2.size() - 1 - i] : 1;

        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            return false;
        }
    }

    return true;
}

AscppTileShape ElementwiseOpBuilder::GetBroadcastedShape(
    const AscppTileShape& shape1,
    const AscppTileShape& shape2) {

    size_t maxDims = std::max(shape1.size(), shape2.size());
    AscppTileShape result(maxDims);

    for (size_t i = 0; i < maxDims; i++) {
        int64_t dim1 = (i < shape1.size()) ? shape1[shape1.size() - 1 - i] : 1;
        int64_t dim2 = (i < shape2.size()) ? shape2[shape2.size() - 1 - i] : 1;
        result[maxDims - 1 - i] = std::max(dim1, dim2);
    }

    return result;
}

int ElementwiseOpBuilder::CalculateInputTileShape(
    const AscppTileShape& outputTileShape,
    AscppTileShape& inputTileShape) {
    return BaseOpBuilder::CalculateInputTileShape(outputTileShape, inputTileShape);
}

int ElementwiseOpBuilder::CalculateOutputTileShape(
    const AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {
    return BaseOpBuilder::CalculateOutputTileShape(inputTileShape, outputTileShape);
}

// ============================================================================
// ReshapeOpBuilder Implementation - Placeholder
// ============================================================================

int ReshapeOpBuilder::InitAndValidateTileShape() {
    return BaseOpBuilder::InitAndValidateTileShape();
}

int ReshapeOpBuilder::InferTileShape(
    const AscppTileShape& predecessorOutputShape,
    AscppTileShape& tileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {
    return BaseOpBuilder::InferTileShape(
        predecessorOutputShape, tileShape, inputTileShape, outputTileShape);
}

int ReshapeOpBuilder::ValidateConstraints() {
    return BaseOpBuilder::ValidateConstraints();
}

int ReshapeOpBuilder::GetTargetShape(std::vector<int64_t>& targetShape) {
    // Placeholder: retrieve from node attributes
    return npucl::SUCCESS;
}

// ============================================================================
// ActivationOpBuilder Implementation - Placeholder
// ============================================================================

int ActivationOpBuilder::InitAndValidateTileShape() {
    return BaseOpBuilder::InitAndValidateTileShape();
}

int ActivationOpBuilder::InferTileShape(
    const AscppTileShape& predecessorOutputShape,
    AscppTileShape& tileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {
    return BaseOpBuilder::InferTileShape(
        predecessorOutputShape, tileShape, inputTileShape, outputTileShape);
}

int ActivationOpBuilder::ValidateConstraints() {
    return BaseOpBuilder::ValidateConstraints();
}

// ============================================================================
// BatchNormOpBuilder Implementation - Placeholder
// ============================================================================

int BatchNormOpBuilder::InitAndValidateTileShape() {
    return BaseOpBuilder::InitAndValidateTileShape();
}

int BatchNormOpBuilder::InferTileShape(
    const AscppTileShape& predecessorOutputShape,
    AscppTileShape& tileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {
    return BaseOpBuilder::InferTileShape(
        predecessorOutputShape, tileShape, inputTileShape, outputTileShape);
}

int BatchNormOpBuilder::ValidateConstraints() {
    return BaseOpBuilder::ValidateConstraints();
}

// ============================================================================
// ReduceOpBuilder Implementation - Placeholder
// ============================================================================

int ReduceOpBuilder::InitAndValidateTileShape() {
    return BaseOpBuilder::InitAndValidateTileShape();
}

int ReduceOpBuilder::InferTileShape(
    const AscppTileShape& predecessorOutputShape,
    AscppTileShape& tileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {
    return BaseOpBuilder::InferTileShape(
        predecessorOutputShape, tileShape, inputTileShape, outputTileShape);
}

int ReduceOpBuilder::ValidateConstraints() {
    return BaseOpBuilder::ValidateConstraints();
}

int ReduceOpBuilder::GetReduceAttributes(std::vector<int64_t>& axes, bool& keepDims) {
    // Placeholder: retrieve from node attributes
    return npucl::SUCCESS;
}

// ============================================================================
// CastOpBuilder Implementation (passthrough)
// ============================================================================

int CastOpBuilder::InitAndValidateTileShape() {
    return BaseOpBuilder::InitAndValidateTileShape();
}

int CastOpBuilder::InferTileShape(
    const AscppTileShape& predecessorOutputShape,
    AscppTileShape& tileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {
    // Cast is a passthrough: shape unchanged
    inputTileShape = predecessorOutputShape;
    outputTileShape = predecessorOutputShape;
    tileShape = predecessorOutputShape;
    return npucl::SUCCESS;
}

int CastOpBuilder::ValidateConstraints() {
    return BaseOpBuilder::ValidateConstraints();
}

// ============================================================================
// TransDataOpBuilder Implementation (passthrough with format conversion)
// ============================================================================

int TransDataOpBuilder::InitAndValidateTileShape() {
    return BaseOpBuilder::InitAndValidateTileShape();
}

int TransDataOpBuilder::InferTileShape(
    const AscppTileShape& predecessorOutputShape,
    AscppTileShape& tileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {
    // TransData may change shape based on format conversion
    inputTileShape = predecessorOutputShape;

    std::string srcFormat, dstFormat;
    if (GetTransDataFormats(srcFormat, dstFormat) == npucl::SUCCESS) {
        TransformTileShapeByFormat(inputTileShape, srcFormat, dstFormat, outputTileShape);
    } else {
        outputTileShape = inputTileShape;
    }

    tileShape = outputTileShape;
    return npucl::SUCCESS;
}

int TransDataOpBuilder::ValidateConstraints() {
    return BaseOpBuilder::ValidateConstraints();
}

int TransDataOpBuilder::CalculateInputTileShape(
    const AscppTileShape& outputTileShape,
    AscppTileShape& inputTileShape) {
    std::string srcFormat, dstFormat;
    if (GetTransDataFormats(srcFormat, dstFormat) == npucl::SUCCESS) {
        // Reverse transformation
        TransformTileShapeByFormat(outputTileShape, dstFormat, srcFormat, inputTileShape);
    } else {
        inputTileShape = outputTileShape;
    }
    return npucl::SUCCESS;
}

int TransDataOpBuilder::CalculateOutputTileShape(
    const AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {
    std::string srcFormat, dstFormat;
    if (GetTransDataFormats(srcFormat, dstFormat) == npucl::SUCCESS) {
        TransformTileShapeByFormat(inputTileShape, srcFormat, dstFormat, outputTileShape);
    } else {
        outputTileShape = inputTileShape;
    }
    return npucl::SUCCESS;
}

int TransDataOpBuilder::GetTransDataFormats(std::string& srcFormat, std::string& dstFormat) {
    // Placeholder: retrieve from node attributes
    // ge::AttrUtils::GetStr(node_->GetOpDesc(), "src_format", srcFormat);
    // ge::AttrUtils::GetStr(node_->GetOpDesc(), "dst_format", dstFormat);
    return npucl::SUCCESS;
}

int TransDataOpBuilder::TransformTileShapeByFormat(
    const AscppTileShape& srcShape,
    const std::string& srcFormat,
    const std::string& dstFormat,
    AscppTileShape& dstShape) {

    // Placeholder: implement format transformation logic
    // Example: NCHW (1, 64, 224, 224) <-> NC1HWC0 (1, 4, 224, 224, 16)

    if (srcFormat == dstFormat) {
        dstShape = srcShape;
        return npucl::SUCCESS;
    }

    // NCHW -> NC1HWC0
    if (srcFormat == "NCHW" && dstFormat == "NC1HWC0") {
        if (srcShape.size() == 4) {
            int64_t N = srcShape[0];
            int64_t C = srcShape[1];
            int64_t H = srcShape[2];
            int64_t W = srcShape[3];
            int64_t C0 = 16;  // Default C0 value
            int64_t C1 = (C + C0 - 1) / C0;
            dstShape = {N, C1, H, W, C0};
            return npucl::SUCCESS;
        }
    }

    // NC1HWC0 -> NCHW
    if (srcFormat == "NC1HWC0" && dstFormat == "NCHW") {
        if (srcShape.size() == 5) {
            int64_t N = srcShape[0];
            int64_t C1 = srcShape[1];
            int64_t H = srcShape[2];
            int64_t W = srcShape[3];
            int64_t C0 = srcShape[4];
            int64_t C = C1 * C0;
            dstShape = {N, C, H, W};
            return npucl::SUCCESS;
        }
    }

    // Default: no transformation
    dstShape = srcShape;
    return npucl::SUCCESS;
}

// ============================================================================
// ConcatOpBuilder Implementation
// ============================================================================

int ConcatOpBuilder::InitAndValidateTileShape() {
    return BaseOpBuilder::InitAndValidateTileShape();
}

int ConcatOpBuilder::InferTileShape(
    const AscppTileShape& predecessorOutputShape,
    AscppTileShape& tileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {
    return BaseOpBuilder::InferTileShape(
        predecessorOutputShape, tileShape, inputTileShape, outputTileShape);
}

int ConcatOpBuilder::ValidateConstraints() {
    return BaseOpBuilder::ValidateConstraints();
}

int ConcatOpBuilder::InferUnifiedTileShape(
    const std::vector<AscppTileShape>& inputShapes,
    AscppTileShape& unifiedShape,
    std::string& errorMsg) {

    if (inputShapes.empty()) {
        errorMsg = "No input shapes provided for Concat";
        return npucl::FAILED;
    }

    if (inputShapes.size() == 1) {
        unifiedShape = inputShapes[0];
        return npucl::SUCCESS;
    }

    // For Concat: non-concat dimensions must be equal
    // Use first input as base, validate others
    unifiedShape = inputShapes[0];
    int64_t concatAxis = GetConcatAxis();

    for (size_t i = 1; i < inputShapes.size(); i++) {
        const auto& shape = inputShapes[i];

        if (shape.size() != unifiedShape.size()) {
            errorMsg = "Concat input shapes have different dimensions";
            return npucl::FAILED;
        }

        for (size_t dim = 0; dim < shape.size(); dim++) {
            if (static_cast<int64_t>(dim) != concatAxis) {
                if (shape[dim] != unifiedShape[dim]) {
                    errorMsg = "Concat non-concat dimensions must match";
                    return npucl::FAILED;
                }
            } else {
                // For concat axis, use minimum (conservative)
                unifiedShape[dim] = std::min(unifiedShape[dim], shape[dim]);
            }
        }
    }

    return npucl::SUCCESS;
}

int64_t ConcatOpBuilder::GetConcatAxis() {
    // Placeholder: retrieve from node attributes
    // int64_t axis = 0;
    // ge::AttrUtils::GetInt(node_->GetOpDesc(), "axis", axis);
    return 0;
}

// ============================================================================
// OpBuilder Registration
// ============================================================================

void RegisterBuiltinOpBuilders() {
    auto& factory = OpBuilderFactory::Instance();

    // Register element-wise operators
    factory.RegisterBuilder("Add", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ElementwiseOpBuilder>();
    });
    factory.RegisterBuilder("Sub", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ElementwiseOpBuilder>();
    });
    factory.RegisterBuilder("Mul", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ElementwiseOpBuilder>();
    });
    factory.RegisterBuilder("Div", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ElementwiseOpBuilder>();
    });
    factory.RegisterBuilder("RealDiv", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ElementwiseOpBuilder>();
    });
    factory.RegisterBuilder("Maximum", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ElementwiseOpBuilder>();
    });
    factory.RegisterBuilder("Minimum", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ElementwiseOpBuilder>();
    });

    // Register activation operators
    factory.RegisterBuilder("Relu", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ActivationOpBuilder>();
    });
    factory.RegisterBuilder("Sigmoid", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ActivationOpBuilder>();
    });
    factory.RegisterBuilder("Tanh", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ActivationOpBuilder>();
    });
    factory.RegisterBuilder("ReLU6", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ActivationOpBuilder>();
    });
    factory.RegisterBuilder("LeakyRelu", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ActivationOpBuilder>();
    });
    factory.RegisterBuilder("Elu", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ActivationOpBuilder>();
    });
    factory.RegisterBuilder("Gelu", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ActivationOpBuilder>();
    });
    factory.RegisterBuilder("Swish", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ActivationOpBuilder>();
    });
    factory.RegisterBuilder("Softmax", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ActivationOpBuilder>();
    });

    // Register conv operators
    factory.RegisterBuilder("Conv2D", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ConvOpBuilder>();
    });
    factory.RegisterBuilder("Conv2DTranspose", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ConvOpBuilder>();
    });
    factory.RegisterBuilder("Conv3D", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ConvOpBuilder>();
    });
    factory.RegisterBuilder("DepthwiseConv2D", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ConvOpBuilder>();
    });

    // Register pool operators
    factory.RegisterBuilder("MaxPool", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<PoolOpBuilder>();
    });
    factory.RegisterBuilder("AvgPool", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<PoolOpBuilder>();
    });
    factory.RegisterBuilder("MaxPoolV2", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<PoolOpBuilder>();
    });
    factory.RegisterBuilder("AvgPoolV2", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<PoolOpBuilder>();
    });
    factory.RegisterBuilder("MaxPool3D", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<PoolOpBuilder>();
    });
    factory.RegisterBuilder("AvgPool3D", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<PoolOpBuilder>();
    });

    // Register matmul operators
    factory.RegisterBuilder("MatMul", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<MatmulOpBuilder>();
    });
    factory.RegisterBuilder("BatchMatMul", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<MatmulOpBuilder>();
    });
    factory.RegisterBuilder("BatchMatMulV2", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<MatmulOpBuilder>();
    });

    // Register reshape operators
    factory.RegisterBuilder("Reshape", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ReshapeOpBuilder>();
    });
    factory.RegisterBuilder("Flatten", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ReshapeOpBuilder>();
    });
    factory.RegisterBuilder("ExpandDims", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ReshapeOpBuilder>();
    });
    factory.RegisterBuilder("Squeeze", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ReshapeOpBuilder>();
    });

    // Register batchnorm operators
    factory.RegisterBuilder("BatchNorm", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<BatchNormOpBuilder>();
    });
    factory.RegisterBuilder("BatchNormalization", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<BatchNormOpBuilder>();
    });
    factory.RegisterBuilder("InstanceNorm", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<BatchNormOpBuilder>();
    });
    factory.RegisterBuilder("LayerNorm", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<BatchNormOpBuilder>();
    });

    // Register reduce operators
    factory.RegisterBuilder("ReduceSum", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ReduceOpBuilder>();
    });
    factory.RegisterBuilder("ReduceMean", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ReduceOpBuilder>();
    });
    factory.RegisterBuilder("ReduceMax", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ReduceOpBuilder>();
    });
    factory.RegisterBuilder("ReduceMin", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ReduceOpBuilder>();
    });
    factory.RegisterBuilder("ReduceProd", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ReduceOpBuilder>();
    });
    factory.RegisterBuilder("ReduceAll", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ReduceOpBuilder>();
    });
    factory.RegisterBuilder("ReduceAny", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ReduceOpBuilder>();
    });

    // Register passthrough operators
    factory.RegisterBuilder("Cast", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<CastOpBuilder>();
    });
    factory.RegisterBuilder("TransData", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<TransDataOpBuilder>();
    });

    // Register concat operators
    factory.RegisterBuilder("Concat", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ConcatOpBuilder>();
    });
    factory.RegisterBuilder("ConcatV2", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ConcatOpBuilder>();
    });
    factory.RegisterBuilder("ConcatD", []() -> std::unique_ptr<OpBuilder> {
        return std::make_unique<ConcatOpBuilder>();
    });
}

} // namespace tile_shape
