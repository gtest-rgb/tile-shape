/**
 * @file op_builders.cpp
 * @brief Implementation of concrete OpBuilder classes
 */

#include "op_builders.h"

namespace tile_shape {

// ============================================================================
// BaseOpBuilder Implementation
// ============================================================================

int BaseOpBuilder::InitAndValidateTileShape(
    ge::NodePtr node,
    const AscppTileShape& userTileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {

    if (!node || !node->GetOpDesc()) {
        return npucl::FAILED;
    }

    // Calculate input and output TileShapes based on user configuration
    int ret = CalculateOutputTileShape(node, userTileShape, outputTileShape);
    if (ret != npucl::SUCCESS) {
        return npucl::FAILED;
    }

    ret = CalculateInputTileShape(node, outputTileShape, inputTileShape);
    if (ret != npucl::SUCCESS) {
        return npucl::FAILED;
    }

    // Validate constraints
    return ValidateConstraints(node);
}

int BaseOpBuilder::InferTileShape(
    ge::NodePtr node,
    const AscppTileShape& predecessorOutputShape,
    AscppTileShape& tileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {

    if (!node || !node->GetOpDesc()) {
        return npucl::FAILED;
    }

    // Conservative strategy: use predecessor's output as our input
    inputTileShape = predecessorOutputShape;

    // Calculate our output TileShape
    int ret = CalculateOutputTileShape(node, inputTileShape, outputTileShape);
    if (ret != npucl::SUCCESS) {
        return npucl::FAILED;
    }

    // Final tile shape is typically the output shape
    tileShape = outputTileShape;

    return ValidateConstraints(node);
}

int BaseOpBuilder::ValidateConstraints(ge::NodePtr node) {
    // Default: no additional constraints
    return npucl::SUCCESS;
}

int BaseOpBuilder::CalculateInputTileShape(
    ge::NodePtr node,
    const AscppTileShape& outputTileShape,
    AscppTileShape& inputTileShape) {

    // Default: input shape equals output shape (for element-wise ops)
    inputTileShape = outputTileShape;
    return npucl::SUCCESS;
}

int BaseOpBuilder::CalculateOutputTileShape(
    ge::NodePtr node,
    const AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {

    // Default: output shape equals input shape (for element-wise ops)
    outputTileShape = inputTileShape;
    return npucl::SUCCESS;
}

int BaseOpBuilder::GetNumDimensions(ge::NodePtr node) {
    // Default: 4D (NCHW format)
    return 4;
}

bool BaseOpBuilder::ValidateDimensions(const AscppTileShape& tileShape, int expectedDims) {
    if (tileShape.size() != static_cast<size_t>(expectedDims)) {
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
// OpBuilder Default Implementations for N2/N3
// ============================================================================

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
    ge::NodePtr node,
    const std::vector<AscppTileShape>& inputShapes,
    AscppTileShape& unifiedShape,
    std::string& errorMsg) {

    // Default implementation: require all shapes to be equal
    if (inputShapes.empty()) {
        errorMsg = "No input shapes provided";
        return npucl::FAILED;
    }

    unifiedShape = inputShapes[0];

    for (size_t i = 1; i < inputShapes.size(); i++) {
        if (inputShapes[i] != unifiedShape) {
            errorMsg = "Input shapes are not equal (default strategy requires equal shapes)";
            return npucl::FAILED;
        }
    }

    return npucl::SUCCESS;
}

// ============================================================================
// ConvOpBuilder Implementation
// ============================================================================

int ConvOpBuilder::InitAndValidateTileShape(
    ge::NodePtr node,
    const AscppTileShape& userTileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {

    // For conv, user tile shape is typically the output tile shape
    outputTileShape = userTileShape;

    // Calculate corresponding input tile shape
    int ret = CalculateInputTileShape(node, outputTileShape, inputTileShape);
    if (ret != npucl::SUCCESS) {
        return npucl::FAILED;
    }

    return ValidateConstraints(node);
}

int ConvOpBuilder::InferTileShape(
    ge::NodePtr node,
    const AscppTileShape& predecessorOutputShape,
    AscppTileShape& tileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {

    // For conv, predecessor's output is our input (feature map)
    inputTileShape = predecessorOutputShape;

    // Calculate output tile shape based on conv parameters
    int ret = CalculateOutputTileShape(node, inputTileShape, outputTileShape);
    if (ret != npucl::SUCCESS) {
        return npucl::FAILED;
    }

    // Final tile shape is the output shape
    tileShape = outputTileShape;

    return ValidateConstraints(node);
}

int ConvOpBuilder::ValidateConstraints(ge::NodePtr node) {
    // Validate buffer constraints specific to convolution
    // Implementation depends on hardware-specific limits
    return npucl::SUCCESS;
}

int ConvOpBuilder::CalculateInputTileShape(
    ge::NodePtr node,
    const AscppTileShape& outputTileShape,
    AscppTileShape& inputTileShape) {

    std::vector<int64_t> strides, pads, dilations, kernels;
    int ret = GetConvAttributes(node, strides, pads, dilations, kernels);
    if (ret != npucl::SUCCESS) {
        return npucl::FAILED;
    }

    // Output tile shape: [N, C_out, H_out, W_out]
    // Input tile shape: [N, C_in, H_in, W_in]
    inputTileShape.resize(outputTileShape.size());

    // N dimension stays the same
    inputTileShape[0] = outputTileShape[0];

    // C_in needs to be determined from the weight tensor
    // For now, we'll keep it as is (will be adjusted based on weights)
    inputTileShape[1] = outputTileShape[1]; // This may need adjustment

    // H and W dimensions need to be calculated based on conv params
    // For 2D conv with pads = [top, bottom, left, right]
    if (strides.size() >= 2 && kernels.size() >= 2) {
        // H dimension
        int64_t padTop = pads.size() >= 4 ? pads[0] : 0;
        int64_t padBottom = pads.size() >= 4 ? pads[1] : 0;
        int64_t padLeft = pads.size() >= 4 ? pads[2] : 0;
        int64_t padRight = pads.size() >= 4 ? pads[3] : 0;

        inputTileShape[2] = CalculateInputSize(
            outputTileShape[2], kernels[0], strides[0],
            padTop, padBottom, dilations.size() > 0 ? dilations[0] : 1);

        // W dimension
        inputTileShape[3] = CalculateInputSize(
            outputTileShape[3], kernels[1], strides[1],
            padLeft, padRight, dilations.size() > 1 ? dilations[1] : 1);
    }

    return npucl::SUCCESS;
}

int ConvOpBuilder::CalculateOutputTileShape(
    ge::NodePtr node,
    const AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {

    std::vector<int64_t> strides, pads, dilations, kernels;
    int ret = GetConvAttributes(node, strides, pads, dilations, kernels);
    if (ret != npucl::SUCCESS) {
        return npucl::FAILED;
    }

    // Input tile shape: [N, C_in, H_in, W_in]
    // Output tile shape: [N, C_out, H_out, W_out]
    outputTileShape.resize(inputTileShape.size());

    // N dimension stays the same
    outputTileShape[0] = inputTileShape[0];

    // C_out is determined by the number of filters (from weights)
    // For now, keep it the same (should be updated based on weights)
    outputTileShape[1] = inputTileShape[1];

    // H and W dimensions
    if (strides.size() >= 2 && kernels.size() >= 2) {
        int64_t padTop = pads.size() >= 4 ? pads[0] : 0;
        int64_t padBottom = pads.size() >= 4 ? pads[1] : 0;
        int64_t padLeft = pads.size() >= 4 ? pads[2] : 0;
        int64_t padRight = pads.size() >= 4 ? pads[3] : 0;

        outputTileShape[2] = CalculateOutputSize(
            inputTileShape[2], kernels[0], strides[0],
            padTop, padBottom, dilations.size() > 0 ? dilations[0] : 1);

        outputTileShape[3] = CalculateOutputSize(
            inputTileShape[3], kernels[1], strides[1],
            padLeft, padRight, dilations.size() > 1 ? dilations[1] : 1);
    }

    return npucl::SUCCESS;
}

int ConvOpBuilder::GetConvAttributes(
    ge::NodePtr node,
    std::vector<int64_t>& strides,
    std::vector<int64_t>& pads,
    std::vector<int64_t>& dilations,
    std::vector<int64_t>& kernels) {

    // Implementation should retrieve attributes from node:
    // - strides: ge::AttrUtils::GetListInt(opDesc, "strides", strides)
    // - pads: ge::AttrUtils::GetListInt(opDesc, "pads", pads)
    // - dilations: ge::AttrUtils::GetListInt(opDesc, "dilations", dilations)
    // - kernels: from weight tensor shape

    // Default values for 3x3 conv with stride 1
    strides = {1, 1};
    pads = {0, 0, 0, 0};
    dilations = {1, 1};
    kernels = {3, 3};

    return npucl::SUCCESS;
}

int64_t ConvOpBuilder::CalculateOutputSize(
    int64_t inputSize,
    int64_t kernelSize,
    int64_t stride,
    int64_t padStart,
    int64_t padEnd,
    int64_t dilation) {

    int64_t effectiveKernelSize = (kernelSize - 1) * dilation + 1;
    int64_t outputSize = (inputSize + padStart + padEnd - effectiveKernelSize) / stride + 1;

    return outputSize;
}

int64_t ConvOpBuilder::CalculateInputSize(
    int64_t outputSize,
    int64_t kernelSize,
    int64_t stride,
    int64_t padStart,
    int64_t padEnd,
    int64_t dilation) {

    int64_t effectiveKernelSize = (kernelSize - 1) * dilation + 1;
    int64_t inputSize = (outputSize - 1) * stride + effectiveKernelSize - padStart - padEnd;

    return inputSize;
}

// ============================================================================
// MatmulOpBuilder Implementation
// ============================================================================

int MatmulOpBuilder::InitAndValidateTileShape(
    ge::NodePtr node,
    const AscppTileShape& userTileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {

    outputTileShape = userTileShape;
    int ret = CalculateInputTileShape(node, outputTileShape, inputTileShape);
    if (ret != npucl::SUCCESS) {
        return npucl::FAILED;
    }

    return ValidateConstraints(node);
}

int MatmulOpBuilder::InferTileShape(
    ge::NodePtr node,
    const AscppTileShape& predecessorOutputShape,
    AscppTileShape& tileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {

    inputTileShape = predecessorOutputShape;

    int ret = CalculateOutputTileShape(node, inputTileShape, outputTileShape);
    if (ret != npucl::SUCCESS) {
        return npucl::FAILED;
    }

    tileShape = outputTileShape;
    return ValidateConstraints(node);
}

int MatmulOpBuilder::ValidateConstraints(ge::NodePtr node) {
    // Validate matmul-specific constraints
    return npucl::SUCCESS;
}

int MatmulOpBuilder::CalculateInputTileShape(
    ge::NodePtr node,
    const AscppTileShape& outputTileShape,
    AscppTileShape& inputTileShape) {

    bool transposeA = false, transposeB = false;
    GetTransposeAttrs(node, transposeA, transposeB);

    // For matmul: C = A * B
    // If A is [M, K] and B is [K, N], then C is [M, N]
    // Input tile shape depends on which input we're calculating for
    // This is a simplified version - actual implementation needs to handle
    // batch dimensions and transposition

    inputTileShape = outputTileShape;
    return npucl::SUCCESS;
}

int MatmulOpBuilder::CalculateOutputTileShape(
    ge::NodePtr node,
    const AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {

    bool transposeA = false, transposeB = false;
    GetTransposeAttrs(node, transposeA, transposeB);

    // Simplified: output shape derived from input shape
    // Actual implementation needs to consider batch dims and transpose
    outputTileShape = inputTileShape;
    return npucl::SUCCESS;
}

int MatmulOpBuilder::GetTransposeAttrs(
    ge::NodePtr node,
    bool& transposeA,
    bool& transposeB) {

    // Implementation should retrieve:
    // ge::AttrUtils::GetBool(opDesc, "transpose_x1", transposeA)
    // ge::AttrUtils::GetBool(opDesc, "transpose_x2", transposeB)

    transposeA = false;
    transposeB = false;
    return npucl::SUCCESS;
}

// ============================================================================
// PoolOpBuilder Implementation
// ============================================================================

int PoolOpBuilder::InitAndValidateTileShape(
    ge::NodePtr node,
    const AscppTileShape& userTileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {

    outputTileShape = userTileShape;
    int ret = CalculateInputTileShape(node, outputTileShape, inputTileShape);
    if (ret != npucl::SUCCESS) {
        return npucl::FAILED;
    }

    return ValidateConstraints(node);
}

int PoolOpBuilder::InferTileShape(
    ge::NodePtr node,
    const AscppTileShape& predecessorOutputShape,
    AscppTileShape& tileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {

    inputTileShape = predecessorOutputShape;

    int ret = CalculateOutputTileShape(node, inputTileShape, outputTileShape);
    if (ret != npucl::SUCCESS) {
        return npucl::FAILED;
    }

    tileShape = outputTileShape;
    return ValidateConstraints(node);
}

int PoolOpBuilder::ValidateConstraints(ge::NodePtr node) {
    return npucl::SUCCESS;
}

int PoolOpBuilder::CalculateInputTileShape(
    ge::NodePtr node,
    const AscppTileShape& outputTileShape,
    AscppTileShape& inputTileShape) {

    std::vector<int64_t> kernelSize, strides, pads;
    bool ceilMode = false;

    int ret = GetPoolAttributes(node, kernelSize, strides, pads, ceilMode);
    if (ret != npucl::SUCCESS) {
        return npucl::FAILED;
    }

    // Similar to conv, calculate input from output
    inputTileShape = outputTileShape;

    if (kernelSize.size() >= 2 && strides.size() >= 2) {
        // Simplified calculation - actual should consider padding
        inputTileShape[2] = (outputTileShape[2] - 1) * strides[0] + kernelSize[0];
        inputTileShape[3] = (outputTileShape[3] - 1) * strides[1] + kernelSize[1];
    }

    return npucl::SUCCESS;
}

int PoolOpBuilder::CalculateOutputTileShape(
    ge::NodePtr node,
    const AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {

    std::vector<int64_t> kernelSize, strides, pads;
    bool ceilMode = false;

    int ret = GetPoolAttributes(node, kernelSize, strides, pads, ceilMode);
    if (ret != npucl::SUCCESS) {
        return npucl::FAILED;
    }

    outputTileShape = inputTileShape;

    if (kernelSize.size() >= 2 && strides.size() >= 2) {
        outputTileShape[2] = (inputTileShape[2] - kernelSize[0]) / strides[0] + 1;
        outputTileShape[3] = (inputTileShape[3] - kernelSize[1]) / strides[1] + 1;
    }

    return npucl::SUCCESS;
}

int PoolOpBuilder::GetPoolAttributes(
    ge::NodePtr node,
    std::vector<int64_t>& kernelSize,
    std::vector<int64_t>& strides,
    std::vector<int64_t>& pads,
    bool& ceilMode) {

    // Implementation should retrieve from node attributes
    kernelSize = {2, 2};
    strides = {2, 2};
    pads = {0, 0, 0, 0};
    ceilMode = false;

    return npucl::SUCCESS;
}

// ============================================================================
// ElementwiseOpBuilder Implementation
// ============================================================================

int ElementwiseOpBuilder::InitAndValidateTileShape(
    ge::NodePtr node,
    const AscppTileShape& userTileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {

    // For element-wise ops, input and output shapes are the same
    inputTileShape = userTileShape;
    outputTileShape = userTileShape;

    return ValidateConstraints(node);
}

int ElementwiseOpBuilder::InferTileShape(
    ge::NodePtr node,
    const AscppTileShape& predecessorOutputShape,
    AscppTileShape& tileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {

    // Element-wise: pass through the shape
    inputTileShape = predecessorOutputShape;
    outputTileShape = predecessorOutputShape;
    tileShape = predecessorOutputShape;

    return ValidateConstraints(node);
}

int ElementwiseOpBuilder::ValidateConstraints(ge::NodePtr node) {
    return npucl::SUCCESS;
}

int ElementwiseOpBuilder::CalculateInputTileShape(
    ge::NodePtr node,
    const AscppTileShape& outputTileShape,
    AscppTileShape& inputTileShape) {

    inputTileShape = outputTileShape;
    return npucl::SUCCESS;
}

int ElementwiseOpBuilder::CalculateOutputTileShape(
    ge::NodePtr node,
    const AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {

    outputTileShape = inputTileShape;
    return npucl::SUCCESS;
}

int ElementwiseOpBuilder::InferUnifiedTileShape(
    ge::NodePtr node,
    const std::vector<AscppTileShape>& inputShapes,
    AscppTileShape& unifiedShape,
    std::string& errorMsg) {

    if (inputShapes.empty()) {
        errorMsg = "No input shapes provided";
        return npucl::FAILED;
    }

    // Strategy: compute broadcast-compatible shape (maximum along each dimension)
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

    // Broadcast rules: dimensions are compatible when:
    // 1. They are equal, or
    // 2. One of them is 1

    // Align from the right (trailing dimensions)
    size_t i = shape1.size();
    size_t j = shape2.size();

    while (i > 0 && j > 0) {
        i--;
        j--;
        if (shape1[i] != shape2[j] &&
            shape1[i] != 1 &&
            shape2[j] != 1) {
            return false;
        }
    }

    // All compared dimensions are compatible
    return true;
}

AscppTileShape ElementwiseOpBuilder::GetBroadcastedShape(
    const AscppTileShape& shape1,
    const AscppTileShape& shape2) {

    // Result shape has max rank
    size_t maxRank = std::max(shape1.size(), shape2.size());
    AscppTileShape result(maxRank, 1);

    // Fill from the right (trailing dimensions)
    size_t i = shape1.size();
    size_t j = shape2.size();
    size_t k = maxRank;

    while (k > 0) {
        k--;
        int64_t dim1 = (i > 0) ? shape1[--i] : 1;
        int64_t dim2 = (j > 0) ? shape2[--j] : 1;

        // Take the larger dimension (broadcast rule)
        result[k] = std::max(dim1, dim2);
    }

    return result;
}

// ============================================================================
// ReshapeOpBuilder Implementation
// ============================================================================

int ReshapeOpBuilder::InitAndValidateTileShape(
    ge::NodePtr node,
    const AscppTileShape& userTileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {

    outputTileShape = userTileShape;

    // For reshape, input tile shape needs to maintain element count
    // but may have different dimensions
    std::vector<int64_t> targetShape;
    GetTargetShape(node, targetShape);

    // Calculate input tile shape that maintains total elements
    // This is a simplified version
    inputTileShape = userTileShape;

    return ValidateConstraints(node);
}

int ReshapeOpBuilder::InferTileShape(
    ge::NodePtr node,
    const AscppTileShape& predecessorOutputShape,
    AscppTileShape& tileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {

    inputTileShape = predecessorOutputShape;

    std::vector<int64_t> targetShape;
    GetTargetShape(node, targetShape);

    // Output tile shape follows the target shape pattern
    // but maintains tile element relationships
    outputTileShape = predecessorOutputShape; // Simplified

    tileShape = outputTileShape;
    return ValidateConstraints(node);
}

int ReshapeOpBuilder::ValidateConstraints(ge::NodePtr node) {
    // Validate that total elements are preserved
    return npucl::SUCCESS;
}

int ReshapeOpBuilder::GetTargetShape(ge::NodePtr node, std::vector<int64_t>& targetShape) {
    // Implementation should retrieve "shape" attribute
    return npucl::SUCCESS;
}

// ============================================================================
// ActivationOpBuilder Implementation
// ============================================================================

int ActivationOpBuilder::InitAndValidateTileShape(
    ge::NodePtr node,
    const AscppTileShape& userTileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {

    // Activation is element-wise, shapes don't change
    inputTileShape = userTileShape;
    outputTileShape = userTileShape;

    return ValidateConstraints(node);
}

int ActivationOpBuilder::InferTileShape(
    ge::NodePtr node,
    const AscppTileShape& predecessorOutputShape,
    AscppTileShape& tileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {

    // Pass through
    inputTileShape = predecessorOutputShape;
    outputTileShape = predecessorOutputShape;
    tileShape = predecessorOutputShape;

    return ValidateConstraints(node);
}

int ActivationOpBuilder::ValidateConstraints(ge::NodePtr node) {
    return npucl::SUCCESS;
}

// ============================================================================
// BatchNormOpBuilder Implementation
// ============================================================================

int BatchNormOpBuilder::InitAndValidateTileShape(
    ge::NodePtr node,
    const AscppTileShape& userTileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {

    // BatchNorm doesn't change shape
    inputTileShape = userTileShape;
    outputTileShape = userTileShape;

    return ValidateConstraints(node);
}

int BatchNormOpBuilder::InferTileShape(
    ge::NodePtr node,
    const AscppTileShape& predecessorOutputShape,
    AscppTileShape& tileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {

    inputTileShape = predecessorOutputShape;
    outputTileShape = predecessorOutputShape;
    tileShape = predecessorOutputShape;

    return ValidateConstraints(node);
}

int BatchNormOpBuilder::ValidateConstraints(ge::NodePtr node) {
    return npucl::SUCCESS;
}

// ============================================================================
// ReduceOpBuilder Implementation
// ============================================================================

int ReduceOpBuilder::InitAndValidateTileShape(
    ge::NodePtr node,
    const AscppTileShape& userTileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {

    outputTileShape = userTileShape;

    std::vector<int64_t> axes;
    bool keepDims = false;
    GetReduceAttributes(node, axes, keepDims);

    // Calculate input tile shape based on reduction axes
    inputTileShape = userTileShape; // Simplified

    return ValidateConstraints(node);
}

int ReduceOpBuilder::InferTileShape(
    ge::NodePtr node,
    const AscppTileShape& predecessorOutputShape,
    AscppTileShape& tileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {

    inputTileShape = predecessorOutputShape;

    std::vector<int64_t> axes;
    bool keepDims = false;
    GetReduceAttributes(node, axes, keepDims);

    // Output shape depends on axes and keepDims
    outputTileShape = predecessorOutputShape;

    if (!keepDims) {
        // Remove reduced dimensions (simplified)
        for (auto axis : axes) {
            if (axis >= 0 && static_cast<size_t>(axis) < outputTileShape.size()) {
                outputTileShape[axis] = 1; // Or remove dimension
            }
        }
    }

    tileShape = outputTileShape;
    return ValidateConstraints(node);
}

int ReduceOpBuilder::ValidateConstraints(ge::NodePtr node) {
    return npucl::SUCCESS;
}

int ReduceOpBuilder::GetReduceAttributes(
    ge::NodePtr node,
    std::vector<int64_t>& axes,
    bool& keepDims) {

    // Implementation should retrieve from node
    keepDims = false;
    return npucl::SUCCESS;
}

// ============================================================================
// CastOpBuilder Implementation
// ============================================================================

int CastOpBuilder::InitAndValidateTileShape(
    ge::NodePtr node,
    const AscppTileShape& userTileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {

    // Cast only does type conversion, shape is unchanged
    inputTileShape = userTileShape;
    outputTileShape = userTileShape;

    return ValidateConstraints(node);
}

int CastOpBuilder::InferTileShape(
    ge::NodePtr node,
    const AscppTileShape& predecessorOutputShape,
    AscppTileShape& tileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {

    // Cast passes through the shape unchanged
    inputTileShape = predecessorOutputShape;
    outputTileShape = predecessorOutputShape;
    tileShape = predecessorOutputShape;

    return ValidateConstraints(node);
}

int CastOpBuilder::ValidateConstraints(ge::NodePtr node) {
    // No special constraints for Cast
    return npucl::SUCCESS;
}

// ============================================================================
// TransDataOpBuilder Implementation
// ============================================================================

int TransDataOpBuilder::InitAndValidateTileShape(
    ge::NodePtr node,
    const AscppTileShape& userTileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {

    // For TransData, user tile shape is the output tile shape
    outputTileShape = userTileShape;

    // Calculate input tile shape based on format conversion
    int ret = CalculateInputTileShape(node, outputTileShape, inputTileShape);
    if (ret != npucl::SUCCESS) {
        return npucl::FAILED;
    }

    return ValidateConstraints(node);
}

int TransDataOpBuilder::InferTileShape(
    ge::NodePtr node,
    const AscppTileShape& predecessorOutputShape,
    AscppTileShape& tileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {

    // Input shape comes from predecessor
    inputTileShape = predecessorOutputShape;

    // Calculate output tile shape based on format conversion
    int ret = CalculateOutputTileShape(node, inputTileShape, outputTileShape);
    if (ret != npucl::SUCCESS) {
        return npucl::FAILED;
    }

    // Final tile shape is the output shape
    tileShape = outputTileShape;

    return ValidateConstraints(node);
}

int TransDataOpBuilder::ValidateConstraints(ge::NodePtr node) {
    // Validate that format conversion is supported
    return npucl::SUCCESS;
}

int TransDataOpBuilder::CalculateInputTileShape(
    ge::NodePtr node,
    const AscppTileShape& outputTileShape,
    AscppTileShape& inputTileShape) {

    std::string srcFormat, dstFormat;
    int ret = GetTransDataFormats(node, srcFormat, dstFormat);
    if (ret != npucl::SUCCESS) {
        // If formats not available, pass through unchanged
        inputTileShape = outputTileShape;
        return npucl::SUCCESS;
    }

    // Transform from dst format back to src format
    return TransformTileShapeByFormat(outputTileShape, dstFormat, srcFormat, inputTileShape);
}

int TransDataOpBuilder::CalculateOutputTileShape(
    ge::NodePtr node,
    const AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {

    std::string srcFormat, dstFormat;
    int ret = GetTransDataFormats(node, srcFormat, dstFormat);
    if (ret != npucl::SUCCESS) {
        // If formats not available, pass through unchanged
        outputTileShape = inputTileShape;
        return npucl::SUCCESS;
    }

    // Transform from src format to dst format
    return TransformTileShapeByFormat(inputTileShape, srcFormat, dstFormat, outputTileShape);
}

int TransDataOpBuilder::GetTransDataFormats(
    ge::NodePtr node,
    std::string& srcFormat,
    std::string& dstFormat) {

    // Implementation should retrieve from node attributes:
    // ge::AttrUtils::GetStr(opDesc, "src_format", srcFormat)
    // ge::AttrUtils::GetStr(opDesc, "dst_format", dstFormat)

    // Default: assume NCHW to NCHW (no conversion)
    srcFormat = "NCHW";
    dstFormat = "NCHW";

    return npucl::SUCCESS;
}

int TransDataOpBuilder::TransformTileShapeByFormat(
    const AscppTileShape& srcShape,
    const std::string& srcFormat,
    const std::string& dstFormat,
    AscppTileShape& dstShape) {

    // Simple pass-through for same format
    if (srcFormat == dstFormat) {
        dstShape = srcShape;
        return npucl::SUCCESS;
    }

    // Handle common format conversions
    // NCHW <-> NC1HWC0 (5D format with C0=16 for Ascend)
    if (srcFormat == "NCHW" && dstFormat == "NC1HWC0") {
        // NCHW [N, C, H, W] -> NC1HWC0 [N, C1, H, W, C0]
        // C1 = ceil(C / C0), typically C0 = 16
        if (srcShape.size() != 4) {
            return npucl::FAILED;
        }
        int64_t C0 = 16;
        int64_t C = srcShape[1];
        int64_t C1 = (C + C0 - 1) / C0;  // ceil division
        dstShape = {srcShape[0], C1, srcShape[2], srcShape[3], C0};
        return npucl::SUCCESS;
    }

    if (srcFormat == "NC1HWC0" && dstFormat == "NCHW") {
        // NC1HWC0 [N, C1, H, W, C0] -> NCHW [N, C, H, W]
        // C = C1 * C0 (but actual C may be less)
        if (srcShape.size() != 5) {
            return npucl::FAILED;
        }
        int64_t C = srcShape[1] * srcShape[4];
        dstShape = {srcShape[0], C, srcShape[2], srcShape[3]};
        return npucl::SUCCESS;
    }

    // For other format conversions, pass through (simplified)
    dstShape = srcShape;
    return npucl::SUCCESS;
}

// ============================================================================
// ConcatOpBuilder Implementation
// ============================================================================

int ConcatOpBuilder::InitAndValidateTileShape(
    ge::NodePtr node,
    const AscppTileShape& userTileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {

    // For Concat, output tile shape follows user configuration
    outputTileShape = userTileShape;
    inputTileShape = userTileShape;

    return ValidateConstraints(node);
}

int ConcatOpBuilder::InferTileShape(
    ge::NodePtr node,
    const AscppTileShape& predecessorOutputShape,
    AscppTileShape& tileShape,
    AscppTileShape& inputTileShape,
    AscppTileShape& outputTileShape) {

    // Use predecessor's output as input tile shape
    inputTileShape = predecessorOutputShape;
    outputTileShape = predecessorOutputShape;
    tileShape = predecessorOutputShape;

    return ValidateConstraints(node);
}

int ConcatOpBuilder::ValidateConstraints(ge::NodePtr node) {
    return npucl::SUCCESS;
}

int ConcatOpBuilder::InferUnifiedTileShape(
    ge::NodePtr node,
    const std::vector<AscppTileShape>& inputShapes,
    AscppTileShape& unifiedShape,
    std::string& errorMsg) {

    if (inputShapes.empty()) {
        errorMsg = "No input shapes provided";
        return npucl::FAILED;
    }

    // Get concat axis
    int64_t axis = GetConcatAxis(node);

    // Initialize with first input
    unifiedShape = inputShapes[0];

    // Check non-concat dimensions are equal, and find minimum for concat axis
    for (size_t i = 1; i < inputShapes.size(); i++) {
        const auto& shape = inputShapes[i];

        if (shape.size() != unifiedShape.size()) {
            errorMsg = "Input shapes have different ranks";
            return npucl::FAILED;
        }

        for (size_t dim = 0; dim < unifiedShape.size(); dim++) {
            if (static_cast<int64_t>(dim) != axis) {
                // Non-concat dimensions must match
                if (unifiedShape[dim] != shape[dim]) {
                    errorMsg = "Non-concat dimensions must be equal";
                    return npucl::FAILED;
                }
            } else {
                // Concat axis: take minimum (conservative strategy)
                unifiedShape[dim] = std::min(unifiedShape[dim], shape[dim]);
            }
        }
    }

    return npucl::SUCCESS;
}

int64_t ConcatOpBuilder::GetConcatAxis(ge::NodePtr node) {
    // Implementation should retrieve axis from node attributes
    // int64_t axis = 0;
    // ge::AttrUtils::GetInt(node->GetOpDesc(), "axis", axis);
    // return axis;
    return 0;  // Default: axis 0
}

// ============================================================================
// OpBuilder Registration
// ============================================================================

// Helper template for registration
namespace {

template<typename T>
void RegisterOpBuilderImpl(const std::string& opType) {
    OpBuilderFactory::Instance().RegisterBuilder(
        opType,
        []() -> std::unique_ptr<OpBuilder> {
            return std::make_unique<T>();
        });
}

} // anonymous namespace

void RegisterBuiltinOpBuilders() {
    // Convolution operators
    RegisterOpBuilderImpl<ConvOpBuilder>("Conv2D");
    RegisterOpBuilderImpl<ConvOpBuilder>("Conv2DTranspose");
    RegisterOpBuilderImpl<ConvOpBuilder>("Conv3D");

    // Matrix multiplication operators
    RegisterOpBuilderImpl<MatmulOpBuilder>("MatMul");
    RegisterOpBuilderImpl<MatmulOpBuilder>("BatchMatMul");

    // Pooling operators
    RegisterOpBuilderImpl<PoolOpBuilder>("MaxPool");
    RegisterOpBuilderImpl<PoolOpBuilder>("AvgPool");
    RegisterOpBuilderImpl<PoolOpBuilder>("MaxPoolV2");
    RegisterOpBuilderImpl<PoolOpBuilder>("AvgPoolV2");

    // Element-wise operators
    RegisterOpBuilderImpl<ElementwiseOpBuilder>("Add");
    RegisterOpBuilderImpl<ElementwiseOpBuilder>("Sub");
    RegisterOpBuilderImpl<ElementwiseOpBuilder>("Mul");
    RegisterOpBuilderImpl<ElementwiseOpBuilder>("Div");
    RegisterOpBuilderImpl<ElementwiseOpBuilder>("RealDiv");

    // Reshape operators
    RegisterOpBuilderImpl<ReshapeOpBuilder>("Reshape");
    RegisterOpBuilderImpl<ReshapeOpBuilder>("Flatten");
    RegisterOpBuilderImpl<ReshapeOpBuilder>("Squeeze");
    RegisterOpBuilderImpl<ReshapeOpBuilder>("ExpandDims");

    // Activation operators
    RegisterOpBuilderImpl<ActivationOpBuilder>("ReLU");
    RegisterOpBuilderImpl<ActivationOpBuilder>("Sigmoid");
    RegisterOpBuilderImpl<ActivationOpBuilder>("Tanh");
    RegisterOpBuilderImpl<ActivationOpBuilder>("Swish");
    RegisterOpBuilderImpl<ActivationOpBuilder>("GELU");

    // Normalization operators
    RegisterOpBuilderImpl<BatchNormOpBuilder>("BatchNorm");
    RegisterOpBuilderImpl<BatchNormOpBuilder>("BatchNormalization");
    RegisterOpBuilderImpl<BatchNormOpBuilder>("LayerNorm");

    // Reduce operators
    RegisterOpBuilderImpl<ReduceOpBuilder>("ReduceSum");
    RegisterOpBuilderImpl<ReduceOpBuilder>("ReduceMean");
    RegisterOpBuilderImpl<ReduceOpBuilder>("ReduceMax");
    RegisterOpBuilderImpl<ReduceOpBuilder>("ReduceMin");

    // Passthrough operators (N1)
    RegisterOpBuilderImpl<CastOpBuilder>("Cast");
    RegisterOpBuilderImpl<TransDataOpBuilder>("TransData");

    // Multi-input operators (N3)
    RegisterOpBuilderImpl<ConcatOpBuilder>("Concat");
    RegisterOpBuilderImpl<ConcatOpBuilder>("ConcatV2");
    RegisterOpBuilderImpl<ConcatOpBuilder>("ConcatD");
}

} // namespace tile_shape
