/**
 * @file op_builders.h
 * @brief Concrete OpBuilder implementations for various operator types
 */

#ifndef OP_BUILDERS_H
#define OP_BUILDERS_H

#include "tile_shape_inference_pass.h"
#include <cmath>

namespace tile_shape {

// ============================================================================
// BaseOpBuilder - Common base class with shared functionality
// ============================================================================

/**
 * @brief Base OpBuilder with common implementation patterns
 *
 * Provides default implementations that can be overridden by
 * specific operator builders.
 */
class BaseOpBuilder : public OpBuilder {
public:
    ~BaseOpBuilder() override = default;

    int InitAndValidateTileShape() override;

    int InferTileShape(
        const AscppTileShape& predecessorOutputShape,
        AscppTileShape& tileShape,
        AscppTileShape& inputTileShape,
        AscppTileShape& outputTileShape) override;

    int ValidateConstraints() override;

protected:
    int CalculateInputTileShape(
        const AscppTileShape& outputTileShape,
        AscppTileShape& inputTileShape) override;

    int CalculateOutputTileShape(
        const AscppTileShape& inputTileShape,
        AscppTileShape& outputTileShape) override;

    /**
     * @brief Get the number of dimensions for the operator
     *
     * @return Number of dimensions (default: 4 for NCHW)
     */
    virtual int GetNumDimensions();

    /**
     * @brief Validate TileShape dimensions
     *
     * @param tileShape TileShape to validate
     * @param expectedDims Expected number of dimensions
     * @return true if valid, false otherwise
     */
    bool ValidateDimensions(const AscppTileShape& tileShape, int expectedDims);
};

// ============================================================================
// ConvOpBuilder - Convolution operator builder
// ============================================================================

/**
 * @brief OpBuilder for Convolution operators
 *
 * Handles TileShape inference for Conv2D, Conv3D, and similar operations.
 * Conv operators typically have input/output TileShapes that differ based
 * on padding, stride, and kernel size.
 */
class ConvOpBuilder : public BaseOpBuilder {
public:
    ~ConvOpBuilder() override = default;

    int InitAndValidateTileShape() override;

    int InferTileShape(
        const AscppTileShape& predecessorOutputShape,
        AscppTileShape& tileShape,
        AscppTileShape& inputTileShape,
        AscppTileShape& outputTileShape) override;

    int ValidateConstraints() override;

protected:
    int CalculateInputTileShape(
        const AscppTileShape& outputTileShape,
        AscppTileShape& inputTileShape) override;

    int CalculateOutputTileShape(
        const AscppTileShape& inputTileShape,
        AscppTileShape& outputTileShape) override;

private:
    /**
     * @brief Get convolution attributes (stride, pad, dilation, kernel)
     *
     * @param strides [out] Stride values
     * @param pads [out] Padding values
     * @param dilations [out] Dilation values
     * @param kernels [out] Kernel size values
     * @return npucl::SUCCESS or npucl::FAILED
     */
    int GetConvAttributes(
        std::vector<int64_t>& strides,
        std::vector<int64_t>& pads,
        std::vector<int64_t>& dilations,
        std::vector<int64_t>& kernels);

    /**
     * @brief Calculate output size from input size and conv params
     *
     * @param inputSize Input spatial dimension size
     * @param kernelSize Kernel size
     * @param stride Stride
     * @param padStart Padding start
     * @param padEnd Padding end
     * @param dilation Dilation
     * @return Output size
     */
    int64_t CalculateOutputSize(
        int64_t inputSize,
        int64_t kernelSize,
        int64_t stride,
        int64_t padStart,
        int64_t padEnd,
        int64_t dilation);

    /**
     * @brief Calculate input size from output size and conv params (reverse)
     *
     * @param outputSize Output spatial dimension size
     * @param kernelSize Kernel size
     * @param stride Stride
     * @param padStart Padding start
     * @param padEnd Padding end
     * @param dilation Dilation
     * @return Input size
     */
    int64_t CalculateInputSize(
        int64_t outputSize,
        int64_t kernelSize,
        int64_t stride,
        int64_t padStart,
        int64_t padEnd,
        int64_t dilation);
};

// ============================================================================
// MatmulOpBuilder - Matrix multiplication operator builder
// ============================================================================

/**
 * @brief OpBuilder for Matrix Multiplication operators
 *
 * Handles TileShape inference for MatMul, BatchMatMul, and similar operations.
 */
class MatmulOpBuilder : public BaseOpBuilder {
public:
    ~MatmulOpBuilder() override = default;

    int InitAndValidateTileShape() override;

    int InferTileShape(
        const AscppTileShape& predecessorOutputShape,
        AscppTileShape& tileShape,
        AscppTileShape& inputTileShape,
        AscppTileShape& outputTileShape) override;

    int ValidateConstraints() override;

protected:
    int CalculateInputTileShape(
        const AscppTileShape& outputTileShape,
        AscppTileShape& inputTileShape) override;

    int CalculateOutputTileShape(
        const AscppTileShape& inputTileShape,
        AscppTileShape& outputTileShape) override;

private:
    /**
     * @brief Check if matmul is transposed
     *
     * @param transposeA [out] Whether first input is transposed
     * @param transposeB [out] Whether second input is transposed
     * @return npucl::SUCCESS or npucl::FAILED
     */
    int GetTransposeAttrs(bool& transposeA, bool& transposeB);
};

// ============================================================================
// PoolOpBuilder - Pooling operator builder
// ============================================================================

/**
 * @brief OpBuilder for Pooling operators (MaxPool, AvgPool, etc.)
 */
class PoolOpBuilder : public BaseOpBuilder {
public:
    ~PoolOpBuilder() override = default;

    int InitAndValidateTileShape() override;

    int InferTileShape(
        const AscppTileShape& predecessorOutputShape,
        AscppTileShape& tileShape,
        AscppTileShape& inputTileShape,
        AscppTileShape& outputTileShape) override;

    int ValidateConstraints() override;

protected:
    int CalculateInputTileShape(
        const AscppTileShape& outputTileShape,
        AscppTileShape& inputTileShape) override;

    int CalculateOutputTileShape(
        const AscppTileShape& inputTileShape,
        AscppTileShape& outputTileShape) override;

private:
    /**
     * @brief Get pooling attributes
     *
     * @param kernelSize [out] Kernel size values
     * @param strides [out] Stride values
     * @param pads [out] Padding values
     * @param ceilMode [out] Whether to use ceil mode
     * @return npucl::SUCCESS or npucl::FAILED
     */
    int GetPoolAttributes(
        std::vector<int64_t>& kernelSize,
        std::vector<int64_t>& strides,
        std::vector<int64_t>& pads,
        bool& ceilMode);
};

// ============================================================================
// ElementwiseOpBuilder - Element-wise operator builder
// ============================================================================

/**
 * @brief OpBuilder for element-wise operators (Add, Sub, Mul, Div, etc.)
 *
 * For element-wise operations, input and output TileShapes are typically
 * the same after broadcasting.
 */
class ElementwiseOpBuilder : public BaseOpBuilder {
public:
    ~ElementwiseOpBuilder() override = default;

    int InitAndValidateTileShape() override;

    int InferTileShape(
        const AscppTileShape& predecessorOutputShape,
        AscppTileShape& tileShape,
        AscppTileShape& inputTileShape,
        AscppTileShape& outputTileShape) override;

    int ValidateConstraints() override;

    /**
     * @brief Infer unified TileShape from multiple broadcast-compatible inputs
     *
     * For element-wise ops: uses broadcast rules to find compatible shape.
     */
    int InferUnifiedTileShape(
        const std::vector<AscppTileShape>& inputShapes,
        AscppTileShape& unifiedShape,
        std::string& errorMsg) override;

    /**
     * @brief Check if two shapes are broadcast compatible
     *
     * @param shape1 First shape
     * @param shape2 Second shape
     * @return true if shapes can be broadcast together
     */
    static bool CanBroadcastShapes(
        const AscppTileShape& shape1,
        const AscppTileShape& shape2);

    /**
     * @brief Get broadcasted shape from two shapes
     *
     * @param shape1 First shape
     * @param shape2 Second shape
     * @return Broadcasted (output) shape
     */
    static AscppTileShape GetBroadcastedShape(
        const AscppTileShape& shape1,
        const AscppTileShape& shape2);

protected:
    int CalculateInputTileShape(
        const AscppTileShape& outputTileShape,
        AscppTileShape& inputTileShape) override;

    int CalculateOutputTileShape(
        const AscppTileShape& inputTileShape,
        AscppTileShape& outputTileShape) override;
};

// ============================================================================
// ReshapeOpBuilder - Reshape operator builder
// ============================================================================

/**
 * @brief OpBuilder for Reshape operators
 *
 * Reshape operations don't change the total number of elements,
 * only the dimension layout.
 */
class ReshapeOpBuilder : public BaseOpBuilder {
public:
    ~ReshapeOpBuilder() override = default;

    int InitAndValidateTileShape() override;

    int InferTileShape(
        const AscppTileShape& predecessorOutputShape,
        AscppTileShape& tileShape,
        AscppTileShape& inputTileShape,
        AscppTileShape& outputTileShape) override;

    int ValidateConstraints() override;

private:
    /**
     * @brief Get target shape from reshape node
     *
     * @param targetShape [out] Target shape values
     * @return npucl::SUCCESS or npucl::FAILED
     */
    int GetTargetShape(std::vector<int64_t>& targetShape);
};

// ============================================================================
// ActivationOpBuilder - Activation operator builder
// ============================================================================

/**
 * @brief OpBuilder for activation operators (ReLU, Sigmoid, Tanh, etc.)
 *
 * Activation operations are element-wise and don't change the shape.
 */
class ActivationOpBuilder : public BaseOpBuilder {
public:
    ~ActivationOpBuilder() override = default;

    int InitAndValidateTileShape() override;

    int InferTileShape(
        const AscppTileShape& predecessorOutputShape,
        AscppTileShape& tileShape,
        AscppTileShape& inputTileShape,
        AscppTileShape& outputTileShape) override;

    int ValidateConstraints() override;
};

// ============================================================================
// BatchNormOpBuilder - Batch Normalization operator builder
// ============================================================================

/**
 * @brief OpBuilder for BatchNorm operators
 */
class BatchNormOpBuilder : public BaseOpBuilder {
public:
    ~BatchNormOpBuilder() override = default;

    int InitAndValidateTileShape() override;

    int InferTileShape(
        const AscppTileShape& predecessorOutputShape,
        AscppTileShape& tileShape,
        AscppTileShape& inputTileShape,
        AscppTileShape& outputTileShape) override;

    int ValidateConstraints() override;
};

// ============================================================================
// ReduceOpBuilder - Reduce operator builder
// ============================================================================

/**
 * @brief OpBuilder for reduce operators (ReduceSum, ReduceMean, etc.)
 */
class ReduceOpBuilder : public BaseOpBuilder {
public:
    ~ReduceOpBuilder() override = default;

    int InitAndValidateTileShape() override;

    int InferTileShape(
        const AscppTileShape& predecessorOutputShape,
        AscppTileShape& tileShape,
        AscppTileShape& inputTileShape,
        AscppTileShape& outputTileShape) override;

    int ValidateConstraints() override;

private:
    /**
     * @brief Get reduce axes
     *
     * @param axes [out] Axes to reduce
     * @param keepDims [out] Whether to keep dimensions
     * @return npucl::SUCCESS or npucl::FAILED
     */
    int GetReduceAttributes(std::vector<int64_t>& axes, bool& keepDims);
};

// ============================================================================
// CastOpBuilder - Cast operator builder (passthrough)
// ============================================================================

/**
 * @brief OpBuilder for Cast operator
 *
 * Cast performs type conversion without changing shape.
 * TileShape is directly passed through from input to output.
 */
class CastOpBuilder : public BaseOpBuilder {
public:
    ~CastOpBuilder() override = default;

    int InitAndValidateTileShape() override;

    int InferTileShape(
        const AscppTileShape& predecessorOutputShape,
        AscppTileShape& tileShape,
        AscppTileShape& inputTileShape,
        AscppTileShape& outputTileShape) override;

    int ValidateConstraints() override;
};

// ============================================================================
// TransDataOpBuilder - TransData operator builder (passthrough)
// ============================================================================

/**
 * @brief OpBuilder for TransData operator
 *
 * TransData performs data rearrangement (e.g., NC1HWC0 <-> NCHW).
 * May change shape based on src_format and dst_format.
 */
class TransDataOpBuilder : public BaseOpBuilder {
public:
    ~TransDataOpBuilder() override = default;

    int InitAndValidateTileShape() override;

    int InferTileShape(
        const AscppTileShape& predecessorOutputShape,
        AscppTileShape& tileShape,
        AscppTileShape& inputTileShape,
        AscppTileShape& outputTileShape) override;

    int ValidateConstraints() override;

protected:
    int CalculateInputTileShape(
        const AscppTileShape& outputTileShape,
        AscppTileShape& inputTileShape) override;

    int CalculateOutputTileShape(
        const AscppTileShape& inputTileShape,
        AscppTileShape& outputTileShape) override;

private:
    /**
     * @brief Get TransData format attributes
     *
     * @param srcFormat [out] Source format (e.g., "NCHW", "NC1HWC0")
     * @param dstFormat [out] Destination format
     * @return npucl::SUCCESS or npucl::FAILED
     */
    int GetTransDataFormats(std::string& srcFormat, std::string& dstFormat);

    /**
     * @brief Transform TileShape based on format conversion
     *
     * @param srcShape Source TileShape
     * @param srcFormat Source format
     * @param dstFormat Destination format
     * @param dstShape [out] Transformed TileShape
     * @return npucl::SUCCESS or npucl::FAILED
     */
    int TransformTileShapeByFormat(
        const AscppTileShape& srcShape,
        const std::string& srcFormat,
        const std::string& dstFormat,
        AscppTileShape& dstShape);
};

// ============================================================================
// ConcatOpBuilder - Concat operator builder
// ============================================================================

/**
 * @brief OpBuilder for Concat operator
 *
 * Concat concatenates tensors along a specified axis.
 * Multi-input handling: non-concat dimensions must match.
 */
class ConcatOpBuilder : public BaseOpBuilder {
public:
    ~ConcatOpBuilder() override = default;

    int InitAndValidateTileShape() override;

    int InferTileShape(
        const AscppTileShape& predecessorOutputShape,
        AscppTileShape& tileShape,
        AscppTileShape& inputTileShape,
        AscppTileShape& outputTileShape) override;

    int ValidateConstraints() override;

    /**
     * @brief Infer unified TileShape from multiple inputs
     *
     * For Concat: non-concat dimensions must be equal,
     * concat axis takes minimum across inputs (conservative).
     */
    int InferUnifiedTileShape(
        const std::vector<AscppTileShape>& inputShapes,
        AscppTileShape& unifiedShape,
        std::string& errorMsg) override;

private:
    /**
     * @brief Get concat axis
     *
     * @return Concatenation axis
     */
    int64_t GetConcatAxis();
};

// ============================================================================
// OpBuilder Registration Helper
// ============================================================================

/**
 * @brief Helper class to register OpBuilders at startup
 *
 * Usage:
 *   static OpBuilderRegistrar<ConvOpBuilder> convRegistrar("Conv2D");
 */
template<typename T>
class OpBuilderRegistrar {
public:
    explicit OpBuilderRegistrar(const std::string& opType) {
        OpBuilderFactory::Instance().RegisterBuilder(
            opType,
            []() -> std::unique_ptr<OpBuilder> {
                return std::make_unique<T>();
            });
    }
};

// Registration function - call this to register all built-in builders
void RegisterBuiltinOpBuilders();

} // namespace tile_shape

#endif // OP_BUILDERS_H
