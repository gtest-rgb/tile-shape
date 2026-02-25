/**
 * @file tile_shape_inference_pass.h
 * @brief TileShape Inference Pass for neural network compilation
 *
 * This pass automatically infers TileShape configurations for operators
 * in a compute graph based on user configurations and conservative
 * propagation strategies.
 */

#ifndef TILE_SHAPE_INFERENCE_PASS_H
#define TILE_SHAPE_INFERENCE_PASS_H

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <unordered_set>

// Forward declarations for external types
namespace ge {
class ComputeGraph;
using ComputeGraphPtr = std::shared_ptr<ComputeGraph>;
class Node;
using NodePtr = std::shared_ptr<Node>;
class OpDesc;
using OpDescPtr = std::shared_ptr<OpDesc>;
}

namespace npucl {
constexpr int SUCCESS = 0;
constexpr int FAILED = 1;
}

namespace fe {
enum OptStatus {
    SUCCESS = 0,
    FAILED = 1
};
}

// TileShape type definitions
using AscppTileShape = std::vector<int64_t>;
using IoTileShapePair = std::pair<AscppTileShape, AscppTileShape>;

namespace tile_shape {

// ============================================================================
// Error Codes
// ============================================================================

enum class ErrorCode {
    E001_INPUT_NODE_NOT_CONFIGURED = 1,    // Input node must have TileShape configured
    E002_VALIDATION_FAILED = 2,             // TileShape validation failed
    E003_INFER_FAILED = 3,                  // Cannot infer TileShape
    E004_BUFFER_CONSTRAINT_VIOLATION = 4,   // TileShape violates buffer constraints
    E005_OPBUILDER_CREATE_FAILED = 5        // Cannot create OpBuilder
};

// ============================================================================
// OpBuilder Abstract Base Class
// ============================================================================

/**
 * @brief Abstract base class for operation-specific TileShape builders
 *
 * Each operator type should have its own OpBuilder implementation that
 * handles TileShape initialization, validation, and inference logic.
 */
class OpBuilder {
public:
    virtual ~OpBuilder() = default;

    /**
     * @brief Initialize and validate user-configured TileShape
     *
     * @param node The node to initialize
     * @param userTileShape User-provided TileShape configuration
     * @param inputTileShape [out] Calculated input N-dim TileShape
     * @param outputTileShape [out] Calculated output N-dim TileShape
     * @return npucl::SUCCESS or npucl::FAILED
     */
    virtual int InitAndValidateTileShape(
        ge::NodePtr node,
        const AscppTileShape& userTileShape,
        AscppTileShape& inputTileShape,
        AscppTileShape& outputTileShape) = 0;

    /**
     * @brief Infer TileShape from predecessor node (conservative strategy)
     *
     * @param node The node to infer TileShape for
     * @param predecessorOutputShape Output TileShape from predecessor
     * @param tileShape [out] Inferred final TileShape
     * @param inputTileShape [out] Inferred input N-dim TileShape
     * @param outputTileShape [out] Inferred output N-dim TileShape
     * @return npucl::SUCCESS or npucl::FAILED
     */
    virtual int InferTileShape(
        ge::NodePtr node,
        const AscppTileShape& predecessorOutputShape,
        AscppTileShape& tileShape,
        AscppTileShape& inputTileShape,
        AscppTileShape& outputTileShape) = 0;

    /**
     * @brief Validate operator-specific constraints
     *
     * @param node The node to validate
     * @return npucl::SUCCESS or npucl::FAILED
     */
    virtual int ValidateConstraints(ge::NodePtr node) = 0;

protected:
    /**
     * @brief Calculate input TileShape based on output TileShape
     *
     * @param node The node
     * @param outputTileShape Output TileShape
     * @param inputTileShape [out] Calculated input TileShape
     * @return npucl::SUCCESS or npucl::FAILED
     */
    virtual int CalculateInputTileShape(
        ge::NodePtr node,
        const AscppTileShape& outputTileShape,
        AscppTileShape& inputTileShape) = 0;

    /**
     * @brief Calculate output TileShape based on input TileShape
     *
     * @param node The node
     * @param inputTileShape Input TileShape
     * @param outputTileShape [out] Calculated output TileShape
     * @return npucl::SUCCESS or npucl::FAILED
     */
    virtual int CalculateOutputTileShape(
        ge::NodePtr node,
        const AscppTileShape& inputTileShape,
        AscppTileShape& outputTileShape) = 0;
};

// ============================================================================
// OpBuilder Factory (Singleton)
// ============================================================================

/**
 * @brief Singleton factory for creating OpBuilder instances
 *
 * This factory creates appropriate OpBuilder instances based on
 * operator type and library name.
 */
class OpBuilderFactory {
public:
    /**
     * @brief Get singleton instance
     */
    static OpBuilderFactory& Instance();

    /**
     * @brief Create an OpBuilder for the given node
     *
     * @param node The node to create builder for
     * @param libName Library name (e.g., "ascendcpp_lib")
     * @return Pointer to OpBuilder, or nullptr on failure
     */
    std::unique_ptr<OpBuilder> CreateOpBuilder(
        ge::NodePtr node,
        const std::string& libName);

    /**
     * @brief Register a builder creator function for an operator type
     *
     * @param opType Operator type name
     * @param creator Function that creates OpBuilder instance
     */
    void RegisterBuilder(
        const std::string& opType,
        std::function<std::unique_ptr<OpBuilder>()> creator);

private:
    OpBuilderFactory() = default;
    ~OpBuilderFactory() = default;
    OpBuilderFactory(const OpBuilderFactory&) = delete;
    OpBuilderFactory& operator=(const OpBuilderFactory&) = delete;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================================
// TileShapeInferencePass
// ============================================================================

/**
 * @brief Main pass class for TileShape inference
 *
 * This pass implements a three-phase approach:
 * 1. Initialization and input constraint checking
 * 2. TileShape inference for unconfigured nodes
 * 3. Final validation
 *
 * Design Principles:
 * - User Priority: User-configured TileShapes are never modified
 * - Conservative Strategy: Maintain consistency between connected operators
 * - Responsibility Separation: Each operator's logic handled by its OpBuilder
 */
class TileShapeInferencePass {
public:
    TileShapeInferencePass();
    ~TileShapeInferencePass();

    /**
     * @brief Run the TileShape inference pass
     *
     * @param graph The compute graph to process
     * @return fe::OptStatus::SUCCESS or fe::OptStatus::FAILED
     */
    fe::OptStatus Run(ge::ComputeGraphPtr graph);

    // ========== Utility Methods (public for testing) ==========

    /**
     * @brief Check if operator type is data-related (should be skipped)
     *
     * @param opType Operator type name
     * @return true if data-related, false otherwise
     */
    bool IsDataRelatedOp(const std::string& opType);

    /**
     * @brief Check if TileShape is valid
     *
     * @param tileShape TileShape to check
     * @return true if valid, false otherwise
     */
    bool IsValidTileShape(const AscppTileShape& tileShape);

    /**
     * @brief Format error message with error code
     *
     * @param code Error code
     * @param nodeName Node name for context
     * @param suggestion Optional suggestion text
     * @return Formatted error message
     */
    std::string FormatError(
        ErrorCode code,
        const std::string& nodeName,
        const std::string& suggestion = "");

private:
    // ========== Step 1 Methods: Initialization ==========

    /**
     * @brief Check that all input nodes have TileShape configured
     *
     * @param graph The compute graph
     * @return npucl::SUCCESS or npucl::FAILED
     */
    int CheckInputNodesConstraint(ge::ComputeGraphPtr graph);

    /**
     * @brief Get nodes in topological order
     *
     * @param graph The compute graph
     * @param topoNodes [out] Vector of nodes in topological order
     * @return npucl::SUCCESS or npucl::FAILED
     */
    int GetTopologicalOrder(
        ge::ComputeGraphPtr graph,
        std::vector<ge::NodePtr>& topoNodes);

    /**
     * @brief Get TileShape from node attributes
     *
     * @param node The node
     * @return TileShape vector, empty if not configured
     */
    AscppTileShape GetTileShapeFromNode(ge::NodePtr node);

    /**
     * @brief Initialize and validate user-configured nodes
     *
     * @param graph The compute graph
     * @param topoNodes Nodes in topological order
     * @return npucl::SUCCESS or npucl::FAILED
     */
    int InitializeUserConfiguredNodes(
        ge::ComputeGraphPtr graph,
        const std::vector<ge::NodePtr>& topoNodes);

    // ========== Step 2 Methods: Inference ==========

    /**
     * @brief Conservative TileShape inference from predecessors
     *
     * @param node The node to infer TileShape for
     * @param inputTileShape [out] Inferred input N-dim TileShape
     * @param outputTileShape [out] Inferred output N-dim TileShape
     * @return npucl::SUCCESS or npucl::FAILED
     */
    int ConservativeTileShapeInference(
        ge::NodePtr node,
        AscppTileShape& inputTileShape,
        AscppTileShape& outputTileShape);

    /**
     * @brief Get predecessor's output TileShape
     *
     * @param node The node
     * @param predecessorOutputShape [out] Predecessor's output TileShape
     * @return npucl::SUCCESS or npucl::FAILED
     */
    int GetPredecessorOutputTileShape(
        ge::NodePtr node,
        AscppTileShape& predecessorOutputShape);

    /**
     * @brief Convert IoTileShapePair to AscppTileShape
     *
     * @param ioShape Input/output TileShape pair
     * @param node The node
     * @param tileShape [out] Final TileShape
     * @return npucl::SUCCESS or npucl::FAILED
     */
    int ConvertToAscppTileShape(
        const IoTileShapePair& ioShape,
        ge::NodePtr node,
        AscppTileShape& tileShape);

    /**
     * @brief Run TileShape inference for all unconfigured nodes
     *
     * @param graph The compute graph
     * @param topoNodes Nodes in topological order
     * @return npucl::SUCCESS or npucl::FAILED
     */
    int RunInference(
        ge::ComputeGraphPtr graph,
        const std::vector<ge::NodePtr>& topoNodes);

    // ========== Step 3 Methods: Validation ==========

    /**
     * @brief Final validation of all nodes
     *
     * @param graph The compute graph
     * @return npucl::SUCCESS or npucl::FAILED
     */
    int FinalValidation(ge::ComputeGraphPtr graph);

    /**
     * @brief Validate graph consistency
     *
     * @param graph The compute graph
     * @return npucl::SUCCESS or npucl::FAILED
     */
    int ValidateGraphConsistency(ge::ComputeGraphPtr graph);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================================
// Constants
// ============================================================================

/**
 * @brief List of data-related operator types that should be skipped
 * during TileShape inference
 */
const std::vector<std::string> kDataRelatedOps = {
    "Data",
    "Const",
    "Placeholder",
    "Variable",
    "ReadVariableOp",
    "Assign",
    "AssignVariableOp"
};

// ============================================================================
// Error Macros
// ============================================================================

#define ASCPP_CHK_TRUE_RETURN_FAIL(condition, fmt, ...) \
    do { \
        if (condition) { \
            return fe::OptStatus::FAILED; \
        } \
    } while (0)

#define ASCPP_CHK_NULL_RETURN_FAIL(ptr, fmt, ...) \
    do { \
        if ((ptr) == nullptr) { \
            return fe::OptStatus::FAILED; \
        } \
    } while (0)

#define DUMP_GRAPH(graph, stage) \
    do { \
        /* Graph dumping for debugging - implement as needed */ \
    } while (0)

} // namespace tile_shape

#endif // TILE_SHAPE_INFERENCE_PASS_H
