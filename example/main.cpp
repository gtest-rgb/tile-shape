/**
 * @file main.cpp
 * @brief Example usage of TileShapeInferencePass
 *
 * This example demonstrates how to use the TileShapeInferencePass
 * to automatically infer TileShape configurations for operators
 * in a neural network compute graph.
 */

#include "tile_shape_inference_pass.h"
#include "op_builders.h"
#include <iostream>

using namespace tile_shape;

/**
 * @brief Example: Creating and running the TileShape inference pass
 *
 * This function demonstrates the typical usage pattern:
 * 1. Create the pass instance
 * 2. Run on a compute graph
 * 3. Check results
 */
void ExampleUsage() {
    std::cout << "=== TileShapeInferencePass Example ===" << std::endl;
    std::cout << std::endl;

    // In a real scenario, you would have a compute graph from your framework
    // ge::ComputeGraphPtr graph = CreateComputeGraph();

    // Create the inference pass
    TileShapeInferencePass pass;

    // Run the pass on the graph
    // fe::OptStatus status = pass.Run(graph);

    std::cout << "Pass created successfully." << std::endl;
    std::cout << std::endl;

    // Demonstrate the three-phase approach
    std::cout << "The pass follows a three-phase approach:" << std::endl;
    std::cout << "  1. Initialization and input constraint checking" << std::endl;
    std::cout << "  2. TileShape inference for unconfigured nodes" << std::endl;
    std::cout << "  3. Final validation" << std::endl;
    std::cout << std::endl;
}

/**
 * @brief Example: Using OpBuilderFactory
 */
void ExampleOpBuilderFactory() {
    std::cout << "=== OpBuilderFactory Example ===" << std::endl;
    std::cout << std::endl;

    // The factory is a singleton
    OpBuilderFactory& factory = OpBuilderFactory::Instance();

    std::cout << "OpBuilderFactory instance obtained." << std::endl;
    std::cout << "Registered builders for operator types:" << std::endl;
    std::cout << "  - Conv2D, Conv2DTranspose, Conv3D" << std::endl;
    std::cout << "  - MatMul, BatchMatMul" << std::endl;
    std::cout << "  - MaxPool, AvgPool" << std::endl;
    std::cout << "  - Add, Sub, Mul, Div" << std::endl;
    std::cout << "  - Reshape, Flatten, Squeeze" << std::endl;
    std::cout << "  - ReLU, Sigmoid, Tanh, GELU" << std::endl;
    std::cout << "  - BatchNorm, LayerNorm" << std::endl;
    std::cout << "  - ReduceSum, ReduceMean, ReduceMax, ReduceMin" << std::endl;
    std::cout << std::endl;
}

/**
 * @brief Example: Custom OpBuilder registration
 */
void ExampleCustomBuilder() {
    std::cout << "=== Custom OpBuilder Registration Example ===" << std::endl;
    std::cout << std::endl;

    // You can register custom OpBuilders for your own operator types
    // Example:
    // class MyCustomOpBuilder : public BaseOpBuilder { ... };
    // REGISTER_OP_BUILDER("MyCustomOp", MyCustomOpBuilder);

    std::cout << "To register a custom OpBuilder:" << std::endl;
    std::cout << "  1. Create a class inheriting from BaseOpBuilder or OpBuilder" << std::endl;
    std::cout << "  2. Override the virtual methods as needed" << std::endl;
    std::cout << "  3. Use REGISTER_OP_BUILDER(\"OpType\", BuilderClass) macro" << std::endl;
    std::cout << std::endl;
}

/**
 * @brief Example: Understanding TileShape attributes
 */
void ExampleTileShapeAttributes() {
    std::cout << "=== TileShape Attributes Example ===" << std::endl;
    std::cout << std::endl;

    std::cout << "Node attributes used by the pass:" << std::endl;
    std::cout << "  - tile_shape: Final tile values (List<int64_t>)" << std::endl;
    std::cout << "  - tile_shape_input_nd: Input N-dim tile shape (List<int64_t>)" << std::endl;
    std::cout << "  - tile_shape_output_nd: Output N-dim tile shape (List<int64_t>)" << std::endl;
    std::cout << "  - tile_shape_user_configured: Whether user configured (bool)" << std::endl;
    std::cout << std::endl;

    std::cout << "Example TileShape for 4D tensor (NCHW):" << std::endl;
    std::cout << "  tile_shape = [1, 32, 16, 16]" << std::endl;
    std::cout << "  This means tiling 1 batch, 32 channels, 16 height, 16 width" << std::endl;
    std::cout << std::endl;
}

/**
 * @brief Example: Error handling
 */
void ExampleErrorHandling() {
    std::cout << "=== Error Handling Example ===" << std::endl;
    std::cout << std::endl;

    std::cout << "Error codes used by the pass:" << std::endl;
    std::cout << "  E001: Input node must have TileShape configured" << std::endl;
    std::cout << "  E002: TileShape validation failed for op" << std::endl;
    std::cout << "  E003: Cannot infer TileShape for op" << std::endl;
    std::cout << "  E004: TileShape violates buffer constraints" << std::endl;
    std::cout << "  E005: Cannot create OpBuilder for op" << std::endl;
    std::cout << std::endl;

    std::cout << "When errors occur, suggestions are provided:" << std::endl;
    std::cout << "  - For E003: 'Suggestion: Configure TileShape explicitly'" << std::endl;
    std::cout << "  - For E002: 'Suggestion: Modify TileShape configuration'" << std::endl;
    std::cout << std::endl;
}

/**
 * @brief Example: Conservative strategy explanation
 */
void ExampleConservativeStrategy() {
    std::cout << "=== Conservative Strategy Example ===" << std::endl;
    std::cout << std::endl;

    std::cout << "The pass uses a conservative strategy:" << std::endl;
    std::cout << "  - User-configured TileShapes are NEVER modified" << std::endl;
    std::cout << "  - Unconfigured nodes inherit from predecessors" << std::endl;
    std::cout << "  - Optimization actions are left to user specification" << std::endl;
    std::cout << std::endl;

    std::cout << "Example graph flow:" << std::endl;
    std::cout << "  Input [1,64,32,32] (user configured)" << std::endl;
    std::cout << "    -> Conv2D [1,64,32,32] (inherited from input)" << std::endl;
    std::cout << "    -> ReLU [1,64,32,32] (inherited from conv)" << std::endl;
    std::cout << "    -> MaxPool [1,64,16,16] (calculated from pool params)" << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║       TileShape Inference Pass - Example Program           ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;

    ExampleUsage();
    ExampleOpBuilderFactory();
    ExampleCustomBuilder();
    ExampleTileShapeAttributes();
    ExampleErrorHandling();
    ExampleConservativeStrategy();

    std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                    End of Examples                         ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;

    return 0;
}
