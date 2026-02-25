/**
 * @file test_pass.cpp
 * @brief Unit tests for TileShapeInferencePass
 */

#include "../tile_shape_inference_pass.h"
#include "../op_builders.h"
#include <iostream>
#include <cassert>

using namespace tile_shape;

// Test helper macros
#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "FAILED: " << message << std::endl; \
            return 1; \
        } \
    } while (0)

#define RUN_TEST(test_func) \
    do { \
        std::cout << "Running " << #test_func << "... "; \
        int result = test_func(); \
        if (result == 0) { \
            std::cout << "PASSED" << std::endl; \
            passed++; \
        } else { \
            std::cout << "FAILED" << std::endl; \
            failed++; \
        } \
    } while (0)

/**
 * Test: OpBuilderFactory singleton
 */
int TestOpBuilderFactorySingleton() {
    OpBuilderFactory& factory1 = OpBuilderFactory::Instance();
    OpBuilderFactory& factory2 = OpBuilderFactory::Instance();

    TEST_ASSERT(&factory1 == &factory2, "Factory singleton should return same instance");
    return 0;
}

/**
 * Test: Error code formatting
 */
int TestErrorCodeFormatting() {
    TileShapeInferencePass pass;

    std::string error = pass.FormatError(ErrorCode::E001_INPUT_NODE_NOT_CONFIGURED, "test_node");
    TEST_ASSERT(error.find("E001") != std::string::npos, "Error should contain code E001");
    TEST_ASSERT(error.find("test_node") != std::string::npos, "Error should contain node name");

    error = pass.FormatError(ErrorCode::E003_INFER_FAILED, "infer_node", "Configure manually");
    TEST_ASSERT(error.find("E003") != std::string::npos, "Error should contain code E003");
    TEST_ASSERT(error.find("Suggestion") != std::string::npos, "Error should contain suggestion");

    return 0;
}

/**
 * Test: TileShape validation
 */
int TestTileShapeValidation() {
    TileShapeInferencePass pass;

    // Valid TileShape
    AscppTileShape validShape = {1, 32, 16, 16};
    TEST_ASSERT(pass.IsValidTileShape(validShape), "Valid TileShape should pass");

    // Empty TileShape
    AscppTileShape emptyShape;
    TEST_ASSERT(!pass.IsValidTileShape(emptyShape), "Empty TileShape should fail");

    // Invalid TileShape (contains -1)
    AscppTileShape invalidShape = {1, -1, 16, 16};
    TEST_ASSERT(!pass.IsValidTileShape(invalidShape), "TileShape with -1 should fail");

    // Invalid TileShape (contains 0)
    AscppTileShape zeroShape = {1, 0, 16, 16};
    TEST_ASSERT(!pass.IsValidTileShape(zeroShape), "TileShape with 0 should fail");

    return 0;
}

/**
 * Test: Data-related operator detection
 */
int TestDataRelatedOpDetection() {
    TileShapeInferencePass pass;

    TEST_ASSERT(pass.IsDataRelatedOp("Data"), "Data should be data-related");
    TEST_ASSERT(pass.IsDataRelatedOp("Const"), "Const should be data-related");
    TEST_ASSERT(pass.IsDataRelatedOp("Placeholder"), "Placeholder should be data-related");
    TEST_ASSERT(pass.IsDataRelatedOp("Variable"), "Variable should be data-related");
    TEST_ASSERT(!pass.IsDataRelatedOp("Conv2D"), "Conv2D should not be data-related");
    TEST_ASSERT(!pass.IsDataRelatedOp("MatMul"), "MatMul should not be data-related");

    return 0;
}

/**
 * Test: IoTileShapePair
 */
int TestIoTileShapePair() {
    AscppTileShape inputShape = {1, 64, 32, 32};
    AscppTileShape outputShape = {1, 32, 16, 16};

    IoTileShapePair pair = std::make_pair(inputShape, outputShape);

    TEST_ASSERT(pair.first == inputShape, "First element should be input shape");
    TEST_ASSERT(pair.second == outputShape, "Second element should be output shape");

    return 0;
}

/**
 * Test: Constants
 */
int TestConstants() {
    TEST_ASSERT(!kDataRelatedOps.empty(), "kDataRelatedOps should not be empty");

    // Check that key data-related ops are in the list
    bool hasData = false;
    for (const auto& op : kDataRelatedOps) {
        if (op == "Data") {
            hasData = true;
            break;
        }
    }
    TEST_ASSERT(hasData, "kDataRelatedOps should contain 'Data'");

    return 0;
}

/**
 * Test: Register builtin builders
 */
int TestRegisterBuiltinBuilders() {
    // This should not throw
    RegisterBuiltinOpBuilders();

    // Factory should now have builders registered
    // Note: Actual builder creation requires a valid node, which we can't test without mock

    return 0;
}

int main() {
    std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║           TileShapeInferencePass Unit Tests                ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;

    int passed = 0;
    int failed = 0;

    RUN_TEST(TestOpBuilderFactorySingleton);
    RUN_TEST(TestErrorCodeFormatting);
    RUN_TEST(TestTileShapeValidation);
    RUN_TEST(TestDataRelatedOpDetection);
    RUN_TEST(TestIoTileShapePair);
    RUN_TEST(TestConstants);
    RUN_TEST(TestRegisterBuiltinBuilders);

    std::cout << std::endl;
    std::cout << "════════════════════════════════════════════════════════════" << std::endl;
    std::cout << "Results: " << passed << " passed, " << failed << " failed" << std::endl;
    std::cout << "════════════════════════════════════════════════════════════" << std::endl;

    return failed > 0 ? 1 : 0;
}
