# TileShape Inference Pass

A C++ implementation of the TileShape Inference Pass for neural network compilation, designed to automatically infer TileShape configurations for operators in a compute graph.

## Overview

This pass implements a three-phase approach:

1. **Initialization and Input Constraint Checking** - Validates input nodes and initializes user-configured TileShapes
2. **TileShape Inference** - Propagates TileShapes through the graph using conservative strategy
3. **Final Validation** - Ensures all nodes have valid TileShapes

### Design Principles

- **User Priority**: User-configured TileShapes are never modified during inference
- **Conservative Strategy**: Maintains consistency between connected operators
- **Responsibility Separation**: Each operator's inference logic is handled by its dedicated OpBuilder

## Project Structure

```
tile-shape/
├── tile_shape_inference_pass.h   # Main pass class definition
├── tile_shape_inference_pass.cpp # Main pass implementation
├── op_builders.h                 # OpBuilder base and concrete classes
├── op_builders.cpp               # OpBuilder implementations
├── demo_pass.cpp                 # Original demo reference
├── CMakeLists.txt                # Build configuration
├── example/
│   └── main.cpp                  # Usage examples
├── tests/
│   └── test_pass.cpp             # Unit tests
└── README.md                     # This file
```

## Core Components

### TileShapeInferencePass

The main pass class that orchestrates the TileShape inference process.

```cpp
#include "tile_shape_inference_pass.h"

tile_shape::TileShapeInferencePass pass;
fe::OptStatus status = pass.Run(graph);
```

### OpBuilder (Abstract Base Class)

Defines the interface for operator-specific TileShape handling:

```cpp
class OpBuilder {
public:
    virtual int InitAndValidateTileShape(...) = 0;
    virtual int InferTileShape(...) = 0;
    virtual int ValidateConstraints(ge::NodePtr node) = 0;

protected:
    virtual int CalculateInputTileShape(...) = 0;
    virtual int CalculateOutputTileShape(...) = 0;
};
```

### OpBuilderFactory (Singleton)

Factory for creating OpBuilder instances:

```cpp
auto builder = OpBuilderFactory::Instance().CreateOpBuilder(node, "ascendcpp_lib");
```

## Supported Operators

### Convolution
- Conv2D, Conv2DTranspose, Conv3D

### Matrix Operations
- MatMul, BatchMatMul

### Pooling
- MaxPool, AvgPool, MaxPoolV2, AvgPoolV2

### Element-wise
- Add, Sub, Mul, Div, RealDiv

### Reshape
- Reshape, Flatten, Squeeze, ExpandDims

### Activation
- ReLU, Sigmoid, Tanh, Swish, GELU

### Normalization
- BatchNorm, BatchNormalization, LayerNorm

### Reduce
- ReduceSum, ReduceMean, ReduceMax, ReduceMin

## Node Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `tile_shape` | List<int64_t> | Final effective tile values |
| `tile_shape_input_nd` | List<int64_t> | Input N-dim tile shape (for inference) |
| `tile_shape_output_nd` | List<int64_t> | Output N-dim tile shape (for inference) |
| `tile_shape_user_configured` | bool | Whether user configured this TileShape |

## Error Codes

| Code | Description |
|------|-------------|
| E001 | Input node must have TileShape configured |
| E002 | TileShape validation failed for op |
| E003 | Cannot infer TileShape for op |
| E004 | TileShape violates buffer constraints |
| E005 | Cannot create OpBuilder for op |

## Building

```bash
mkdir build && cd build
cmake ..
make
```

### Build Options

- `BUILD_TESTS=ON` - Build unit tests (default: OFF)
- `BUILD_EXAMPLES=ON` - Build example programs (default: ON)

## Usage

### Basic Usage

```cpp
#include "tile_shape_inference_pass.h"
#include "op_builders.h"

// Register built-in OpBuilders
tile_shape::RegisterBuiltinOpBuilders();

// Create and run the pass
tile_shape::TileShapeInferencePass pass;
fe::OptStatus status = pass.Run(computeGraph);

if (status == fe::OptStatus::SUCCESS) {
    // TileShapes inferred successfully
}
```

### Custom OpBuilder

```cpp
class MyCustomOpBuilder : public tile_shape::BaseOpBuilder {
public:
    // Override methods as needed
    int InferTileShape(...) override {
        // Custom inference logic
    }
};

// Register your custom builder
tile_shape::OpBuilderFactory::Instance().RegisterBuilder(
    "MyCustomOp",
    []() -> std::unique_ptr<tile_shape::OpBuilder> {
        return std::make_unique<MyCustomOpBuilder>();
    }
);
```

## Constraints

| ID | Constraint | Description |
|----|------------|-------------|
| C1 | Input nodes must be configured | All network input nodes require TileShape configuration |
| C2 | User configuration protection | User-configured TileShapes are never modified |
| C3 | Conservative strategy | Prefer maintaining consistency between connected operators |

## License

This project is provided as-is for educational and reference purposes.
