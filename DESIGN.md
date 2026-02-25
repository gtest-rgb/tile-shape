# TileShape 推导 Pass 软件设计文档

## 1. 概述

### 1.1 目标

设计一个 TileShape 推导 Pass，用于神经网络编译过程中自动推导算子的 TileShape 配置。

### 1.2 设计原则

| 原则 | 描述 |
|------|------|
| **用户优先** | 用户配置的 TileShape 为最高优先级，推导过程不修改 |
| **保守策略** | 保持前后算子 TileShape 一致性，优化动作由用户指定 |
| **责任分离** | 每个算子的推导逻辑由对应的 OpBuilder 负责 |

### 1.3 适用范围

本设计适用于需要自动推导 TileShape 配置的神经网络编译器，支持常见算子类型。

---

## 2. 约束条件

| 编号 | 约束 | 说明 |
|------|------|------|
| C1 | 输入节点必配置 | 网络所有输入 Node 必须配置 TileShape |
| C2 | 用户配置保护 | 用户配置的 TileShape 推导过程中不修改 |
| C3 | 保守策略 | 优先保持前后算子 TileShape 一致 |

---

## 3. 核心数据结构

### 3.1 TileShape 类型定义

```cpp
// TileShape 基础类型
using AscppTileShape = std::vector<int64_t>;

// 输入/输出 TileShape 对
using IoTileShapePair = std::pair<AscppTileShape, AscppTileShape>;
```

### 3.2 Node 属性定义

| 属性名 | 类型 | 说明 |
|--------|------|------|
| `tile_shape` | List<int64_t> | 最终生效的 tile 值 |
| `tile_shape_input_nd` | List<int64_t> | 输入 N 维 tile shape，仅用于推导 |
| `tile_shape_output_nd` | List<int64_t> | 输出 N 维 tile shape，仅用于推导 |
| `tile_shape_user_configured` | bool | 是否用户配置 |

### 3.3 错误码定义

```cpp
enum class ErrorCode {
    E001_INPUT_NODE_NOT_CONFIGURED = 1,    // 输入节点必须配置 TileShape
    E002_VALIDATION_FAILED = 2,             // TileShape 校验失败
    E003_INFER_FAILED = 3,                  // 无法推导 TileShape
    E004_BUFFER_CONSTRAINT_VIOLATION = 4,   // TileShape 违反 buffer 约束
    E005_OPBUILDER_CREATE_FAILED = 5        // 无法创建 OpBuilder
};
```

---

## 4. 模块设计

### 4.1 模块架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    TileShapeInferencePass                        │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                         Run()                              │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐   │  │
│  │  │  Step 1:    │  │  Step 2:    │  │     Step 3:      │   │  │
│  │  │ 初始化与校验 │→│ TileShape   │→│    最终验证      │   │  │
│  │  │             │  │ 推导        │  │                  │   │  │
│  │  └─────────────┘  └─────────────┘  └──────────────────┘   │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  OpBuilderFactory (单例)                         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │           CreateOpBuilder(node, lib_name)                  │  │
│  │           RegisterBuilder(opType, creator)                 │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  OpBuilder (抽象基类)                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ + InitAndValidateTileShape()     // 初始化并校验           │  │
│  │ + InferTileShape()               // 推导 TileShape         │  │
│  │ + CalculateInputTileShape()      // 计算输入 TileShape     │  │
│  │ + CalculateOutputTileShape()     // 计算输出 TileShape     │  │
│  │ + ValidateConstraints()          // 校验约束               │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    ┌──────────┐        ┌──────────┐        ┌──────────┐
    │ConvOpBuilder│      │MatmulOpBuilder│    │PoolOpBuilder│
    └──────────┘        └──────────┘        └──────────┘
          │                   │                   │
          ▼                   ▼                   ▼
    ┌──────────┐        ┌──────────┐        ┌──────────┐
    │Elementwise│       │ReshapeOpBuilder│   │ActivationOp│
    │OpBuilder  │       └──────────┘        │Builder    │
    └──────────┘                            └──────────┘
          │                   │                   │
          ▼                   ▼                   ▼
    ┌──────────┐        ┌──────────┐        ┌──────────┐
    │BatchNorm │        │ReduceOpBuilder│    │...        │
    │OpBuilder │        └──────────┘        └──────────┘
    └──────────┘
```

### 4.2 类图

```
┌─────────────────────────────────────────────────────────────────┐
│                      <<interface>>                               │
│                        OpBuilder                                 │
├─────────────────────────────────────────────────────────────────┤
│ + InitAndValidateTileShape(node, userTileShape,                 │
│                           inputTileShape, outputTileShape): int │
│ + InferTileShape(node, predecessorOutputShape, tileShape,       │
│                  inputTileShape, outputTileShape): int          │
│ + ValidateConstraints(node): int                                │
│ # CalculateInputTileShape(node, outputTileShape,                │
│                           inputTileShape): int                  │
│ # CalculateOutputTileShape(node, inputTileShape,                │
│                            outputTileShape): int                │
└─────────────────────────────────────────────────────────────────┘
                              △
                              │
┌─────────────────────────────────────────────────────────────────┐
│                       BaseOpBuilder                              │
├─────────────────────────────────────────────────────────────────┤
│ # GetNumDimensions(node): int                                   │
│ # ValidateDimensions(tileShape, expectedDims): bool             │
└─────────────────────────────────────────────────────────────────┘
                              △
          ┌───────────────────┼───────────────────┐
          │                   │                   │
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  ConvOpBuilder  │  │ MatmulOpBuilder │  │  PoolOpBuilder  │
├─────────────────┤  ├─────────────────┤  ├─────────────────┤
│ - GetConvAttrs()│  │ - GetTranspose()│  │ - GetPoolAttrs()│
│ - CalcOutputSize│  │   Attrs()       │  │                 │
│ - CalcInputSize │  │                 │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### 4.3 OpBuilder 接口定义

```cpp
class OpBuilder {
public:
    virtual ~OpBuilder() = default;

    /**
     * @brief 初始化并校验用户配置的 TileShape
     *
     * @param node 目标节点
     * @param userTileShape 用户配置的 TileShape
     * @param inputTileShape [out] 计算得到的输入 TileShape
     * @param outputTileShape [out] 计算得到的输出 TileShape
     * @return npucl::SUCCESS 或 npucl::FAILED
     */
    virtual int InitAndValidateTileShape(
        ge::NodePtr node,
        const AscppTileShape& userTileShape,
        AscppTileShape& inputTileShape,
        AscppTileShape& outputTileShape) = 0;

    /**
     * @brief 从前驱节点推导 TileShape（保守策略）
     *
     * @param node 目标节点
     * @param predecessorOutputShape 前驱节点的输出 TileShape
     * @param tileShape [out] 推导得到的最终 TileShape
     * @param inputTileShape [out] 推导得到的输入 TileShape
     * @param outputTileShape [out] 推导得到的输出 TileShape
     * @return npucl::SUCCESS 或 npucl::FAILED
     */
    virtual int InferTileShape(
        ge::NodePtr node,
        const AscppTileShape& predecessorOutputShape,
        AscppTileShape& tileShape,
        AscppTileShape& inputTileShape,
        AscppTileShape& outputTileShape) = 0;

    /**
     * @brief 校验算子特定约束
     *
     * @param node 目标节点
     * @return npucl::SUCCESS 或 npucl::FAILED
     */
    virtual int ValidateConstraints(ge::NodePtr node) = 0;

protected:
    /**
     * @brief 根据输出 TileShape 计算输入 TileShape
     */
    virtual int CalculateInputTileShape(
        ge::NodePtr node,
        const AscppTileShape& outputTileShape,
        AscppTileShape& inputTileShape) = 0;

    /**
     * @brief 根据输入 TileShape 计算输出 TileShape
     */
    virtual int CalculateOutputTileShape(
        ge::NodePtr node,
        const AscppTileShape& inputTileShape,
        AscppTileShape& outputTileShape) = 0;
};
```

---

## 5. 流程设计

### 5.1 Step 1: 初始化与输入约束检查

#### 5.1.1 目标

获取用户配置，调用 OpBuilder 计算并校验。

#### 5.1.2 流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                     Step 1: 初始化与校验                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│            1.1 检查所有输入节点是否配置 TileShape                 │
│                    CheckInputNodesConstraint()                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   1.2 获取拓扑排序节点列表                        │
│                    GetTopologicalOrder()                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    1.3 遍历每个节点                              │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    ┌──────────────────┐            ┌──────────────────┐
    │   已配置 TileShape │            │  未配置 TileShape │
    └──────────────────┘            └──────────────────┘
              │                               │
              ▼                               ▼
    ┌──────────────────┐            ┌──────────────────┐
    │ 标记 user_config  │            │ 设置默认值 {-1...}│
    │    = true        │            └──────────────────┘
    └──────────────────┘
              │
              ▼
    ┌──────────────────┐
    │ 创建 OpBuilder    │
    └──────────────────┘
              │
              ▼
    ┌──────────────────┐
    │ InitAndValidate  │
    │   TileShape()    │
    └──────────────────┘
              │
              ▼
    ┌──────────────────┐
    │ 存储计算结果到    │
    │   节点属性        │
    └──────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              1.4 最终检查：所有输入节点必须配置                    │
└─────────────────────────────────────────────────────────────────┘
```

#### 5.1.3 伪代码

```cpp
int InitializeUserConfiguredNodes(graph, topoNodes) {
    for (node in topoNodes) {
        currentTileShape = GetTileShapeFromNode(node);

        if (!currentTileShape.empty()) {
            // 标记为用户配置
            SetBool(node, "tile_shape_user_configured", true);

            // 创建 OpBuilder
            opBuilder = OpBuilderFactory::Instance().CreateOpBuilder(node, "ascendcpp_lib");
            if (opBuilder == nullptr) {
                return FAILED;  // Error E005
            }

            // 初始化并校验
            result = opBuilder->InitAndValidateTileShape(
                node, currentTileShape, inputTileShape, outputTileShape);
            if (result == FAILED) {
                return FAILED;  // Error E002
            }

            // 存储结果
            SetListInt(node, "tile_shape_input_nd", inputTileShape);
            SetListInt(node, "tile_shape_output_nd", outputTileShape);
        } else {
            // 未配置：设置默认值
            SetListInt(node, "tile_shape_input_nd", {-1, -1, ...});
            SetListInt(node, "tile_shape_output_nd", {-1, -1, ...});
        }
    }
    return SUCCESS;
}
```

### 5.2 Step 2: TileShape 推导

#### 5.2.1 目标

按拓扑顺序推导未配置节点的 TileShape。

#### 5.2.2 流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                     Step 2: TileShape 推导                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   2.1 按拓扑顺序遍历节点                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   获取节点类型   │
                    └─────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    ┌──────────────────┐            ┌──────────────────┐
    │ 数据相关算子？    │──Yes─────→│     跳过         │
    │ (kDataRelatedOps)│            └──────────────────┘
    └──────────────────┘
              │ No
              ▼
    ┌──────────────────┐
    │ 已用户配置？      │──Yes─────→│     跳过         │
    └──────────────────┘            └──────────────────┘
              │ No
              ▼
┌─────────────────────────────────────────────────────────────────┐
│              2.2 保守策略：从前驱推导                             │
│           ConservativeTileShapeInference()                       │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
    ┌──────────────────┐
    │ 获取前驱节点的    │
    │ outputTileShape  │
    └──────────────────┘
              │
              ▼
    ┌──────────────────┐
    │ 创建 OpBuilder    │
    └──────────────────┘
              │
              ▼
    ┌──────────────────┐
    │ InferTileShape() │
    └──────────────────┘
              │
              ▼
    ┌──────────────────┐
    │ 存储推导结果：    │
    │ - tile_shape     │
    │ - tile_shape_    │
    │   input_nd       │
    │ - tile_shape_    │
    │   output_nd      │
    └──────────────────┘
              │
              ▼
    ┌──────────────────┐
    │ 推导失败？        │──Yes──→ Error E003
    └──────────────────┘
```

#### 5.2.3 伪代码

```cpp
int RunInference(graph, topoNodes) {
    for (node in topoNodes) {
        nodeType = node->GetType();

        // 跳过数据相关节点
        if (IsDataRelatedOp(nodeType)) {
            continue;
        }

        currentTileShape = GetTileShapeFromNode(node);

        // 跳过已配置节点（保护用户配置）
        isUserConfigured = GetBool(node, "tile_shape_user_configured");
        if (isUserConfigured || !currentTileShape.empty()) {
            continue;
        }

        // 保守策略：优先从前驱推导
        result = ConservativeTileShapeInference(
            node, inputTileShape, outputTileShape);
        if (result == FAILED) {
            return FAILED;  // Error E003: 建议用户手动配置
        }

        // 设置推导结果
        SetListInt(node, "tile_shape", currentTileShape);
        SetListInt(node, "tile_shape_input_nd", inputTileShape);
        SetListInt(node, "tile_shape_output_nd", outputTileShape);
    }
    return SUCCESS;
}
```

### 5.3 Step 3: 最终验证

#### 5.3.1 目标

确保所有节点都有有效的 TileShape。

#### 5.3.2 流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                     Step 3: 最终验证                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   3.1 遍历所有节点                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ 检查 tile_shape │
                    │   是否有效       │
                    └─────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    ┌──────────────────┐            ┌──────────────────┐
    │     有效         │            │     无效         │
    └──────────────────┘            │  记录错误信息     │
              │                     └──────────────────┘
              │                               │
              ▼                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                 3.2 检查图一致性                                  │
│              ValidateGraphConsistency()                          │
│                                                                  │
│   检查相连节点的 tile_shape_output_nd 和 tile_shape_input_nd     │
│   是否兼容                                                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   3.3 报告剩余问题                               │
└─────────────────────────────────────────────────────────────────┘
```

#### 5.3.3 伪代码

```cpp
int FinalValidation(graph) {
    // 检查所有节点
    for (node in graph->GetAllNodes()) {
        tileShape = GetTileShapeFromNode(node);
        if (!IsValidTileShape(tileShape)) {
            ReportError("Node %s has invalid TileShape", node->GetName());
            return FAILED;
        }
    }

    // 检查图一致性
    return ValidateGraphConsistency(graph);
}

int ValidateGraphConsistency(graph) {
    for (edge in graph->GetAllEdges()) {
        srcNode = edge->GetSrcNode();
        dstNode = edge->GetDstNode();

        srcOutputShape = GetListInt(srcNode, "tile_shape_output_nd");
        dstInputShape = GetListInt(dstNode, "tile_shape_input_nd");

        if (srcOutputShape != dstInputShape) {
            ReportError("Inconsistency between %s and %s",
                srcNode->GetName(), dstNode->GetName());
            return FAILED;
        }
    }
    return SUCCESS;
}
```

---

## 6. 时序图

```
┌─────────┐     ┌────────────┐     ┌──────────────┐     ┌───────────┐     ┌──────┐
│  Pass   │     │ComputeGraph│     │OpBuilderFactory│    │ OpBuilder │     │ Node │
└────┬────┘     └─────┬──────┘     └──────┬───────┘     └─────┬─────┘     └──┬───┘
     │                │                   │                   │              │
     │  Run(graph)    │                   │                   │              │
     │───────────────>│                   │                   │              │
     │                │                   │                   │              │
     │ ═══════════════════ Step 1: 初始化与校验 ═════════════════════════════│
     │                │                   │                   │              │
     │ CheckInputNodes│                   │                   │              │
     │ Constraint()   │                   │                   │              │
     │───────────────>│                   │                   │              │
     │                │                   │                   │              │
     │ GetTopological │                   │                   │              │
     │ Order()        │                   │                   │              │
     │───────────────>│                   │                   │              │
     │<───────────────│                   │                   │              │
     │   topoNodes    │                   │                   │              │
     │                │                   │                   │              │
     │                │                   │                   │              │
     │     ┌──────────────────────────────────────────────────────────────┐   │
     │     │              for each node in topoNodes                      │   │
     │     └──────────────────────────────────────────────────────────────┘   │
     │                │                   │                   │              │
     │                                    │                   │              │
     │                │   CreateOpBuilder │                   │              │
     │                │   (node, lib)     │                   │              │
     │                │──────────────────>│                   │              │
     │                │                   │                   │              │
     │                │                   │  new OpBuilder()  │              │
     │                │                   │──────────────────>│              │
     │                │                   │<──────────────────│              │
     │                │<──────────────────│                   │              │
     │                │   opBuilder       │                   │              │
     │                │                   │                   │              │
     │                                    │                   │              │
     │                │                   │                   │              │
     │                │                   │ InitAndValidate   │              │
     │                │                   │ TileShape()       │              │
     │                │                   │──────────────────>│              │
     │                │                   │                   │              │
     │                │                   │                   │ GetAttrs()   │
     │                │                   │                   │─────────────>│
     │                │                   │                   │<─────────────│
     │                │                   │                   │              │
     │                │                   │                   │ Validate()   │
     │                │                   │                   │─────────────>│
     │                │                   │                   │<─────────────│
     │                │                   │<──────────────────│              │
     │                │                   │  inputShape,      │              │
     │                │                   │  outputShape      │              │
     │                │                   │                   │              │
     │                │                   │                   │ SetAttrs()   │
     │                │                   │                   │─────────────>│
     │                │                   │                   │              │
     │                │                   │                   │              │
     │ ═══════════════════ Step 2: 推导 ═════════════════════════════════════│
     │                │                   │                   │              │
     │     ┌──────────────────────────────────────────────────────────────┐   │
     │     │              for each node in topoNodes                      │   │
     │     └──────────────────────────────────────────────────────────────┘   │
     │                │                   │                   │              │
     │                │                   │                   │              │
     │  GetPredecessor│                   │                   │              │
     │  OutputShape() │                   │                   │              │
     │───────────────>│                   │                   │              │
     │<───────────────│                   │                   │              │
     │  predShape     │                   │                   │              │
     │                │                   │                   │              │
     │                │   CreateOpBuilder │                   │              │
     │                │──────────────────>│                   │              │
     │                │<──────────────────│                   │              │
     │                │   opBuilder       │                   │              │
     │                │                   │                   │              │
     │                │                   │ InferTileShape()  │              │
     │                │                   │──────────────────>│              │
     │                │                   │                   │              │
     │                │                   │                   │ Calculate    │
     │                │                   │                   │ InputShape() │
     │                │                   │                   │─────────────>│
     │                │                   │                   │              │
     │                │                   │                   │ Calculate    │
     │                │                   │                   │ OutputShape()│
     │                │                   │                   │─────────────>│
     │                │                   │<──────────────────│              │
     │                │                   │  tileShape,       │              │
     │                │                   │  inputShape,      │              │
     │                │                   │  outputShape      │              │
     │                │                   │                   │              │
     │                │                   │                   │ SetAttrs()   │
     │                │                   │                   │─────────────>│
     │                │                   │                   │              │
     │ ═══════════════════ Step 3: 验证 ═════════════════════════════════════│
     │                │                   │                   │              │
     │ FinalValidation│                   │                   │              │
     │───────────────>│                   │                   │              │
     │                │                   │                   │              │
     │ ValidateGraph  │                   │                   │              │
     │ Consistency()  │                   │                   │              │
     │───────────────>│                   │                   │              │
     │<───────────────│                   │                   │              │
     │                │                   │                   │              │
     │<───────────────│                   │                   │              │
     │  SUCCESS       │                   │                   │              │
     │                │                   │                   │              │
```

---

## 7. 错误处理策略

### 7.1 错误码与消息

| 错误场景 | 错误码 | 错误信息模板 | 建议 |
|----------|--------|--------------|------|
| 输入节点未配置 TileShape | E001 | "Input node '%s' must have TileShape configured" | 检查输入节点配置 |
| OpBuilder 校验失败 | E002 | "TileShape validation failed for op '%s': %s" | 修改 TileShape 配置 |
| 推导失败 | E003 | "Cannot infer TileShape for op '%s'" | 手动配置 TileShape |
| Buffer 约束违反 | E004 | "TileShape violates buffer constraints for op '%s'" | 调整 TileShape 大小 |
| OpBuilder 创建失败 | E005 | "Cannot create OpBuilder for op '%s'" | 检查算子类型支持 |

### 7.2 错误处理代码示例

```cpp
std::string FormatError(ErrorCode code, const std::string& nodeName, const std::string& suggestion) {
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
```

---

## 8. 支持的算子类型

### 8.1 算子分类

| 类别 | 算子类型 | OpBuilder |
|------|----------|-----------|
| 卷积 | Conv2D, Conv2DTranspose, Conv3D | ConvOpBuilder |
| 矩阵乘 | MatMul, BatchMatMul | MatmulOpBuilder |
| 池化 | MaxPool, AvgPool, MaxPoolV2, AvgPoolV2 | PoolOpBuilder |
| 逐元素 | Add, Sub, Mul, Div, RealDiv | ElementwiseOpBuilder |
| 变形 | Reshape, Flatten, Squeeze, ExpandDims | ReshapeOpBuilder |
| 激活 | ReLU, Sigmoid, Tanh, Swish, GELU | ActivationOpBuilder |
| 归一化 | BatchNorm, BatchNormalization, LayerNorm | BatchNormOpBuilder |
| 规约 | ReduceSum, ReduceMean, ReduceMax, ReduceMin | ReduceOpBuilder |

### 8.2 数据相关算子（跳过推导）

```cpp
const std::vector<std::string> kDataRelatedOps = {
    "Data",
    "Const",
    "Placeholder",
    "Variable",
    "ReadVariableOp",
    "Assign",
    "AssignVariableOp"
};
```

---

## 9. OpBuilder 实现示例

### 9.1 ConvOpBuilder

```cpp
class ConvOpBuilder : public BaseOpBuilder {
public:
    int CalculateOutputTileShape(
        ge::NodePtr node,
        const AscppTileShape& inputTileShape,
        AscppTileShape& outputTileShape) override {

        // 获取卷积属性
        std::vector<int64_t> strides, pads, dilations, kernels;
        GetConvAttributes(node, strides, pads, dilations, kernels);

        // 计算输出 shape
        // output = (input + pad - kernel) / stride + 1
        outputTileShape.resize(inputTileShape.size());
        outputTileShape[0] = inputTileShape[0];  // N
        outputTileShape[1] = inputTileShape[1];  // C (实际由 filter 决定)

        for (int i = 2; i < inputTileShape.size(); i++) {
            outputTileShape[i] = CalculateOutputSize(
                inputTileShape[i],
                kernels[i-2],
                strides[i-2],
                pads[(i-2)*2],
                pads[(i-2)*2+1],
                dilations[i-2]);
        }

        return npucl::SUCCESS;
    }

private:
    int64_t CalculateOutputSize(
        int64_t inputSize,
        int64_t kernelSize,
        int64_t stride,
        int64_t padStart,
        int64_t padEnd,
        int64_t dilation) {

        int64_t effectiveKernel = (kernelSize - 1) * dilation + 1;
        return (inputSize + padStart + padEnd - effectiveKernel) / stride + 1;
    }
};
```

### 9.2 ElementwiseOpBuilder

```cpp
class ElementwiseOpBuilder : public BaseOpBuilder {
public:
    int InferTileShape(
        ge::NodePtr node,
        const AscppTileShape& predecessorOutputShape,
        AscppTileShape& tileShape,
        AscppTileShape& inputTileShape,
        AscppTileShape& outputTileShape) override {

        // 逐元素操作：输入输出 shape 相同
        inputTileShape = predecessorOutputShape;
        outputTileShape = predecessorOutputShape;
        tileShape = predecessorOutputShape;

        return ValidateConstraints(node);
    }
};
```

---

## 10. 扩展指南

### 10.1 添加新算子支持

1. 创建新的 OpBuilder 类：

```cpp
class MyCustomOpBuilder : public BaseOpBuilder {
public:
    int InitAndValidateTileShape(...) override { /* ... */ }
    int InferTileShape(...) override { /* ... */ }
    int ValidateConstraints(ge::NodePtr node) override { /* ... */ }
};
```

2. 注册 OpBuilder：

```cpp
void RegisterBuiltinOpBuilders() {
    // ... 其他注册 ...

    // 注册自定义算子
    RegisterOpBuilderImpl<MyCustomOpBuilder>("MyCustomOp");
}
```

### 10.2 自定义约束校验

```cpp
int MyCustomOpBuilder::ValidateConstraints(ge::NodePtr node) {
    // 获取 TileShape
    AscppTileShape tileShape = GetTileShapeFromNode(node);

    // 自定义约束检查
    if (tileShape[0] > MAX_BATCH_SIZE) {
        return npucl::FAILED;  // 违反约束
    }

    return npucl::SUCCESS;
}
```

---

## 11. 文件结构

```
tile-shape/
├── tile_shape_inference_pass.h   # 主 Pass 类定义
├── tile_shape_inference_pass.cpp # 主 Pass 实现
├── op_builders.h                 # OpBuilder 基类和具体类定义
├── op_builders.cpp               # OpBuilder 实现
├── CMakeLists.txt                # 构建配置
├── example/
│   └── main.cpp                  # 使用示例
├── tests/
│   └── test_pass.cpp             # 单元测试
├── README.md                     # 项目说明
└── DESIGN.md                     # 本设计文档
```

---

## 12. 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| 1.0 | 2024-02-25 | 初始设计 |

---

## 附录 A: 常量定义

```cpp
// 数据相关算子列表
const std::vector<std::string> kDataRelatedOps = {
    "Data",
    "Const",
    "Placeholder",
    "Variable",
    "ReadVariableOp",
    "Assign",
    "AssignVariableOp"
};

// 状态码
namespace npucl {
    constexpr int SUCCESS = 0;
    constexpr int FAILED = 1;
}

// Pass 返回状态
namespace fe {
    enum OptStatus {
        SUCCESS = 0,
        FAILED = 1
    };
}
```

## 附录 B: 宏定义

```cpp
// 条件检查宏
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

// 图转储宏
#define DUMP_GRAPH(graph, stage) \
    do { \
        /* 用于调试的图转储 */ \
    } while (0)
```
