fe::OptStatus AscendcppTileShapeInferencePass::Run(ge::ComputeGraphPtr graph)
{
    DUMP_GRAPH(graph, "before_ascendcpp_tile_shape_inference_pass");

    // ========== Step 1: 初始化与输入约束检查 ==========
    
    // 1.1 检查所有输入节点是否配置 TileShape
    ASCPP_CHK_TRUE_RETURN_FAIL(CheckInputNodesConstraint(graph) == npucl::FAILED,
        "Input node constraint check failed");

    // 1.2 获取拓扑顺序
    std::vector<ge::NodePtr> topoNodes;
    ASCPP_CHK_TRUE_RETURN_FAIL(GetTopologicalOrder(graph, topoNodes) == npucl::FAILED,
        "GetTopologicalOrder failed");

    // 1.3 初始化用户配置的节点
    for (auto node : topoNodes) {
        AscppTileShape currentTileShape = GetTileShapeFromNode(node);
        if (!currentTileShape.empty()) {
            // 标记为用户配置
            ge::AttrUtils::SetBool(node->GetOpDesc(), "tile_shape_user_configured", true);
            
            // 调用 OpBuilder 初始化并校验
            auto opBuilder = OpBuilderFactory::Instance().CreateOpBuilder(node, "ascendcpp_lib");
            ASCPP_CHK_NULL_RETURN_FAIL(opBuilder, "Can not create opbuilder for op[%s]", 
                node->GetOpDesc()->GetName().c_str());
            
            ASCPP_CHK_TRUE_RETURN_FAIL(
                opBuilder->InitAndValidateTileShape(node, currentTileShape) == npucl::FAILED,
                "InitAndValidateTileShape failed for op[%s]", 
                node->GetOpDesc()->GetName().c_str());
        }
    }

    DUMP_GRAPH(graph, "middle_ascendcpp_tile_shape_inference_pass");

    // ========== Step 2: 推导 TileShape ==========
    
    for (auto node : topoNodes) {
        std::string nodeName = node->GetOpDesc()->GetName();
        std::string nodeType = node->GetOpDesc()->GetType();
        
        // 跳过数据相关节点
        if (std::find(kDataRelatedOps.begin(), kDataRelatedOps.end(), nodeType) != kDataRelatedOps.end()) {
            continue;
        }

        AscppTileShape currentTileShape = GetTileShapeFromNode(node);
        
        // 跳过已配置节点（保护用户配置）
        bool isUserConfigured = false;
        ge::AttrUtils::GetBool(node->GetOpDesc(), "tile_shape_user_configured", isUserConfigured);
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
        IoTileShapePair nodeInferredIoShape = std::make_pair(nodeInferredNdInputShape, nodeInferredNdOutputShape);
        ASCPP_CHK_TRUE_RETURN_FAIL(
            ConvertToAscppTileShape(nodeInferredIoShape, node, currentTileShape) == npucl::FAILED,
            "ConvertToAscppTileShape failed for node %s", nodeName.c_str());

        ge::AttrUtils::SetListInt(node->GetOpDesc(), "tile_shape", currentTileShape);
        ge::AttrUtils::SetListInt(node->GetOpDesc(), "tile_shape_input_nd", nodeInferredNdInputShape);
        ge::AttrUtils::SetListInt(node->GetOpDesc(), "tile_shape_output_nd", nodeInferredNdOutputShape);
    }

    // ========== Step 3: 最终验证 ==========
    
    ASCPP_CHK_TRUE_RETURN_FAIL(FinalValidation(graph) == npucl::FAILED,
        "Final validation failed");

    DUMP_GRAPH(graph, "after_ascendcpp_tile_shape_inference_pass");

    return npucl::SUCCESS;
}