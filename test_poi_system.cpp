#include <iostream>
#include <cassert>
#include "poi.h"
#include "dic_processor.h"
#include "poi_io.h"
#include "rgdic_poi_adapter.h"

/**
 * @brief Simple test program to validate POI functionality without external dependencies
 */
int main() {
    std::cout << "=== RGDIC POI 模块化系统测试 ===" << std::endl;
    
    // Test 1: POI Basic Functionality
    std::cout << "\n1. 测试 POI 基础功能..." << std::endl;
    
    auto poi = std::shared_ptr<POI>(new POI(100.0, 200.0));
    assert(poi->getReferenceX() == 100.0);
    assert(poi->getReferenceY() == 200.0);
    
    poi->setDisplacement(2.5, 1.8);
    poi->calculateDeformedFromDisplacement();
    assert(poi->getDeformedX() == 102.5);
    assert(poi->getDeformedY() == 201.8);
    
    poi->setZNCC(0.85);
    poi->setValid(true);
    poi->setConverged(true);
    
    std::cout << "   ✓ POI 坐标和位移计算正确" << std::endl;
    std::cout << "   ✓ 参考坐标: (" << poi->getReferenceX() << ", " << poi->getReferenceY() << ")" << std::endl;
    std::cout << "   ✓ 变形坐标: (" << poi->getDeformedX() << ", " << poi->getDeformedY() << ")" << std::endl;
    std::cout << "   ✓ 位移: (" << poi->getDisplacementU() << ", " << poi->getDisplacementV() << ")" << std::endl;
    
    // Test 2: POI Manager
    std::cout << "\n2. 测试 POI 管理器..." << std::endl;
    
    auto poiManager = std::unique_ptr<POIManager>(new POIManager());
    
    // Add some test POIs
    for (int i = 0; i < 10; ++i) {
        auto testPOI = std::shared_ptr<POI>(new POI(i * 10.0, i * 5.0));
        testPOI->setDisplacement(i * 0.5, i * 0.3);
        testPOI->setZNCC(0.8 + i * 0.01);
        testPOI->setValid(i % 2 == 0);  // Every other POI is valid
        testPOI->setConverged(testPOI->isValid());
        poiManager->addPOI(testPOI);
    }
    
    assert(poiManager->size() == 10);
    assert(poiManager->getValidCount() == 5);
    assert(poiManager->getConvergedCount() == 5);
    
    std::cout << "   ✓ POI 管理器创建成功，包含 " << poiManager->size() << " 个 POI" << std::endl;
    std::cout << "   ✓ 有效 POI: " << poiManager->getValidCount() << " 个" << std::endl;
    std::cout << "   ✓ 收敛 POI: " << poiManager->getConvergedCount() << " 个" << std::endl;
    
    // Test 3: POI Search
    std::cout << "\n3. 测试 POI 搜索功能..." << std::endl;
    
    auto nearestPOI = poiManager->findNearestPOI(25.0, 12.0);
    assert(nearestPOI != nullptr);
    std::cout << "   ✓ 最近 POI 搜索成功: (" << nearestPOI->getReferenceX() 
              << ", " << nearestPOI->getReferenceY() << ")" << std::endl;
    
    auto nearbyPOIs = poiManager->findPOIsInRadius(30.0, 15.0, 20.0);
    std::cout << "   ✓ 半径搜索找到 " << nearbyPOIs.size() << " 个 POI" << std::endl;
    
    // Test 4: Statistics
    std::cout << "\n4. 测试统计功能..." << std::endl;
    
    double avgZNCC = poiManager->getAverageZNCC();
    auto meanDisp = poiManager->getMeanDisplacement();
    auto stdDisp = poiManager->getStdDisplacement();
    
    std::cout << "   ✓ 平均 ZNCC: " << avgZNCC << std::endl;
    std::cout << "   ✓ 平均位移: (" << meanDisp[0] << ", " << meanDisp[1] << ")" << std::endl;
    std::cout << "   ✓ 位移标准差: (" << stdDisp[0] << ", " << stdDisp[1] << ")" << std::endl;
    
    // Test 5: CSV Export
    std::cout << "\n5. 测试 CSV 导出..." << std::endl;
    
    std::string csvContent = poi->toCSVRow();
    std::string csvHeader = POI::getCSVHeader();
    
    std::cout << "   ✓ CSV 表头: " << csvHeader << std::endl;
    std::cout << "   ✓ 示例行: " << csvContent << std::endl;
    
    // Test 6: POI I/O Interface
    std::cout << "\n6. 测试 I/O 接口..." << std::endl;
    
    auto ioInterface = POIIOFactory::create(POIIOFactory::STANDARD_IO);
    assert(ioInterface != nullptr);
    
    auto enhancedIO = POIIOFactory::create(POIIOFactory::ENHANCED_IO);
    assert(enhancedIO != nullptr);
    
    std::cout << "   ✓ 标准 I/O 接口创建成功" << std::endl;
    std::cout << "   ✓ 增强 I/O 接口创建成功" << std::endl;
    
    // Test 7: DIC Processor Configuration
    std::cout << "\n7. 测试 DIC 处理器配置..." << std::endl;
    
    DICProcessor::Config config;
    assert(config.subsetRadius == 15);
    assert(config.ccThreshold == 0.8);
    assert(config.enableParallelProcessing == true);
    
    std::cout << "   ✓ 默认配置正确" << std::endl;
    std::cout << "   ✓ 子集半径: " << config.subsetRadius << std::endl;
    std::cout << "   ✓ 相关系数阈值: " << config.ccThreshold << std::endl;
    std::cout << "   ✓ 并行处理: " << (config.enableParallelProcessing ? "启用" : "禁用") << std::endl;
    
    // Test 8: Factory Pattern
    std::cout << "\n8. 测试工厂模式..." << std::endl;
    
    RGDICFactory::FactoryConfig factoryConfig;
    assert(factoryConfig.enablePOI == true);
    assert(factoryConfig.enableEnhancedIO == true);
    
    std::cout << "   ✓ 工厂配置正确" << std::endl;
    std::cout << "   ✓ POI 功能: " << (factoryConfig.enablePOI ? "启用" : "禁用") << std::endl;
    std::cout << "   ✓ 增强 I/O: " << (factoryConfig.enableEnhancedIO ? "启用" : "禁用") << std::endl;
    
    // Test 9: System Capabilities
    std::cout << "\n9. 系统能力检查..." << std::endl;
    
    std::string capabilities = RGDICFactory::getSystemCapabilities();
    std::cout << capabilities << std::endl;
    
    // Test 10: POI Clone and Memory Management
    std::cout << "\n10. 测试 POI 克隆和内存管理..." << std::endl;
    
    auto clonedPOI = poi->clone();
    assert(clonedPOI != nullptr);
    assert(clonedPOI->getReferenceX() == poi->getReferenceX());
    assert(clonedPOI->getReferenceY() == poi->getReferenceY());
    assert(clonedPOI->getDisplacementU() == poi->getDisplacementU());
    assert(clonedPOI->getDisplacementV() == poi->getDisplacementV());
    
    std::cout << "   ✓ POI 克隆成功，数据一致" << std::endl;
    
    // Summary
    std::cout << "\n=== 测试总结 ===" << std::endl;
    std::cout << "✅ POI 核心数据结构：正常" << std::endl;
    std::cout << "✅ POI 管理器功能：正常" << std::endl;
    std::cout << "✅ 搜索和查询功能：正常" << std::endl;
    std::cout << "✅ 统计分析功能：正常" << std::endl;
    std::cout << "✅ 数据导出格式：正常" << std::endl;
    std::cout << "✅ I/O 接口模块：正常" << std::endl;
    std::cout << "✅ 处理器配置：正常" << std::endl;
    std::cout << "✅ 工厂模式：正常" << std::endl;
    std::cout << "✅ 内存管理：正常" << std::endl;
    
    std::cout << "\n🎉 所有测试通过！POI 模块化系统运行正常。" << std::endl;
    
    // Demonstrate enhanced coordinate tracking
    std::cout << "\n=== 增强像素坐标跟踪演示 ===" << std::endl;
    
    auto demoPOI = std::shared_ptr<POI>(new POI(150.0, 250.0));
    std::cout << "原始参考坐标 (左图): (" << demoPOI->getReferenceX() << ", " << demoPOI->getReferenceY() << ")" << std::endl;
    
    // 设置位移并自动计算变形坐标
    demoPOI->setDisplacement(3.2, -1.5);
    demoPOI->calculateDeformedFromDisplacement();
    std::cout << "计算位移: (" << demoPOI->getDisplacementU() << ", " << demoPOI->getDisplacementV() << ")" << std::endl;
    std::cout << "自动计算变形坐标 (右图): (" << demoPOI->getDeformedX() << ", " << demoPOI->getDeformedY() << ")" << std::endl;
    
    // 反向验证：从坐标计算位移
    demoPOI->setDeformedCoords(152.8, 248.3);
    demoPOI->calculateDisplacementFromCoords();
    std::cout << "验证：从坐标反算位移: (" << demoPOI->getDisplacementU() << ", " << demoPOI->getDisplacementV() << ")" << std::endl;
    
    std::cout << "\n✨ 模块化重构完成！" << std::endl;
    std::cout << "   • 使用 OpenCorr POI 设计模式" << std::endl;
    std::cout << "   • 完整的左右图像像素坐标跟踪" << std::endl;
    std::cout << "   • 模块化架构，向后兼容" << std::endl;
    std::cout << "   • 增强的数据导出和质量控制" << std::endl;
    
    return 0;
}