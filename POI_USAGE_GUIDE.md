# RGDIC POI 模块化设计使用指南

## 概述

本项目已成功重构为基于 POI (Point of Interest) 设计的模块化架构，完全兼容原有 RGDIC 接口的同时，提供了更强大的功能和更灵活的使用方式。

## 主要特性

### 1. POI (Point of Interest) 核心设计
- **完整坐标跟踪**: 自动维护参考图像和变形图像的像素坐标
- **数据封装**: 每个 POI 包含位移、应变、相关系数、质量指标等完整信息
- **邻域管理**: 支持 POI 间的邻域关系建立和查询

### 2. 模块化架构
- **DIC 处理器**: 独立的计算引擎，支持 CPU 并行处理
- **I/O 接口**: 支持 CSV、JSON 等多种格式的数据导入导出
- **质量控制**: 多层次的质量检查和统计分析
- **适配器模式**: 完全兼容现有 RGDIC 接口

### 3. 增强功能
- **统计分析**: 自动计算处理统计信息和质量指标
- **并行处理**: OpenMP 多线程支持
- **进度回调**: 实时处理进度反馈
- **结果验证**: 与传统方法的对比验证工具

## 快速开始

### 传统 RGDIC 接口（零修改迁移）

```cpp
#include "rgdic_poi_adapter.h"

// 原有代码无需修改，自动使用 POI 增强功能
auto dic = createRGDICWithPOI(false, 15, 0.00001, 30, 0.8, 1.0, SECOND_ORDER, 5);
auto result = dic->compute(refImage, defImage, roi);

// 新增：访问 POI 增强功能
auto poiAdapter = dynamic_cast<RGDICPOIAdapter*>(dic.get());
if (poiAdapter) {
    // 导出增强 CSV（包含完整 POI 数据）
    poiAdapter->exportEnhancedCSV("enhanced_results.csv");
    
    // 导出 JSON 格式
    poiAdapter->exportJSON("results.json");
    
    // 导出处理统计
    poiAdapter->exportStatistics("processing_stats.csv");
}
```

### 直接使用 POI 接口

```cpp
#include "poi.h"
#include "dic_processor.h"
#include "poi_io.h"

// 1. 创建 POI 管理器并从 ROI 生成 POI
auto poiManager = std::unique_ptr<POIManager>(new POIManager());
poiManager->generatePOIsFromROI(roi, 5);  // 5像素间距

// 2. 配置 DIC 处理器
DICProcessor::Config config;
config.subsetRadius = 15;
config.ccThreshold = 0.8;
config.enableParallelProcessing = true;

auto processor = DICProcessorFactory::createOptimal(config);

// 3. 设置进度回调
processor->setProgressCallback([](size_t current, size_t total, const std::string& msg) {
    std::cout << "Progress: " << current << "/" << total << " - " << msg << std::endl;
});

// 4. 执行 DIC 处理
auto stats = processor->processPOIs(refImage, defImage, *poiManager);

// 5. 导出结果
auto ioInterface = POIIOFactory::create(POIIOFactory::ENHANCED_IO);
ioInterface->exportPOIs(*poiManager, "poi_results.csv");

// 6. 访问处理结果
std::cout << "处理统计:" << std::endl;
std::cout << "  总点数: " << stats.totalPoints << std::endl;
std::cout << "  有效点数: " << stats.validPoints << std::endl;
std::cout << "  覆盖率: " << (stats.coverageRatio * 100.0) << "%" << std::endl;
```

### POI 数据访问

```cpp
// 访问单个 POI 数据
auto poi = poiManager->getPOI(0);
if (poi->isValid()) {
    std::cout << "参考坐标: (" << poi->getReferenceX() << ", " << poi->getReferenceY() << ")" << std::endl;
    std::cout << "变形坐标: (" << poi->getDeformedX() << ", " << poi->getDeformedY() << ")" << std::endl;
    std::cout << "位移: (" << poi->getDisplacementU() << ", " << poi->getDisplacementV() << ")" << std::endl;
    std::cout << "ZNCC: " << poi->getZNCC() << std::endl;
    
    if (poi->hasStrain()) {
        std::cout << "应变: εxx=" << poi->getStrainExx() 
                  << ", εyy=" << poi->getStrainEyy() 
                  << ", εxy=" << poi->getStrainExy() << std::endl;
    }
}

// 搜索邻近 POI
auto nearbyPOIs = poiManager->findPOIsInRadius(100.0, 150.0, 20.0);
std::cout << "半径20像素内找到 " << nearbyPOIs.size() << " 个POI" << std::endl;

// 统计分析
std::cout << "平均位移: " << poiManager->getMeanDisplacement() << std::endl;
std::cout << "位移标准差: " << poiManager->getStdDisplacement() << std::endl;
```

## 输出格式

### 增强 CSV 格式
```csv
left_x,left_y,right_x,right_y,u,v,exx,eyy,exy,zncc
106.000000,171.000000,108.304899,171.291096,2.304899,0.291096,0.001359,0.001569,-0.021258,0.823456
```

### JSON 格式
```json
{
  "metadata": {
    "timestamp": "2024-01-01 12:00:00",
    "software_version": "RGDIC POI v1.0",
    "image_size": {"width": 500, "height": 400}
  },
  "pois": [
    {
      "reference": [106.0, 171.0],
      "deformed": [108.304899, 171.291096],
      "displacement": [2.304899, 0.291096],
      "zncc": 0.823456,
      "strain": {
        "exx": 0.001359,
        "eyy": 0.001569,
        "exy": -0.021258
      },
      "valid": true,
      "converged": true
    }
  ]
}
```

## 编译说明

### 新增依赖
所有新模块都基于 C++11 标准，无新增外部依赖：

```bash
# 编译 POI 模块
g++ -std=c++11 -I. -IEigen3 -Iopencv/include -O2 -fopenmp -c poi.cpp
g++ -std=c++11 -I. -IEigen3 -Iopencv/include -O2 -fopenmp -c dic_processor.cpp
g++ -std=c++11 -I. -IEigen3 -Iopencv/include -O2 -fopenmp -c poi_io.cpp
g++ -std=c++11 -I. -IEigen3 -Iopencv/include -O2 -fopenmp -c rgdic_poi_adapter.cpp

# 链接（需要配置正确的 OpenCV 路径）
g++ -o your_program *.obj -lopencv_core -lopencv_imgproc -lopencv_imgcodecs
```

### 使用工厂自动选择
```cpp
// 自动选择最优实现
RGDICFactory::FactoryConfig config;
config.enablePOI = true;  // 启用 POI 增强功能
auto dic = RGDICFactory::create(config, 15, 0.00001, 30, 0.8, 1.0, SECOND_ORDER, 5);
```

## 迁移指南

### 从传统 RGDIC 迁移

1. **零修改方案**: 
   ```cpp
   // 原代码
   auto dic = createRGDIC(false, 15, 0.00001, 30, 0.8, 1.0, SECOND_ORDER, 5);
   
   // 新代码（功能增强，接口不变）
   auto dic = createRGDICWithPOI(false, 15, 0.00001, 30, 0.8, 1.0, SECOND_ORDER, 5);
   ```

2. **逐步迁移**:
   - 先使用适配器接口，获得增强功能
   - 再逐步采用直接 POI 接口，获得更大灵活性

### 验证迁移结果
```cpp
// 使用迁移辅助工具对比结果
auto legacyResult = legacyDIC->compute(refImage, defImage, roi);
auto poiResult = poiDIC->compute(refImage, defImage, roi);

auto validationStats = RGDICMigrationHelper::validatePOIResults(legacyResult, poiResult, 1e-6);
std::cout << "验证通过率: " << (validationStats.coverageRatio * 100.0) << "%" << std::endl;
```

## 性能特点

- **并行处理**: OpenMP 多线程支持，充分利用 CPU 多核
- **内存优化**: 智能指针管理，避免内存泄漏
- **批量处理**: 支持大规模 POI 集合的高效处理
- **向前兼容**: 为 CUDA POI 实现预留接口

## 扩展计划

- **CUDA POI 处理器**: GPU 加速的 POI 处理实现
- **HDF5 支持**: 大数据集的高效存储格式
- **实时处理**: 流式 POI 处理接口
- **可视化增强**: 基于 POI 的高级可视化功能

---

通过这次重构，RGDIC 项目获得了：
✅ 完全模块化的架构设计  
✅ OpenCorr POI 设计模式  
✅ 增强的像素坐标跟踪  
✅ 向后兼容性保证  
✅ 丰富的数据导出格式  
✅ 完整的质量控制体系  

欢迎使用新的 POI 模块化 RGDIC 系统！