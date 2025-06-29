# CUDA加速的RGDIC (Reliability-Guided Digital Image Correlation)

本项目实现了CUDA加速的RGDIC算法，用于高性能的数字图像相关性分析。

## 功能特性

### 核心功能
- **CUDA加速**: 利用GPU并行计算大幅提升DIC计算性能
- **高度模块化**: 可复用的CUDA核函数和类设计
- **自动回退**: 当CUDA不可用时自动使用CPU实现
- **批处理优化**: 智能批处理以最大化GPU利用率
- **内存优化**: 高效的GPU内存管理和数据传输
- **性能监控**: 详细的性能统计和分析

### 算法特性
- **可靠性引导搜索**: 基于相关性的智能点传播
- **多阶形状函数**: 支持一阶和二阶形状函数
- **ICGN优化**: 反向组合高斯-牛顿迭代优化
- **异常值过滤**: 智能异常值检测和过滤
- **ROI支持**: 灵活的感兴趣区域定义

## 系统要求

### 硬件要求
- **GPU**: NVIDIA GPU，计算能力 >= 7.5 (推荐RTX 20系列及以上)
- **内存**: 至少4GB GPU内存 (取决于图像大小和点数量)
- **CPU**: 支持OpenMP的多核处理器

### 软件要求
- **CUDA Toolkit**: 版本 >= 11.0 (推荐12.0+)
- **OpenCV**: 版本 >= 4.0
- **Eigen3**: 线性代数库
- **编译器**: 
  - Windows: MinGW-w64 或 Visual Studio 2019+
  - Linux: GCC 9+ 或 Clang 10+

## 安装和编译

### 1. 依赖安装

#### Windows (推荐使用MSYS2)
```bash
# 安装MSYS2后
pacman -S mingw-w64-x86_64-gcc
pacman -S mingw-w64-x86_64-opencv
pacman -S mingw-w64-x86_64-eigen3
pacman -S mingw-w64-x86_64-cmake

# 下载并安装CUDA Toolkit
# https://developer.nvidia.com/cuda-toolkit
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install nvidia-cuda-toolkit
sudo apt-get install libopencv-dev
sudo apt-get install libeigen3-dev
sudo apt-get install build-essential cmake
```

### 2. 编译方法

#### 方法1: 使用CMake (推荐)
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

#### 方法2: 使用Makefile
```bash
# 测试CUDA安装
make -f Makefile_cuda_new test-cuda

# 编译
make -f Makefile_cuda_new all

# 针对特定GPU架构编译
make -f Makefile_cuda_new sm_80  # RTX 30系列
make -f Makefile_cuda_new sm_86  # RTX 30系列高端
```

### 3. 验证安装
```bash
# 运行程序
./CUDA_RGDIC

# 或者使用Makefile编译的版本
./main_cuda.exe
```

## 使用方法

### 基本用法

```cpp
#include "cuda_rgdic.h"

// 创建CUDA加速的RGDIC对象
auto dic = createRGDIC(
    true,           // 使用CUDA加速
    15,             // 子集半径
    0.00001,        // 收敛阈值
    30,             // 最大迭代次数
    0.8,            // 相关系数阈值
    1.0,            // 位移跳跃阈值
    SECOND_ORDER,   // 二阶形状函数
    1               // 邻域步长
);

// 加载图像
cv::Mat refImage = cv::imread("reference.png", cv::IMREAD_GRAYSCALE);
cv::Mat defImage = cv::imread("deformed.png", cv::IMREAD_GRAYSCALE);

// 定义ROI
cv::Mat roi = cv::Mat::ones(refImage.size(), CV_8U);

// 执行DIC分析
auto result = dic->compute(refImage, defImage, roi);

// 获取结果
cv::Mat u = result.u;  // x方向位移
cv::Mat v = result.v;  // y方向位移
cv::Mat validMask = result.validMask;  // 有效点掩码
```

### 高级配置

```cpp
// 自定义CUDA参数
CudaRGDIC cudaDic(
    15,             // 子集半径
    0.00001,        // 收敛阈值
    30,             // 最大迭代次数
    0.8,            // 相关系数阈值
    1.0,            // 位移跳跃阈值
    SECOND_ORDER,   // 形状函数阶数
    1,              // 邻域步长
    5000            // 最大批处理大小
);

// 执行计算
auto result = cudaDic.compute(refImage, defImage, roi);

// 获取性能统计
auto stats = cudaDic.getLastPerformanceStats();
std::cout << "GPU加速比: " << stats.speedup << "x" << std::endl;
std::cout << "GPU利用率: " << (stats.gpuComputeTime / stats.totalTime * 100) << "%" << std::endl;
```

## 性能优化

### 1. GPU内存优化
- **批处理大小**: 根据GPU内存调整批处理大小
- **图像尺寸**: 对于超大图像，考虑分块处理
- **数据类型**: 使用单精度浮点数以节省内存

### 2. 计算优化
- **子集大小**: 平衡精度和性能，推荐半径15-25
- **形状函数**: 一阶函数更快，二阶函数更精确
- **收敛阈值**: 适当放松阈值可提高速度

### 3. 系统优化
- **CUDA架构**: 针对目标GPU编译相应架构
- **并行度**: 确保有足够的线程并行度
- **内存带宽**: 优化内存访问模式

## 典型性能表现

### 基准测试 (500x500图像, RTX 3080)
- **CPU (单线程)**: ~30秒
- **CPU (OpenMP 8线程)**: ~8秒
- **CUDA加速**: ~1.2秒
- **加速比**: 约25倍

### 内存使用
- **图像**: 2 × W × H × 4 bytes (ref + def images)
- **子集**: N × subset_size² × 4 bytes
- **参数**: N × 12 × 4 bytes (二阶)
- **临时数据**: 约3-5倍基础内存

## 故障排除

### 常见问题

1. **CUDA初始化失败**
   ```
   错误: CUDA initialization failed
   解决: 检查CUDA驱动和Toolkit安装
   ```

2. **内存不足**
   ```
   错误: Failed to allocate GPU memory
   解决: 减少批处理大小或图像尺寸
   ```

3. **编译错误**
   ```
   错误: nvcc command not found
   解决: 确保CUDA Toolkit在PATH中
   ```

### 调试技巧

1. **启用详细输出**
   ```bash
   export CUDA_LAUNCH_BLOCKING=1
   ./CUDA_RGDIC
   ```

2. **检查GPU状态**
   ```bash
   nvidia-smi
   nvcc --version
   ```

3. **性能分析**
   ```bash
   nvprof ./CUDA_RGDIC
   ```

## API参考

### CudaRGDIC类

```cpp
class CudaRGDIC : public RGDIC {
public:
    // 构造函数
    CudaRGDIC(int subsetRadius = 15, 
              double convergenceThreshold = 0.00001,
              int maxIterations = 30,
              double ccThreshold = 0.8,
              double deltaDispThreshold = 1.0,
              ShapeFunctionOrder order = SECOND_ORDER,
              int neighborStep = 5,
              int maxBatchSize = 5000);
    
    // 主计算函数
    virtual DisplacementResult compute(const cv::Mat& refImage, 
                                     const cv::Mat& defImage,
                                     const cv::Mat& roi) override;
    
    // 获取性能统计
    PerformanceStats getLastPerformanceStats() const;
};
```

### 工厂函数

```cpp
std::unique_ptr<RGDIC> createRGDIC(
    bool useCuda = true,                    // 是否使用CUDA
    int subsetRadius = 15,                  // 子集半径
    double convergenceThreshold = 0.00001,  // 收敛阈值
    int maxIterations = 30,                 // 最大迭代次数
    double ccThreshold = 0.8,               // 相关系数阈值
    double deltaDispThreshold = 1.0,        // 位移跳跃阈值
    ShapeFunctionOrder order = SECOND_ORDER, // 形状函数阶数
    int neighborStep = 5                    // 邻域步长
);
```

## 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 创建Pull Request

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 致谢

- OpenCV计算机视觉库
- NVIDIA CUDA并行计算平台
- Eigen线性代数库
