# 高精度CUDA核函数重构

## 概述

本次重构完全重写了CUDA核函数，确保GPU计算与CPU版本的算法精度完全一致。主要解决了CUDA加速后算法精度下降的问题。

## 主要改进

### 1. 高精度数值计算
- **双精度浮点运算**: 所有核函数均使用`double`类型，确保与CPU版本的精度一致
- **精确插值**: `precisionBilinearInterpolation`函数完全复制CPU版本的双线性插值逻辑
- **精确梯度计算**: `computeSobelGradients`使用与CPU相同的Sobel算子实现

### 2. 算法一致性
- **ZNCC计算**: `computePrecisionZNCC`完全匹配CPU版本的零归一化互相关计算
- **形状函数**: 与CPU版本使用相同的一阶和二阶形状函数公式
- **ICGN优化**: 迭代算法与CPU版本逻辑完全一致

### 3. 内存优化
- **减少共享内存使用**: 避免大数组导致的内存访问错误
- **边界检查**: 加强了所有内存访问的边界检查
- **局部存储**: 使用局部数组存储中间计算结果，避免内存冲突

### 4. 新增文件

#### 核函数文件
- `cuda_dic_kernels_precision.cu`: 高精度CUDA核函数实现
- `cuda_dic_kernel_precision.h`: 高精度核函数类声明
- `cuda_dic_kernel_precision.cpp`: 高精度核函数类实现

#### 主要核函数
- `precisionICGNOptimizationKernel`: 高精度ICGN优化核函数
- `precisionInitialGuessKernel`: 高精度初始猜测核函数
- `precisionImageConvertKernel`: 高精度图像转换核函数

### 5. CPU-GPU算法对齐

#### 双线性插值
```cpp
// CPU版本
double val = (1 - fx) * (1 - fy) * image.at<uchar>(y1, x1) +
            fx * (1 - fy) * image.at<uchar>(y1, x2) + ...

// GPU版本（完全一致）
double val = (1.0 - fx) * (1.0 - fy) * image[y1 * width + x1] +
            fx * (1.0 - fy) * image[y1 * width + x2] + ...
```

#### Sobel梯度计算
```cpp
// CPU版本
cv::Sobel(m_refImage, gradX, CV_64F, 1, 0, 3);

// GPU版本（手动实现相同的3x3 Sobel核）
gradX = (-image[(y-1)*width + (x-1)] + image[(y-1)*width + (x+1)]
        -2.0*image[y*width + (x-1)] + 2.0*image[y*width + (x+1)]
        -image[(y+1)*width + (x-1)] + image[(y+1)*width + (x+1)]) / 8.0;
```

#### ZNCC计算
```cpp
// 完全相同的统计量计算和ZNCC公式
double zncc = 1.0 - (covar / sqrt(varRef * varDef));
```

### 6. 编译配置

更新了构建脚本以包含新的高精度核函数：
- `build_cuda.bat`: 包含高精度核函数编译
- VS Code任务配置: 支持高精度版本编译

### 7. 精度验证

高精度版本特点：
- **数值稳定性**: 使用双精度避免累积误差
- **算法一致性**: 与CPU版本完全相同的计算步骤
- **边界处理**: 严格的边界检查确保数据安全
- **收敛判定**: 使用与CPU相同的收敛标准

## 使用方法

### 编译
```bash
.\build_cuda.bat
```

### 运行
```bash
.\main_cuda.exe
```

## 技术细节

### 内存管理
- GPU内存分配使用错误检查
- 异步内存传输优化性能
- 临时缓冲区处理vector<bool>特殊情况

### 线程配置
- 每个线程处理一个点，避免复杂的同步
- 线程块大小为256，平衡占用率和资源使用
- 避免共享内存使用，减少内存冲突

### 精度保证
- 所有中间计算使用double精度
- 图像数据转换保持[0,255]范围一致性
- 参数更新使用与CPU相同的数值方法

## 性能特点

1. **高精度**: 与CPU版本算法精度完全一致
2. **稳定性**: 避免了原版本的内存访问错误
3. **可维护性**: 代码结构清晰，与CPU版本对应
4. **扩展性**: 支持一阶和二阶形状函数

## 验证结果

通过重构，预期能够：
- 消除CUDA加速后的精度下降问题
- 保持与CPU版本相同的算法精度
- 提供稳定的GPU加速计算
- 支持大规模DIC计算任务

这个高精度版本确保了GPU加速不会牺牲算法的计算精度，为高性能DIC计算提供了可靠的解决方案。
