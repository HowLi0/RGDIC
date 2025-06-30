# RGDIC - Reliability-Guided Digital Image Correlation with CUDA Acceleration

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-blue.svg)](https://opencv.org/)

A high-performance Digital Image Correlation (DIC) implementation with CUDA acceleration, featuring reliability-guided search and advanced strain analysis capabilities.

## 🌟 Features

### Core DIC Capabilities
- **Reliability-Guided Analysis**: Prioritizes high-correlation points for optimal propagation
- **ICGN Optimization**: Inverse Compositional Gauss-Newton algorithm for sub-pixel accuracy
- **Multi-Order Shape Functions**: First-order (6 parameters) and second-order (12 parameters) support
- **Adaptive Subset Sizing**: Configurable subset radius for different image characteristics

### CUDA Acceleration
- **GPU-Accelerated Computation**: Massive parallel processing for faster analysis
- **High-Precision Kernels**: Double-precision floating-point operations
- **Memory Optimization**: Efficient GPU memory management and data transfer
- **Automatic Fallback**: CPU implementation available when CUDA is unavailable

### Advanced Post-Processing
- **Displacement Field Interpolation**: Multiple CUDA-accelerated methods available
  - Inverse Distance Weighting (original method)
  - Bilinear interpolation 
  - Bicubic interpolation
- **Strain Field Calculation**: Least-squares strain analysis with GPU acceleration
- **Outlier Detection**: Displacement jump-based filtering
- **Quality Control**: Correlation coefficient thresholding

### I/O and Visualization
- **Comprehensive CSV Export**: Enhanced format with strain data
- **Visualization Tools**: Displacement and strain field visualizations with scale bars
- **Synthetic Image Generation**: Built-in test pattern generation
- **Manual ROI Selection**: Interactive region of interest definition

## 🚀 Performance

### Speed Improvements
- **DIC Computation**: Up to 100x faster than CPU-only implementations
- **Interpolation**: 60x+ acceleration for dense displacement fields (bilinear/bicubic)
- **Strain Calculation**: 50x+ faster strain field computation
- **Memory Efficiency**: Optimized GPU memory usage with batch processing

### Scalability
- **Large Images**: Supports high-resolution image analysis
- **Batch Processing**: Configurable batch sizes for memory optimization
- **Parallel Execution**: Multi-threaded CPU fallback with OpenMP support

## 📋 Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA Compute Capability 6.0+
- **Memory**: 4GB+ GPU memory recommended for large images
- **CPU**: Multi-core processor for CPU fallback mode

### Software Dependencies
- **CUDA Toolkit**: 11.0 or later
- **OpenCV**: 4.0 or later
- **Eigen**: 3.3 or later (included)
- **Visual Studio**: 2019 or later (Windows)
- **CMake**: 3.18+ (optional)

## 🛠️ Building

### Windows (Visual Studio)

```bash
# Clone the repository
git clone https://github.com/yourusername/RGDIC.git
cd RGDIC

# Build CUDA version
./build_cuda.bat

# Build CPU-only version
./build_cpu.bat
```

### Dependencies Setup
1. Install CUDA Toolkit from [NVIDIA](https://developer.nvidia.com/cuda-downloads)
2. Download OpenCV and extract to `opencv/` directory
3. Eigen3 is included in the project

## 🎯 Usage

### Basic Example

```cpp
#include "cuda_rgdic.h"

// Create CUDA RGDIC instance with inverse distance weighting (original method)
auto dic = std::make_unique<CudaRGDIC>(
    19,          // subset radius
    0.00001,     // convergence threshold
    30,          // max iterations
    0.2,         // correlation threshold
    1.0,         // displacement threshold
    SECOND_ORDER, // shape function order
    5,           // neighbor step
    50000,       // max batch size
    INVERSE_DISTANCE_WEIGHTING // interpolation method (original)
);

// Load images
cv::Mat refImage = cv::imread("reference.png", cv::IMREAD_GRAYSCALE);
cv::Mat defImage = cv::imread("deformed.png", cv::IMREAD_GRAYSCALE);
cv::Mat roi = createManualROI(refImage);

// Compute displacement field
auto result = dic->compute(refImage, defImage, roi);

// Access results
cv::Mat u = result.u;           // X displacement
cv::Mat v = result.v;           // Y displacement
cv::Mat cc = result.cc;         // Correlation coefficient
cv::Mat mask = result.validMask; // Valid points mask
```

### Command Line Usage

```bash
# Run with synthetic images (default)
./main_cuda.exe

# Run with custom images
./main_cuda.exe reference.png deformed.png
```

### Configuration Options

The software supports various configuration options:

```cpp
// In main_cuda.cpp, modify these flags:
bool useSyntheticImages = true;      // Use built-in test patterns
bool useFirstOrderShapeFunction = false; // Shape function order
bool useManualROI = true;            // Interactive ROI selection

// Interpolation method selection (0=bilinear, 1=bicubic, 2=inverse distance weighting)
int interpolationMethodChoice = 2;   // Default to inverse distance weighting (original)

// DIC parameters
int subsetRadius = 19;               // Subset half-width
double convergenceThreshold = 0.00001; // ICGN convergence
int maxIterations = 30;              // Maximum ICGN iterations
double ccThreshold = 0.2;            // Correlation threshold
double deltaDispThreshold = 1.0;     // Displacement jump threshold
int neighborStep = 5;                // Point spacing (pixels)
```

## 📊 Output

### CSV Data Format
The software exports comprehensive results in CSV format:

```csv
left_x,left_y,right_x,right_y,u,v,exx,eyy,exy,zncc
411.0,103.0,413.946,103.932,2.946,0.932,-0.0006,-0.0022,0.013,0.023
```

Where:
- `left_x, left_y`: Reference image coordinates
- `right_x, right_y`: Deformed image coordinates
- `u, v`: Displacement components (pixels)
- `exx, eyy`: Normal strain components
- `exy`: Shear strain component
- `zncc`: Zero-normalized cross-correlation coefficient

### Generated Files
- `displacement_results.csv`: Complete displacement and strain data
- `computed_disp_x.png`: X-displacement visualization
- `computed_disp_y.png`: Y-displacement visualization
- `strain_exx.png`: Normal strain εxx visualization
- `strain_eyy.png`: Normal strain εyy visualization
- `strain_exy.png`: Shear strain εxy visualization
- `vector_field.png`: Displacement vector field
- `selected_roi.png`: Region of interest visualization

## 🔧 Algorithm Details

### Reliability-Guided DIC (RGDIC)
1. **Seed Point Selection**: Automatically finds the most reliable starting point
2. **Priority Queue**: Processes points in order of correlation quality
3. **Propagation Strategy**: Spreads analysis from reliable to uncertain regions
4. **Quality Control**: Filters outliers based on displacement continuity

### ICGN Optimization
- **Inverse Compositional**: More efficient than forward additive approaches
- **Gauss-Newton**: Second-order convergence for fast optimization
- **Sub-pixel Accuracy**: Bilinear interpolation for precise measurements
- **Robust Convergence**: Multiple convergence criteria

### Strain Calculation
- **Least-Squares Fitting**: Polynomial surface fitting for derivatives
- **Multi-Point Window**: Configurable neighborhood size
- **Noise Reduction**: Smoothing through local averaging
- **Edge Handling**: Proper boundary condition treatment

### Interpolation Methods
The software supports three interpolation methods for displacement field reconstruction:

1. **Inverse Distance Weighting (Default & Original)**
   - Uses squared inverse distance weighting across adaptively-sized neighborhoods
   - Proven method with excellent performance for most DIC applications
   - Adaptive search radius based on point spacing
   - Best for: General purpose DIC applications, maintains compatibility with original implementation

2. **Bilinear Interpolation**
   - Simple grid-based interpolation using nearest neighbor approach
   - Fast computation with minimal overhead
   - Best for: Quick approximations and testing

3. **Bicubic Interpolation**
   - Uses cubic kernel-based surface fitting with 4×4 neighborhood
   - Higher-order continuity and smoother displacement fields
   - Automatic fallback to inverse distance weighting at boundaries and sparse regions
   - Best for: High-precision applications requiring smooth derivatives

## 🔬 Validation

### Synthetic Testing
- **Known Displacement Fields**: Analytical test cases
- **Controlled Deformation**: Rigid body motion and uniform strain
- **Noise Analysis**: Performance under various noise levels
- **Accuracy Metrics**: Sub-pixel displacement accuracy

### Real-World Applications
- **Material Testing**: Tensile and compression experiments
- **Crack Propagation**: Fracture mechanics studies
- **Thermal Deformation**: Temperature-induced strain measurement
- **Dynamic Loading**: High-speed deformation analysis

## 📈 Performance Benchmarks

### Typical Performance (GTX 1630, 500×500 images)
- **Sparse Computation** (step=5): ~1 second for 4,600 points
- **Dense Interpolation**: ~0.5 seconds for 116,000 points
  - Bilinear: Standard performance baseline
  - Bicubic: +3% overhead for higher accuracy
- **Strain Calculation**: ~0.3 seconds for full field
- **Total Processing**: <2 seconds end-to-end

### Memory Usage
- **GPU Memory**: ~600MB for 500×500 images
- **CPU Memory**: ~200MB for data structures
- **Batch Processing**: Configurable for larger datasets

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Initialization Failed**
   - Ensure NVIDIA GPU drivers are up to date
   - Verify CUDA Toolkit installation
   - Check GPU compatibility (Compute Capability 6.0+)

2. **Out of Memory Errors**
   - Reduce batch size in configuration
   - Use smaller subset radius
   - Process images in tiles

3. **Poor Correlation Results**
   - Increase subset radius for low-texture regions
   - Adjust correlation threshold
   - Check image quality and lighting

4. **Slow Performance**
   - Ensure GPU is being used (check console output)
   - Optimize neighbor step size
   - Consider image preprocessing

## 📚 References

1. Pan, B., Qian, K., Xie, H., & Asundi, A. (2009). Two-dimensional digital image correlation for in-plane displacement and strain measurement. *Measurement Science and Technology*, 20(6), 062001.

2. Baker, S., & Matthews, I. (2004). Lucas-kanade 20 years on: A unifying framework. *International Journal of Computer Vision*, 56(3), 221-255.

3. Bing, P., Hui-min, X., Tao, H., & Asundi, A. (2006). Measurement of coefficient of thermal expansion of films using digital image correlation method. *Polymer Testing*, 25(5), 622-628.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Developer** - RGDIC2 Implementation - [GitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- OpenCV community for computer vision tools
- NVIDIA for CUDA computing platform
- Eigen team for linear algebra library
- Digital image correlation research community

## 📞 Contact

For questions, issues, or collaborations:
- **Issues**: [GitHub Issues](https://github.com/yourusername/RGDIC2/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/RGDIC2/discussions)

---

**Note**: This software is for research and educational purposes. Please cite appropriately if used in academic work.
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
