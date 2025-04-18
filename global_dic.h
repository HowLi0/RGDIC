#ifndef PYRAMID_GLOBAL_DIC_FFT_H
#define PYRAMID_GLOBAL_DIC_FFT_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <memory>
#include <unordered_map>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <fftw3.h>
#include <omp.h>

class PyramidGlobalDIC {
public:
    // 形函数阶数枚举
    enum ShapeFunctionOrder {
        FIRST_ORDER = 1,  // 6 parameters: u, v, du/dx, du/dy, dv/dx, dv/dy
        SECOND_ORDER = 2  // 12 parameters: first order + second derivatives
    };
    
    // 正则化类型枚举
    enum RegularizationType {
        TIKHONOV = 0,     // 简单Tikhonov正则化（默认）
        DIFFUSION = 1,    // 基于扩散的正则化 (更好地保留边缘)
        TOTAL_VARIATION = 2 // 全变分正则化 (保留不连续性)
    };
    
    // 初始猜测方法
    enum InitialGuessMethod {
        ZERO = 0,         // 零初始猜测
        FFTCC = 1,        // 使用FFT互相关 (默认，更快)
        NCC_SEARCH = 2    // 穷举NCC搜索 (更准确但更慢)
    };
    
    // 金字塔层结构
    struct Layer {
        cv::Mat img;         // 此金字塔级别的图像
        cv::Mat mask;        // 此金字塔级别的掩码
        int octave;          // 当前阶段
        double scale;        // 相对于原始图像的比例因子
        double sigma;        // 此层的高斯模糊sigma值
    };
    
    // 结果结构体，保存位移和应变场
    struct Result {
        cv::Mat u;           // x方向位移场
        cv::Mat v;           // y方向位移场
        cv::Mat exx;         // x方向正应变
        cv::Mat eyy;         // y方向正应变
        cv::Mat exy;         // 剪切应变
        cv::Mat cc;          // 相关系数
        cv::Mat validMask;   // 有效点掩码
        
        // 置信度度量
        cv::Mat confidence;  // 置信度指标 (0-1)
        double meanResidual; // 收敛后的平均残差
        int iterations;      // 执行的迭代次数
        
        // 构造函数，初始化所有矩阵
        Result(const cv::Size& size);
    };
    
    // FFTW包装器，用于快速傅里叶变换
    class FFTW {
    public:
        int subset_width, subset_height;
        int subset_size;
        
        float* ref_subset;
        float* tar_subset;
        fftwf_complex* ref_freq;
        fftwf_complex* tar_freq;
        fftwf_complex* zncc_freq;
        float* zncc;
        
        fftwf_plan ref_plan;
        fftwf_plan tar_plan;
        fftwf_plan zncc_plan;
        
        static std::unique_ptr<FFTW> allocate(int radius_x, int radius_y) {
            auto instance = std::make_unique<FFTW>();
            
            instance->subset_width = radius_x * 2;
            instance->subset_height = radius_y * 2;
            instance->subset_size = instance->subset_width * instance->subset_height;
            
            // 分配内存
            instance->ref_subset = (float*)fftwf_malloc(sizeof(float) * instance->subset_size);
            instance->tar_subset = (float*)fftwf_malloc(sizeof(float) * instance->subset_size);
            instance->zncc = (float*)fftwf_malloc(sizeof(float) * instance->subset_size);
            
            // 分配频域内存 (复数)
            instance->ref_freq = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (instance->subset_height * (instance->subset_width / 2 + 1)));
            instance->tar_freq = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (instance->subset_height * (instance->subset_width / 2 + 1)));
            instance->zncc_freq = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (instance->subset_height * (instance->subset_width / 2 + 1)));
            
            // 创建FFTW计划
            instance->ref_plan = fftwf_plan_dft_r2c_2d(instance->subset_height, instance->subset_width, 
                                                     instance->ref_subset, instance->ref_freq, FFTW_MEASURE);
            instance->tar_plan = fftwf_plan_dft_r2c_2d(instance->subset_height, instance->subset_width, 
                                                     instance->tar_subset, instance->tar_freq, FFTW_MEASURE);
            instance->zncc_plan = fftwf_plan_dft_c2r_2d(instance->subset_height, instance->subset_width, 
                                                       instance->zncc_freq, instance->zncc, FFTW_MEASURE);
            
            return instance;
        }
        
        static void release(std::unique_ptr<FFTW>& instance) {
            // 销毁FFTW计划
            fftwf_destroy_plan(instance->ref_plan);
            fftwf_destroy_plan(instance->tar_plan);
            fftwf_destroy_plan(instance->zncc_plan);
            
            // 释放内存
            fftwf_free(instance->ref_subset);
            fftwf_free(instance->tar_subset);
            fftwf_free(instance->zncc);
            fftwf_free(instance->ref_freq);
            fftwf_free(instance->tar_freq);
            fftwf_free(instance->zncc_freq);
        }
    };
    
    // 参数结构体，集中所有参数
    struct Parameters {
        int nodeSpacing;              // 节点间距（像素）
        int subsetRadius;             // 每个节点周围子集的半径
        double regularizationWeight;  // 正则化项的权重
        RegularizationType regType;   // 使用的正则化类型
        double convergenceThreshold;  // 收敛准则
        int maxIterations;            // 最大迭代次数
        ShapeFunctionOrder order;     // 形函数阶数
        bool useMultiScaleApproach;   // 使用多尺度方法
        int numScaleLevels;           // 尺度级别数
        double scaleFactor;           // 级别间的比例因子
        bool useParallel;             // 使用并行化
        int numThreads;               // OpenMP线程数
        cv::Size minImageSize;        // 允许的最小图像尺寸
        
        // 优化选项
        bool useEigenSolver;          // 使用Eigen高效求解器
        bool useSparseMatrix;         // 使用稀疏矩阵
        bool useFastInterpolation;    // 使用快速插值
        bool useSSE;                  // 使用SSE指令集
        bool useCaching;              // 使用缓存
        InitialGuessMethod initialGuessMethod; // 初始猜测方法
        bool useFFTCC;                // 使用FFT加速互相关
        int fftCCSearchRadius;        // FFTCC搜索半径
        
        // 默认值构造函数
        Parameters();
    };
    
    // 构造函数
    PyramidGlobalDIC(const Parameters& params = Parameters());
    ~PyramidGlobalDIC();
    
    // 主函数，执行全局DIC分析
    Result compute(const cv::Mat& refImage, 
                   const cv::Mat& defImage,
                   const cv::Mat& roi);
    
    // 通过位移场计算应变场
    void calculateStrains(Result& result, bool useWindowedLeastSquares = true, int windowSize = 5);
    
    // 显示结果的实用函数
    void displayResults(const cv::Mat& refImage, const Result& result, 
                        bool showDisplacement = true, bool showStrain = true,
                        bool useEnhancedVisualization = true);
    
    // 获取/设置参数
    Parameters& getParameters() { return m_params; }
    void setParameters(const Parameters& params) { m_params = params; }

private:
    // 参数
    Parameters m_params;
    
    // 基于ROI创建节点网格
    void createNodeGrid(const cv::Mat& roi, 
                        std::vector<cv::Point>& nodePoints);
    
    // 计算全局DIC系统矩阵和向量
    void buildGlobalSystem(const cv::Mat& refImage, 
                          const cv::Mat& defImage,
                          const std::vector<cv::Point>& nodePoints,
                          cv::Mat& systemMatrix,
                          cv::Mat& systemVector,
                          const cv::Mat& nodeDisplacements);
    
    // 向系统添加正则化项
    void addRegularization(const std::vector<cv::Point>& nodePoints,
                          cv::Mat& systemMatrix);
    
    // 高效求解系统
    bool solveSystem(const cv::Mat& systemMatrix, const cv::Mat& systemVector, 
                    cv::Mat& solution, double& residualNorm);
    
    // 使用FEM形函数在任意点插值位移
    void interpolateDisplacement(const cv::Point& point,
                                const std::vector<cv::Point>& nodePoints,
                                const cv::Mat& nodeDisplacements,
                                double& u, double& v);
    
    // 计算参考图像和变形图像之间的相关系数
    double calculateCorrelation(const cv::Mat& refImage, 
                               const cv::Mat& defImage,
                               const std::vector<cv::Point>& nodePoints,
                               const cv::Mat& nodeDisplacements);
    
    // 在非整数坐标插值强度
    double interpolate(const cv::Mat& image, const cv::Point2f& pt) const;
    
    // 从节点位移生成完整位移场
    void generateDisplacementField(const std::vector<cv::Point>& nodePoints,
                                  const cv::Mat& nodeDisplacements,
                                  Result& result,
                                  const cv::Mat& roi);
                                  
    // 创建图像金字塔用于多尺度方法 (OpenCorr启发)
    void createImagePyramid(const cv::Mat& image, 
                           const cv::Mat& mask,
                           std::vector<Layer>& pyramid);
    
    // 使用不同尺度计算位移场
    Result computeMultiScale(const cv::Mat& refImage, 
                           const cv::Mat& defImage,
                           const cv::Mat& roi);
    
    // 计算元素形函数和导数
    void computeShapeFunctions(const cv::Point& point, 
                              const std::vector<cv::Point>& nodePoints,
                              std::vector<double>& N,
                              std::vector<double>& dNdx,
                              std::vector<double>& dNdy);
                              
    // 计算置信度指标
    void calculateConfidence(Result& result, 
                           const cv::Mat& refImage, 
                           const cv::Mat& defImage);
    
    // 计算每个节点的平均残差
    double calculateResidual(const cv::Mat& refImage, 
                           const cv::Mat& defImage,
                           const std::vector<cv::Point>& nodePoints,
                           const cv::Mat& nodeDisplacements);
                           
    // 使用FFTCC计算初始位移场
    void calculateInitialGuessFFTCC(const cv::Mat& refImage,
                                  const cv::Mat& defImage,
                                  const cv::Mat& roi,
                                  cv::Mat& initialU,
                                  cv::Mat& initialV);
                                  
    // 使用FFT-CC计算单点的初始位移
    bool computeFFTCCDisplacement(const cv::Mat& refImage,
                                const cv::Mat& defImage,
                                const cv::Point& point,
                                double& u, double& v,
                                double& zncc);
                                
    // 缓存结构
    mutable std::unordered_map<int64_t, double> m_interpolationCache;
    mutable std::unordered_map<int64_t, std::pair<double, double>> m_gradientCache;
    mutable std::unordered_map<size_t, std::pair<double, double>> m_displacementCache;

    // 缓存的梯度图像
    mutable cv::Mat m_gradX, m_gradY;
    mutable bool m_gradientsComputed;
    
    // 形函数值的内部缓存
    using ShapeFunctionCache = std::unordered_map<size_t, std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>>;
    mutable ShapeFunctionCache m_shapeFunctionCache;
    
    // 预计算梯度以用于缓存
    void precomputeGradients(const cv::Mat& image);
    
    // 优化的插值函数
    inline double fastInterpolate(const cv::Mat& image, const cv::Point2f& pt) const;
    
    // 获取梯度值，带缓存优化
    inline void getGradientValue(const cv::Point& point, double& gx, double& gy) const;
    
    // 使用稀疏矩阵的全局系统构建
    void buildGlobalSystemSparse(const cv::Mat& refImage, const cv::Mat& defImage,
                               const std::vector<cv::Point>& nodePoints,
                               Eigen::SparseMatrix<double>& systemMatrix,
                               Eigen::VectorXd& systemVector,
                               const cv::Mat& nodeDisplacements);
                               
    // 向稀疏矩阵添加正则化
    void addRegularizationSparse(const std::vector<cv::Point>& nodePoints,
                               Eigen::SparseMatrix<double>& systemMatrix);
                               
    // 求解稀疏系统
    bool solveSystemSparse(const Eigen::SparseMatrix<double>& systemMatrix,
                         const Eigen::VectorXd& systemVector,
                         Eigen::VectorXd& solution,
                         double& residualNorm);
                         
    // 查找K个最近的节点 (用于形函数计算优化)
    void findKNearestNodes(const cv::Point& point, 
                         const std::vector<cv::Point>& nodePoints,
                         int K,
                         std::vector<size_t>& indices,
                         std::vector<double>& distances);
                         
    // FFTW实例池，用于并行处理
    std::vector<std::unique_ptr<FFTW>> m_fftwPool;
    
    // 获取线程ID对应的FFTW实例
    std::unique_ptr<FFTW>& getFFTWInstance(int threadId);
};

#endif // PYRAMID_GLOBAL_DIC_FFT_H