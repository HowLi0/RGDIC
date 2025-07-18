#ifndef CUDA_RGDIC_H
#define CUDA_RGDIC_H

#include "rgdic.h"
#include "cuda_dic_kernel_precision.h"
#include <memory>
#include <vector>
#include <queue>
#include <map>
#include <string>

class CudaRGDIC : public RGDIC {
public:
    // Constructor
    CudaRGDIC(int subsetRadius = 15, 
              double convergenceThreshold = 0.00001,
              int maxIterations = 30,
              double ccThreshold = 0.8,
              double deltaDispThreshold = 1.0,
              ShapeFunctionOrder order = SECOND_ORDER,
              int neighborStep = 5,
              int maxBatchSize = 5000,
              InterpolationMethod interpolationMethod = BILINEAR_INTERPOLATION);
    
    // Destructor
    virtual ~CudaRGDIC();
    
    // Override the main compute function to use CUDA acceleration
    virtual DisplacementResult compute(const cv::Mat& refImage, 
                              const cv::Mat& defImage,
                              const cv::Mat& roi) override;
    
    // Batch processing function for better GPU utilization
    DisplacementResult computeBatch(const cv::Mat& refImage,
                                   const cv::Mat& defImage,
                                   const std::vector<cv::Point>& points,
                                   const std::vector<cv::Vec2f>& initialGuess);
    
    // GPU memory management
    void initializeGPU();
    void cleanupGPU();
    
    // Performance monitoring
    struct PerformanceStats {
        double totalTime;
        double gpuComputeTime;
        double memoryTransferTime;
        double cpuProcessingTime;
        int pointsProcessed;
        int validPoints;
        double speedup;
    };
    
    // Strain field structure
    struct StrainField {
        cv::Mat exx;    // Normal strain in x direction
        cv::Mat eyy;    // Normal strain in y direction  
        cv::Mat exy;    // Shear strain
        cv::Mat validMask; // Valid strain points mask
    };
    
    PerformanceStats getLastPerformanceStats() const { return m_lastStats; }
    
    // Set interpolation method
    void setInterpolationMethod(InterpolationMethod method) { m_interpolationMethod = method; }
    InterpolationMethod getInterpolationMethod() const { return m_interpolationMethod; }
    
    // Get last computed strain field (for IO operations)
    StrainField getLastStrainField() const { return m_lastStrainField; }
    bool hasStrainField() const { return m_hasStrainField; }
    
    // Interpolation and strain calculation methods (CUDA-accelerated)
    DisplacementResult interpolateDisplacementField(const DisplacementResult& sparseResult, 
                                                   const cv::Mat& roi, 
                                                   InterpolationMethod method = BILINEAR_INTERPOLATION);
    StrainField calculateStrainField(const DisplacementResult& displacementResult);
    
    // CPU fallback versions (for comparison or when CUDA is not available)
    DisplacementResult interpolateDisplacementFieldCPU(const DisplacementResult& sparseResult, 
                                                      const cv::Mat& roi);
    StrainField calculateStrainFieldCPU(const DisplacementResult& displacementResult);
    
private:
    // High-precision CUDA kernel wrapper
    std::unique_ptr<CudaDICKernelPrecision> m_precisionKernel;
    
    // Configuration
    int m_maxBatchSize;
    bool m_gpuInitialized;
    InterpolationMethod m_interpolationMethod;
    
    // Performance tracking
    PerformanceStats m_lastStats;
    
    // Strain field storage
    StrainField m_lastStrainField;
    bool m_hasStrainField;
    
    // Helper functions
    std::vector<cv::Point> extractROIPoints(const cv::Mat& roi);
    std::vector<cv::Vec2f> generateInitialGuess(const std::vector<cv::Point>& points, 
                                               const DisplacementResult& seedResult,
                                               cv::Point seedPoint);
    
    // Batch processing helpers
    void processBatch(const std::vector<cv::Point>& points,
                     const std::vector<cv::Vec2f>& initialGuess,
                     std::vector<DICResult>& batchResults);
    
    // Result integration
    void integrateResults(const std::vector<cv::Point>& points,
                         const std::vector<DICResult>& batchResults,
                         DisplacementResult& result);
    
    // Quality control
    void filterOutliers(DisplacementResult& result, const cv::Mat& roi);
    
    // Memory optimization
    std::vector<std::vector<cv::Point>> createBatches(const std::vector<cv::Point>& points);
    
    // CPU fallback for comparison and validation
    DisplacementResult computeCPUFallback(const cv::Mat& refImage,
                                         const cv::Mat& defImage,
                                         const cv::Mat& roi);
    
    // Timing utilities
    double getCurrentTime() const;
    void startTiming();
    void endTiming(const std::string& operation);
    
    double m_startTime;
    std::map<std::string, double> m_timings;
};

// Utility class for CUDA device management
class CudaDeviceManager {
public:
    static CudaDeviceManager& getInstance();
    
    bool initializeCuda();
    void printDeviceInfo();
    bool isInitialized() const { return m_initialized; }
    int getDeviceCount() const { return m_deviceCount; }
    int getCurrentDevice() const { return m_currentDevice; }
    
    // Memory information
    size_t getAvailableMemory() const;
    size_t getTotalMemory() const;
    
    // Performance characteristics
    int getMultiProcessorCount() const;
    int getMaxThreadsPerBlock() const;
    int getWarpSize() const;
    
private:
    CudaDeviceManager() : m_initialized(false), m_deviceCount(0), m_currentDevice(0) {}
    ~CudaDeviceManager() = default;
    
    // Delete copy constructor and assignment operator
    CudaDeviceManager(const CudaDeviceManager&) = delete;
    CudaDeviceManager& operator=(const CudaDeviceManager&) = delete;
    
    bool m_initialized;
    int m_deviceCount;
    int m_currentDevice;
    cudaDeviceProp m_deviceProp;
};
#endif // CUDA_RGDIC_H
