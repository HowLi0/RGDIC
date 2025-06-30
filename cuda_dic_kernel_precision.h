#ifndef CUDA_DIC_KERNEL_PRECISION_H
#define CUDA_DIC_KERNEL_PRECISION_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolver_common.h>
#include <cusolverDn.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include "icgn_optimizer.h"

// Error checking macros
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t stat = call; \
        if (stat != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d - %d\n", __FILE__, __LINE__, stat); \
            exit(1); \
        } \
    } while(0)

#define CHECK_CUSOLVER(call) \
    do { \
        cusolverStatus_t stat = call; \
        if (stat != CUSOLVER_STATUS_SUCCESS) { \
            fprintf(stderr, "cuSOLVER error at %s:%d - %d\n", __FILE__, __LINE__, stat); \
            exit(1); \
        } \
    } while(0)

// Constants
const int MAX_THREADS_PER_BLOCK = 1024;
const int WARP_SIZE = 32;

// Interpolation method enumeration
enum InterpolationMethod {
    BILINEAR_INTERPOLATION = 0,     // Standard bilinear interpolation
    BICUBIC_INTERPOLATION = 1,      // Bicubic interpolation with 4x4 kernel
    INVERSE_DISTANCE_WEIGHTING = 2  // Inverse distance weighting (original method)
};

// Point structure for GPU
struct Point2D {
    int x, y;
    __host__ __device__ Point2D() : x(0), y(0) {}
    __host__ __device__ Point2D(int x_, int y_) : x(x_), y(y_) {}
};

struct Point2Df {
    float x, y;
    __host__ __device__ Point2Df() : x(0.0f), y(0.0f) {}
    __host__ __device__ Point2Df(float x_, float y_) : x(x_), y(y_) {}
};

// DIC result structure for GPU - 使用双精度
struct DICResult {
    double u, v;           // Displacement components (double precision)
    double zncc;           // Zero-normalized cross correlation (double precision)
    bool valid;           // Whether the computation was successful
    int iterations;       // Number of iterations used
    
    __host__ __device__ DICResult() : u(0.0), v(0.0), zncc(1.0), valid(false), iterations(0) {}
};

// High-precision CUDA DIC kernel class
class CudaDICKernelPrecision {
public:
    CudaDICKernelPrecision(int maxPoints, ShapeFunctionOrder order);
    ~CudaDICKernelPrecision();
    
    // Initialize with image dimensions
    bool initialize(int imageWidth, int imageHeight, int subsetRadius);
    
    // Set reference and deformed images
    bool setImages(const cv::Mat& refImage, const cv::Mat& defImage);
    
    // Compute DIC for a batch of points
    bool computeDIC(const std::vector<cv::Point>& points,
                   const std::vector<cv::Vec2f>& initialGuess,
                   std::vector<DICResult>& results,
                   double convergenceThreshold = 0.00001,
                   int maxIterations = 30);
    
    // Compute initial guess for points
    bool computeInitialGuess(const std::vector<cv::Point>& points,
                           std::vector<cv::Vec2f>& initialGuess,
                           int searchRadius = 15);
    
    // CUDA-accelerated interpolation for displacement field
    bool interpolateDisplacementField(const cv::Mat& sparseU, const cv::Mat& sparseV,
                                    const cv::Mat& sparseMask, const cv::Mat& roi,
                                    const std::vector<cv::Point>& sparsePoints,
                                    cv::Mat& interpU, cv::Mat& interpV, cv::Mat& interpMask,
                                    int step, InterpolationMethod method = BILINEAR_INTERPOLATION);
    
    // CUDA-accelerated strain field calculation
    bool calculateStrainField(const cv::Mat& u, const cv::Mat& v, const cv::Mat& validMask,
                            cv::Mat& strainExx, cv::Mat& strainEyy, cv::Mat& strainExy,
                            cv::Mat& strainMask, int windowSize);
    
    // Cleanup resources
    void cleanup();
    
    // Check if initialized
    bool isInitialized() const { return m_initialized; }
    
private:
    // GPU memory pointers
    double* d_refImage;           // Reference image (double precision)
    double* d_defImage;           // Deformed image (double precision)
    Point2D* d_points;            // Points to process
    double* d_initialParams;      // Initial warp parameters
    double* d_finalU;             // Final U displacement
    double* d_finalV;             // Final V displacement
    double* d_finalZNCC;          // Final ZNCC values
    bool* d_validMask;            // Valid point mask
    double* d_tempBuffer;         // Temporary buffer for processing
    
    // Host memory for results
    std::vector<double> h_finalU;
    std::vector<double> h_finalV;
    std::vector<double> h_finalZNCC;
    std::vector<bool> h_validMask;
    
    // Parameters
    int m_maxPoints;              // Maximum number of points to process
    int m_imageWidth;             // Image width
    int m_imageHeight;            // Image height
    int m_subsetRadius;           // Subset radius
    int m_subsetSize;             // Subset size (pixels)
    ShapeFunctionOrder m_order;   // Shape function order
    int m_numParams;              // Number of warp parameters
    bool m_initialized;           // Initialization flag
    bool m_imagesSet;             // Images set flag
    
    // CUDA stream for asynchronous operations
    cudaStream_t m_stream;
    
    // Internal methods
    bool allocateGPUMemory();
    void deallocateGPUMemory();
    bool initializeCUDA();
    bool copyPointsToGPU(const std::vector<cv::Point>& points);
    bool copyInitialGuessToGPU(const std::vector<cv::Vec2f>& initialGuess);
    bool copyResultsFromGPU(std::vector<DICResult>& results, int numPoints);
};

// High-precision CUDA kernel function declarations
extern "C" {

// High-precision ICGN optimization kernel
void launchPrecisionICGNOptimizationKernel(double* finalU, double* finalV, double* finalZNCC, bool* validMask,
                                          const double* refImage, const double* defImage,
                                          const Point2D* points, const double* initialParams,
                                          int numPoints, int imageWidth, int imageHeight,
                                          int subsetRadius, int numParams, int maxIterations,
                                          double convergenceThreshold, cudaStream_t stream);

// High-precision initial guess kernel
void launchPrecisionInitialGuessKernel(double* initialParams, double* initialZNCC, bool* validMask,
                                      const double* refImage, const double* defImage,
                                      const Point2D* points, int numPoints,
                                      int imageWidth, int imageHeight, int subsetRadius,
                                      int numParams, int searchRadius, cudaStream_t stream);

// High-precision image conversion kernel
void launchPrecisionImageConvertKernel(double* dst, const unsigned char* src, 
                                      int width, int height, cudaStream_t stream);

// High-precision interpolation kernel for displacement field
void launchPrecisionInterpolationKernel(double* interpU, double* interpV, unsigned char* interpMask,
                                       const double* sparseU, const double* sparseV, 
                                       const unsigned char* sparseMask, const unsigned char* roi,
                                       const Point2D* sparsePoints, int numSparsePoints,
                                       int width, int height, int step, int interpolationMethod, 
                                       cudaStream_t stream);

// High-precision strain calculation kernel using least squares
void launchPrecisionStrainCalculationKernel(double* strainExx, double* strainEyy, double* strainExy,
                                           unsigned char* strainMask, const double* u, const double* v,
                                           const unsigned char* validMask, int width, int height,
                                           int windowSize, cudaStream_t stream);

}

#endif // CUDA_DIC_KERNEL_PRECISION_H
