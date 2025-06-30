#include "cuda_dic_kernel_precision.h"
#include <iostream>
#include <algorithm>
#include <cmath>

CudaDICKernelPrecision::CudaDICKernelPrecision(int maxPoints, ShapeFunctionOrder order)
    : d_refImage(nullptr), d_defImage(nullptr), d_points(nullptr),
      d_initialParams(nullptr), d_finalU(nullptr), d_finalV(nullptr),
      d_finalZNCC(nullptr), d_validMask(nullptr), d_tempBuffer(nullptr),
      m_maxPoints(maxPoints), m_imageWidth(0), m_imageHeight(0),
      m_subsetRadius(0), m_subsetSize(0), m_order(order),
      m_numParams((order == FIRST_ORDER) ? 6 : 12), 
      m_initialized(false), m_imagesSet(false), m_stream(nullptr)
{
    initializeCUDA();
}

CudaDICKernelPrecision::~CudaDICKernelPrecision() {
    cleanup();
}

bool CudaDICKernelPrecision::initializeCUDA() {
    // Create CUDA stream for asynchronous operations
    cudaError_t err = cudaStreamCreate(&m_stream);
    if (err != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    std::cout << "High-precision CUDA DIC kernel initialized" << std::endl;
    return true;
}

bool CudaDICKernelPrecision::initialize(int imageWidth, int imageHeight, int subsetRadius) {
    if (m_initialized) {
        cleanup();
    }
    
    m_imageWidth = imageWidth;
    m_imageHeight = imageHeight;
    m_subsetRadius = subsetRadius;
    m_subsetSize = (2 * subsetRadius + 1) * (2 * subsetRadius + 1);
    
    // Allocate GPU memory
    if (!allocateGPUMemory()) {
        std::cerr << "Failed to allocate GPU memory for precision kernel" << std::endl;
        return false;
    }
    
    // Resize host buffers
    h_finalU.resize(m_maxPoints);
    h_finalV.resize(m_maxPoints);
    h_finalZNCC.resize(m_maxPoints);
    h_validMask.resize(m_maxPoints);
    
    m_initialized = true;
    std::cout << "High-precision CUDA DIC Kernel initialized successfully" << std::endl;
    std::cout << "  - Image size: " << m_imageWidth << "x" << m_imageHeight << std::endl;
    std::cout << "  - Subset radius: " << m_subsetRadius << " (size: " << m_subsetSize << ")" << std::endl;
    std::cout << "  - Max points: " << m_maxPoints << std::endl;
    std::cout << "  - Shape function: " << m_numParams << " parameters (" 
              << (m_order == FIRST_ORDER ? "first" : "second") << "-order)" << std::endl;
    
    return true;
}

bool CudaDICKernelPrecision::allocateGPUMemory() {
    try {
        // Calculate memory sizes
        size_t imageSize = m_imageWidth * m_imageHeight * sizeof(double);
        size_t pointsSize = m_maxPoints * sizeof(Point2D);
        size_t paramsSize = m_maxPoints * m_numParams * sizeof(double);
        size_t displacementSize = m_maxPoints * sizeof(double);
        size_t validMaskSize = m_maxPoints * sizeof(bool);
        size_t tempBufferSize = m_maxPoints * m_subsetSize * sizeof(double);
        
        // Allocate image memory
        cudaError_t err = cudaMalloc(&d_refImage, imageSize);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate reference image memory: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        err = cudaMalloc(&d_defImage, imageSize);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate deformed image memory: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        // Allocate point processing memory
        err = cudaMalloc(&d_points, pointsSize);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate points memory: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        err = cudaMalloc(&d_initialParams, paramsSize);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate initial parameters memory: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        err = cudaMalloc(&d_finalU, displacementSize);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate final U memory: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        err = cudaMalloc(&d_finalV, displacementSize);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate final V memory: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        err = cudaMalloc(&d_finalZNCC, displacementSize);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate final ZNCC memory: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        err = cudaMalloc(&d_validMask, validMaskSize);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate valid mask memory: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        err = cudaMalloc(&d_tempBuffer, tempBufferSize);
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate temporary buffer memory: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        // Calculate total memory usage
        size_t totalMemory = 2 * imageSize + pointsSize + paramsSize + 3 * displacementSize + validMaskSize + tempBufferSize;
        std::cout << "Allocated " << totalMemory / (1024 * 1024) << " MB of GPU memory" << std::endl;
        
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception during GPU memory allocation: " << e.what() << std::endl;
        return false;
    }
}

void CudaDICKernelPrecision::deallocateGPUMemory() {
    if (d_refImage) { cudaFree(d_refImage); d_refImage = nullptr; }
    if (d_defImage) { cudaFree(d_defImage); d_defImage = nullptr; }
    if (d_points) { cudaFree(d_points); d_points = nullptr; }
    if (d_initialParams) { cudaFree(d_initialParams); d_initialParams = nullptr; }
    if (d_finalU) { cudaFree(d_finalU); d_finalU = nullptr; }
    if (d_finalV) { cudaFree(d_finalV); d_finalV = nullptr; }
    if (d_finalZNCC) { cudaFree(d_finalZNCC); d_finalZNCC = nullptr; }
    if (d_validMask) { cudaFree(d_validMask); d_validMask = nullptr; }
    if (d_tempBuffer) { cudaFree(d_tempBuffer); d_tempBuffer = nullptr; }
}

bool CudaDICKernelPrecision::setImages(const cv::Mat& refImage, const cv::Mat& defImage) {
    if (!m_initialized) {
        std::cerr << "Precision kernel not initialized" << std::endl;
        return false;
    }
    
    if (refImage.size() != defImage.size() || 
        refImage.cols != m_imageWidth || refImage.rows != m_imageHeight) {
        std::cerr << "Image size mismatch in precision kernel" << std::endl;
        return false;
    }
    
    // Convert images to double precision and scale to [0,255] range for consistency with CPU
    cv::Mat refDouble, defDouble;
    
    if (refImage.type() == CV_8UC1) {
        refImage.convertTo(refDouble, CV_64F);
    } else if (refImage.type() == CV_64F) {
        // If already double but normalized to [0,1], scale to [0,255]
        double minVal, maxVal;
        cv::minMaxLoc(refImage, &minVal, &maxVal);
        if (maxVal <= 1.0) {
            refImage.convertTo(refDouble, CV_64F, 255.0);
        } else {
            refDouble = refImage.clone();
        }
    } else {
        refImage.convertTo(refDouble, CV_64F);
    }
    
    if (defImage.type() == CV_8UC1) {
        defImage.convertTo(defDouble, CV_64F);
    } else if (defImage.type() == CV_64F) {
        double minVal, maxVal;
        cv::minMaxLoc(defImage, &minVal, &maxVal);
        if (maxVal <= 1.0) {
            defImage.convertTo(defDouble, CV_64F, 255.0);
        } else {
            defDouble = defImage.clone();
        }
    } else {
        defImage.convertTo(defDouble, CV_64F);
    }
    
    // Copy images to GPU
    size_t imageSize = m_imageWidth * m_imageHeight * sizeof(double);
    cudaError_t err1 = cudaMemcpyAsync(d_refImage, refDouble.ptr<double>(), imageSize, 
                                      cudaMemcpyHostToDevice, m_stream);
    cudaError_t err2 = cudaMemcpyAsync(d_defImage, defDouble.ptr<double>(), imageSize, 
                                      cudaMemcpyHostToDevice, m_stream);
    
    if (err1 != cudaSuccess || err2 != cudaSuccess) {
        std::cerr << "Failed to copy images to GPU in precision kernel" << std::endl;
        return false;
    }
    
    // Synchronize to ensure transfer is complete
    cudaStreamSynchronize(m_stream);
    
    m_imagesSet = true;
    std::cout << "High-precision images set successfully" << std::endl;
    return true;
}

bool CudaDICKernelPrecision::copyPointsToGPU(const std::vector<cv::Point>& points) {
    if (points.size() > static_cast<size_t>(m_maxPoints)) {
        std::cerr << "Too many points for precision kernel: " << points.size() << " > " << m_maxPoints << std::endl;
        return false;
    }
    
    // Convert cv::Point to Point2D
    std::vector<Point2D> gpuPoints(points.size());
    for (size_t i = 0; i < points.size(); i++) {
        gpuPoints[i] = Point2D(points[i].x, points[i].y);
    }
    
    // Copy to GPU
    size_t pointsSize = points.size() * sizeof(Point2D);
    cudaError_t err = cudaMemcpyAsync(d_points, gpuPoints.data(), pointsSize, 
                                     cudaMemcpyHostToDevice, m_stream);
    
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy points to GPU: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    return true;
}

bool CudaDICKernelPrecision::copyInitialGuessToGPU(const std::vector<cv::Vec2f>& initialGuess) {
    if (initialGuess.size() > static_cast<size_t>(m_maxPoints)) {
        std::cerr << "Too many initial guesses: " << initialGuess.size() << " > " << m_maxPoints << std::endl;
        return false;
    }
    
    // Convert initial guess to warp parameters format
    std::vector<double> params(initialGuess.size() * m_numParams, 0.0);
    
    for (size_t i = 0; i < initialGuess.size(); i++) {
        params[i * m_numParams + 0] = static_cast<double>(initialGuess[i][0]); // u
        params[i * m_numParams + 1] = static_cast<double>(initialGuess[i][1]); // v
        // Higher order parameters remain zero
    }
    
    // Copy to GPU
    size_t paramsSize = initialGuess.size() * m_numParams * sizeof(double);
    cudaError_t err = cudaMemcpyAsync(d_initialParams, params.data(), paramsSize, 
                                     cudaMemcpyHostToDevice, m_stream);
    
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy initial parameters to GPU: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    return true;
}

bool CudaDICKernelPrecision::copyResultsFromGPU(std::vector<DICResult>& results, int numPoints) {
    if (numPoints > m_maxPoints) {
        std::cerr << "Too many points to copy back: " << numPoints << " > " << m_maxPoints << std::endl;
        return false;
    }
    
    // Copy results from GPU
    size_t displacementSize = numPoints * sizeof(double);
    size_t validMaskSize = numPoints * sizeof(bool);
    
    cudaError_t err1 = cudaMemcpyAsync(h_finalU.data(), d_finalU, displacementSize, 
                                      cudaMemcpyDeviceToHost, m_stream);
    cudaError_t err2 = cudaMemcpyAsync(h_finalV.data(), d_finalV, displacementSize, 
                                      cudaMemcpyDeviceToHost, m_stream);
    cudaError_t err3 = cudaMemcpyAsync(h_finalZNCC.data(), d_finalZNCC, displacementSize, 
                                      cudaMemcpyDeviceToHost, m_stream);
    
    // For vector<bool>, we need to use a temporary buffer
    std::vector<bool> tempValidMask(numPoints);
    bool* tempBuffer = new bool[numPoints];
    
    cudaError_t err4 = cudaMemcpyAsync(tempBuffer, d_validMask, validMaskSize, 
                                      cudaMemcpyDeviceToHost, m_stream);
    
    if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess || err4 != cudaSuccess) {
        std::cerr << "Failed to copy results from GPU" << std::endl;
        delete[] tempBuffer;
        return false;
    }
    
    // Synchronize to ensure transfer is complete
    cudaStreamSynchronize(m_stream);
    
    // Copy bool buffer to vector<bool>
    for (int i = 0; i < numPoints; i++) {
        h_validMask[i] = tempBuffer[i];
    }
    delete[] tempBuffer;
    
    // Convert to DICResult format
    results.resize(numPoints);
    for (int i = 0; i < numPoints; i++) {
        results[i].u = h_finalU[i];
        results[i].v = h_finalV[i];
        results[i].zncc = h_finalZNCC[i];
        results[i].valid = h_validMask[i];
    }
    
    return true;
}

bool CudaDICKernelPrecision::computeInitialGuess(const std::vector<cv::Point>& points,
                                                std::vector<cv::Vec2f>& initialGuess,
                                                int searchRadius) {
    if (!m_initialized || !m_imagesSet) {
        std::cerr << "Precision kernel not properly initialized for initial guess" << std::endl;
        return false;
    }
    
    int numPoints = static_cast<int>(points.size());
    if (numPoints > m_maxPoints) {
        std::cerr << "Too many points for initial guess: " << numPoints << " > " << m_maxPoints << std::endl;
        return false;
    }
    
    // Copy points to GPU
    if (!copyPointsToGPU(points)) {
        return false;
    }
    
    // Launch initial guess kernel
    launchPrecisionInitialGuessKernel(d_initialParams, d_finalZNCC, d_validMask,
                                     d_refImage, d_defImage, d_points, numPoints,
                                     m_imageWidth, m_imageHeight, m_subsetRadius,
                                     m_numParams, searchRadius, m_stream);
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Initial guess kernel failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Copy initial parameters back
    std::vector<double> params(numPoints * m_numParams);
    size_t paramsSize = numPoints * m_numParams * sizeof(double);
    err = cudaMemcpyAsync(params.data(), d_initialParams, paramsSize, 
                         cudaMemcpyDeviceToHost, m_stream);
    
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy initial guess results: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    cudaStreamSynchronize(m_stream);
    
    // Convert to cv::Vec2f format
    initialGuess.resize(numPoints);
    for (int i = 0; i < numPoints; i++) {
        initialGuess[i][0] = static_cast<float>(params[i * m_numParams + 0]);
        initialGuess[i][1] = static_cast<float>(params[i * m_numParams + 1]);
    }
    
    std::cout << "Initial guess computed for " << numPoints << " points" << std::endl;
    return true;
}

bool CudaDICKernelPrecision::computeDIC(const std::vector<cv::Point>& points,
                                       const std::vector<cv::Vec2f>& initialGuess,
                                       std::vector<DICResult>& results,
                                       double convergenceThreshold,
                                       int maxIterations) {
    
    if (!m_initialized || !m_imagesSet) {
        std::cerr << "Precision kernel not properly initialized for DIC computation" << std::endl;
        return false;
    }
    
    if (points.size() != initialGuess.size()) {
        std::cerr << "Points and initial guess size mismatch" << std::endl;
        return false;
    }
    
    int numPoints = static_cast<int>(points.size());
    if (numPoints > m_maxPoints) {
        std::cerr << "Too many points for DIC computation: " << numPoints << " > " << m_maxPoints << std::endl;
        return false;
    }
    
    if (numPoints == 0) {
        results.clear();
        return true;
    }
    
    // Copy points and initial guess to GPU
    if (!copyPointsToGPU(points) || !copyInitialGuessToGPU(initialGuess)) {
        return false;
    }
    
    // Launch high-precision ICGN optimization kernel
    launchPrecisionICGNOptimizationKernel(d_finalU, d_finalV, d_finalZNCC, d_validMask,
                                         d_refImage, d_defImage, d_points, d_initialParams,
                                         numPoints, m_imageWidth, m_imageHeight,
                                         m_subsetRadius, m_numParams, maxIterations,
                                         convergenceThreshold, m_stream);
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Precision ICGN kernel failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Synchronize to ensure kernel completion
    cudaStreamSynchronize(m_stream);
    
    // Copy results back from GPU
    if (!copyResultsFromGPU(results, numPoints)) {
        return false;
    }
    
    // Count valid results
    int validCount = 0;
    for (const auto& result : results) {
        if (result.valid) validCount++;
    }
    
    std::cout << "High-precision DIC computation completed: " << validCount 
              << "/" << numPoints << " valid results" << std::endl;
    
    return true;
}

void CudaDICKernelPrecision::cleanup() {
    deallocateGPUMemory();
    
    if (m_stream) {
        cudaStreamDestroy(m_stream);
        m_stream = nullptr;
    }
    
    m_initialized = false;
    m_imagesSet = false;
    
    std::cout << "High-precision CUDA DIC kernel cleaned up" << std::endl;
}

// CUDA-accelerated interpolation for displacement field
bool CudaDICKernelPrecision::interpolateDisplacementField(const cv::Mat& sparseU, const cv::Mat& sparseV,
                                                         const cv::Mat& sparseMask, const cv::Mat& roi,
                                                         const std::vector<cv::Point>& sparsePoints,
                                                         cv::Mat& interpU, cv::Mat& interpV, cv::Mat& interpMask,
                                                         int step, InterpolationMethod method) {
    
    if (!m_initialized) {
        std::cerr << "CUDA kernel not initialized for interpolation" << std::endl;
        return false;
    }
    
    int width = sparseU.cols;
    int height = sparseU.rows;
    int totalPixels = width * height;
    
    // Allocate GPU memory for interpolation
    double* d_sparseU = nullptr;
    double* d_sparseV = nullptr;
    unsigned char* d_sparseMask = nullptr;
    unsigned char* d_roi = nullptr;
    Point2D* d_sparsePoints = nullptr;
    double* d_interpU = nullptr;
    double* d_interpV = nullptr;
    unsigned char* d_interpMask = nullptr;
    
    cudaError_t err;
    
    // Allocate memory
    err = cudaMalloc(&d_sparseU, totalPixels * sizeof(double));
    if (err != cudaSuccess) return false;
    err = cudaMalloc(&d_sparseV, totalPixels * sizeof(double));
    if (err != cudaSuccess) { cudaFree(d_sparseU); return false; }
    err = cudaMalloc(&d_sparseMask, totalPixels * sizeof(unsigned char));
    if (err != cudaSuccess) { cudaFree(d_sparseU); cudaFree(d_sparseV); return false; }
    err = cudaMalloc(&d_roi, totalPixels * sizeof(unsigned char));
    if (err != cudaSuccess) { 
        cudaFree(d_sparseU); cudaFree(d_sparseV); cudaFree(d_sparseMask); 
        return false; 
    }
    err = cudaMalloc(&d_sparsePoints, sparsePoints.size() * sizeof(Point2D));
    if (err != cudaSuccess) { 
        cudaFree(d_sparseU); cudaFree(d_sparseV); cudaFree(d_sparseMask); cudaFree(d_roi);
        return false; 
    }
    err = cudaMalloc(&d_interpU, totalPixels * sizeof(double));
    if (err != cudaSuccess) { 
        cudaFree(d_sparseU); cudaFree(d_sparseV); cudaFree(d_sparseMask); 
        cudaFree(d_roi); cudaFree(d_sparsePoints);
        return false; 
    }
    err = cudaMalloc(&d_interpV, totalPixels * sizeof(double));
    if (err != cudaSuccess) { 
        cudaFree(d_sparseU); cudaFree(d_sparseV); cudaFree(d_sparseMask); 
        cudaFree(d_roi); cudaFree(d_sparsePoints); cudaFree(d_interpU);
        return false; 
    }
    err = cudaMalloc(&d_interpMask, totalPixels * sizeof(unsigned char));
    if (err != cudaSuccess) { 
        cudaFree(d_sparseU); cudaFree(d_sparseV); cudaFree(d_sparseMask); 
        cudaFree(d_roi); cudaFree(d_sparsePoints); cudaFree(d_interpU); cudaFree(d_interpV);
        return false; 
    }
    
    // Copy data to GPU
    err = cudaMemcpy(d_sparseU, sparseU.ptr<double>(), totalPixels * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup_interp;
    err = cudaMemcpy(d_sparseV, sparseV.ptr<double>(), totalPixels * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup_interp;
    err = cudaMemcpy(d_sparseMask, sparseMask.ptr<unsigned char>(), totalPixels * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup_interp;
    err = cudaMemcpy(d_roi, roi.ptr<unsigned char>(), totalPixels * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup_interp;
    
    // Convert sparse points to GPU format
    {
        std::vector<Point2D> gpuPoints(sparsePoints.size());
        for (size_t i = 0; i < sparsePoints.size(); i++) {
            gpuPoints[i] = Point2D(sparsePoints[i].x, sparsePoints[i].y);
        }
        err = cudaMemcpy(d_sparsePoints, gpuPoints.data(), sparsePoints.size() * sizeof(Point2D), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto cleanup_interp;
    }
    
    // Launch interpolation kernel
    launchPrecisionInterpolationKernel(d_interpU, d_interpV, d_interpMask,
                                      d_sparseU, d_sparseV, d_sparseMask, d_roi,
                                      d_sparsePoints, sparsePoints.size(),
                                      width, height, step, static_cast<int>(method), m_stream);
    
    // Wait for completion
    err = cudaStreamSynchronize(m_stream);
    if (err != cudaSuccess) goto cleanup_interp;
    
    // Copy results back
    err = cudaMemcpy(interpU.ptr<double>(), d_interpU, totalPixels * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto cleanup_interp;
    err = cudaMemcpy(interpV.ptr<double>(), d_interpV, totalPixels * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto cleanup_interp;
    err = cudaMemcpy(interpMask.ptr<unsigned char>(), d_interpMask, totalPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto cleanup_interp;
    
    // Cleanup
cleanup_interp:
    cudaFree(d_sparseU);
    cudaFree(d_sparseV);
    cudaFree(d_sparseMask);
    cudaFree(d_roi);
    cudaFree(d_sparsePoints);
    cudaFree(d_interpU);
    cudaFree(d_interpV);
    cudaFree(d_interpMask);
    
    return err == cudaSuccess;
}

// CUDA-accelerated strain field calculation
bool CudaDICKernelPrecision::calculateStrainField(const cv::Mat& u, const cv::Mat& v, const cv::Mat& validMask,
                                                 cv::Mat& strainExx, cv::Mat& strainEyy, cv::Mat& strainExy,
                                                 cv::Mat& strainMask, int windowSize) {
    
    if (!m_initialized) {
        std::cerr << "CUDA kernel not initialized for strain calculation" << std::endl;
        return false;
    }
    
    int width = u.cols;
    int height = u.rows;
    int totalPixels = width * height;
    
    // Allocate GPU memory for strain calculation
    double* d_u = nullptr;
    double* d_v = nullptr;
    unsigned char* d_validMask = nullptr;
    double* d_strainExx = nullptr;
    double* d_strainEyy = nullptr;
    double* d_strainExy = nullptr;
    unsigned char* d_strainMask = nullptr;
    
    cudaError_t err;
    
    // Allocate memory
    err = cudaMalloc(&d_u, totalPixels * sizeof(double));
    if (err != cudaSuccess) return false;
    err = cudaMalloc(&d_v, totalPixels * sizeof(double));
    if (err != cudaSuccess) { cudaFree(d_u); return false; }
    err = cudaMalloc(&d_validMask, totalPixels * sizeof(unsigned char));
    if (err != cudaSuccess) { cudaFree(d_u); cudaFree(d_v); return false; }
    err = cudaMalloc(&d_strainExx, totalPixels * sizeof(double));
    if (err != cudaSuccess) { 
        cudaFree(d_u); cudaFree(d_v); cudaFree(d_validMask); 
        return false; 
    }
    err = cudaMalloc(&d_strainEyy, totalPixels * sizeof(double));
    if (err != cudaSuccess) { 
        cudaFree(d_u); cudaFree(d_v); cudaFree(d_validMask); cudaFree(d_strainExx);
        return false; 
    }
    err = cudaMalloc(&d_strainExy, totalPixels * sizeof(double));
    if (err != cudaSuccess) { 
        cudaFree(d_u); cudaFree(d_v); cudaFree(d_validMask); 
        cudaFree(d_strainExx); cudaFree(d_strainEyy);
        return false; 
    }
    err = cudaMalloc(&d_strainMask, totalPixels * sizeof(unsigned char));
    if (err != cudaSuccess) { 
        cudaFree(d_u); cudaFree(d_v); cudaFree(d_validMask); 
        cudaFree(d_strainExx); cudaFree(d_strainEyy); cudaFree(d_strainExy);
        return false; 
    }
    
    // Copy data to GPU
    err = cudaMemcpy(d_u, u.ptr<double>(), totalPixels * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup_strain;
    err = cudaMemcpy(d_v, v.ptr<double>(), totalPixels * sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup_strain;
    err = cudaMemcpy(d_validMask, validMask.ptr<unsigned char>(), totalPixels * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup_strain;
    
    // Launch strain calculation kernel
    launchPrecisionStrainCalculationKernel(d_strainExx, d_strainEyy, d_strainExy, d_strainMask,
                                          d_u, d_v, d_validMask, width, height, windowSize, m_stream);
    
    // Wait for completion
    err = cudaStreamSynchronize(m_stream);
    if (err != cudaSuccess) goto cleanup_strain;
    
    // Copy results back
    err = cudaMemcpy(strainExx.ptr<double>(), d_strainExx, totalPixels * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto cleanup_strain;
    err = cudaMemcpy(strainEyy.ptr<double>(), d_strainEyy, totalPixels * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto cleanup_strain;
    err = cudaMemcpy(strainExy.ptr<double>(), d_strainExy, totalPixels * sizeof(double), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto cleanup_strain;
    err = cudaMemcpy(strainMask.ptr<unsigned char>(), d_strainMask, totalPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto cleanup_strain;
    
    // Cleanup
cleanup_strain:
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_validMask);
    cudaFree(d_strainExx);
    cudaFree(d_strainEyy);
    cudaFree(d_strainExy);
    cudaFree(d_strainMask);
    
    return err == cudaSuccess;
}
