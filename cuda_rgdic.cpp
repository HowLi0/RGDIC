#include "cuda_rgdic.h"
#include "cuda_dic_kernel_precision.h"
#include <chrono>
#include <algorithm>
#include <iostream>
#include <cmath>

CudaRGDIC::CudaRGDIC(int subsetRadius, double convergenceThreshold, int maxIterations,
                     double ccThreshold, double deltaDispThreshold, ShapeFunctionOrder order,
                     int neighborStep, int maxBatchSize)
    : RGDIC(subsetRadius, convergenceThreshold, maxIterations, ccThreshold, 
            deltaDispThreshold, order, neighborStep),
      m_maxBatchSize(maxBatchSize), m_gpuInitialized(false), m_startTime(0.0)
{
    // Initialize CUDA device manager
    auto& deviceManager = CudaDeviceManager::getInstance();
    if (!deviceManager.initializeCuda()) {
        std::cerr << "Warning: CUDA initialization failed, falling back to CPU" << std::endl;
        return;
    }
    
    deviceManager.printDeviceInfo();
    initializeGPU();
}

CudaRGDIC::~CudaRGDIC() {
    cleanupGPU();
}

void CudaRGDIC::initializeGPU() {
    try {
        // Convert enum to CUDA kernel enum
        ShapeFunctionOrder cudaOrder = (m_order == FIRST_ORDER) ? 
            FIRST_ORDER : SECOND_ORDER;
        
        // Initialize high-precision kernel
        m_precisionKernel = std::make_unique<CudaDICKernelPrecision>(m_maxBatchSize, cudaOrder);
        
        m_gpuInitialized = true;
        
        std::cout << "High-precision CUDA RGDIC initialized successfully" << std::endl;
        std::cout << "  - Max batch size: " << m_maxBatchSize << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Failed to initialize CUDA RGDIC: " << e.what() << std::endl;
        m_gpuInitialized = false;
    }
}

void CudaRGDIC::cleanupGPU() {
    if (m_precisionKernel) {
        m_precisionKernel->cleanup();
        m_precisionKernel.reset();
    }
    m_gpuInitialized = false;
}

RGDIC::DisplacementResult CudaRGDIC::compute(const cv::Mat& refImage, 
                                           const cv::Mat& defImage,
                                           const cv::Mat& roi) {
    
    startTiming();
    m_lastStats = PerformanceStats();
    
    // Fallback to CPU if CUDA is not available
    if (!m_gpuInitialized) {
        std::cout << "Using CPU fallback for DIC computation" << std::endl;
        auto result = computeCPUFallback(refImage, defImage, roi);
        endTiming("CPU computation");
        m_lastStats.totalTime = m_timings["CPU computation"];
        m_lastStats.cpuProcessingTime = m_lastStats.totalTime;
        return result;
    }
    
    // Initialize CUDA kernel with image dimensions
    if (m_precisionKernel) {
        if (!m_precisionKernel->initialize(refImage.cols, refImage.rows, m_subsetRadius)) {
            std::cerr << "Failed to initialize precision CUDA kernel, falling back to CPU" << std::endl;
            return computeCPUFallback(refImage, defImage, roi);
        }
    } else {
        std::cerr << "No CUDA kernel available, falling back to CPU" << std::endl;
        return computeCPUFallback(refImage, defImage, roi);
    }
    
    // Set images on GPU
    startTiming();
    bool imageSetResult = false;
    if (m_precisionKernel) {
        imageSetResult = m_precisionKernel->setImages(refImage, defImage);
    }
    
    if (!imageSetResult) {
        std::cerr << "Failed to set images on GPU, falling back to CPU" << std::endl;
        return computeCPUFallback(refImage, defImage, roi);
    }
    endTiming("Image transfer to GPU");
    m_lastStats.memoryTransferTime += m_timings["Image transfer to GPU"];
    
    // Extract ROI points
    startTiming();
    auto roiPoints = extractROIPoints(roi);
    m_lastStats.pointsProcessed = roiPoints.size();
    
    if (roiPoints.empty()) {
        std::cerr << "No valid points found in ROI" << std::endl;
        return DisplacementResult();
    }
    endTiming("ROI extraction");
    m_lastStats.cpuProcessingTime += m_timings["ROI extraction"];
    
    // Find seed point using CPU method (small overhead)
    startTiming();
    cv::Mat sda = calculateSDA(roi);
    cv::Point seedPoint = findSeedPoint(roi, sda);
    
    // Compute seed point on CPU to get initial parameters
    ICGNOptimizer optimizer(refImage, defImage, m_subsetRadius, m_order, 
                          m_convergenceThreshold, m_maxIterations);
    
    cv::Mat warpParams = cv::Mat::zeros(m_numParams, 1, CV_64F);
    double initialZNCC = 0.0;
    
    if (!optimizer.initialGuess(seedPoint, warpParams, initialZNCC) ||
        !optimizer.optimize(seedPoint, warpParams, initialZNCC)) {
        std::cerr << "Failed to compute seed point, falling back to CPU" << std::endl;
        return computeCPUFallback(refImage, defImage, roi);
    }
    
    DisplacementResult seedResult;
    seedResult.u = cv::Mat::zeros(roi.size(), CV_64F);
    seedResult.v = cv::Mat::zeros(roi.size(), CV_64F);
    seedResult.u.at<double>(seedPoint) = warpParams.at<double>(0);
    seedResult.v.at<double>(seedPoint) = warpParams.at<double>(1);
    
    endTiming("Seed point computation");
    m_lastStats.cpuProcessingTime += m_timings["Seed point computation"];
    
    // Generate initial guess for all points
    startTiming();
    auto initialGuess = generateInitialGuess(roiPoints, seedResult, seedPoint);
    endTiming("Initial guess generation");
    m_lastStats.cpuProcessingTime += m_timings["Initial guess generation"];
    
    // Compute DIC on GPU using batch processing
    startTiming();
    auto result = computeBatch(refImage, defImage, roiPoints, initialGuess);
    endTiming("GPU batch computation");
    m_lastStats.gpuComputeTime = m_timings["GPU batch computation"];
    
    // Post-processing and outlier filtering
    startTiming();
    filterOutliers(result, roi);
    endTiming("Post-processing");
    m_lastStats.cpuProcessingTime += m_timings["Post-processing"];
    
    // Calculate performance statistics
    m_lastStats.totalTime = getCurrentTime() - m_startTime;
    m_lastStats.validPoints = cv::countNonZero(result.validMask);
    
    // Compare with CPU timing (estimate)
    double estimatedCpuTime = m_lastStats.pointsProcessed * 0.1; // Rough estimate
    m_lastStats.speedup = estimatedCpuTime / m_lastStats.totalTime;
    
    std::cout << "CUDA RGDIC Performance Summary:" << std::endl;
    std::cout << "  - Total time: " << m_lastStats.totalTime << " seconds" << std::endl;
    std::cout << "  - GPU compute time: " << m_lastStats.gpuComputeTime << " seconds" << std::endl;
    std::cout << "  - Memory transfer time: " << m_lastStats.memoryTransferTime << " seconds" << std::endl;
    std::cout << "  - CPU processing time: " << m_lastStats.cpuProcessingTime << " seconds" << std::endl;
    std::cout << "  - Points processed: " << m_lastStats.pointsProcessed << std::endl;
    std::cout << "  - Valid points: " << m_lastStats.validPoints << std::endl;
    std::cout << "  - Estimated speedup: " << m_lastStats.speedup << "x" << std::endl;
    
    return result;
}

RGDIC::DisplacementResult CudaRGDIC::computeBatch(const cv::Mat& refImage,
                                                const cv::Mat& defImage,
                                                const std::vector<cv::Point>& points,
                                                const std::vector<cv::Vec2f>& initialGuess) {
    
    DisplacementResult result;
    result.u = cv::Mat::zeros(refImage.size(), CV_64F);
    result.v = cv::Mat::zeros(refImage.size(), CV_64F);
    result.cc = cv::Mat::zeros(refImage.size(), CV_64F);
    result.validMask = cv::Mat::zeros(refImage.size(), CV_8U);
    
    // Create batches for processing
    auto batches = createBatches(points);
    
    for (size_t batchIdx = 0; batchIdx < batches.size(); batchIdx++) {
        const auto& batchPoints = batches[batchIdx];
        
        // Extract initial guess for this batch
        std::vector<cv::Vec2f> batchInitialGuess;
        for (const auto& point : batchPoints) {
            // Find the corresponding initial guess
            auto it = std::find(points.begin(), points.end(), point);
            if (it != points.end()) {
                size_t idx = std::distance(points.begin(), it);
                batchInitialGuess.push_back(initialGuess[idx]);
            } else {
                batchInitialGuess.push_back(cv::Vec2f(0.0f, 0.0f));
            }
        }
        
        // Process this batch
        std::vector<DICResult> batchResults;
        processBatch(batchPoints, batchInitialGuess, batchResults);
        
        // Integrate results
        integrateResults(batchPoints, batchResults, result);
        
        std::cout << "Processed batch " << (batchIdx + 1) << "/" << batches.size() 
                  << " (" << batchPoints.size() << " points)" << std::endl;
    }
    
    return result;
}

void CudaRGDIC::processBatch(const std::vector<cv::Point>& points,
                           const std::vector<cv::Vec2f>& initialGuess,
                           std::vector<DICResult>& batchResults) {
    
    bool success = false;
    
    if (m_precisionKernel) {
        success = m_precisionKernel->computeDIC(points, initialGuess, batchResults,
                                              m_convergenceThreshold, m_maxIterations);
    }
    
    if (!success) {
        std::cerr << "CUDA DIC computation failed for batch" << std::endl;
        
        // Fill with invalid results
        batchResults.resize(points.size());
        for (auto& result : batchResults) {
            result.valid = false;
        }
    }
}

void CudaRGDIC::integrateResults(const std::vector<cv::Point>& points,
                               const std::vector<DICResult>& batchResults,
                               DisplacementResult& result) {
    
    for (size_t i = 0; i < points.size() && i < batchResults.size(); i++) {
        const auto& point = points[i];
        const auto& dicResult = batchResults[i];
        
        if (dicResult.valid && dicResult.zncc < m_ccThreshold) {
            result.u.at<double>(point) = dicResult.u;
            result.v.at<double>(point) = dicResult.v;
            result.cc.at<double>(point) = dicResult.zncc;
            result.validMask.at<uchar>(point) = 255;
        }
    }
}

std::vector<cv::Point> CudaRGDIC::extractROIPoints(const cv::Mat& roi) {
    std::vector<cv::Point> points;
    
    for (int y = m_subsetRadius; y < roi.rows - m_subsetRadius; y++) {
        for (int x = m_subsetRadius; x < roi.cols - m_subsetRadius; x++) {
            if (roi.at<uchar>(y, x) > 0) {
                points.emplace_back(x, y);
            }
        }
    }
    
    return points;
}

std::vector<cv::Vec2f> CudaRGDIC::generateInitialGuess(const std::vector<cv::Point>& points,
                                                      const DisplacementResult& seedResult,
                                                      cv::Point seedPoint) {
    
    std::vector<cv::Vec2f> initialGuess(points.size());
    
    // Use seed point displacement as initial guess for all points
    float seedU = static_cast<float>(seedResult.u.at<double>(seedPoint));
    float seedV = static_cast<float>(seedResult.v.at<double>(seedPoint));
    
    for (size_t i = 0; i < points.size(); i++) {
        initialGuess[i] = cv::Vec2f(seedU, seedV);
    }
    
    return initialGuess;
}

std::vector<std::vector<cv::Point>> CudaRGDIC::createBatches(const std::vector<cv::Point>& points) {
    std::vector<std::vector<cv::Point>> batches;
    
    for (size_t i = 0; i < points.size(); i += m_maxBatchSize) {
        size_t endIdx = std::min(i + m_maxBatchSize, points.size());
        std::vector<cv::Point> batch(points.begin() + i, points.begin() + endIdx);
        batches.push_back(batch);
    }
    
    return batches;
}

void CudaRGDIC::filterOutliers(DisplacementResult& result, const cv::Mat& roi) {
    // Use the parent class outlier filtering method
    cv::Mat validMaskCopy = result.validMask.clone();
    
    for (int y = 0; y < result.validMask.rows; y++) {
        for (int x = 0; x < result.validMask.cols; x++) {
            if (result.validMask.at<uchar>(y, x)) {
                bool isOutlier = false;
                
                auto validNeighbors = m_neighborUtils.getValidNeighbors(cv::Point(x, y), roi);
                
                for (const auto& neighborPoint : validNeighbors) {
                    if (result.validMask.at<uchar>(neighborPoint)) {
                        double du = result.u.at<double>(y, x) - result.u.at<double>(neighborPoint);
                        double dv = result.v.at<double>(y, x) - result.v.at<double>(neighborPoint);
                        double dispJump = std::sqrt(du*du + dv*dv);
                        
                        if (dispJump > m_deltaDispThreshold) {
                            isOutlier = true;
                            break;
                        }
                    }
                }
                
                if (isOutlier) {
                    validMaskCopy.at<uchar>(y, x) = 0;
                }
            }
        }
    }
    
    result.validMask = validMaskCopy;
}

RGDIC::DisplacementResult CudaRGDIC::computeCPUFallback(const cv::Mat& refImage,
                                                      const cv::Mat& defImage,
                                                      const cv::Mat& roi) {
    // Use the parent class CPU implementation
    return RGDIC::compute(refImage, defImage, roi);
}

double CudaRGDIC::getCurrentTime() const {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}

void CudaRGDIC::startTiming() {
    m_startTime = getCurrentTime();
}

void CudaRGDIC::endTiming(const std::string& operation) {
    double endTime = getCurrentTime();
    m_timings[operation] = endTime - m_startTime;
    m_startTime = endTime;
}

// CudaDeviceManager implementation
CudaDeviceManager& CudaDeviceManager::getInstance() {
    static CudaDeviceManager instance;
    return instance;
}

bool CudaDeviceManager::initializeCuda() {
    if (m_initialized) {
        return true;
    }
    
    // Check for CUDA devices
    cudaError_t err = cudaGetDeviceCount(&m_deviceCount);
    if (err != cudaSuccess || m_deviceCount == 0) {
        std::cerr << "No CUDA devices found" << std::endl;
        return false;
    }
    
    // Get device properties
    err = cudaGetDeviceProperties(&m_deviceProp, m_currentDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to get device properties" << std::endl;
        return false;
    }
    
    // Set device
    err = cudaSetDevice(m_currentDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device" << std::endl;
        return false;
    }
    
    m_initialized = true;
    return true;
}

void CudaDeviceManager::printDeviceInfo() {
    if (!m_initialized) {
        std::cerr << "CUDA not initialized" << std::endl;
        return;
    }
    
    std::cout << "CUDA Device Information:" << std::endl;
    std::cout << "  - Device count: " << m_deviceCount << std::endl;
    std::cout << "  - Current device: " << m_currentDevice << std::endl;
    std::cout << "  - Device name: " << m_deviceProp.name << std::endl;
    std::cout << "  - Compute capability: " << m_deviceProp.major << "." << m_deviceProp.minor << std::endl;
    std::cout << "  - Total global memory: " << (m_deviceProp.totalGlobalMem / 1024 / 1024) << " MB" << std::endl;
    std::cout << "  - Multiprocessor count: " << m_deviceProp.multiProcessorCount << std::endl;
    std::cout << "  - Max threads per block: " << m_deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "  - Warp size: " << m_deviceProp.warpSize << std::endl;
}

size_t CudaDeviceManager::getAvailableMemory() const {
    if (!m_initialized) return 0;
    
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return free;
}

size_t CudaDeviceManager::getTotalMemory() const {
    if (!m_initialized) return 0;
    return m_deviceProp.totalGlobalMem;
}

int CudaDeviceManager::getMultiProcessorCount() const {
    if (!m_initialized) return 0;
    return m_deviceProp.multiProcessorCount;
}

int CudaDeviceManager::getMaxThreadsPerBlock() const {
    if (!m_initialized) return 0;
    return m_deviceProp.maxThreadsPerBlock;
}

int CudaDeviceManager::getWarpSize() const {
    if (!m_initialized) return 0;
    return m_deviceProp.warpSize;
}
