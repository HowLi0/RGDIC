#include "dic_processor.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <queue>
#include <chrono>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

// === DICProcessor Implementation ===

DICProcessor::DICProcessor(const Config& config) 
    : config_(config)
    , neighborUtils_(config.neighborStep, NeighborUtils::FOUR_CONNECTED)
    , progressCallback_(nullptr)
{
}

DICProcessor::Statistics DICProcessor::processPOIs(const cv::Mat& refImage, 
                                                  const cv::Mat& defImage,
                                                  POIManager& poiManager) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Initialize optimizer
    initializeOptimizer(refImage, defImage);
    
    // Initialize statistics
    Statistics stats;
    stats.totalPoints = poiManager.size();
    stats.processedPoints = 0;
    stats.validPoints = 0;
    stats.convergedPoints = 0;
    
    if (stats.totalPoints == 0) {
        return stats;
    }
    
    updateProgress(0, stats.totalPoints, "Starting DIC processing...");
    
    // Find and process seed point first
    std::shared_ptr<POI> seedPOI = nullptr;
    if (config_.enableQualityControl) {
        // Find the POI closest to the center of the collection
        double centerX = 0, centerY = 0;
        for (size_t i = 0; i < poiManager.size(); ++i) {
            auto poi = poiManager.getPOI(i);
            centerX += poi->getReferenceX();
            centerY += poi->getReferenceY();
        }
        centerX /= stats.totalPoints;
        centerY /= stats.totalPoints;
        
        seedPOI = poiManager.findNearestPOI(centerX, centerY);
        
        if (seedPOI && processSinglePOI(refImage, defImage, seedPOI)) {
            updateProgress(1, stats.totalPoints, "Seed point processed");
        }
    }
    
    // Process remaining POIs
    size_t processed = seedPOI ? 1 : 0;
    
    for (size_t i = 0; i < poiManager.size(); ++i) {
        auto poi = poiManager.getPOI(i);
        
        // Skip seed POI if already processed
        if (poi == seedPOI) continue;
        
        // Generate initial guess from neighbors if available
        if (config_.enableQualityControl && processed > 0) {
            std::vector<std::shared_ptr<POI>> processedPOIs;
            for (size_t j = 0; j < i; ++j) {
                auto prevPOI = poiManager.getPOI(j);
                if (prevPOI->isValid() && prevPOI->hasDisplacement()) {
                    processedPOIs.push_back(prevPOI);
                }
            }
            
            if (!processedPOIs.empty()) {
                cv::Vec2f initialGuess = generateInitialGuess(poi, processedPOIs);
                poi->setDisplacement(initialGuess[0], initialGuess[1]);
            }
        }
        
        // Process the POI
        bool success = processSinglePOI(refImage, defImage, poi);
        processed++;
        
        updateProgress(processed, stats.totalPoints, 
                      "Processing POI " + std::to_string(processed) + "/" + std::to_string(stats.totalPoints));
        
        if (success) {
            stats.validPoints++;
            if (poi->isConverged()) {
                stats.convergedPoints++;
            }
        }
    }
    
    // Apply quality control if enabled
    if (config_.enableQualityControl) {
        applyQualityControl(poiManager);
        updateProgress(stats.totalPoints, stats.totalPoints, "Quality control applied");
    }
    
    // Calculate final statistics
    stats = calculateStatistics(poiManager);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    stats.processingTime = duration.count() / 1000.0;
    
    lastStats_ = stats;
    return stats;
}

bool DICProcessor::processSinglePOI(const cv::Mat& refImage,
                                   const cv::Mat& defImage,
                                   std::shared_ptr<POI> poi) {
    if (!poi || !optimizer_) {
        return false;
    }
    
    cv::Point refPoint(static_cast<int>(poi->getReferenceX()), 
                      static_cast<int>(poi->getReferenceY()));
    
    // Check bounds
    int margin = config_.subsetRadius + 1;
    if (refPoint.x < margin || refPoint.y < margin ||
        refPoint.x >= refImage.cols - margin || 
        refPoint.y >= refImage.rows - margin) {
        poi->setValid(false);
        return false;
    }
    
    // Initialize warp parameters
    cv::Mat warpParams = cv::Mat::zeros(optimizer_->getNumParams(), 1, CV_64F);
    
    // Use existing displacement as initial guess if available
    if (poi->hasDisplacement()) {
        warpParams.at<double>(0, 0) = poi->getDisplacementU();
        warpParams.at<double>(1, 0) = poi->getDisplacementV();
    }
    
    double zncc = 0.0;
    
    // Perform initial guess if no displacement provided
    if (!poi->hasDisplacement()) {
        bool initialSuccess = optimizer_->initialGuess(refPoint, warpParams, zncc);
        if (!initialSuccess) {
            poi->setValid(false);
            return false;
        }
    }
    
    // Perform ICGN optimization
    bool optimizeSuccess = optimizer_->optimize(refPoint, warpParams, zncc);
    
    // Extract results
    double u = warpParams.at<double>(0, 0);
    double v = warpParams.at<double>(1, 0);
    
    // Set POI data
    poi->setDisplacement(u, v);
    poi->calculateDeformedFromDisplacement();
    poi->setZNCC(zncc);
    poi->setShapeParameters(warpParams);
    poi->setConverged(optimizeSuccess);
    
    // Validate result
    bool isValid = optimizeSuccess && 
                   zncc >= config_.ccThreshold &&
                   std::abs(u) < 50.0 && std::abs(v) < 50.0;  // Reasonable displacement bounds
    
    poi->setValid(isValid);
    
    return isValid;
}

std::unique_ptr<POIManager> DICProcessor::processROI(const cv::Mat& refImage,
                                                    const cv::Mat& defImage, 
                                                    const cv::Mat& roi) {
    std::unique_ptr<POIManager> poiManager(new POIManager());
    
    // Generate POIs from ROI
    poiManager->generatePOIsFromROI(roi, config_.neighborStep);
    
    // Process the POIs
    processPOIs(refImage, defImage, *poiManager);
    
    return poiManager;
}

std::shared_ptr<POI> DICProcessor::findAndProcessSeedPoint(const cv::Mat& refImage,
                                                          const cv::Mat& defImage,
                                                          const cv::Mat& roi) {
    // Calculate SDA for seed point selection
    cv::Mat sda = calculateSDA(roi);
    
    // Find seed point
    cv::Point seedPoint = findSeedPoint(roi, sda);
    
    // Create seed POI
    auto seedPOI = std::make_shared<POI>(static_cast<double>(seedPoint.x), 
                                        static_cast<double>(seedPoint.y));
    
    // Initialize optimizer
    initializeOptimizer(refImage, defImage);
    
    // Process seed POI
    if (processSinglePOI(refImage, defImage, seedPOI)) {
        return seedPOI;
    }
    
    return nullptr;
}

void DICProcessor::applyQualityControl(POIManager& poiManager) {
    // Filter by ZNCC threshold
    poiManager.filterByZNCC(config_.ccThreshold);
    
    // Filter non-converged points
    poiManager.filterByConvergence();
    
    // Filter displacement outliers
    filterDisplacementOutliers(poiManager);
    
    // Filter invalid points
    poiManager.filterInvalid();
}

bool DICProcessor::validatePOI(std::shared_ptr<POI> poi, 
                              const std::vector<std::shared_ptr<POI>>& neighbors) {
    if (!poi->isValid() || !poi->hasDisplacement()) {
        return false;
    }
    
    // Check against neighbors for continuity
    if (neighbors.empty()) {
        return true;  // No neighbors to check against
    }
    
    double poiU = poi->getDisplacementU();
    double poiV = poi->getDisplacementV();
    
    // Calculate average neighbor displacement
    double avgU = 0.0, avgV = 0.0;
    int validNeighbors = 0;
    
    for (const auto& neighbor : neighbors) {
        if (neighbor->isValid() && neighbor->hasDisplacement()) {
            avgU += neighbor->getDisplacementU();
            avgV += neighbor->getDisplacementV();
            validNeighbors++;
        }
    }
    
    if (validNeighbors == 0) {
        return true;  // No valid neighbors
    }
    
    avgU /= validNeighbors;
    avgV /= validNeighbors;
    
    // Check displacement jump
    double deltaU = poiU - avgU;
    double deltaV = poiV - avgV;
    double displacement_jump = std::sqrt(deltaU * deltaU + deltaV * deltaV);
    
    return displacement_jump <= config_.deltaDispThreshold;
}

DICProcessor::Statistics DICProcessor::calculateStatistics(const POIManager& poiManager) {
    Statistics stats;
    
    stats.totalPoints = poiManager.size();
    stats.validPoints = poiManager.getValidCount();
    stats.convergedPoints = poiManager.getConvergedCount();
    stats.averageZNCC = poiManager.getAverageZNCC();
    stats.meanDisplacement = poiManager.getMeanDisplacement();
    stats.stdDisplacement = poiManager.getStdDisplacement();
    
    if (stats.totalPoints > 0) {
        stats.coverageRatio = static_cast<double>(stats.validPoints) / stats.totalPoints;
    }
    
    // Calculate average iterations
    double totalIterations = 0.0;
    int iterationCount = 0;
    for (size_t i = 0; i < poiManager.size(); ++i) {
        auto poi = poiManager.getPOI(i);
        if (poi->isValid()) {
            totalIterations += poi->getIterationCount();
            iterationCount++;
        }
    }
    
    if (iterationCount > 0) {
        stats.averageIterations = totalIterations / iterationCount;
    }
    
    return stats;
}

cv::Vec2f DICProcessor::generateInitialGuess(std::shared_ptr<POI> poi,
                                            const std::vector<std::shared_ptr<POI>>& processedPOIs) {
    if (processedPOIs.empty()) {
        return cv::Vec2f(0.0f, 0.0f);
    }
    
    double poiX = poi->getReferenceX();
    double poiY = poi->getReferenceY();
    
    // Find nearest processed POIs
    std::vector<std::pair<double, cv::Vec2f>> weightedDisplacements;
    
    for (const auto& processedPOI : processedPOIs) {
        if (!processedPOI->hasDisplacement()) continue;
        
        double dx = processedPOI->getReferenceX() - poiX;
        double dy = processedPOI->getReferenceY() - poiY;
        double distance = std::sqrt(dx * dx + dy * dy);
        
        if (distance < 50.0) {  // Only consider nearby POIs
            double weight = 1.0 / (distance + 1.0);  // Inverse distance weighting
            cv::Vec2f displacement(processedPOI->getDisplacementU(), 
                                  processedPOI->getDisplacementV());
            weightedDisplacements.push_back({weight, displacement});
        }
    }
    
    if (weightedDisplacements.empty()) {
        return cv::Vec2f(0.0f, 0.0f);
    }
    
    // Calculate weighted average
    double totalWeight = 0.0;
    cv::Vec2f weightedSum(0.0f, 0.0f);
    
    for (const auto& pair : weightedDisplacements) {
        totalWeight += pair.first;
        weightedSum += pair.first * pair.second;
    }
    
    if (totalWeight > 0.0) {
        return weightedSum / totalWeight;
    }
    
    return cv::Vec2f(0.0f, 0.0f);
}

cv::Mat DICProcessor::calculateSDA(const cv::Mat& roi) {
    cv::Mat sda;
    cv::distanceTransform(roi, sda, cv::DIST_L2, 3);
    return sda;
}

cv::Point DICProcessor::findSeedPoint(const cv::Mat& roi, const cv::Mat& sda) {
    cv::Point maxLoc;
    cv::minMaxLoc(sda, nullptr, nullptr, nullptr, &maxLoc);
    return maxLoc;
}

void DICProcessor::initializeOptimizer(const cv::Mat& refImage, const cv::Mat& defImage) {
    optimizer_ = std::unique_ptr<ICGNOptimizer>(new ICGNOptimizer(
        refImage, defImage,
        config_.subsetRadius, config_.order,
        config_.convergenceThreshold, config_.maxIterations
    ));
}

void DICProcessor::updateProgress(size_t current, size_t total, const std::string& message) {
    if (progressCallback_) {
        progressCallback_(current, total, message);
    }
}

void DICProcessor::filterDisplacementOutliers(POIManager& poiManager) {
    poiManager.filterByDisplacementJump(config_.deltaDispThreshold);
}

std::vector<size_t> DICProcessor::setupProcessingOrder(const POIManager& poiManager,
                                                      std::shared_ptr<POI> seedPOI) {
    std::vector<size_t> order;
    if (poiManager.empty()) return order;
    
    // Simple approach: start with seed, then process by distance
    // In a full implementation, this would use a priority queue based on reliability
    
    std::vector<std::pair<double, size_t>> distanceIndexPairs;
    
    for (size_t i = 0; i < poiManager.size(); ++i) {
        auto poi = poiManager.getPOI(i);
        if (poi == seedPOI) {
            order.push_back(i);  // Seed first
        } else if (seedPOI) {
            double distance = poi->distanceTo(*seedPOI);
            distanceIndexPairs.push_back({distance, i});
        } else {
            order.push_back(i);  // No seed, just add in order
        }
    }
    
    // Sort by distance and add to order
    std::sort(distanceIndexPairs.begin(), distanceIndexPairs.end());
    for (const auto& pair : distanceIndexPairs) {
        order.push_back(pair.second);
    }
    
    return order;
}

// === CPUDICProcessor Implementation ===

DICProcessor::Statistics CPUDICProcessor::processPOIs(const cv::Mat& refImage, 
                                                     const cv::Mat& defImage,
                                                     POIManager& poiManager) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Initialize optimizer
    initializeOptimizer(refImage, defImage);
    
    Statistics stats;
    stats.totalPoints = poiManager.size();
    
    if (stats.totalPoints == 0) {
        return stats;
    }
    
    updateProgress(0, stats.totalPoints, "Starting CPU DIC processing...");
    
    // Determine number of threads
    int numThreads = 1;
#ifdef _OPENMP
    if (config_.enableParallelProcessing) {
        numThreads = omp_get_max_threads();
    }
#endif
    
    // Process POIs in parallel batches
    const size_t batchSize = std::max(static_cast<size_t>(1), stats.totalPoints / numThreads);
    
    for (size_t startIdx = 0; startIdx < stats.totalPoints; startIdx += batchSize) {
        size_t endIdx = std::min(startIdx + batchSize, stats.totalPoints);
        
        std::vector<size_t> indices;
        for (size_t i = startIdx; i < endIdx; ++i) {
            indices.push_back(i);
        }
        
#ifdef _OPENMP
        if (config_.enableParallelProcessing) {
            #pragma omp parallel
            {
                int threadId = omp_get_thread_num();
                size_t threadBatchSize = indices.size() / numThreads;
                size_t threadStart = threadId * threadBatchSize;
                size_t threadEnd = (threadId == numThreads - 1) ? indices.size() : threadStart + threadBatchSize;
                
                processPointsBatch(refImage, defImage, poiManager, indices, threadStart, threadEnd);
            }
        } else {
            processPointsBatch(refImage, defImage, poiManager, indices, 0, indices.size());
        }
#else
        processPointsBatch(refImage, defImage, poiManager, indices, 0, indices.size());
#endif
        
        updateProgress(endIdx, stats.totalPoints, 
                      "Processed batch " + std::to_string(endIdx) + "/" + std::to_string(stats.totalPoints));
    }
    
    // Apply quality control
    if (config_.enableQualityControl) {
        applyQualityControl(poiManager);
        updateProgress(stats.totalPoints, stats.totalPoints, "Quality control applied");
    }
    
    // Calculate final statistics
    stats = calculateStatistics(poiManager);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    stats.processingTime = duration.count() / 1000.0;
    
    lastStats_ = stats;
    return stats;
}

void CPUDICProcessor::processPointsBatch(const cv::Mat& refImage,
                                        const cv::Mat& defImage,
                                        POIManager& poiManager,
                                        const std::vector<size_t>& indices,
                                        size_t startIdx, size_t endIdx) {
    // Each thread needs its own optimizer to avoid conflicts
    ICGNOptimizer threadOptimizer(refImage, defImage,
                                 config_.subsetRadius, config_.order,
                                 config_.convergenceThreshold, config_.maxIterations);
    
    for (size_t i = startIdx; i < endIdx; ++i) {
        if (i >= indices.size()) break;
        
        auto poi = poiManager.getPOI(indices[i]);
        
        cv::Point refPoint(static_cast<int>(poi->getReferenceX()), 
                          static_cast<int>(poi->getReferenceY()));
        
        // Check bounds
        int margin = config_.subsetRadius + 1;
        if (refPoint.x < margin || refPoint.y < margin ||
            refPoint.x >= refImage.cols - margin || 
            refPoint.y >= refImage.rows - margin) {
            poi->setValid(false);
            continue;
        }
        
        // Initialize warp parameters
        cv::Mat warpParams = cv::Mat::zeros(threadOptimizer.getNumParams(), 1, CV_64F);
        double zncc = 0.0;
        
        // Perform initial guess
        bool initialSuccess = threadOptimizer.initialGuess(refPoint, warpParams, zncc);
        if (!initialSuccess) {
            poi->setValid(false);
            continue;
        }
        
        // Perform ICGN optimization
        bool optimizeSuccess = threadOptimizer.optimize(refPoint, warpParams, zncc);
        
        // Extract results
        double u = warpParams.at<double>(0, 0);
        double v = warpParams.at<double>(1, 0);
        
        // Set POI data
        poi->setDisplacement(u, v);
        poi->calculateDeformedFromDisplacement();
        poi->setZNCC(zncc);
        poi->setShapeParameters(warpParams);
        poi->setConverged(optimizeSuccess);
        
        // Validate result
        bool isValid = optimizeSuccess && 
                       zncc >= config_.ccThreshold &&
                       std::abs(u) < 50.0 && std::abs(v) < 50.0;
        
        poi->setValid(isValid);
    }
}

// === DICProcessorFactory Implementation ===

std::unique_ptr<DICProcessor> DICProcessorFactory::create(ProcessorType type, 
                                                         const DICProcessor::Config& config) {
    switch (type) {
        case CPU_PROCESSOR:
            return std::unique_ptr<DICProcessor>(new CPUDICProcessor(config));
        case CUDA_PROCESSOR:
            // TODO: Implement CUDA processor
            throw std::runtime_error("CUDA processor not yet implemented");
        default:
            throw std::invalid_argument("Unknown processor type");
    }
}

std::unique_ptr<DICProcessor> DICProcessorFactory::createOptimal(const DICProcessor::Config& config) {
    // For now, always return CPU processor
    // In the future, check for CUDA availability
    return create(CPU_PROCESSOR, config);
}