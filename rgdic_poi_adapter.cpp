#include "rgdic_poi_adapter.h"
#include "common_functions.h"
#include <chrono>
#include <iomanip>

// === RGDICPOIAdapter Implementation ===

RGDICPOIAdapter::RGDICPOIAdapter(int subsetRadius, 
                                double convergenceThreshold,
                                int maxIterations,
                                double ccThreshold,
                                double deltaDispThreshold,
                                ShapeFunctionOrder order,
                                int neighborStep)
    : RGDIC(subsetRadius, convergenceThreshold, maxIterations, 
            ccThreshold, deltaDispThreshold, order, neighborStep)
{
    setupProcessorConfig();
    initializePOISystem();
}

RGDIC::DisplacementResult RGDICPOIAdapter::compute(const cv::Mat& refImage, 
                                                   const cv::Mat& defImage,
                                                   const cv::Mat& roi) {
    // Create new POI manager for this computation
    poiManager_ = std::make_shared<POIManager>();
    
    // Generate POIs from ROI
    poiManager_->generatePOIsFromROI(roi, processorConfig_.neighborStep);
    
    // Set up progress callback if available
    if (progressCallback_) {
        dicProcessor_->setProgressCallback(progressCallback_);
    }
    
    // Process POIs using DIC processor
    auto stats = dicProcessor_->processPOIs(refImage, defImage, *poiManager_);
    
    std::cout << "POI Processing completed:" << std::endl;
    std::cout << "  Total POIs: " << stats.totalPoints << std::endl;
    std::cout << "  Valid POIs: " << stats.validPoints << std::endl;
    std::cout << "  Converged POIs: " << stats.convergedPoints << std::endl;
    std::cout << "  Average ZNCC: " << stats.averageZNCC << std::endl;
    std::cout << "  Processing time: " << stats.processingTime << " seconds" << std::endl;
    std::cout << "  Coverage: " << (stats.coverageRatio * 100.0) << "%" << std::endl;
    
    // Convert POI results to traditional DisplacementResult format
    return convertToDisplacementResult(refImage.size());
}

bool RGDICPOIAdapter::exportEnhancedCSV(const std::string& filename, bool includeStrainData) {
    if (!poiManager_) {
        return false;
    }
    
    auto enhancedIO = std::dynamic_pointer_cast<POIIOEnhanced>(ioInterface_);
    if (!enhancedIO) {
        enhancedIO = std::make_shared<POIIOEnhanced>();
    }
    
    POIIOEnhanced::EnhancedExportOptions options;
    options.includeStrainFields = includeStrainData;
    options.includeInvalid = false;
    options.includeMetadata = true;
    options.includeStatistics = true;
    
    return enhancedIO->exportWithStrainData(*poiManager_, filename, 
                                           POIIOInterface::CSV_FORMAT, options);
}

bool RGDICPOIAdapter::exportJSON(const std::string& filename) {
    if (!poiManager_) {
        return false;
    }
    
    return ioInterface_->exportPOIs(*poiManager_, filename, POIIOInterface::JSON_FORMAT);
}

bool RGDICPOIAdapter::exportStatistics(const std::string& filename) {
    if (!dicProcessor_) {
        return false;
    }
    
    auto enhancedIO = std::dynamic_pointer_cast<POIIOEnhanced>(ioInterface_);
    if (!enhancedIO) {
        enhancedIO = std::shared_ptr<POIIOEnhanced>(new POIIOEnhanced());
    }
    
    auto stats = dicProcessor_->getLastStatistics();
    
    // Convert stats to map for simplified export
    std::map<std::string, double> statsMap;
    statsMap["total_points"] = static_cast<double>(stats.totalPoints);
    statsMap["processed_points"] = static_cast<double>(stats.processedPoints);
    statsMap["valid_points"] = static_cast<double>(stats.validPoints);
    statsMap["converged_points"] = static_cast<double>(stats.convergedPoints);
    statsMap["average_zncc"] = stats.averageZNCC;
    statsMap["processing_time"] = stats.processingTime;
    statsMap["average_iterations"] = stats.averageIterations;
    statsMap["mean_displacement_u"] = stats.meanDisplacement[0];
    statsMap["mean_displacement_v"] = stats.meanDisplacement[1];
    statsMap["std_displacement_u"] = stats.stdDisplacement[0];
    statsMap["std_displacement_v"] = stats.stdDisplacement[1];
    statsMap["coverage_ratio"] = stats.coverageRatio;
    
    return enhancedIO->exportStatistics(filename, statsMap);
}

RGDIC::DisplacementResult RGDICPOIAdapter::importFromPOIFile(const std::string& filename, 
                                                             const cv::Size& imageSize) {
    auto importedPOIManager = ioInterface_->importPOIs(filename, POIIOInterface::CSV_FILE);
    
    if (!importedPOIManager) {
        // Return empty result
        DisplacementResult result;
        result.u = cv::Mat::zeros(imageSize, CV_64F);
        result.v = cv::Mat::zeros(imageSize, CV_64F);
        result.cc = cv::Mat::zeros(imageSize, CV_64F);
        result.validMask = cv::Mat::zeros(imageSize, CV_8U);
        return result;
    }
    
    // Convert unique_ptr to shared_ptr
    poiManager_ = std::shared_ptr<POIManager>(importedPOIManager.release());
    return convertToDisplacementResult(imageSize);
}

void RGDICPOIAdapter::setProgressCallback(std::function<void(size_t, size_t, const std::string&)> callback) {
    progressCallback_ = callback;
    if (dicProcessor_) {
        dicProcessor_->setProgressCallback(callback);
    }
}

DICProcessor::Statistics RGDICPOIAdapter::getProcessingStatistics() const {
    if (dicProcessor_) {
        return dicProcessor_->getLastStatistics();
    }
    return DICProcessor::Statistics();
}

void RGDICPOIAdapter::setParallelProcessing(bool enable) {
    processorConfig_.enableParallelProcessing = enable;
    if (dicProcessor_) {
        dicProcessor_->setConfig(processorConfig_);
    }
}

void RGDICPOIAdapter::setQualityControl(bool enable) {
    processorConfig_.enableQualityControl = enable;
    if (dicProcessor_) {
        dicProcessor_->setConfig(processorConfig_);
    }
}

RGDIC::DisplacementResult RGDICPOIAdapter::convertToDisplacementResult(const cv::Size& imageSize) {
    if (!poiManager_) {
        // Return empty result
        DisplacementResult result;
        result.u = cv::Mat::zeros(imageSize, CV_64F);
        result.v = cv::Mat::zeros(imageSize, CV_64F);
        result.cc = cv::Mat::zeros(imageSize, CV_64F);
        result.validMask = cv::Mat::zeros(imageSize, CV_8U);
        return result;
    }
    
    DisplacementResult result;
    ioInterface_->convertToRGDICResult(*poiManager_, result.u, result.v, 
                                      result.cc, result.validMask, imageSize);
    
    return result;
}

void RGDICPOIAdapter::setupProcessorConfig() {
    processorConfig_.subsetRadius = m_subsetRadius;
    processorConfig_.convergenceThreshold = m_convergenceThreshold;
    processorConfig_.maxIterations = m_maxIterations;
    processorConfig_.ccThreshold = m_ccThreshold;
    processorConfig_.deltaDispThreshold = m_deltaDispThreshold;
    processorConfig_.order = m_order;
    processorConfig_.neighborStep = m_neighborUtils.getStep();
    processorConfig_.enableQualityControl = true;
    processorConfig_.enableParallelProcessing = true;
}

void RGDICPOIAdapter::initializePOISystem() {
    // Create DIC processor
    dicProcessor_ = DICProcessorFactory::createOptimal(processorConfig_);
    
    // Create I/O interface
    ioInterface_ = POIIOFactory::create(POIIOFactory::ENHANCED_IO);
    
    // Setup metadata
    POIIOInterface::Metadata metadata;
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time_t);
    
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    metadata.timestamp = oss.str();
    metadata.software_version = "RGDIC POI v1.0";
    metadata.description = "RGDIC with POI-based modular processing";
    metadata.subsetRadius = m_subsetRadius;
    metadata.convergenceThreshold = m_convergenceThreshold;
    metadata.maxIterations = m_maxIterations;
    metadata.shapeFunctionOrder = (m_order == FIRST_ORDER) ? "first-order" : "second-order";
    
    ioInterface_->setMetadata(metadata);
}

// === Factory Functions ===

std::unique_ptr<RGDIC> createRGDICWithPOI(bool useCuda,
                                          int subsetRadius,
                                          double convergenceThreshold,
                                          int maxIterations,
                                          double ccThreshold,
                                          double deltaDispThreshold,
                                          ShapeFunctionOrder order,
                                          int neighborStep) {
    if (useCuda) {
        // For now, fall back to original CUDA implementation
        // TODO: Implement CUDA POI adapter
        return createRGDIC(true, subsetRadius, convergenceThreshold, maxIterations,
                          ccThreshold, deltaDispThreshold, order, neighborStep);
    } else {
        return std::unique_ptr<RGDIC>(new RGDICPOIAdapter(subsetRadius, convergenceThreshold,
                                                maxIterations, ccThreshold,
                                                deltaDispThreshold, order, neighborStep));
    }
}

// === RGDICFactory Implementation ===

std::unique_ptr<RGDIC> RGDICFactory::create(const FactoryConfig& config,
                                           int subsetRadius,
                                           double convergenceThreshold,
                                           int maxIterations,
                                           double ccThreshold,
                                           double deltaDispThreshold,
                                           ShapeFunctionOrder order,
                                           int neighborStep) {
    ImplementationType actualType = config.type;
    
    if (actualType == AUTO_SELECT) {
        // Simple auto-selection logic
        if (config.enablePOI) {
            actualType = POI_CPU;  // Default to POI CPU for now
        } else {
            actualType = LEGACY_CPU;
        }
    }
    
    switch (actualType) {
        case POI_CPU:
            return std::unique_ptr<RGDIC>(new RGDICPOIAdapter(subsetRadius, convergenceThreshold,
                                                    maxIterations, ccThreshold,
                                                    deltaDispThreshold, order, neighborStep));
        
        case POI_CUDA:
            // TODO: Implement POI CUDA adapter
            std::cout << "POI CUDA not yet implemented, falling back to POI CPU" << std::endl;
            return std::unique_ptr<RGDIC>(new RGDICPOIAdapter(subsetRadius, convergenceThreshold,
                                                    maxIterations, ccThreshold,
                                                    deltaDispThreshold, order, neighborStep));
        
        case LEGACY_CPU:
            return createRGDIC(false, subsetRadius, convergenceThreshold, maxIterations,
                              ccThreshold, deltaDispThreshold, order, neighborStep);
        
        case LEGACY_CUDA:
            return createRGDIC(true, subsetRadius, convergenceThreshold, maxIterations,
                              ccThreshold, deltaDispThreshold, order, neighborStep);
        
        default:
            return std::unique_ptr<RGDIC>(new RGDICPOIAdapter(subsetRadius, convergenceThreshold,
                                                    maxIterations, ccThreshold,
                                                    deltaDispThreshold, order, neighborStep));
    }
}

RGDICFactory::ImplementationType RGDICFactory::getRecommendedImplementation(
    const cv::Size& imageSize,
    const cv::Size& roiSize,
    size_t pointCount) {
    
    // Simple heuristics for implementation selection
    size_t totalPixels = imageSize.width * imageSize.height;
    size_t roiPixels = roiSize.width * roiSize.height;
    
    if (totalPixels > 1000000 || pointCount > 10000) {
        // Large images or many points - prefer CUDA if available
        if (isCudaPOIAvailable()) {
            return POI_CUDA;
        } else {
            return POI_CPU;
        }
    } else {
        // Smaller problems - CPU is fine
        return POI_CPU;
    }
}

bool RGDICFactory::isCudaPOIAvailable() {
    // TODO: Check for CUDA POI implementation availability
    return false;
}

std::string RGDICFactory::getSystemCapabilities() {
    std::ostringstream oss;
    oss << "RGDIC System Capabilities:" << std::endl;
    oss << "  POI CPU: Available" << std::endl;
    oss << "  POI CUDA: " << (isCudaPOIAvailable() ? "Available" : "Not Available") << std::endl;
    oss << "  Legacy CPU: Available" << std::endl;
    oss << "  Legacy CUDA: " << (createRGDIC(true) ? "Available" : "Not Available") << std::endl;
    oss << "  Enhanced I/O: Available" << std::endl;
    oss << "  JSON Export: Available" << std::endl;
    oss << "  Statistics Export: Available" << std::endl;
    
#ifdef _OPENMP
    oss << "  OpenMP: Available" << std::endl;
#else
    oss << "  OpenMP: Not Available" << std::endl;
#endif
    
    return oss.str();
}

// === RGDICMigrationHelper Implementation ===

int RGDICMigrationHelper::convertedMain(int argc, char** argv, bool usePOI) {
    std::cout << "=== RGDIC with POI Support ===" << std::endl;
    
    // Configuration flags (same as original main)
    bool useSyntheticImages = true;
    bool useFirstOrderShapeFunction = false;
    bool useManualROI = true;
    
    cv::Mat refImage, defImage;
    cv::Mat trueDispX, trueDispY;
    
    // Load or generate images (same as original)
    if (useSyntheticImages) {
        std::cout << "Generating synthetic speckle pattern images..." << std::endl;
        generateSyntheticImages(refImage, defImage, trueDispX, trueDispY);
        
        cv::imwrite("reference.png", refImage);
        cv::imwrite("deformed.png", defImage);
        
        cv::imshow("Reference Image", refImage);
        cv::imshow("Deformed Image", defImage);
        cv::waitKey(100);
    } else {
        if (argc < 3) {
            std::cout << "Usage: " << argv[0] << " <reference_image> <deformed_image>" << std::endl;
            return -1;
        }
        
        refImage = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
        defImage = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
        
        if (refImage.empty() || defImage.empty()) {
            std::cerr << "Error loading images!" << std::endl;
            return -1;
        }
        
        cv::imshow("Reference Image", refImage);
        cv::imshow("Deformed Image", defImage);
        cv::waitKey(100);
    }
    
    // Create ROI (same as original)
    cv::Mat roi;
    if (useManualROI) {
        roi = createManualROI(refImage);
    } else {
        int borderWidth = 25;
        roi = cv::Mat::ones(refImage.size(), CV_8UC1);
        cv::rectangle(roi, cv::Point(0, 0), cv::Point(roi.cols-1, roi.rows-1), 0, borderWidth);
    }
    
    // Create RGDIC instance (POI or legacy)
    ShapeFunctionOrder order = useFirstOrderShapeFunction ? FIRST_ORDER : SECOND_ORDER;
    
    std::unique_ptr<RGDIC> dic;
    if (usePOI) {
        std::cout << "Using POI-based RGDIC implementation..." << std::endl;
        dic = createRGDICWithPOI(false, 15, 0.00001, 30, 0.8, 1.0, order, 5);
        
        // Set up progress callback for POI adapter
        auto poiAdapter = dynamic_cast<RGDICPOIAdapter*>(dic.get());
        if (poiAdapter) {
            poiAdapter->setProgressCallback([](size_t current, size_t total, const std::string& msg) {
                std::cout << "Progress: " << current << "/" << total << " - " << msg << std::endl;
            });
        }
    } else {
        std::cout << "Using legacy RGDIC implementation..." << std::endl;
        dic = createRGDIC(false, 15, 0.00001, 30, 0.8, 1.0, order, 5);
    }
    
    // Measure execution time
    double t = (double)cv::getTickCount();
    
    // Run RGDIC algorithm
    auto result = dic->compute(refImage, defImage, roi);
    
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "Computation completed in " << t << " seconds." << std::endl;
    
    // Enhanced export for POI-based results
    if (usePOI) {
        auto poiAdapter = dynamic_cast<RGDICPOIAdapter*>(dic.get());
        if (poiAdapter) {
            // Export enhanced CSV with full POI data
            poiAdapter->exportEnhancedCSV("./result/poi_displacement_results.csv");
            
            // Export JSON format
            poiAdapter->exportJSON("./result/poi_displacement_results.json");
            
            // Export processing statistics
            poiAdapter->exportStatistics("./result/processing_statistics.csv");
            
            std::cout << "Enhanced POI export completed." << std::endl;
        }
    }
    
    // Process and save traditional results (for compatibility)
    processAndSaveResults(refImage, defImage, trueDispX, trueDispY,
                         result.u, result.v, result.validMask, useSyntheticImages);
    
    std::cout << "Press any key to exit..." << std::endl;
    cv::waitKey(0);
    return 0;
}

DICProcessor::Config RGDICMigrationHelper::createConfigFromLegacyParams(
    int subsetRadius,
    double convergenceThreshold,
    int maxIterations,
    double ccThreshold,
    double deltaDispThreshold,
    ShapeFunctionOrder order,
    int neighborStep) {
    
    DICProcessor::Config config;
    config.subsetRadius = subsetRadius;
    config.convergenceThreshold = convergenceThreshold;
    config.maxIterations = maxIterations;
    config.ccThreshold = ccThreshold;
    config.deltaDispThreshold = deltaDispThreshold;
    config.order = order;
    config.neighborStep = neighborStep;
    config.enableQualityControl = true;
    config.enableParallelProcessing = true;
    
    return config;
}

bool RGDICMigrationHelper::exportResultComparison(const RGDIC::DisplacementResult& legacyResult,
                                                  const RGDIC::DisplacementResult& poiResult,
                                                  const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    file << "# RGDIC Result Comparison - Legacy vs POI" << std::endl;
    file << "x,y,legacy_u,legacy_v,legacy_cc,poi_u,poi_v,poi_cc,diff_u,diff_v,diff_cc" << std::endl;
    
    // Compare point by point
    for (int y = 0; y < legacyResult.u.rows; ++y) {
        for (int x = 0; x < legacyResult.u.cols; ++x) {
            bool legacyValid = legacyResult.validMask.at<uchar>(y, x) > 0;
            bool poiValid = poiResult.validMask.at<uchar>(y, x) > 0;
            
            if (legacyValid || poiValid) {
                double legacy_u = legacyValid ? legacyResult.u.at<double>(y, x) : 0.0;
                double legacy_v = legacyValid ? legacyResult.v.at<double>(y, x) : 0.0;
                double legacy_cc = legacyValid ? legacyResult.cc.at<double>(y, x) : 0.0;
                
                double poi_u = poiValid ? poiResult.u.at<double>(y, x) : 0.0;
                double poi_v = poiValid ? poiResult.v.at<double>(y, x) : 0.0;
                double poi_cc = poiValid ? poiResult.cc.at<double>(y, x) : 0.0;
                
                double diff_u = poi_u - legacy_u;
                double diff_v = poi_v - legacy_v;
                double diff_cc = poi_cc - legacy_cc;
                
                file << x << "," << y << ","
                     << std::fixed << std::setprecision(6)
                     << legacy_u << "," << legacy_v << "," << legacy_cc << ","
                     << poi_u << "," << poi_v << "," << poi_cc << ","
                     << diff_u << "," << diff_v << "," << diff_cc << std::endl;
            }
        }
    }
    
    file.close();
    return true;
}

DICProcessor::Statistics RGDICMigrationHelper::validatePOIResults(
    const RGDIC::DisplacementResult& legacyResult,
    const RGDIC::DisplacementResult& poiResult,
    double tolerance) {
    
    DICProcessor::Statistics stats;
    
    double sumDiffU = 0.0, sumDiffV = 0.0, sumDiffCC = 0.0;
    double sumSqDiffU = 0.0, sumSqDiffV = 0.0, sumSqDiffCC = 0.0;
    size_t validComparisons = 0;
    
    // Compare valid points
    for (int y = 0; y < legacyResult.u.rows; ++y) {
        for (int x = 0; x < legacyResult.u.cols; ++x) {
            bool legacyValid = legacyResult.validMask.at<uchar>(y, x) > 0;
            bool poiValid = poiResult.validMask.at<uchar>(y, x) > 0;
            
            if (legacyValid && poiValid) {
                double legacy_u = legacyResult.u.at<double>(y, x);
                double legacy_v = legacyResult.v.at<double>(y, x);
                double legacy_cc = legacyResult.cc.at<double>(y, x);
                
                double poi_u = poiResult.u.at<double>(y, x);
                double poi_v = poiResult.v.at<double>(y, x);
                double poi_cc = poiResult.cc.at<double>(y, x);
                
                double diff_u = poi_u - legacy_u;
                double diff_v = poi_v - legacy_v;
                double diff_cc = poi_cc - legacy_cc;
                
                sumDiffU += diff_u;
                sumDiffV += diff_v;
                sumDiffCC += diff_cc;
                
                sumSqDiffU += diff_u * diff_u;
                sumSqDiffV += diff_v * diff_v;
                sumSqDiffCC += diff_cc * diff_cc;
                
                validComparisons++;
                
                // Check if differences are within tolerance
                if (std::abs(diff_u) <= tolerance && 
                    std::abs(diff_v) <= tolerance && 
                    std::abs(diff_cc) <= tolerance) {
                    stats.validPoints++;
                }
            }
        }
    }
    
    if (validComparisons > 0) {
        stats.totalPoints = validComparisons;
        stats.meanDisplacement = cv::Scalar(sumDiffU / validComparisons, sumDiffV / validComparisons);
        
        double varU = (sumSqDiffU / validComparisons) - (sumDiffU / validComparisons) * (sumDiffU / validComparisons);
        double varV = (sumSqDiffV / validComparisons) - (sumDiffV / validComparisons) * (sumDiffV / validComparisons);
        stats.stdDisplacement = cv::Scalar(std::sqrt(varU), std::sqrt(varV));
        
        stats.averageZNCC = sumDiffCC / validComparisons;
        stats.coverageRatio = static_cast<double>(stats.validPoints) / stats.totalPoints;
    }
    
    return stats;
}