#include <iostream>
#include <opencv2/opencv.hpp>
#include "rgdic_poi_adapter.h"
#include "common_functions.h"

/**
 * @brief Example demonstrating the modular POI-based RGDIC system
 */
int main(int argc, char** argv) {
    std::cout << "=== RGDIC POI Modular Demo ===" << std::endl;
    
    // Print system capabilities
    std::cout << std::endl << RGDICFactory::getSystemCapabilities() << std::endl;
    
    // Configuration flags
    bool useSyntheticImages = true;
    bool useFirstOrderShapeFunction = false;
    bool useManualROI = false;  // Use automatic ROI for demo
    bool demonstratePOIFeatures = true;
    
    cv::Mat refImage, defImage;
    cv::Mat trueDispX, trueDispY;
    
    // Load or generate images
    if (useSyntheticImages) {
        std::cout << "Generating synthetic speckle pattern images..." << std::endl;
        generateSyntheticImages(refImage, defImage, trueDispX, trueDispY, 400, 300);  // Smaller for demo
        
        cv::imwrite("reference.png", refImage);
        cv::imwrite("deformed.png", defImage);
        
        std::cout << "Generated images: " << refImage.size() << std::endl;
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
    }
    
    // Create ROI
    cv::Mat roi;
    if (useManualROI) {
        roi = createManualROI(refImage);
    } else {
        // Create automatic ROI (exclude border regions)
        int borderWidth = 30;
        roi = cv::Mat::ones(refImage.size(), CV_8UC1);
        cv::rectangle(roi, cv::Point(0, 0), cv::Point(roi.cols-1, roi.rows-1), 0, borderWidth);
        std::cout << "Created automatic ROI with " << borderWidth << " pixel border" << std::endl;
    }
    
    // === Demonstrate POI-based Processing ===
    
    std::cout << std::endl << "=== POI-Based Processing Demo ===" << std::endl;
    
    // Create POI-based RGDIC with factory
    RGDICFactory::FactoryConfig factoryConfig;
    factoryConfig.type = RGDICFactory::POI_CPU;
    factoryConfig.enablePOI = true;
    factoryConfig.enableEnhancedIO = true;
    factoryConfig.enableStatistics = true;
    
    ShapeFunctionOrder order = useFirstOrderShapeFunction ? FIRST_ORDER : SECOND_ORDER;
    
    auto dicPOI = RGDICFactory::create(factoryConfig, 15, 0.00001, 30, 0.2, 1.0, order, 8);
    
    // Cast to POI adapter to access enhanced features
    auto poiAdapter = dynamic_cast<RGDICPOIAdapter*>(dicPOI.get());
    if (!poiAdapter) {
        std::cerr << "Failed to create POI adapter!" << std::endl;
        return -1;
    }
    
    // Set up progress callback
    poiAdapter->setProgressCallback([](size_t current, size_t total, const std::string& message) {
        if (current % 50 == 0 || current == total) {  // Don't spam console
            std::cout << "Progress: [" << current << "/" << total << "] " << message << std::endl;
        }
    });
    
    // Measure execution time
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Run POI-based RGDIC
    auto poiResult = dicPOI->compute(refImage, defImage, roi);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    std::cout << "POI-based computation completed in " << duration.count() << " ms" << std::endl;
    
    // === Demonstrate POI Features ===
    
    if (demonstratePOIFeatures) {
        std::cout << std::endl << "=== POI Feature Demonstration ===" << std::endl;
        
        // Get POI manager
        auto poiManager = poiAdapter->getPOIManager();
        if (poiManager) {
            std::cout << "POI Manager Statistics:" << std::endl;
            std::cout << "  Total POIs: " << poiManager->size() << std::endl;
            std::cout << "  Valid POIs: " << poiManager->getValidCount() << std::endl;
            std::cout << "  Converged POIs: " << poiManager->getConvergedCount() << std::endl;
            std::cout << "  Average ZNCC: " << poiManager->getAverageZNCC() << std::endl;
            
            cv::Scalar meanDisp = poiManager->getMeanDisplacement();
            cv::Scalar stdDisp = poiManager->getStdDisplacement();
            std::cout << "  Mean Displacement: (" << meanDisp[0] << ", " << meanDisp[1] << ")" << std::endl;
            std::cout << "  Std Displacement: (" << stdDisp[0] << ", " << stdDisp[1] << ")" << std::endl;
            
            // Export POI data in different formats
            std::cout << std::endl << "Exporting POI data..." << std::endl;
            
            // Enhanced CSV export
            if (poiAdapter->exportEnhancedCSV("./result/poi_enhanced_results.csv", true)) {
                std::cout << "  Enhanced CSV exported successfully" << std::endl;
            }
            
            // JSON export
            if (poiAdapter->exportJSON("./result/poi_results.json")) {
                std::cout << "  JSON exported successfully" << std::endl;
            }
            
            // Statistics export
            if (poiAdapter->exportStatistics("./result/poi_processing_stats.csv")) {
                std::cout << "  Processing statistics exported successfully" << std::endl;
            }
            
            // Demonstrate POI access
            std::cout << std::endl << "Sample POI data (first 5 valid POIs):" << std::endl;
            int count = 0;
            for (size_t i = 0; i < poiManager->size() && count < 5; ++i) {
                auto poi = poiManager->getPOI(i);
                if (poi->isValid()) {
                    std::cout << "  POI " << count + 1 << ": "
                              << "Ref(" << poi->getReferenceX() << ", " << poi->getReferenceY() << ") -> "
                              << "Def(" << poi->getDeformedX() << ", " << poi->getDeformedY() << "), "
                              << "Disp(" << poi->getDisplacementU() << ", " << poi->getDisplacementV() << "), "
                              << "ZNCC=" << poi->getZNCC() << std::endl;
                    count++;
                }
            }
            
            // Demonstrate neighbor search
            auto centerPOI = poiManager->findNearestPOI(refImage.cols / 2.0, refImage.rows / 2.0);
            if (centerPOI) {
                auto nearbyPOIs = poiManager->findPOIsInRadius(centerPOI->getReferenceX(), 
                                                              centerPOI->getReferenceY(), 20.0);
                std::cout << std::endl << "POIs within 20 pixels of center: " << nearbyPOIs.size() << std::endl;
            }
        }
        
        // Get processing statistics
        auto stats = poiAdapter->getProcessingStatistics();
        std::cout << std::endl << "Detailed Processing Statistics:" << std::endl;
        std::cout << "  Processing time: " << stats.processingTime << " seconds" << std::endl;
        std::cout << "  Coverage ratio: " << (stats.coverageRatio * 100.0) << "%" << std::endl;
        std::cout << "  Average iterations: " << stats.averageIterations << std::endl;
    }
    
    // === Compare with Legacy Implementation ===
    
    std::cout << std::endl << "=== Legacy vs POI Comparison ===" << std::endl;
    
    // Run legacy implementation for comparison
    auto dicLegacy = createRGDIC(false, 15, 0.00001, 30, 0.2, 1.0, order, 8);
    
    auto legacyStartTime = std::chrono::high_resolution_clock::now();
    auto legacyResult = dicLegacy->compute(refImage, defImage, roi);
    auto legacyEndTime = std::chrono::high_resolution_clock::now();
    auto legacyDuration = std::chrono::duration_cast<std::chrono::milliseconds>(legacyEndTime - legacyStartTime);
    
    std::cout << "Legacy computation completed in " << legacyDuration.count() << " ms" << std::endl;
    
    // Export comparison
    if (RGDICMigrationHelper::exportResultComparison(legacyResult, poiResult, "./result/legacy_vs_poi_comparison.csv")) {
        std::cout << "Result comparison exported successfully" << std::endl;
    }
    
    // Validate results
    auto validationStats = RGDICMigrationHelper::validatePOIResults(legacyResult, poiResult, 1e-3);
    std::cout << "Validation results (tolerance=1e-3):" << std::endl;
    std::cout << "  Points within tolerance: " << validationStats.validPoints 
              << "/" << validationStats.totalPoints 
              << " (" << (validationStats.coverageRatio * 100.0) << "%)" << std::endl;
    std::cout << "  Mean difference: (" << validationStats.meanDisplacement[0] 
              << ", " << validationStats.meanDisplacement[1] << ")" << std::endl;
    std::cout << "  Std difference: (" << validationStats.stdDisplacement[0] 
              << ", " << validationStats.stdDisplacement[1] << ")" << std::endl;
    
    // === Traditional Result Processing (for compatibility) ===
    
    std::cout << std::endl << "Saving traditional visualization results..." << std::endl;
    processAndSaveResults(refImage, defImage, trueDispX, trueDispY,
                         poiResult.u, poiResult.v, poiResult.validMask, useSyntheticImages);
    
    // === Summary ===
    
    std::cout << std::endl << "=== Demo Summary ===" << std::endl;
    std::cout << "✓ POI-based modular processing demonstrated" << std::endl;
    std::cout << "✓ Enhanced data export formats (CSV, JSON)" << std::endl;
    std::cout << "✓ Processing statistics and quality metrics" << std::endl;
    std::cout << "✓ Backward compatibility with legacy interface" << std::endl;
    std::cout << "✓ Pixel coordinate tracking for left/right images" << std::endl;
    std::cout << "✓ Result validation and comparison tools" << std::endl;
    
    std::cout << std::endl << "Check the ./result/ directory for all exported files." << std::endl;
    std::cout << "Demo completed successfully!" << std::endl;
    
    return 0;
}