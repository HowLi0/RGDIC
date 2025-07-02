#include <iostream>
#include <opencv2/opencv.hpp>
#include "rgdic.h"
#include "poi.h"
#include "common_functions.h"

int main() {
    std::cout << "=== RGDIC with POI Support Demo ===" << std::endl;
    
    // Generate synthetic test images
    cv::Mat refImage, defImage, trueDispX, trueDispY;
    generateSyntheticImages(refImage, defImage, trueDispX, trueDispY, 200, 200);
    
    std::cout << "Generated synthetic images: " << refImage.size() << std::endl;
    
    // Create ROI
    cv::Mat roi = cv::Mat::ones(refImage.size(), CV_8UC1) * 255;
    cv::Rect roiRect(50, 50, 100, 100);
    roi = cv::Mat::zeros(refImage.size(), CV_8UC1);
    roi(roiRect) = 255;
    
    std::cout << "Created ROI: " << roiRect << std::endl;
    
    // Create RGDIC instance
    auto dic = std::make_unique<RGDIC>();
    
    std::cout << "Computing displacement field..." << std::endl;
    auto result = dic->compute(refImage, defImage, roi);
    
    std::cout << "Computation complete. Valid points: " << cv::countNonZero(result.validMask) << std::endl;
    
    // Enable POI mode and convert data
    std::cout << "\nConverting to POI format..." << std::endl;
    result.enablePOIs(true);
    result.convertMatrixToPOIs();
    
    std::cout << "POI conversion complete." << std::endl;
    std::cout << "Total POIs: " << result.pois.size() << std::endl;
    std::cout << "Valid POIs: " << result.pois.getValidCount() << std::endl;
    
    // Filter high-quality POIs
    auto goodPOIs = result.pois.filterByCorrelation(0.8);
    std::cout << "High quality POIs (correlation >= 0.8): " << goodPOIs.size() << std::endl;
    
    // Export results
    std::cout << "\nExporting results..." << std::endl;
    
    bool csvSuccess = goodPOIs.exportToCSV("poi_results.csv");
    bool poiSuccess = goodPOIs.exportToPOIFormat("poi_results.poi");
    bool matlabSuccess = goodPOIs.exportToMatlab("poi_results.m");
    
    std::cout << "CSV export: " << (csvSuccess ? "Success" : "Failed") << std::endl;
    std::cout << "POI format export: " << (poiSuccess ? "Success" : "Failed") << std::endl;
    std::cout << "MATLAB export: " << (matlabSuccess ? "Success" : "Failed") << std::endl;
    
    // Test data roundtrip
    std::cout << "\nTesting data roundtrip..." << std::endl;
    
    rgdic::POICollection importedPOIs;
    bool importSuccess = importedPOIs.importFromCSV("poi_results.csv");
    
    if (importSuccess) {
        std::cout << "Import successful. Imported " << importedPOIs.size() << " POIs" << std::endl;
        
        // Convert back to matrices
        cv::Mat newU, newV, newCC, newMask;
        importedPOIs.convertToMatrices(newU, newV, newCC, newMask);
        
        std::cout << "Converted back to matrices: " << newU.size() << std::endl;
        std::cout << "Valid points in new mask: " << cv::countNonZero(newMask) << std::endl;
    }
    
    // Display POI statistics
    dic->displayPOIStatistics(goodPOIs);
    
    // Test POI filtering by region
    std::cout << "\nTesting regional filtering..." << std::endl;
    cv::Rect filterRegion(60, 60, 40, 40);
    auto regionalPOIs = goodPOIs.filterByRegion(filterRegion);
    std::cout << "POIs in region " << filterRegion << ": " << regionalPOIs.size() << std::endl;
    
    std::cout << "\n=== POI Demo Complete ===" << std::endl;
    
    return 0;
}