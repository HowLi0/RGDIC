#include <iostream>
#include <opencv2/opencv.hpp>
#include "rgdic.h"
#include "common_functions.h"

int main() {
    std::cout << "=== RGDIC POI Demonstration ===" << std::endl;
    
    // Generate synthetic images for demonstration
    cv::Mat refImage, defImage, trueDispX, trueDispY;
    generateSyntheticImages(refImage, defImage, trueDispX, trueDispY, 200, 200);
    
    // Create a ROI excluding borders
    cv::Mat roi = cv::Mat::ones(refImage.size(), CV_8UC1);
    cv::rectangle(roi, cv::Point(0, 0), cv::Point(roi.cols-1, roi.rows-1), 0, 20);
    
    // Create RGDIC instance
    auto dic = createRGDIC(false, 10, 0.00001, 30, 0.8, 1.0, SECOND_ORDER, 3);
    
    std::cout << "Running RGDIC computation..." << std::endl;
    auto result = dic->compute(refImage, defImage, roi);
    
    // Show results
    int validPoints = cv::countNonZero(result.validMask);
    int totalRoiPoints = cv::countNonZero(roi);
    double coverage = 100.0 * validPoints / totalRoiPoints;
    
    std::cout << "\n=== Analysis Results ===" << std::endl;
    std::cout << "Valid points: " << validPoints << " of " << totalRoiPoints << std::endl;
    std::cout << "Coverage: " << coverage << "%" << std::endl;
    
    // POI Analysis
    std::cout << "\n=== POI Analysis ===" << std::endl;
    std::cout << "Total POIs: " << result.pois.size() << std::endl;
    std::cout << "Valid POIs: " << result.pois.getValidCount() << std::endl;
    
    // Show first few POIs
    int showCount = std::min(5, static_cast<int>(result.pois.size()));
    for (int i = 0; i < showCount; i++) {
        const auto& poi = result.pois.pois[i];
        std::cout << "POI " << i+1 << ":" << std::endl;
        std::cout << "  Left: (" << poi.leftCoord.x << ", " << poi.leftCoord.y << ")" << std::endl;
        std::cout << "  Right: (" << poi.rightCoord.x << ", " << poi.rightCoord.y << ")" << std::endl;
        std::cout << "  Displacement: (" << poi.displacement[0] << ", " << poi.displacement[1] << ")" << std::endl;
        std::cout << "  Correlation: " << poi.correlation << std::endl;
    }
    
    // Export POI data
    std::cout << "\nExporting POI data..." << std::endl;
    result.exportToCSV("demo_poi_results.csv");
    
    // Test round-trip conversion
    std::cout << "\nTesting round-trip conversion..." << std::endl;
    RGDIC::DisplacementResult result2;
    result2.pois = result.pois;
    result2.convertPOIsToMatrix(refImage.size());
    
    int convertedValidCount = cv::countNonZero(result2.validMask);
    std::cout << "Original valid count: " << validPoints << std::endl;
    std::cout << "Converted valid count: " << convertedValidCount << std::endl;
    std::cout << "Conversion accuracy: " << (100.0 * convertedValidCount / validPoints) << "%" << std::endl;
    
    std::cout << "\nPOI demonstration completed successfully!" << std::endl;
    
    return 0;
}