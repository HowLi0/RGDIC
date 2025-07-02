#include <iostream>
#include <opencv2/opencv.hpp>
#include "rgdic.h"
#include "poi.h"

void testPOIFunctionality() {
    std::cout << "=== Testing POI Functionality ===" << std::endl;
    
    // Create a simple test case
    cv::Size imageSize(100, 100);
    
    // Create a displacement result with some test data
    RGDIC::DisplacementResult result;
    result.u = cv::Mat::zeros(imageSize, CV_64F);
    result.v = cv::Mat::zeros(imageSize, CV_64F);
    result.cc = cv::Mat::ones(imageSize, CV_64F) * 0.5; // Good correlation
    result.validMask = cv::Mat::zeros(imageSize, CV_8UC1);
    
    // Add some test points
    for (int i = 10; i < 90; i += 10) {
        for (int j = 10; j < 90; j += 10) {
            result.u.at<double>(j, i) = i * 0.1; // Small displacement in x
            result.v.at<double>(j, i) = j * 0.05; // Small displacement in y
            result.cc.at<double>(j, i) = 0.3; // Good correlation
            result.validMask.at<uchar>(j, i) = 255;
        }
    }
    
    // Create a simple ROI
    cv::Mat roi = cv::Mat::ones(imageSize, CV_8UC1);
    
    // Test conversion to POIs
    std::cout << "Converting matrix to POIs..." << std::endl;
    result.convertMatrixToPOIs(roi);
    
    std::cout << "Number of POIs created: " << result.pois.size() << std::endl;
    std::cout << "Valid POIs: " << result.pois.getValidCount() << std::endl;
    
    // Test POI properties
    if (!result.pois.empty()) {
        const auto& firstPOI = result.pois.pois[0];
        std::cout << "First POI:" << std::endl;
        std::cout << "  Left coord: (" << firstPOI.leftCoord.x << ", " << firstPOI.leftCoord.y << ")" << std::endl;
        std::cout << "  Right coord: (" << firstPOI.rightCoord.x << ", " << firstPOI.rightCoord.y << ")" << std::endl;
        std::cout << "  Displacement: (" << firstPOI.displacement[0] << ", " << firstPOI.displacement[1] << ")" << std::endl;
        std::cout << "  Correlation: " << firstPOI.correlation << std::endl;
        std::cout << "  Valid: " << (firstPOI.valid ? "true" : "false") << std::endl;
    }
    
    // Test CSV export
    std::cout << "Exporting to CSV..." << std::endl;
    if (!result.pois.empty()) {
        result.exportToCSV("test_poi_output.csv");
    }
    
    // Test conversion back to matrix
    std::cout << "Converting POIs back to matrix..." << std::endl;
    RGDIC::DisplacementResult result2;
    result2.pois = result.pois;
    result2.convertPOIsToMatrix(imageSize);
    
    // Verify conversion
    int validCount1 = cv::countNonZero(result.validMask);
    int validCount2 = cv::countNonZero(result2.validMask);
    std::cout << "Original valid count: " << validCount1 << std::endl;
    std::cout << "Converted valid count: " << validCount2 << std::endl;
    
    std::cout << "POI functionality test completed!" << std::endl;
}

int main() {
    testPOIFunctionality();
    return 0;
}