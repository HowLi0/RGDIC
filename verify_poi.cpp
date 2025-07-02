#include <iostream>
#include <opencv2/opencv.hpp>
#include "poi.h"

// Simple test without OpenCV linking - just test the POI functionality
int main() {
    std::cout << "=== Minimal POI Verification Test ===" << std::endl;
    
    // Test 1: Basic POI operations
    rgdic::POI poi1;
    poi1.leftCoord = cv::Point2f(10.5f, 20.3f);
    poi1.displacement = cv::Vec2f(1.6f, 1.5f);
    poi1.updateRightCoord();
    poi1.correlation = 0.95;
    poi1.valid = true;
    
    std::cout << "POI 1 - Left: (" << poi1.leftCoord.x << ", " << poi1.leftCoord.y << ")" << std::endl;
    std::cout << "       Right: (" << poi1.rightCoord.x << ", " << poi1.rightCoord.y << ")" << std::endl;
    std::cout << "       Displacement: (" << poi1.displacement[0] << ", " << poi1.displacement[1] << ")" << std::endl;
    std::cout << "       Correlation: " << poi1.correlation << std::endl;
    
    // Test 2: POI Collection
    rgdic::POICollection collection(cv::Size(100, 100), "Test Collection");
    
    // Create test data
    for (int i = 0; i < 20; ++i) {
        rgdic::POI poi;
        poi.leftCoord = cv::Point2f(i * 2.0f, i * 1.5f);
        poi.displacement = cv::Vec2f(i * 0.1f, i * 0.05f);
        poi.updateRightCoord();
        poi.correlation = 0.7 + (i % 10) * 0.03; // Varies from 0.7 to 0.97
        poi.valid = (i % 3 != 0); // Most are valid
        collection.addPOI(poi);
    }
    
    std::cout << "\nCollection Statistics:" << std::endl;
    std::cout << "  Total POIs: " << collection.size() << std::endl;
    std::cout << "  Valid POIs: " << collection.getValidCount() << std::endl;
    std::cout << "  Mean correlation: " << collection.getMeanCorrelation() << std::endl;
    
    cv::Vec2f meanDisp = collection.getMeanDisplacement();
    std::cout << "  Mean displacement: (" << meanDisp[0] << ", " << meanDisp[1] << ")" << std::endl;
    
    // Test 3: Filtering
    auto highQuality = collection.filterByCorrelation(0.9);
    std::cout << "  High quality POIs (>= 0.9): " << highQuality.size() << std::endl;
    
    auto regionFiltered = collection.filterByRegion(cv::Rect(0, 0, 20, 15));
    std::cout << "  POIs in region (0,0,20,15): " << regionFiltered.size() << std::endl;
    
    // Test 4: Matrix conversion
    cv::Mat u, v, cc, validMask;
    collection.convertToMatrices(u, v, cc, validMask);
    
    std::cout << "\nMatrix Conversion:" << std::endl;
    std::cout << "  Matrix size: " << u.size() << std::endl;
    std::cout << "  Non-zero U elements: " << cv::countNonZero(u != 0) << std::endl;
    std::cout << "  Non-zero V elements: " << cv::countNonZero(v != 0) << std::endl;
    std::cout << "  Valid mask count: " << cv::countNonZero(validMask) << std::endl;
    
    // Test 5: Export/Import
    bool exported = collection.exportToCSV("test_minimal.csv");
    std::cout << "\nExport test: " << (exported ? "Success" : "Failed") << std::endl;
    
    if (exported) {
        rgdic::POICollection imported;
        bool imported_success = imported.importFromCSV("test_minimal.csv");
        std::cout << "Import test: " << (imported_success ? "Success" : "Failed") << std::endl;
        
        if (imported_success) {
            std::cout << "  Imported POIs: " << imported.size() << std::endl;
            std::cout << "  Original POIs: " << collection.size() << std::endl;
            std::cout << "  Valid imported: " << imported.getValidCount() << std::endl;
        }
    }
    
    // Test 6: String serialization
    std::string serialized = poi1.toString();
    std::cout << "\nSerialization test:" << std::endl;
    std::cout << "  Serialized: " << serialized << std::endl;
    
    rgdic::POI deserialized = rgdic::POI::fromString(serialized);
    std::cout << "  Deserialized left coord: (" << deserialized.leftCoord.x 
              << ", " << deserialized.leftCoord.y << ")" << std::endl;
    std::cout << "  Correlation match: " << (std::abs(deserialized.correlation - poi1.correlation) < 1e-6) << std::endl;
    
    std::cout << "\n=== Verification Complete ===" << std::endl;
    
    return 0;
}