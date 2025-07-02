#include <iostream>
#include <opencv2/opencv.hpp>
#include "rgdic.h"
#include "poi.h"
#include "common_functions.h"

int main() {
    std::cout << "=== POI Functionality Test ===" << std::endl;
    
    // Test 1: Basic POI creation and manipulation
    std::cout << "\n1. Testing POI class..." << std::endl;
    
    rgdic::POI poi1(cv::Point2f(10.5, 20.3), cv::Point2f(12.1, 21.8), 
                    cv::Vec2f(1.6, 1.5), 0.95, true);
    
    std::cout << "Created POI: " << poi1.toString() << std::endl;
    
    poi1.strain.exx = 0.001;
    poi1.strain.eyy = -0.0005;
    poi1.strain.exy = 0.0002;
    poi1.strain.computed = true;
    
    std::cout << "POI with strain: " << poi1.toString() << std::endl;
    
    // Test 2: POI Collection
    std::cout << "\n2. Testing POI Collection..." << std::endl;
    
    rgdic::POICollection collection(cv::Size(100, 100), "Test Collection");
    
    // Add some test POIs
    for (int i = 0; i < 10; ++i) {
        rgdic::POI poi;
        poi.leftCoord = cv::Point2f(i * 10.0f, i * 5.0f);
        poi.displacement = cv::Vec2f(i * 0.1f, i * 0.05f);
        poi.updateRightCoord();
        poi.correlation = 0.8 + i * 0.02;
        poi.valid = (i % 2 == 0); // Every other POI is valid
        collection.addPOI(poi);
    }
    
    std::cout << "Collection size: " << collection.size() << std::endl;
    std::cout << "Valid POIs: " << collection.getValidCount() << std::endl;
    std::cout << "Mean correlation: " << collection.getMeanCorrelation() << std::endl;
    
    // Test 3: Filtering
    std::cout << "\n3. Testing POI filtering..." << std::endl;
    
    auto highQuality = collection.filterByCorrelation(0.9);
    std::cout << "High quality POIs (corr >= 0.9): " << highQuality.size() << std::endl;
    
    // Test 4: Matrix conversion
    std::cout << "\n4. Testing matrix conversion..." << std::endl;
    
    cv::Mat u, v, cc, validMask;
    collection.convertToMatrices(u, v, cc, validMask);
    
    std::cout << "Generated matrices - U: " << u.size() << ", V: " << v.size() << std::endl;
    std::cout << "Non-zero elements in U: " << cv::countNonZero(u != 0) << std::endl;
    std::cout << "Valid mask sum: " << cv::sum(validMask)[0] << std::endl;
    
    // Test 5: DisplacementResult with POI support
    std::cout << "\n5. Testing DisplacementResult POI integration..." << std::endl;
    
    RGDIC::DisplacementResult result;
    result.u = u.clone();
    result.v = v.clone();
    result.cc = cc.clone();
    result.validMask = validMask.clone();
    
    // Convert to POI format
    result.convertMatrixToPOIs();
    std::cout << "Converted to POI format. POIs count: " << result.pois.size() << std::endl;
    std::cout << "POI mode enabled: " << (result.isPOIsEnabled() ? "Yes" : "No") << std::endl;
    
    // Test 6: Export functionality
    std::cout << "\n6. Testing export functionality..." << std::endl;
    
    bool csvExported = collection.exportToCSV("test_poi_results.csv");
    bool poiExported = collection.exportToPOIFormat("test_poi_results.poi");
    
    std::cout << "CSV export: " << (csvExported ? "Success" : "Failed") << std::endl;
    std::cout << "POI export: " << (poiExported ? "Success" : "Failed") << std::endl;
    
    // Test 7: Import functionality
    std::cout << "\n7. Testing import functionality..." << std::endl;
    
    rgdic::POICollection importedCollection;
    bool csvImported = importedCollection.importFromCSV("test_poi_results.csv");
    
    std::cout << "CSV import: " << (csvImported ? "Success" : "Failed") << std::endl;
    if (csvImported) {
        std::cout << "Imported POIs: " << importedCollection.size() << std::endl;
        std::cout << "Original POIs: " << collection.size() << std::endl;
    }
    
    std::cout << "\n=== POI Test Complete ===" << std::endl;
    
    return 0;
}