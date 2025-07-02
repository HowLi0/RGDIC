#include <iostream>
#include <opencv2/opencv.hpp>
#include "rgdic.h"
#include "common_functions.h"

// Simple strain calculation for demonstration
void calculateStrainForPOIs(RGDIC::DisplacementResult& result) {
    if (!result.hasPOIData()) {
        std::cout << "No POI data available for strain calculation" << std::endl;
        return;
    }
    
    // Simple finite difference strain calculation
    std::vector<POI>& pois = result.pois.pois;
    
    for (size_t i = 0; i < pois.size(); i++) {
        POI& poi = pois[i];
        if (!poi.isValid()) continue;
        
        // Find neighboring POIs for strain calculation
        std::vector<size_t> neighbors;
        const double searchRadius = 10.0;
        
        for (size_t j = 0; j < pois.size(); j++) {
            if (i == j || !pois[j].isValid()) continue;
            
            double dx = poi.leftCoord.x - pois[j].leftCoord.x;
            double dy = poi.leftCoord.y - pois[j].leftCoord.y;
            double dist = std::sqrt(dx*dx + dy*dy);
            
            if (dist < searchRadius) {
                neighbors.push_back(j);
            }
        }
        
        // Simple strain estimation if we have enough neighbors
        if (neighbors.size() >= 4) {
            // Simplified finite difference approximation
            double dudx = 0, dudy = 0, dvdx = 0, dvdy = 0;
            int count = 0;
            
            for (size_t j : neighbors) {
                double dx = pois[j].leftCoord.x - poi.leftCoord.x;
                double dy = pois[j].leftCoord.y - poi.leftCoord.y;
                double du = pois[j].displacement[0] - poi.displacement[0];
                double dv = pois[j].displacement[1] - poi.displacement[1];
                
                if (std::abs(dx) > 1e-6) {
                    dudx += du / dx;
                    dvdx += dv / dx;
                    count++;
                }
                if (std::abs(dy) > 1e-6) {
                    dudy += du / dy;
                    dvdy += dv / dy;
                }
            }
            
            if (count > 0) {
                dudx /= count;
                dudy /= count;
                dvdx /= count;
                dvdy /= count;
                
                // Calculate strain components
                double exx = dudx;
                double eyy = dvdy;
                double exy = 0.5 * (dudy + dvdx);
                
                poi.setStrain(exx, eyy, exy);
            }
        }
    }
    
    std::cout << "Strain calculation completed for " << result.pois.getStrainComputedCount() << " POIs" << std::endl;
}

int main() {
    std::cout << "=== RGDIC POI with Strain Demonstration ===" << std::endl;
    
    // Generate synthetic images 
    cv::Mat refImage, defImage, trueDispX, trueDispY;
    generateSyntheticImages(refImage, defImage, trueDispX, trueDispY, 300, 300);
    
    // Create a ROI excluding borders
    cv::Mat roi = cv::Mat::ones(refImage.size(), CV_8UC1);
    cv::rectangle(roi, cv::Point(0, 0), cv::Point(roi.cols-1, roi.rows-1), 0, 25);
    
    // Use CPU version with larger neighborhood step for better point distribution
    auto dic = createRGDIC(false, 15, 0.00001, 30, 0.8, 1.0, SECOND_ORDER, 3);
    
    std::cout << "Running CPU RGDIC computation..." << std::endl;
    auto result = dic->compute(refImage, defImage, roi);
    
    // Calculate strain for POIs
    std::cout << "Calculating strain for POIs..." << std::endl;
    calculateStrainForPOIs(result);
    
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
    std::cout << "POIs with strain: " << result.pois.getStrainComputedCount() << std::endl;
    
    // Show examples of POIs with strain data
    int strainCount = 0;
    for (const auto& poi : result.pois.pois) {
        if (poi.strain.computed && strainCount < 5) {
            std::cout << "POI with strain " << (strainCount+1) << ":" << std::endl;
            std::cout << "  Position: (" << poi.leftCoord.x << ", " << poi.leftCoord.y << ")" << std::endl;
            std::cout << "  Displacement: (" << poi.displacement[0] << ", " << poi.displacement[1] << ")" << std::endl;
            std::cout << "  Strain: exx=" << poi.strain.exx << ", eyy=" << poi.strain.eyy << ", exy=" << poi.strain.exy << std::endl;
            strainCount++;
        }
    }
    
    // Export complete POI data with strain
    std::cout << "\nExporting complete POI data with strain..." << std::endl;
    result.exportToCSV("complete_poi_with_strain.csv");
    
    // Compare with traditional matrix-based export
    exportToCSV(result.u, result.v, result.validMask, "traditional_matrix_export.csv");
    
    std::cout << "\nDemonstration completed successfully!" << std::endl;
    std::cout << "Files created:" << std::endl;
    std::cout << "  - complete_poi_with_strain.csv (POI format with coordinates and strain)" << std::endl;
    std::cout << "  - traditional_matrix_export.csv (traditional format)" << std::endl;
    
    return 0;
}