#include "poi.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>

// POI class implementation

POI::POI() 
    : leftCoord(0, 0), rightCoord(0, 0), displacement(0, 0), 
      correlation(1.0), valid(false) {
}

POI::POI(const cv::Point2f& leftCoord, const cv::Point2f& rightCoord)
    : leftCoord(leftCoord), rightCoord(rightCoord), displacement(0, 0),
      correlation(1.0), valid(false) {
    updateDisplacement();
}

void POI::updateRightCoord() {
    rightCoord.x = leftCoord.x + displacement[0];
    rightCoord.y = leftCoord.y + displacement[1];
}

void POI::updateDisplacement() {
    displacement[0] = rightCoord.x - leftCoord.x;
    displacement[1] = rightCoord.y - leftCoord.y;
}

bool POI::isValid() const {
    return valid && correlation < 1.0; // ZNCC should be less than 1.0 for meaningful correlation
}

double POI::getDisplacementMagnitude() const {
    return std::sqrt(displacement[0] * displacement[0] + displacement[1] * displacement[1]);
}

void POI::setStrain(double exx, double eyy, double exy) {
    strain.exx = exx;
    strain.eyy = eyy;
    strain.exy = exy;
    strain.computed = true;
}

void POI::clearStrain() {
    strain.exx = strain.eyy = strain.exy = 0.0;
    strain.computed = false;
}

// POICollection class implementation

POICollection::POICollection(const std::vector<POI>& pois) : pois(pois) {
}

void POICollection::addPOI(const POI& poi) {
    pois.push_back(poi);
}

size_t POICollection::getValidCount() const {
    size_t count = 0;
    for (const auto& poi : pois) {
        if (poi.isValid()) {
            count++;
        }
    }
    return count;
}

size_t POICollection::getStrainComputedCount() const {
    size_t count = 0;
    for (const auto& poi : pois) {
        if (poi.strain.computed) {
            count++;
        }
    }
    return count;
}

void POICollection::clear() {
    pois.clear();
}

void POICollection::exportToCSV(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    
    // Write header
    file << "left_x,left_y,right_x,right_y,u,v,exx,eyy,exy,zncc" << std::endl;
    
    // Write data
    for (const auto& poi : pois) {
        file << std::fixed << std::setprecision(6);
        file << poi.leftCoord.x << ","
             << poi.leftCoord.y << ","
             << poi.rightCoord.x << ","
             << poi.rightCoord.y << ","
             << poi.displacement[0] << ","
             << poi.displacement[1] << ",";
        
        if (poi.strain.computed) {
            file << poi.strain.exx << ","
                 << poi.strain.eyy << ","
                 << poi.strain.exy << ",";
        } else {
            file << "NaN,NaN,NaN,";
        }
        
        file << poi.correlation << std::endl;
    }
    
    std::cout << "Exported " << pois.size() << " POIs to " << filename << std::endl;
    std::cout << "  Valid POIs: " << getValidCount() << std::endl;
    std::cout << "  POIs with strain: " << getStrainComputedCount() << std::endl;
}

cv::Rect2f POICollection::getBoundingBox() const {
    if (pois.empty()) {
        return cv::Rect2f(0, 0, 0, 0);
    }
    
    float minX = pois[0].leftCoord.x;
    float maxX = pois[0].leftCoord.x;
    float minY = pois[0].leftCoord.y;
    float maxY = pois[0].leftCoord.y;
    
    for (const auto& poi : pois) {
        if (poi.isValid()) {
            minX = std::min(minX, poi.leftCoord.x);
            maxX = std::max(maxX, poi.leftCoord.x);
            minY = std::min(minY, poi.leftCoord.y);
            maxY = std::max(maxY, poi.leftCoord.y);
        }
    }
    
    return cv::Rect2f(minX, minY, maxX - minX, maxY - minY);
}

POICollection POICollection::getValidPOIs() const {
    POICollection validCollection;
    for (const auto& poi : pois) {
        if (poi.isValid()) {
            validCollection.addPOI(poi);
        }
    }
    return validCollection;
}