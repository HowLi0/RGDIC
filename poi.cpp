#include "poi.h"
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>

namespace rgdic {

// POI class implementation
POI::POI() 
    : leftCoord(0, 0), rightCoord(0, 0), displacement(0, 0), 
      correlation(0.0), valid(false) {
}

POI::POI(const cv::Point2f& left, const cv::Point2f& right, 
         const cv::Vec2f& disp, double corr, bool isValid)
    : leftCoord(left), rightCoord(right), displacement(disp), 
      correlation(corr), valid(isValid) {
}

void POI::updateRightCoord() {
    rightCoord.x = leftCoord.x + displacement[0];
    rightCoord.y = leftCoord.y + displacement[1];
}

std::string POI::toString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << leftCoord.x << "," << leftCoord.y << ","
        << rightCoord.x << "," << rightCoord.y << ","
        << displacement[0] << "," << displacement[1] << ","
        << correlation << "," << (valid ? 1 : 0);
    
    if (strain.computed) {
        oss << "," << strain.exx << "," << strain.eyy << "," << strain.exy;
    }
    
    return oss.str();
}

POI POI::fromString(const std::string& str) {
    std::istringstream iss(str);
    std::string token;
    std::vector<std::string> tokens;
    
    while (std::getline(iss, token, ',')) {
        tokens.push_back(token);
    }
    
    if (tokens.size() < 8) {
        return POI(); // Invalid POI
    }
    
    POI poi;
    poi.leftCoord.x = std::stof(tokens[0]);
    poi.leftCoord.y = std::stof(tokens[1]);
    poi.rightCoord.x = std::stof(tokens[2]);
    poi.rightCoord.y = std::stof(tokens[3]);
    poi.displacement[0] = std::stof(tokens[4]);
    poi.displacement[1] = std::stof(tokens[5]);
    poi.correlation = std::stod(tokens[6]);
    poi.valid = (std::stoi(tokens[7]) != 0);
    
    // Optional strain information
    if (tokens.size() >= 11) {
        poi.strain.exx = std::stod(tokens[8]);
        poi.strain.eyy = std::stod(tokens[9]);
        poi.strain.exy = std::stod(tokens[10]);
        poi.strain.computed = true;
    }
    
    return poi;
}

// POICollection class implementation
POICollection::POICollection() : m_imageSize(0, 0) {
}

POICollection::POICollection(const cv::Size& imageSize, const std::string& desc)
    : m_imageSize(imageSize), m_description(desc) {
}

void POICollection::addPOI(const POI& poi) {
    m_pois.push_back(poi);
}

void POICollection::removePOI(size_t index) {
    if (index < m_pois.size()) {
        m_pois.erase(m_pois.begin() + index);
    }
}

void POICollection::clear() {
    m_pois.clear();
}

POICollection POICollection::filterByCorrelation(double threshold) const {
    POICollection filtered(m_imageSize, m_description);
    
    for (const auto& poi : m_pois) {
        if (poi.valid && poi.correlation >= threshold) {
            filtered.addPOI(poi);
        }
    }
    
    return filtered;
}

POICollection POICollection::filterByRegion(const cv::Rect& region) const {
    POICollection filtered(m_imageSize, m_description);
    
    for (const auto& poi : m_pois) {
        if (region.contains(cv::Point(static_cast<int>(poi.leftCoord.x), 
                                     static_cast<int>(poi.leftCoord.y)))) {
            filtered.addPOI(poi);
        }
    }
    
    return filtered;
}

std::vector<POI> POICollection::getValidPOIs() const {
    std::vector<POI> validPOIs;
    
    for (const auto& poi : m_pois) {
        if (poi.valid) {
            validPOIs.push_back(poi);
        }
    }
    
    return validPOIs;
}

double POICollection::getMeanCorrelation() const {
    if (m_pois.empty()) return 0.0;
    
    double sum = 0.0;
    size_t count = 0;
    
    for (const auto& poi : m_pois) {
        if (poi.valid) {
            sum += poi.correlation;
            count++;
        }
    }
    
    return (count > 0) ? sum / count : 0.0;
}

cv::Vec2f POICollection::getMeanDisplacement() const {
    if (m_pois.empty()) return cv::Vec2f(0, 0);
    
    cv::Vec2f sum(0, 0);
    size_t count = 0;
    
    for (const auto& poi : m_pois) {
        if (poi.valid) {
            sum += poi.displacement;
            count++;
        }
    }
    
    return (count > 0) ? sum / static_cast<float>(count) : cv::Vec2f(0, 0);
}

size_t POICollection::getValidCount() const {
    size_t count = 0;
    for (const auto& poi : m_pois) {
        if (poi.valid) count++;
    }
    return count;
}

bool POICollection::exportToCSV(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    // Write header
    file << "left_x,left_y,right_x,right_y,u,v,zncc,valid";
    
    // Check if any POI has strain information
    bool hasStrain = false;
    for (const auto& poi : m_pois) {
        if (poi.strain.computed) {
            hasStrain = true;
            break;
        }
    }
    
    if (hasStrain) {
        file << ",exx,eyy,exy";
    }
    file << std::endl;
    
    // Write data
    for (const auto& poi : m_pois) {
        file << std::fixed << std::setprecision(6);
        file << poi.leftCoord.x << "," << poi.leftCoord.y << ","
             << poi.rightCoord.x << "," << poi.rightCoord.y << ","
             << poi.displacement[0] << "," << poi.displacement[1] << ","
             << poi.correlation << "," << (poi.valid ? 1 : 0);
        
        if (hasStrain) {
            if (poi.strain.computed) {
                file << "," << poi.strain.exx << "," << poi.strain.eyy << "," << poi.strain.exy;
            } else {
                file << ",0.0,0.0,0.0";
            }
        }
        file << std::endl;
    }
    
    return true;
}

bool POICollection::exportToPOIFormat(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    // Write metadata
    file << "# RGDIC POI Format v1.0" << std::endl;
    file << "# Description: " << m_description << std::endl;
    file << "# Image Size: " << m_imageSize.width << "x" << m_imageSize.height << std::endl;
    file << "# Total POIs: " << m_pois.size() << std::endl;
    file << "# Valid POIs: " << getValidCount() << std::endl;
    file << "#" << std::endl;
    
    // Write each POI
    for (const auto& poi : m_pois) {
        file << poi.toString() << std::endl;
    }
    
    return true;
}

bool POICollection::exportToMatlab(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    file << "% RGDIC POI Results - MATLAB Format" << std::endl;
    file << "% Generated by RGDIC POI system" << std::endl;
    file << std::endl;
    
    // Write arrays
    file << "left_coords = [";
    for (size_t i = 0; i < m_pois.size(); ++i) {
        if (i > 0) file << "; ";
        file << m_pois[i].leftCoord.x << " " << m_pois[i].leftCoord.y;
    }
    file << "];" << std::endl;
    
    file << "displacements = [";
    for (size_t i = 0; i < m_pois.size(); ++i) {
        if (i > 0) file << "; ";
        file << m_pois[i].displacement[0] << " " << m_pois[i].displacement[1];
    }
    file << "];" << std::endl;
    
    file << "correlations = [";
    for (size_t i = 0; i < m_pois.size(); ++i) {
        if (i > 0) file << "; ";
        file << m_pois[i].correlation;
    }
    file << "];" << std::endl;
    
    return true;
}

bool POICollection::importFromCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    std::string line;
    bool first_line = true;
    
    while (std::getline(file, line)) {
        if (first_line) {
            first_line = false;
            continue; // Skip header
        }
        
        if (line.empty() || line[0] == '#') continue;
        
        POI poi = POI::fromString(line);
        if (poi.valid || poi.correlation > 0) { // Basic validity check
            addPOI(poi);
        }
    }
    
    return true;
}

bool POICollection::importFromPOIFormat(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        POI poi = POI::fromString(line);
        addPOI(poi);
    }
    
    return true;
}

void POICollection::convertToMatrices(cv::Mat& u, cv::Mat& v, cv::Mat& cc, cv::Mat& validMask) const {
    if (m_imageSize.width <= 0 || m_imageSize.height <= 0) {
        // Initialize with default size if not set
        cv::Size size(512, 512);
        u = cv::Mat::zeros(size, CV_64F);
        v = cv::Mat::zeros(size, CV_64F);
        cc = cv::Mat::zeros(size, CV_64F);
        validMask = cv::Mat::zeros(size, CV_8UC1);
        return;
    }
    
    u = cv::Mat::zeros(m_imageSize, CV_64F);
    v = cv::Mat::zeros(m_imageSize, CV_64F);
    cc = cv::Mat::zeros(m_imageSize, CV_64F);
    validMask = cv::Mat::zeros(m_imageSize, CV_8UC1);
    
    for (const auto& poi : m_pois) {
        int x = static_cast<int>(std::round(poi.leftCoord.x));
        int y = static_cast<int>(std::round(poi.leftCoord.y));
        
        if (x >= 0 && x < m_imageSize.width && y >= 0 && y < m_imageSize.height) {
            u.at<double>(y, x) = poi.displacement[0];
            v.at<double>(y, x) = poi.displacement[1];
            cc.at<double>(y, x) = poi.correlation;
            validMask.at<uchar>(y, x) = poi.valid ? 255 : 0;
        }
    }
}

void POICollection::convertFromMatrices(const cv::Mat& u, const cv::Mat& v, 
                                       const cv::Mat& cc, const cv::Mat& validMask) {
    clear();
    m_imageSize = u.size();
    
    for (int y = 0; y < u.rows; ++y) {
        for (int x = 0; x < u.cols; ++x) {
            // Only add POIs for points that have valid data or are in ROI
            if (validMask.at<uchar>(y, x) > 0 || 
                std::abs(u.at<double>(y, x)) > 1e-9 || 
                std::abs(v.at<double>(y, x)) > 1e-9) {
                
                POI poi;
                poi.leftCoord = cv::Point2f(static_cast<float>(x), static_cast<float>(y));
                poi.displacement[0] = static_cast<float>(u.at<double>(y, x));
                poi.displacement[1] = static_cast<float>(v.at<double>(y, x));
                poi.updateRightCoord();
                poi.correlation = cc.at<double>(y, x);
                poi.valid = (validMask.at<uchar>(y, x) > 0);
                
                addPOI(poi);
            }
        }
    }
}

} // namespace rgdic