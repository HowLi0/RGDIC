#include "poi.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// === POI Implementation ===

POI::POI(double ref_x, double ref_y) 
    : ref_x_(ref_x), ref_y_(ref_y)
    , def_x_(0.0), def_y_(0.0), deformed_coords_set_(false)
    , u_(0.0), v_(0.0), displacement_set_(false)
    , zncc_(0.0), ncc_(0.0), zncc_set_(false), ncc_set_(false)
    , exx_(0.0), eyy_(0.0), exy_(0.0), strain_set_(false)
    , iterations_(0), converged_(false), valid_(false)
    , subset_radius_(15)
{
}

std::string POI::toCSVRow() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    
    // Reference coordinates (left image)
    oss << ref_x_ << "," << ref_y_ << ",";
    
    // Deformed coordinates (right image)
    if (deformed_coords_set_) {
        oss << def_x_ << "," << def_y_ << ",";
    } else {
        oss << ",,";  // Empty values
    }
    
    // Displacement
    if (displacement_set_) {
        oss << u_ << "," << v_ << ",";
    } else {
        oss << ",,";
    }
    
    // Strain components
    if (strain_set_) {
        oss << exx_ << "," << eyy_ << "," << exy_ << ",";
    } else {
        oss << ",,,";
    }
    
    // ZNCC
    if (zncc_set_) {
        oss << zncc_;
    } else {
        oss << "";
    }
    
    return oss.str();
}

std::string POI::getCSVHeader() {
    return "left_x,left_y,right_x,right_y,u,v,exx,eyy,exy,zncc";
}

double POI::distanceTo(const POI& other) const {
    double dx = ref_x_ - other.ref_x_;
    double dy = ref_y_ - other.ref_y_;
    return std::sqrt(dx * dx + dy * dy);
}

void POI::reset() {
    def_x_ = def_y_ = 0.0;
    deformed_coords_set_ = false;
    
    u_ = v_ = 0.0;
    displacement_set_ = false;
    
    zncc_ = ncc_ = 0.0;
    zncc_set_ = ncc_set_ = false;
    
    exx_ = eyy_ = exy_ = 0.0;
    strain_set_ = false;
    
    iterations_ = 0;
    converged_ = false;
    valid_ = false;
    
    shape_params_.release();
    neighbors_.clear();
}

std::unique_ptr<POI> POI::clone() const {
    std::unique_ptr<POI> cloned(new POI(ref_x_, ref_y_));
    
    if (deformed_coords_set_) {
        cloned->setDeformedCoords(def_x_, def_y_);
    }
    
    if (displacement_set_) {
        cloned->setDisplacement(u_, v_);
    }
    
    if (zncc_set_) {
        cloned->setZNCC(zncc_);
    }
    
    if (ncc_set_) {
        cloned->setNCC(ncc_);
    }
    
    if (strain_set_) {
        cloned->setStrainComponents(exx_, eyy_, exy_);
    }
    
    cloned->setIterationCount(iterations_);
    cloned->setConverged(converged_);
    cloned->setValid(valid_);
    cloned->setSubsetRadius(subset_radius_);
    
    if (!shape_params_.empty()) {
        cloned->setShapeParameters(shape_params_);
    }
    
    // Note: neighbors are not cloned to avoid circular references
    
    return cloned;
}

// === POIManager Implementation ===

void POIManager::addPOI(std::shared_ptr<POI> poi) {
    if (poi) {
        pois_.push_back(poi);
    }
}

void POIManager::addPOI(double ref_x, double ref_y) {
    auto poi = std::make_shared<POI>(ref_x, ref_y);
    addPOI(poi);
}

void POIManager::removePOI(size_t index) {
    validateIndex(index);
    pois_.erase(pois_.begin() + index);
}

void POIManager::clear() {
    pois_.clear();
    spatial_index_.clear();
}

std::shared_ptr<POI> POIManager::getPOI(size_t index) const {
    validateIndex(index);
    return pois_[index];
}

void POIManager::generatePOIsFromROI(const cv::Mat& roi, int step) {
    if (roi.empty() || step <= 0) return;
    
    clear();
    
    for (int y = 0; y < roi.rows; y += step) {
        for (int x = 0; x < roi.cols; x += step) {
            if (roi.at<uchar>(y, x) > 0) {  // Inside ROI
                addPOI(static_cast<double>(x), static_cast<double>(y));
            }
        }
    }
    
    buildSpatialIndex();
}

void POIManager::generateRegularGrid(const cv::Rect& region, int step) {
    if (step <= 0) return;
    
    clear();
    
    for (int y = region.y; y < region.y + region.height; y += step) {
        for (int x = region.x; x < region.x + region.width; x += step) {
            addPOI(static_cast<double>(x), static_cast<double>(y));
        }
    }
    
    buildSpatialIndex();
}

void POIManager::filterByZNCC(double threshold) {
    auto it = std::remove_if(pois_.begin(), pois_.end(), 
        [threshold](const std::shared_ptr<POI>& poi) {
            return !poi->hasZNCC() || poi->getZNCC() < threshold;
        });
    pois_.erase(it, pois_.end());
}

void POIManager::filterByConvergence() {
    auto it = std::remove_if(pois_.begin(), pois_.end(), 
        [](const std::shared_ptr<POI>& poi) {
            return !poi->isConverged();
        });
    pois_.erase(it, pois_.end());
}

void POIManager::filterByDisplacementJump(double threshold) {
    if (pois_.empty()) return;
    
    // Calculate average displacement
    cv::Scalar meanDisp = getMeanDisplacement();
    double meanU = meanDisp[0];
    double meanV = meanDisp[1];
    
    auto it = std::remove_if(pois_.begin(), pois_.end(), 
        [threshold, meanU, meanV](const std::shared_ptr<POI>& poi) {
            if (!poi->hasDisplacement()) return true;
            
            double du = poi->getDisplacementU() - meanU;
            double dv = poi->getDisplacementV() - meanV;
            double jumpMagnitude = std::sqrt(du * du + dv * dv);
            
            return jumpMagnitude > threshold;
        });
    pois_.erase(it, pois_.end());
}

void POIManager::filterInvalid() {
    auto it = std::remove_if(pois_.begin(), pois_.end(), 
        [](const std::shared_ptr<POI>& poi) {
            return !poi->isValid();
        });
    pois_.erase(it, pois_.end());
}

std::vector<size_t> POIManager::getValidIndices() const {
    std::vector<size_t> indices;
    for (size_t i = 0; i < pois_.size(); ++i) {
        if (pois_[i]->isValid()) {
            indices.push_back(i);
        }
    }
    return indices;
}

std::vector<size_t> POIManager::getConvergedIndices() const {
    std::vector<size_t> indices;
    for (size_t i = 0; i < pois_.size(); ++i) {
        if (pois_[i]->isConverged()) {
            indices.push_back(i);
        }
    }
    return indices;
}

void POIManager::exportToCSV(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    // Write header
    file << POI::getCSVHeader() << std::endl;
    
    // Write data
    for (const auto& poi : pois_) {
        file << poi->toCSVRow() << std::endl;
    }
    
    file.close();
}

void POIManager::exportValidToCSV(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    // Write header
    file << POI::getCSVHeader() << std::endl;
    
    // Write only valid POIs
    for (const auto& poi : pois_) {
        if (poi->isValid()) {
            file << poi->toCSVRow() << std::endl;
        }
    }
    
    file.close();
}

void POIManager::exportToMats(cv::Mat& ref_coords, cv::Mat& def_coords, 
                             cv::Mat& displacement, cv::Mat& zncc,
                             cv::Mat& valid_mask) const {
    size_t validCount = getValidCount();
    if (validCount == 0) {
        // Create empty matrices
        ref_coords = cv::Mat::zeros(0, 2, CV_64F);
        def_coords = cv::Mat::zeros(0, 2, CV_64F);
        displacement = cv::Mat::zeros(0, 2, CV_64F);
        zncc = cv::Mat::zeros(0, 1, CV_64F);
        valid_mask = cv::Mat::zeros(0, 1, CV_8U);
        return;
    }
    
    // Allocate matrices
    ref_coords = cv::Mat::zeros(validCount, 2, CV_64F);
    def_coords = cv::Mat::zeros(validCount, 2, CV_64F);
    displacement = cv::Mat::zeros(validCount, 2, CV_64F);
    zncc = cv::Mat::zeros(validCount, 1, CV_64F);
    valid_mask = cv::Mat::ones(validCount, 1, CV_8U);
    
    size_t idx = 0;
    for (const auto& poi : pois_) {
        if (poi->isValid() && idx < validCount) {
            // Reference coordinates
            ref_coords.at<double>(idx, 0) = poi->getReferenceX();
            ref_coords.at<double>(idx, 1) = poi->getReferenceY();
            
            // Deformed coordinates
            if (poi->hasDeformedCoords()) {
                def_coords.at<double>(idx, 0) = poi->getDeformedX();
                def_coords.at<double>(idx, 1) = poi->getDeformedY();
            }
            
            // Displacement
            if (poi->hasDisplacement()) {
                displacement.at<double>(idx, 0) = poi->getDisplacementU();
                displacement.at<double>(idx, 1) = poi->getDisplacementV();
            }
            
            // ZNCC
            if (poi->hasZNCC()) {
                zncc.at<double>(idx, 0) = poi->getZNCC();
            }
            
            idx++;
        }
    }
}

size_t POIManager::getValidCount() const {
    return std::count_if(pois_.begin(), pois_.end(), 
        [](const std::shared_ptr<POI>& poi) {
            return poi->isValid();
        });
}

size_t POIManager::getConvergedCount() const {
    return std::count_if(pois_.begin(), pois_.end(), 
        [](const std::shared_ptr<POI>& poi) {
            return poi->isConverged();
        });
}

double POIManager::getAverageZNCC() const {
    double sum = 0.0;
    size_t count = 0;
    
    for (const auto& poi : pois_) {
        if (poi->hasZNCC()) {
            sum += poi->getZNCC();
            count++;
        }
    }
    
    return count > 0 ? sum / count : 0.0;
}

cv::Scalar POIManager::getMeanDisplacement() const {
    double sumU = 0.0, sumV = 0.0;
    size_t count = 0;
    
    for (const auto& poi : pois_) {
        if (poi->hasDisplacement()) {
            sumU += poi->getDisplacementU();
            sumV += poi->getDisplacementV();
            count++;
        }
    }
    
    if (count > 0) {
        return cv::Scalar(sumU / count, sumV / count);
    }
    return cv::Scalar(0.0, 0.0);
}

cv::Scalar POIManager::getStdDisplacement() const {
    cv::Scalar mean = getMeanDisplacement();
    double meanU = mean[0];
    double meanV = mean[1];
    
    double sumU2 = 0.0, sumV2 = 0.0;
    size_t count = 0;
    
    for (const auto& poi : pois_) {
        if (poi->hasDisplacement()) {
            double du = poi->getDisplacementU() - meanU;
            double dv = poi->getDisplacementV() - meanV;
            sumU2 += du * du;
            sumV2 += dv * dv;
            count++;
        }
    }
    
    if (count > 1) {
        return cv::Scalar(std::sqrt(sumU2 / (count - 1)), 
                         std::sqrt(sumV2 / (count - 1)));
    }
    return cv::Scalar(0.0, 0.0);
}

void POIManager::buildNeighborConnections(int connectivity) {
    clearNeighborConnections();
    
    if (connectivity != 4 && connectivity != 8) {
        throw std::invalid_argument("Connectivity must be 4 or 8");
    }
    
    // Build spatial index for efficient neighbor search
    buildSpatialIndex();
    
    for (auto& poi : pois_) {
        double x = poi->getReferenceX();
        double y = poi->getReferenceY();
        
        // Find neighbors within a small radius (e.g., 1.5 pixels for adjacent points)
        auto nearbyPOIs = findPOIsInRadius(x, y, 1.5);
        
        for (auto& neighbor : nearbyPOIs) {
            if (neighbor != poi) {  // Don't add self as neighbor
                poi->addNeighbor(neighbor);
            }
        }
    }
}

void POIManager::clearNeighborConnections() {
    for (auto& poi : pois_) {
        poi->clearNeighbors();
    }
}

std::shared_ptr<POI> POIManager::findNearestPOI(double x, double y) const {
    if (pois_.empty()) return nullptr;
    
    std::shared_ptr<POI> nearest = pois_[0];
    double minDist = std::numeric_limits<double>::max();
    
    for (const auto& poi : pois_) {
        double dx = poi->getReferenceX() - x;
        double dy = poi->getReferenceY() - y;
        double dist = dx * dx + dy * dy;  // Squared distance for efficiency
        
        if (dist < minDist) {
            minDist = dist;
            nearest = poi;
        }
    }
    
    return nearest;
}

std::vector<std::shared_ptr<POI>> POIManager::findPOIsInRadius(double x, double y, double radius) const {
    std::vector<std::shared_ptr<POI>> result;
    double radiusSquared = radius * radius;
    
    for (const auto& poi : pois_) {
        double dx = poi->getReferenceX() - x;
        double dy = poi->getReferenceY() - y;
        double distSquared = dx * dx + dy * dy;
        
        if (distSquared <= radiusSquared) {
            result.push_back(poi);
        }
    }
    
    return result;
}

void POIManager::importFromRGDICResult(const cv::Mat& u, const cv::Mat& v, 
                                      const cv::Mat& cc, const cv::Mat& validMask,
                                      const cv::Mat& roi, int step) {
    clear();
    
    if (u.empty() || v.empty() || roi.empty()) return;
    
    // Generate POIs from ROI with the specified step
    for (int y = 0; y < roi.rows; y += step) {
        for (int x = 0; x < roi.cols; x += step) {
            if (roi.at<uchar>(y, x) > 0) {  // Inside ROI
                auto poi = std::make_shared<POI>(static_cast<double>(x), static_cast<double>(y));
                
                // Set displacement if available
                if (x < u.cols && y < u.rows) {
                    double disp_u = u.at<double>(y, x);
                    double disp_v = v.at<double>(y, x);
                    poi->setDisplacement(disp_u, disp_v);
                    poi->calculateDeformedFromDisplacement();
                }
                
                // Set correlation coefficient if available
                if (!cc.empty() && x < cc.cols && y < cc.rows) {
                    poi->setZNCC(cc.at<double>(y, x));
                }
                
                // Set validity
                bool isValid = true;
                if (!validMask.empty() && x < validMask.cols && y < validMask.rows) {
                    isValid = validMask.at<uchar>(y, x) > 0;
                }
                poi->setValid(isValid);
                poi->setConverged(isValid);  // Assume converged if valid
                
                addPOI(poi);
            }
        }
    }
    
    buildSpatialIndex();
}

void POIManager::buildSpatialIndex() {
    spatial_index_.clear();
    
    for (size_t i = 0; i < pois_.size(); ++i) {
        int key = getSpatialKey(pois_[i]->getReferenceX(), pois_[i]->getReferenceY());
        spatial_index_[key].push_back(i);
    }
}

int POIManager::getSpatialKey(double x, double y) const {
    // Simple grid-based spatial indexing with 10-pixel cells
    const int cellSize = 10;
    int gridX = static_cast<int>(x / cellSize);
    int gridY = static_cast<int>(y / cellSize);
    return gridY * 10000 + gridX;  // Assuming images won't exceed 100k pixels wide
}

void POIManager::validateIndex(size_t index) const {
    if (index >= pois_.size()) {
        throw std::out_of_range("POI index out of range");
    }
}