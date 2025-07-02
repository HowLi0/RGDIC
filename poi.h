#ifndef POI_H
#define POI_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <unordered_map>

/**
 * @brief Point of Interest (POI) class following OpenCorr design
 * 
 * This class encapsulates all information related to a single correlation point,
 * including reference and deformed image coordinates, displacement vectors,
 * correlation metrics, and quality measures.
 */
class POI {
public:
    // Constructor
    POI(double ref_x = 0.0, double ref_y = 0.0);
    
    // Copy constructor and assignment operator
    POI(const POI& other) = default;
    POI& operator=(const POI& other) = default;
    
    // Destructor
    ~POI() = default;
    
    // === Reference Image Coordinates (Left Image) ===
    void setReferenceCoords(double x, double y) { ref_x_ = x; ref_y_ = y; }
    double getReferenceX() const { return ref_x_; }
    double getReferenceY() const { return ref_y_; }
    cv::Point2f getReferencePoint() const { return cv::Point2f(ref_x_, ref_y_); }
    
    // === Deformed Image Coordinates (Right Image) ===
    void setDeformedCoords(double x, double y) { def_x_ = x; def_y_ = y; deformed_coords_set_ = true; }
    double getDeformedX() const { return def_x_; }
    double getDeformedY() const { return def_y_; }
    cv::Point2f getDeformedPoint() const { return cv::Point2f(def_x_, def_y_); }
    bool hasDeformedCoords() const { return deformed_coords_set_; }
    
    // === Displacement Vectors ===
    void setDisplacement(double u, double v) { u_ = u; v_ = v; displacement_set_ = true; }
    double getDisplacementU() const { return u_; }
    double getDisplacementV() const { return v_; }
    cv::Vec2f getDisplacementVector() const { return cv::Vec2f(u_, v_); }
    bool hasDisplacement() const { return displacement_set_; }
    
    // Calculate deformed coordinates from displacement (convenience method)
    void calculateDeformedFromDisplacement() {
        if (displacement_set_) {
            setDeformedCoords(ref_x_ + u_, ref_y_ + v_);
        }
    }
    
    // Calculate displacement from coordinates (convenience method)
    void calculateDisplacementFromCoords() {
        if (deformed_coords_set_) {
            setDisplacement(def_x_ - ref_x_, def_y_ - ref_y_);
        }
    }
    
    // === Correlation Metrics ===
    void setZNCC(double zncc) { zncc_ = zncc; zncc_set_ = true; }
    double getZNCC() const { return zncc_; }
    bool hasZNCC() const { return zncc_set_; }
    
    void setNCC(double ncc) { ncc_ = ncc; ncc_set_ = true; }
    double getNCC() const { return ncc_; }
    bool hasNCC() const { return ncc_set_; }
    
    // === Strain Components ===
    void setStrainComponents(double exx, double eyy, double exy) {
        exx_ = exx; eyy_ = eyy; exy_ = exy; strain_set_ = true;
    }
    double getStrainExx() const { return exx_; }
    double getStrainEyy() const { return eyy_; }
    double getStrainExy() const { return exy_; }
    bool hasStrain() const { return strain_set_; }
    
    // === Quality Measures ===
    void setIterationCount(int iterations) { iterations_ = iterations; }
    int getIterationCount() const { return iterations_; }
    
    void setConverged(bool converged) { converged_ = converged; }
    bool isConverged() const { return converged_; }
    
    void setValid(bool valid) { valid_ = valid; }
    bool isValid() const { return valid_; }
    
    // === Shape Function Parameters ===
    void setShapeParameters(const cv::Mat& params) { shape_params_ = params.clone(); }
    cv::Mat getShapeParameters() const { return shape_params_; }
    bool hasShapeParameters() const { return !shape_params_.empty(); }
    
    // === Subset Information ===
    void setSubsetRadius(int radius) { subset_radius_ = radius; }
    int getSubsetRadius() const { return subset_radius_; }
    
    // === Neighbor Information ===
    void addNeighbor(const std::shared_ptr<POI>& neighbor) { neighbors_.push_back(neighbor); }
    void clearNeighbors() { neighbors_.clear(); }
    const std::vector<std::weak_ptr<POI>>& getNeighbors() const { return neighbors_; }
    size_t getNeighborCount() const { return neighbors_.size(); }
    
    // === Utility Methods ===
    
    // Export POI data as CSV row
    std::string toCSVRow() const;
    
    // Get CSV header
    static std::string getCSVHeader();
    
    // Calculate distance to another POI
    double distanceTo(const POI& other) const;
    
    // Check if POI has complete correlation data
    bool isComplete() const {
        return deformed_coords_set_ && displacement_set_ && zncc_set_ && valid_;
    }
    
    // Reset all data
    void reset();
    
    // Clone this POI
    std::unique_ptr<POI> clone() const;

private:
    // Reference image coordinates (left image)
    double ref_x_, ref_y_;
    
    // Deformed image coordinates (right image)
    double def_x_, def_y_;
    bool deformed_coords_set_;
    
    // Displacement vectors
    double u_, v_;
    bool displacement_set_;
    
    // Correlation metrics
    double zncc_, ncc_;
    bool zncc_set_, ncc_set_;
    
    // Strain components
    double exx_, eyy_, exy_;
    bool strain_set_;
    
    // Quality measures
    int iterations_;
    bool converged_;
    bool valid_;
    
    // Shape function parameters
    cv::Mat shape_params_;
    
    // Subset information
    int subset_radius_;
    
    // Neighbor connections (using weak_ptr to avoid circular references)
    std::vector<std::weak_ptr<POI>> neighbors_;
};

/**
 * @brief POI Manager class for handling collections of POIs
 * 
 * This class manages collections of POIs and provides operations
 * for batch processing, filtering, and data export.
 */
class POIManager {
public:
    // Constructor
    POIManager() = default;
    
    // Destructor
    ~POIManager() = default;
    
    // === POI Collection Management ===
    void addPOI(std::shared_ptr<POI> poi);
    void addPOI(double ref_x, double ref_y);
    void removePOI(size_t index);
    void clear();
    
    size_t size() const { return pois_.size(); }
    bool empty() const { return pois_.empty(); }
    
    // Access POIs
    std::shared_ptr<POI> getPOI(size_t index) const;
    std::shared_ptr<POI> operator[](size_t index) const { return getPOI(index); }
    
    // Iterator support
    std::vector<std::shared_ptr<POI>>::iterator begin() { return pois_.begin(); }
    std::vector<std::shared_ptr<POI>>::iterator end() { return pois_.end(); }
    std::vector<std::shared_ptr<POI>>::const_iterator begin() const { return pois_.begin(); }
    std::vector<std::shared_ptr<POI>>::const_iterator end() const { return pois_.end(); }
    
    // === ROI-based POI Generation ===
    void generatePOIsFromROI(const cv::Mat& roi, int step = 5);
    void generateRegularGrid(const cv::Rect& region, int step = 5);
    
    // === Filtering and Quality Control ===
    void filterByZNCC(double threshold);
    void filterByConvergence();
    void filterByDisplacementJump(double threshold);
    void filterInvalid();
    
    // Get filtered indices
    std::vector<size_t> getValidIndices() const;
    std::vector<size_t> getConvergedIndices() const;
    
    // === Data Export ===
    void exportToCSV(const std::string& filename) const;
    void exportValidToCSV(const std::string& filename) const;
    
    // Export to OpenCV Mats (for backward compatibility)
    void exportToMats(cv::Mat& ref_coords, cv::Mat& def_coords, 
                     cv::Mat& displacement, cv::Mat& zncc,
                     cv::Mat& valid_mask) const;
    
    // === Statistics ===
    size_t getValidCount() const;
    size_t getConvergedCount() const;
    double getAverageZNCC() const;
    cv::Scalar getMeanDisplacement() const;
    cv::Scalar getStdDisplacement() const;
    
    // === Neighbor Management ===
    void buildNeighborConnections(int connectivity = 4);
    void clearNeighborConnections();
    
    // === Search and Query ===
    std::shared_ptr<POI> findNearestPOI(double x, double y) const;
    std::vector<std::shared_ptr<POI>> findPOIsInRadius(double x, double y, double radius) const;
    
    // === Import from existing RGDIC results ===
    void importFromRGDICResult(const cv::Mat& u, const cv::Mat& v, 
                              const cv::Mat& cc, const cv::Mat& validMask,
                              const cv::Mat& roi, int step = 5);

private:
    std::vector<std::shared_ptr<POI>> pois_;
    
    // Spatial indexing for fast queries (simple grid-based)
    std::unordered_map<int, std::vector<size_t>> spatial_index_;
    void buildSpatialIndex();
    int getSpatialKey(double x, double y) const;
    
    // Helper methods
    void validateIndex(size_t index) const;
};

#endif // POI_H