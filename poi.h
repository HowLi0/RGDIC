#ifndef POI_H
#define POI_H

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @brief Point of Interest (POI) class representing a single analysis point in DIC
 * 
 * This class follows the OpenCorr-like POI design pattern to provide a more
 * structured approach to managing DIC analysis results at the point level.
 */
class POI {
public:
    // Constructors
    POI();
    POI(const cv::Point2f& leftCoord, const cv::Point2f& rightCoord = cv::Point2f(0, 0));
    
    // Copy constructor and assignment operator
    POI(const POI& other) = default;
    POI& operator=(const POI& other) = default;
    
    // Coordinate information
    cv::Point2f leftCoord;      // Left image (reference) coordinate
    cv::Point2f rightCoord;     // Right image (deformed) coordinate
    
    // Displacement information
    cv::Vec2f displacement;     // Displacement vector (u, v)
    
    // Quality metrics
    double correlation;         // Correlation coefficient (ZNCC)
    bool valid;                 // Whether this POI has valid results
    
    // Strain information (optional)
    struct StrainInfo {
        double exx, eyy, exy;   // Strain components
        bool computed;          // Whether strain has been computed
        
        StrainInfo() : exx(0), eyy(0), exy(0), computed(false) {}
    } strain;
    
    // Utility methods
    /**
     * @brief Calculate right coordinate from left coordinate and displacement
     */
    void updateRightCoord();
    
    /**
     * @brief Calculate displacement from left and right coordinates
     */
    void updateDisplacement();
    
    /**
     * @brief Check if this POI is valid (has meaningful correlation and displacement)
     */
    bool isValid() const;
    
    /**
     * @brief Get displacement magnitude
     */
    double getDisplacementMagnitude() const;
    
    /**
     * @brief Set strain components
     */
    void setStrain(double exx, double eyy, double exy);
    
    /**
     * @brief Clear strain information
     */
    void clearStrain();
};

/**
 * @brief Collection of POIs with utility functions
 */
class POICollection {
public:
    std::vector<POI> pois;
    
    // Constructors
    POICollection() = default;
    POICollection(const std::vector<POI>& pois);
    
    // Access methods
    size_t size() const { return pois.size(); }
    bool empty() const { return pois.empty(); }
    POI& operator[](size_t index) { return pois[index]; }
    const POI& operator[](size_t index) const { return pois[index]; }
    
    // Iterator support
    std::vector<POI>::iterator begin() { return pois.begin(); }
    std::vector<POI>::iterator end() { return pois.end(); }
    std::vector<POI>::const_iterator begin() const { return pois.begin(); }
    std::vector<POI>::const_iterator end() const { return pois.end(); }
    
    // Utility methods
    /**
     * @brief Add a POI to the collection
     */
    void addPOI(const POI& poi);
    
    /**
     * @brief Get count of valid POIs
     */
    size_t getValidCount() const;
    
    /**
     * @brief Get count of POIs with computed strain
     */
    size_t getStrainComputedCount() const;
    
    /**
     * @brief Clear all POIs
     */
    void clear();
    
    /**
     * @brief Export to CSV format with complete coordinate information
     */
    void exportToCSV(const std::string& filename) const;
    
    /**
     * @brief Get bounding box of all valid POIs
     */
    cv::Rect2f getBoundingBox() const;
    
    /**
     * @brief Filter POIs by validity
     */
    POICollection getValidPOIs() const;
};

#endif // POI_H