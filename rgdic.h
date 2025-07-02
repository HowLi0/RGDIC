#ifndef RGDIC_H
#define RGDIC_H

#include <vector>
#include <queue>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "icgn_optimizer.h"
#include "neighbor_utils.h"
#include "poi.h"

class RGDIC {
public:
    // Result structure to hold displacement fields
    struct DisplacementResult {
        // Matrix format (original - for backward compatibility)
        cv::Mat u;           // x-displacement field
        cv::Mat v;           // y-displacement field
        cv::Mat cc;          // correlation coefficient (ZNCC)
        cv::Mat validMask;   // valid points mask
        
        // POI format (new feature)
        rgdic::POICollection pois;    // POI集合
        bool poisEnabled;             // 是否启用POI模式
        
        // Constructor
        DisplacementResult() : poisEnabled(false) {}
        
        // Conversion functions
        void convertMatrixToPOIs();   // 将矩阵格式转换为POI格式
        void convertPOIsToMatrix();   // 将POI格式转换为矩阵格式
        void syncPOIsWithMatrices();  // 同步POI和矩阵数据
        
        // Export functions
        bool exportToCSV(const std::string& filename) const;
        bool exportToPOIFormat(const std::string& filename) const;
        
        // POI mode control
        void enablePOIs(bool enable = true) { poisEnabled = enable; }
        bool isPOIsEnabled() const { return poisEnabled; }
    };
    
    // Constructor
    RGDIC(int subsetRadius = 15, 
          double convergenceThreshold = 0.00001,
          int maxIterations = 30,
          double ccThreshold = 0.8,
          double deltaDispThreshold = 1.0,
          ShapeFunctionOrder order = SECOND_ORDER,
          int neighborStep = 5);
    
    // Virtual destructor for inheritance
    virtual ~RGDIC() = default;
    
    // Main function to perform RGDIC analysis (virtual for override)
    virtual DisplacementResult compute(const cv::Mat& refImage, 
                              const cv::Mat& defImage,
                              const cv::Mat& roi);
    
    // Utility function to display results
    void displayResults(const cv::Mat& refImage, const DisplacementResult& result, 
                      const cv::Mat& trueDispX = cv::Mat(), const cv::Mat& trueDispY = cv::Mat());
    
    // POI-specific visualization functions
    void displayPOIResults(const cv::Mat& refImage, const cv::Mat& defImage, 
                          const DisplacementResult& result);
    void displayPOICorrespondences(const cv::Mat& refImage, const cv::Mat& defImage,
                                  const rgdic::POICollection& pois, int maxPoints = 100);
    void displayPOIStatistics(const rgdic::POICollection& pois);
    
    // Evaluate errors if ground truth is available
    void evaluateErrors(const DisplacementResult& result, 
                      const cv::Mat& trueDispX, const cv::Mat& trueDispY);

protected:
    // Algorithm parameters
    int m_subsetRadius;
    double m_convergenceThreshold;
    int m_maxIterations;
    double m_ccThreshold;
    double m_deltaDispThreshold;
    ShapeFunctionOrder m_order;
    
    // Number of shape function parameters (6 for first-order, 12 for second-order)
    int m_numParams;
    
    // Neighbor utilities for coordinate management
    NeighborUtils m_neighborUtils;
    
    // Representation of warp parameters
    // For first-order: p = [u, v, du/dx, du/dy, dv/dx, dv/dy]
    // For second-order: p = [u, v, du/dx, du/dy, dv/dx, dv/dy, 
    //                        d²u/dx², d²u/dxdy, d²u/dy², d²v/dx², d²v/dxdy, d²v/dy²]
    
    // Calculate signed distance array for ROI (used for seed point selection)
    cv::Mat calculateSDA(const cv::Mat& roi);
    
    // Find initial seed point using SDA
    cv::Point findSeedPoint(const cv::Mat& roi, const cv::Mat& sda);

private:
    
    using PriorityQueue = std::priority_queue<
        std::pair<cv::Point, double>,
        std::vector<std::pair<cv::Point, double>>,
        std::function<bool(const std::pair<cv::Point, double>&, const std::pair<cv::Point, double>&)>>;
    
    // Updated function signature
    bool analyzePoint(const cv::Point& point, ICGNOptimizer& optimizer, 
                    const cv::Mat& roi, DisplacementResult& result, 
                    PriorityQueue& queue, cv::Mat& analyzedPoints);
};

// Factory function for creating RGDIC instance (CPU only)
std::unique_ptr<RGDIC> createRGDIC(bool useCuda = false,
                                   int subsetRadius = 15,
                                   double convergenceThreshold = 0.00001,
                                   int maxIterations = 30,
                                   double ccThreshold = 0.8,
                                   double deltaDispThreshold = 1.0,
                                   ShapeFunctionOrder order = SECOND_ORDER,
                                   int neighborStep = 5);

#endif // RGDIC_H