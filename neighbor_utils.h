#ifndef NEIGHBOR_UTILS_H
#define NEIGHBOR_UTILS_H

#include <opencv2/opencv.hpp>
#include <vector>

// Utility class for handling neighbor coordinates
class NeighborUtils {
public:
    // Neighbor connectivity types
    enum ConnectivityType {
        FOUR_CONNECTED = 4,
        EIGHT_CONNECTED = 8
    };
    
    // Constructor with step size
    NeighborUtils(int step = 1, ConnectivityType connectivity = FOUR_CONNECTED);
    
    // Get neighbor offsets
    const std::vector<cv::Point>& getNeighborOffsets() const { return m_neighborOffsets; }
    
    // Get neighbors for a specific point
    std::vector<cv::Point> getNeighbors(const cv::Point& center) const;
    
    // Get valid neighbors within image bounds
    std::vector<cv::Point> getValidNeighbors(const cv::Point& center, 
                                           const cv::Size& imageSize) const;
    
    // Get valid neighbors within ROI
    std::vector<cv::Point> getValidNeighbors(const cv::Point& center, 
                                           const cv::Mat& roi) const;
    
    // Check if a point is valid within bounds
    bool isValidPoint(const cv::Point& point, const cv::Size& imageSize) const;
    
    // Check if a point is valid within ROI
    bool isValidPoint(const cv::Point& point, const cv::Mat& roi) const;
    
    // Set new step size
    void setStep(int step);
    
    // Set new connectivity type
    void setConnectivity(ConnectivityType connectivity);

    // Get current step size
    int getStep() const { return m_step; }

private:
    int m_step;
    ConnectivityType m_connectivity;
    std::vector<cv::Point> m_neighborOffsets;
    
    // Initialize neighbor offsets based on step and connectivity
    void initializeNeighborOffsets();
};

#endif // NEIGHBOR_UTILS_H
