#include "neighbor_utils.h"

NeighborUtils::NeighborUtils(int step, ConnectivityType connectivity) 
    : m_step(step), m_connectivity(connectivity) {
    initializeNeighborOffsets();
}

std::vector<cv::Point> NeighborUtils::getNeighbors(const cv::Point& center) const {
    std::vector<cv::Point> neighbors;
    neighbors.reserve(m_neighborOffsets.size());
    
    for (const auto& offset : m_neighborOffsets) {
        neighbors.push_back(center + offset);
    }
    
    return neighbors;
}

std::vector<cv::Point> NeighborUtils::getValidNeighbors(const cv::Point& center, 
                                                       const cv::Size& imageSize) const {
    std::vector<cv::Point> validNeighbors;
    
    for (const auto& offset : m_neighborOffsets) {
        cv::Point neighbor = center + offset;
        if (isValidPoint(neighbor, imageSize)) {
            validNeighbors.push_back(neighbor);
        }
    }
    
    return validNeighbors;
}

std::vector<cv::Point> NeighborUtils::getValidNeighbors(const cv::Point& center, 
                                                       const cv::Mat& roi) const {
    std::vector<cv::Point> validNeighbors;
    
    for (const auto& offset : m_neighborOffsets) {
        cv::Point neighbor = center + offset;
        if (isValidPoint(neighbor, roi)) {
            validNeighbors.push_back(neighbor);
        }
    }
    
    return validNeighbors;
}

bool NeighborUtils::isValidPoint(const cv::Point& point, const cv::Size& imageSize) const {
    return point.x >= 0 && point.x < imageSize.width &&
           point.y >= 0 && point.y < imageSize.height;
}

bool NeighborUtils::isValidPoint(const cv::Point& point, const cv::Mat& roi) const {
    return point.x >= 0 && point.x < roi.cols &&
           point.y >= 0 && point.y < roi.rows &&
           roi.at<uchar>(point) > 0;
}

void NeighborUtils::setStep(int step) {
    if (step != m_step) {
        m_step = step;
        initializeNeighborOffsets();
    }
}

void NeighborUtils::setConnectivity(ConnectivityType connectivity) {
    if (connectivity != m_connectivity) {
        m_connectivity = connectivity;
        initializeNeighborOffsets();
    }
}

void NeighborUtils::initializeNeighborOffsets() {
    m_neighborOffsets.clear();
    
    if (m_connectivity == FOUR_CONNECTED) {
        // 4-connected neighbors with step size
        m_neighborOffsets = {
            {0, m_step},      // down
            {m_step, 0},      // right
            {0, -m_step},     // up
            {-m_step, 0}      // left
        };
    } else if (m_connectivity == EIGHT_CONNECTED) {
        // 8-connected neighbors with step size
        m_neighborOffsets = {
            {0, m_step},          // down
            {m_step, 0},          // right
            {0, -m_step},         // up
            {-m_step, 0},         // left
            {m_step, m_step},     // down-right
            {m_step, -m_step},    // up-right
            {-m_step, m_step},    // down-left
            {-m_step, -m_step}    // up-left
        };
    }
}
