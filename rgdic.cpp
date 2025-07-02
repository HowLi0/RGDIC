#include "rgdic.h"
#include <iostream>
#include <functional>

RGDIC::RGDIC(int subsetRadius, double convergenceThreshold, int maxIterations,
           double ccThreshold, double deltaDispThreshold, ShapeFunctionOrder order,
           int neighborStep)
    : m_subsetRadius(subsetRadius),
      m_convergenceThreshold(convergenceThreshold),
      m_maxIterations(maxIterations),
      m_ccThreshold(ccThreshold),
      m_deltaDispThreshold(deltaDispThreshold),
      m_order(order),
      m_neighborUtils(neighborStep, NeighborUtils::FOUR_CONNECTED)
{
    // Set number of parameters based on shape function order
    m_numParams = (order == FIRST_ORDER) ? 6 : 12;
}

RGDIC::DisplacementResult RGDIC::compute(const cv::Mat& refImage, const cv::Mat& defImage, const cv::Mat& roi)
{
    // Initialize results
    DisplacementResult result;
    result.u = cv::Mat::zeros(roi.size(), CV_64F);
    result.v = cv::Mat::zeros(roi.size(), CV_64F);
    result.cc = cv::Mat::zeros(roi.size(), CV_64F);
    result.validMask = cv::Mat::zeros(roi.size(), CV_8U);
    
    // Calculate signed distance array for ROI
    cv::Mat sda = calculateSDA(roi);
    
    // Find seed point
    cv::Point seedPoint = findSeedPoint(roi, sda);
    
    // Create the ICGN optimizer
    ICGNOptimizer optimizer(refImage, defImage, m_subsetRadius, m_order, 
                          m_convergenceThreshold, m_maxIterations);
    
    // Initialize warp parameters matrix
    cv::Mat warpParams = cv::Mat::zeros(m_numParams, 1, CV_64F);
    
    // Compute initial ZNCC
    double initialZNCC = 0.0;
    
    // Try to compute initial guess for seed point
    if (!optimizer.initialGuess(seedPoint, warpParams, initialZNCC)) {
        std::cerr << "Failed to find initial guess for seed point." << std::endl;
        return result;
    }
    
    // Optimize seed point
    if (!optimizer.optimize(seedPoint, warpParams, initialZNCC)) {
        std::cerr << "Failed to optimize seed point." << std::endl;
        return result;
    }
    
    // Check if seed point has good correlation
    if (initialZNCC > m_ccThreshold) {
        std::cerr << "Seed point has poor correlation: " << initialZNCC << std::endl;
        return result;
    }
    
    // Save seed point results
    result.u.at<double>(seedPoint) = warpParams.at<double>(0);
    result.v.at<double>(seedPoint) = warpParams.at<double>(1);
    result.cc.at<double>(seedPoint) = initialZNCC;
    result.validMask.at<uchar>(seedPoint) = 1;
    
    // Define comparator for priority queue
    auto comparator = [](const std::pair<cv::Point, double>& a, const std::pair<cv::Point, double>& b) {
        return a.second > b.second;
    };
    
    // Create priority queue for reliability-guided search
    PriorityQueue queue(comparator);
    
    // Initialize queue with seed point
    queue.push(std::make_pair(seedPoint, initialZNCC));
    
    // Create analyzed points tracker
    cv::Mat analyzedPoints = cv::Mat::zeros(roi.size(), CV_8U);
    analyzedPoints.at<uchar>(seedPoint) = 1;
    
    // Get neighbor offsets from utility class
    const auto& neighborOffsets = m_neighborUtils.getNeighborOffsets();
    
    // Reliability-guided search
    while (!queue.empty()) {
        // Get point with highest reliability (lowest ZNCC value)
        auto current = queue.top();
        queue.pop();
        
        cv::Point currentPoint = current.first;
        
        // Check all neighbors using the neighbor utility
        auto validNeighbors = m_neighborUtils.getValidNeighbors(currentPoint, roi);
        
        for (const auto& neighborPoint : validNeighbors) {
            if (analyzedPoints.at<uchar>(neighborPoint) == 0) {
                // Mark as analyzed
                analyzedPoints.at<uchar>(neighborPoint) = 1;
                
                // Try to analyze this point
                if (analyzePoint(neighborPoint, optimizer, roi, result, queue, analyzedPoints)) {
                    // Point successfully analyzed and added to queue
                }
            }
        }
    }
    
    // Post-process to remove outliers
    // Create a copy of valid mask
    cv::Mat validMaskCopy = result.validMask.clone();
    
    for (int y = 0; y < result.validMask.rows; y++) {
        for (int x = 0; x < result.validMask.cols; x++) {
            if (result.validMask.at<uchar>(y, x)) {
                // Check displacement jumps with valid neighbors
                bool isOutlier = false;
                
                // Use neighbor utility for outlier detection
                auto validNeighbors = m_neighborUtils.getValidNeighbors(cv::Point(x, y), roi);
                
                for (const auto& neighborPoint : validNeighbors) {
                    if (result.validMask.at<uchar>(neighborPoint)) {
                        // Calculate displacement jump
                        double du = result.u.at<double>(y, x) - result.u.at<double>(neighborPoint);
                        double dv = result.v.at<double>(y, x) - result.v.at<double>(neighborPoint);
                        double dispJump = std::sqrt(du*du + dv*dv);
                        
                        // Mark as outlier if displacement jump is too large
                        if (dispJump > m_deltaDispThreshold) {
                            isOutlier = true;
                            break;
                        }
                    }
                }
                
                if (isOutlier) {
                    validMaskCopy.at<uchar>(y, x) = 0;
                }
            }
        }
    }
    
    // Update result with filtered mask
    result.validMask = validMaskCopy;
    
    return result;
}

cv::Mat RGDIC::calculateSDA(const cv::Mat& roi) {
    cv::Mat dist;
    cv::distanceTransform(roi, dist, cv::DIST_L2, cv::DIST_MASK_PRECISE);
    return dist;
}

cv::Point RGDIC::findSeedPoint(const cv::Mat& roi, const cv::Mat& sda) {
    cv::Point maxLoc;
    double maxVal;
    
    // Find point with maximum SDA value (furthest from boundaries)
    cv::minMaxLoc(sda, nullptr, &maxVal, nullptr, &maxLoc, roi);
    
    return maxLoc;
}

bool RGDIC::analyzePoint(const cv::Point& point, ICGNOptimizer& optimizer, 
    const cv::Mat& roi, DisplacementResult& result, 
    PriorityQueue& queue, cv::Mat& analyzedPoints)
{
    // Get neighboring points that have already been analyzed successfully
    std::vector<cv::Point> validNeighbors;
    
    // Use neighbor utility to get valid neighbors
    auto allNeighbors = m_neighborUtils.getValidNeighbors(point, roi);
    
    for (const auto& neighborPoint : allNeighbors) {
        if (result.validMask.at<uchar>(neighborPoint) > 0) {
            validNeighbors.push_back(neighborPoint);
        }
    }
    
    if (validNeighbors.empty()) {
        return false; // No valid neighbors to use as initial guess
    }
    
    // Find neighbor with best correlation coefficient
    cv::Point bestNeighbor = validNeighbors[0];
    double bestCC = result.cc.at<double>(bestNeighbor);
    
    for (size_t i = 1; i < validNeighbors.size(); i++) {
        double cc = result.cc.at<double>(validNeighbors[i]);
        if (cc < bestCC) { // Lower ZNCC value = better correlation
            bestCC = cc;
            bestNeighbor = validNeighbors[i];
        }
    }
    
    // Use warp parameters from best neighbor as initial guess
    cv::Mat warpParams = cv::Mat::zeros(m_numParams, 1, CV_64F);
    warpParams.at<double>(0) = result.u.at<double>(bestNeighbor);
    warpParams.at<double>(1) = result.v.at<double>(bestNeighbor);
    
    // For higher order parameters, we'd need to store them in the result
    // For simplicity, we're only storing and using u and v here
    
    // Run ICGN optimization
    double zncc;
    bool success = optimizer.optimize(point, warpParams, zncc);
    
    if (success && zncc < m_ccThreshold) { // Lower ZNCC value = better correlation
        // Check for displacement jump
        double du = warpParams.at<double>(0) - result.u.at<double>(bestNeighbor);
        double dv = warpParams.at<double>(1) - result.v.at<double>(bestNeighbor);
        double dispJump = std::sqrt(du*du + dv*dv);
        
        if (dispJump <= m_deltaDispThreshold) {
            // Store results
            result.u.at<double>(point) = warpParams.at<double>(0);
            result.v.at<double>(point) = warpParams.at<double>(1);
            result.cc.at<double>(point) = zncc;
            result.validMask.at<uchar>(point) = 1;
            
            // Add to queue for further propagation
            queue.push(std::make_pair(point, zncc));
            return true;
        }
    }
    
    return false;
}

void RGDIC::displayResults(const cv::Mat& refImage, const DisplacementResult& result, 
                         const cv::Mat& trueDispX, const cv::Mat& trueDispY) {
    // Create visualizations for displacement fields
    cv::Mat uViz, vViz;
    
    // Find min/max values for normalization
    double minU, maxU, minV, maxV;
    cv::minMaxLoc(result.u, &minU, &maxU, nullptr, nullptr, result.validMask);
    cv::minMaxLoc(result.v, &minV, &maxV, nullptr, nullptr, result.validMask);
    
    // Normalize displacement fields for visualization
    cv::Mat uNorm, vNorm;
    cv::normalize(result.u, uNorm, 0, 255, cv::NORM_MINMAX, CV_8U, result.validMask);
    cv::normalize(result.v, vNorm, 0, 255, cv::NORM_MINMAX, CV_8U, result.validMask);
    
    // Apply color map
    cv::Mat uColor, vColor;
    cv::applyColorMap(uNorm, uColor, cv::COLORMAP_JET);
    cv::applyColorMap(vNorm, vColor, cv::COLORMAP_JET);
    
    // Apply valid mask
    cv::Mat validMask3Ch;
    cv::cvtColor(result.validMask, validMask3Ch, cv::COLOR_GRAY2BGR);
    uColor = uColor.mul(validMask3Ch, 1.0/255.0);
    vColor = vColor.mul(validMask3Ch, 1.0/255.0);
    
    // Create displacement field visualization
    cv::Mat dispField;
    cv::cvtColor(refImage, dispField, cv::COLOR_GRAY2BGR);
    
    // Draw displacement vectors (subsampled)
    int step = 10;
    for (int y = 0; y < result.u.rows; y += step) {
        for (int x = 0; x < result.u.cols; x += step) {
            if (result.validMask.at<uchar>(y, x)) {
                double u = result.u.at<double>(y, x);
                double v = result.v.at<double>(y, x);
                
                // Scale displacements for visibility
                double scale = 5.0;
                cv::arrowedLine(dispField, cv::Point(x, y), 
                              cv::Point(x + u * scale, y + v * scale),
                              cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
            }
        }
    }
    
    // Display results
    cv::imshow("U Displacement", uColor);
    cv::imshow("V Displacement", vColor);
    cv::imshow("Displacement Field", dispField);
    
    // Save results
    cv::imwrite("E:/code_C++/RGDIC/u_displacement.png", uColor);
    cv::imwrite("E:/code_C++/RGDIC/v_displacement.png", vColor);
    cv::imwrite("E:/code_C++/RGDIC/displacement_field.png", dispField);
    
    // If ground truth is available, compute error maps
    if (!trueDispX.empty() && !trueDispY.empty()) {
        evaluateErrors(result, trueDispX, trueDispY);
    }
    
    cv::waitKey(0);
}

void RGDIC::evaluateErrors(const DisplacementResult& result, 
    const cv::Mat& trueDispX, const cv::Mat& trueDispY) {
    // Convert data types if necessary - ensure all matrices are of the same type
    cv::Mat u, v, trueU, trueV;
    result.u.convertTo(u, CV_32F);
    result.v.convertTo(v, CV_32F);
    trueDispX.convertTo(trueU, CV_32F);
    trueDispY.convertTo(trueV, CV_32F);

    // Calculate error maps
    cv::Mat errorU, errorV;
    cv::subtract(u, trueU, errorU);
    cv::subtract(v, trueV, errorV);

    // Calculate absolute errors - use absdiff to avoid type issues
    cv::Mat absErrorU, absErrorV;
    cv::absdiff(u, trueU, absErrorU);
    cv::absdiff(v, trueV, absErrorV);

    // Convert valid mask to proper type for arithmetic operations
    cv::Mat validMaskFloat;
    result.validMask.convertTo(validMaskFloat, CV_32F, 1.0/255.0);

    // Compute statistics for valid points
    cv::Scalar meanErrorU = cv::mean(absErrorU, result.validMask);
    cv::Scalar meanErrorV = cv::mean(absErrorV, result.validMask);

    double meanU = meanErrorU[0];
    double meanV = meanErrorV[0];

    // Find max errors
    double minU, maxU, minV, maxV;
    cv::minMaxLoc(absErrorU, &minU, &maxU, nullptr, nullptr, result.validMask);
    cv::minMaxLoc(absErrorV, &minV, &maxV, nullptr, nullptr, result.validMask);

    // Calculate RMS error
    cv::Mat errorUSq, errorVSq;
    cv::multiply(errorU, errorU, errorUSq);
    cv::multiply(errorV, errorV, errorVSq);

    // Make sure to convert mask to float for multiplication
    cv::Mat errorUSqMasked, errorVSqMasked;
    cv::multiply(errorUSq, validMaskFloat, errorUSqMasked);
    cv::multiply(errorVSq, validMaskFloat, errorVSqMasked);

    cv::Scalar sumUSq = cv::sum(errorUSqMasked);
    cv::Scalar sumVSq = cv::sum(errorVSqMasked);

    int validPoints = cv::countNonZero(result.validMask);
    double rmsU = std::sqrt(sumUSq[0] / validPoints);
    double rmsV = std::sqrt(sumVSq[0] / validPoints);

    // Print error metrics
    std::cout << "Error Metrics:" << std::endl;
    std::cout << "  U displacement: mean = " << meanU << " px, max = " << maxU << " px, RMS = " << rmsU << " px" << std::endl;
    std::cout << "  V displacement: mean = " << meanV << " px, max = " << maxV << " px, RMS = " << rmsV << " px" << std::endl;

    // Visualize error maps
    cv::Mat errorUNorm, errorVNorm;
    cv::normalize(absErrorU, errorUNorm, 0, 255, cv::NORM_MINMAX, CV_8U, result.validMask);
    cv::normalize(absErrorV, errorVNorm, 0, 255, cv::NORM_MINMAX, CV_8U, result.validMask);

    cv::Mat errorUColor, errorVColor;
    cv::applyColorMap(errorUNorm, errorUColor, cv::COLORMAP_JET);
    cv::applyColorMap(errorVNorm, errorVColor, cv::COLORMAP_JET);

    // Apply valid mask - convert to 3 channels first
    cv::Mat validMask3Ch;
    cv::cvtColor(result.validMask, validMask3Ch, cv::COLOR_GRAY2BGR);

    // Use bitwise AND instead of multiplication for masks
    cv::Mat errorUColorMasked, errorVColorMasked;
    cv::bitwise_and(errorUColor, validMask3Ch, errorUColorMasked);
    cv::bitwise_and(errorVColor, validMask3Ch, errorVColorMasked);

    // Display and save error maps
    cv::imshow("U Error Map", errorUColorMasked);
    cv::imshow("V Error Map", errorVColorMasked);
}

// Factory function for creating RGDIC instance (CPU only)
std::unique_ptr<RGDIC> createRGDIC(bool useCuda,
                                   int subsetRadius,
                                   double convergenceThreshold,
                                   int maxIterations,
                                   double ccThreshold,
                                   double deltaDispThreshold,
                                   ShapeFunctionOrder order,
                                   int neighborStep) {
    // For CPU version, ignore useCuda parameter and always create CPU instance
    return std::unique_ptr<RGDIC>(new RGDIC(subsetRadius, convergenceThreshold, maxIterations,
                                           ccThreshold, deltaDispThreshold, order, neighborStep));
}

// DisplacementResult methods implementation
void RGDIC::DisplacementResult::convertMatrixToPOIs() {
    if (!u.empty() && !v.empty()) {
        pois.convertFromMatrices(u, v, cc, validMask);
        poisEnabled = true;
    }
}

void RGDIC::DisplacementResult::convertPOIsToMatrix() {
    if (!pois.empty()) {
        pois.convertToMatrices(u, v, cc, validMask);
    }
}

void RGDIC::DisplacementResult::syncPOIsWithMatrices() {
    if (poisEnabled && !pois.empty()) {
        pois.convertToMatrices(u, v, cc, validMask);
    } else if (!u.empty() && !v.empty()) {
        pois.convertFromMatrices(u, v, cc, validMask);
        poisEnabled = true;
    }
}

bool RGDIC::DisplacementResult::exportToCSV(const std::string& filename) const {
    if (poisEnabled && !pois.empty()) {
        return pois.exportToCSV(filename);
    }
    return false;
}

bool RGDIC::DisplacementResult::exportToPOIFormat(const std::string& filename) const {
    if (poisEnabled && !pois.empty()) {
        return pois.exportToPOIFormat(filename);
    }
    return false;
}

// POI visualization functions implementation
void RGDIC::displayPOIResults(const cv::Mat& refImage, const cv::Mat& defImage, 
                             const DisplacementResult& result) {
    if (!result.isPOIsEnabled() || result.pois.empty()) {
        std::cout << "No POI data available for visualization." << std::endl;
        return;
    }
    
    // Create visualization of POI points on reference image
    cv::Mat poiViz = refImage.clone();
    if (poiViz.channels() == 1) {
        cv::cvtColor(poiViz, poiViz, cv::COLOR_GRAY2BGR);
    }
    
    // Draw POI points
    for (const auto& poi : result.pois) {
        cv::Scalar color = poi.valid ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::circle(poiViz, cv::Point(static_cast<int>(poi.leftCoord.x), 
                                    static_cast<int>(poi.leftCoord.y)), 
                  3, color, -1);
    }
    
    cv::imshow("POI Results", poiViz);
    
    // Display statistics
    displayPOIStatistics(result.pois);
}

void RGDIC::displayPOICorrespondences(const cv::Mat& refImage, const cv::Mat& defImage,
                                     const rgdic::POICollection& pois, int maxPoints) {
    if (pois.empty()) {
        std::cout << "No POI correspondences to display." << std::endl;
        return;
    }
    
    // Create side-by-side image
    cv::Mat refBGR = refImage.clone();
    cv::Mat defBGR = defImage.clone();
    
    if (refBGR.channels() == 1) cv::cvtColor(refBGR, refBGR, cv::COLOR_GRAY2BGR);
    if (defBGR.channels() == 1) cv::cvtColor(defBGR, defBGR, cv::COLOR_GRAY2BGR);
    
    cv::Mat combined;
    cv::hconcat(refBGR, defBGR, combined);
    
    // Get valid POIs and limit to maxPoints
    auto validPOIs = pois.getValidPOIs();
    int numToShow = std::min(maxPoints, static_cast<int>(validPOIs.size()));
    
    // Draw correspondences
    for (int i = 0; i < numToShow; ++i) {
        const auto& poi = validPOIs[i];
        
        cv::Point leftPt(static_cast<int>(poi.leftCoord.x), static_cast<int>(poi.leftCoord.y));
        cv::Point rightPt(static_cast<int>(poi.rightCoord.x + refImage.cols), 
                         static_cast<int>(poi.rightCoord.y));
        
        // Color based on correlation quality
        cv::Scalar color;
        if (poi.correlation > 0.9) color = cv::Scalar(0, 255, 0);      // Green - excellent
        else if (poi.correlation > 0.8) color = cv::Scalar(0, 255, 255); // Yellow - good
        else color = cv::Scalar(0, 128, 255);                            // Orange - fair
        
        cv::circle(combined, leftPt, 2, color, -1);
        cv::circle(combined, rightPt, 2, color, -1);
        cv::line(combined, leftPt, rightPt, color, 1);
    }
    
    cv::imshow("POI Correspondences", combined);
    std::cout << "Displaying " << numToShow << " POI correspondences out of " 
              << validPOIs.size() << " valid POIs." << std::endl;
}

void RGDIC::displayPOIStatistics(const rgdic::POICollection& pois) {
    if (pois.empty()) {
        std::cout << "No POI statistics to display." << std::endl;
        return;
    }
    
    std::cout << "\n=== POI Statistics ===" << std::endl;
    std::cout << "Total POIs: " << pois.size() << std::endl;
    std::cout << "Valid POIs: " << pois.getValidCount() << std::endl;
    std::cout << "Valid ratio: " << (100.0 * pois.getValidCount() / pois.size()) << "%" << std::endl;
    std::cout << "Mean correlation: " << pois.getMeanCorrelation() << std::endl;
    
    cv::Vec2f meanDisp = pois.getMeanDisplacement();
    std::cout << "Mean displacement: (" << meanDisp[0] << ", " << meanDisp[1] << ")" << std::endl;
    
    // Correlation distribution
    std::vector<int> corrBins(5, 0); // [0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0]
    for (const auto& poi : pois) {
        if (poi.valid) {
            int bin = std::min(4, static_cast<int>(poi.correlation * 5));
            corrBins[bin]++;
        }
    }
    
    std::cout << "Correlation distribution:" << std::endl;
    const char* binLabels[] = {"[0.0-0.2]", "[0.2-0.4]", "[0.4-0.6]", "[0.6-0.8]", "[0.8-1.0]"};
    for (int i = 0; i < 5; ++i) {
        std::cout << "  " << binLabels[i] << ": " << corrBins[i] << " POIs" << std::endl;
    }
    std::cout << "======================" << std::endl;
}