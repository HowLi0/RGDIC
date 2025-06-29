#include "icgn_optimizer.h"
#include <iostream>
#include <limits>

ICGNOptimizer::ICGNOptimizer(const cv::Mat& refImage, const cv::Mat& defImage,
                             int subsetRadius, ShapeFunctionOrder order,
                             double convergenceThreshold, int maxIterations)
    : m_refImage(refImage),
      m_defImage(defImage),
      m_subsetRadius(subsetRadius),
      m_order(order),
      m_convergenceThreshold(convergenceThreshold),
      m_maxIterations(maxIterations)
{
    // Set number of parameters based on shape function order
    m_numParams = (order == FIRST_ORDER) ? 6 : 12;
}

bool ICGNOptimizer::initialGuess(const cv::Point& refPoint, cv::Mat& warpParams, double& zncc) const {
    // Simple grid search for initial translation parameters
    int searchRadius = m_subsetRadius; // Search radius
    double bestZNCC = std::numeric_limits<double>::max(); // Lower ZNCC = better
    cv::Point bestOffset(0, 0);
    
    // Initialize warp params if not already initialized
    if (warpParams.empty()) {
        warpParams = cv::Mat::zeros(m_numParams, 1, CV_64F);
    }
    
    // Perform grid search
    for (int dy = -searchRadius; dy <= searchRadius; dy += 2) {
        for (int dx = -searchRadius; dx <= searchRadius; dx += 2) {
            cv::Point curPoint = refPoint + cv::Point(dx, dy);
            
            // Check if within bounds
            if (curPoint.x >= m_subsetRadius && curPoint.x < m_defImage.cols - m_subsetRadius &&
                curPoint.y >= m_subsetRadius && curPoint.y < m_defImage.rows - m_subsetRadius) {
                
                // Create simple translation warp
                cv::Mat testParams = cv::Mat::zeros(m_numParams, 1, CV_64F);
                testParams.at<double>(0) = dx;
                testParams.at<double>(1) = dy;
                
                // Compute ZNCC
                double testZNCC = computeZNCC(refPoint, testParams);
                
                // Update best match
                if (testZNCC < bestZNCC) {
                    bestZNCC = testZNCC;
                    bestOffset = cv::Point(dx, dy);
                }
            }
        }
    }
    
    // If no good match found
    if (bestZNCC == std::numeric_limits<double>::max()) {
        return false;
    }
    
    // Set initial translation parameters
    warpParams.at<double>(0) = bestOffset.x;
    warpParams.at<double>(1) = bestOffset.y;
    zncc = bestZNCC;
    
    return true;
}

bool ICGNOptimizer::optimize(const cv::Point& refPoint, cv::Mat& warpParams, double& zncc) const {
    // Initial guess if not provided
    if (warpParams.at<double>(0) == 0 && warpParams.at<double>(1) == 0) {
        if (!initialGuess(refPoint, warpParams, zncc)) {
            return false;
        }
    }
    
    // Pre-compute steepest descent images (constant for IC algorithm)
    std::vector<cv::Mat> steepestDescentImages;
    computeSteepestDescentImages(refPoint, steepestDescentImages);
    
    // Pre-compute Hessian matrix (constant for IC algorithm)
    cv::Mat hessian;
    computeHessian(steepestDescentImages, hessian);
    
    // Check if Hessian is invertible
    cv::Mat hessianInv;
    if (cv::invert(hessian, hessianInv, cv::DECOMP_CHOLESKY) == 0) {
        return false; // Not invertible
    }
    
    // ICGN main loop
    double prevZNCC = std::numeric_limits<double>::max();
    int iter = 0;
    
    while (iter < m_maxIterations) {
        // Compute current ZNCC
        zncc = computeZNCC(refPoint, warpParams);
        
        // Check for convergence
        if (std::abs(zncc - prevZNCC) < m_convergenceThreshold) {
            break;
        }
        
        prevZNCC = zncc;
        
        // Prepare to calculate error vector
        cv::Mat errorVector = cv::Mat::zeros(m_numParams, 1, CV_64F);
        
        // For each pixel in subset
        for (int y = -m_subsetRadius; y <= m_subsetRadius; y++) {
            for (int x = -m_subsetRadius; x <= m_subsetRadius; x++) {
                // Reference subset coordinates
                cv::Point2f refSubsetPt(x, y);
                cv::Point refPixel = refPoint + cv::Point(x, y);
                
                // Check if within reference image bounds
                if (refPixel.x < 0 || refPixel.x >= m_refImage.cols ||
                    refPixel.y < 0 || refPixel.y >= m_refImage.rows) {
                    continue;
                }
                
                // Get reference intensity
                double refIntensity = m_refImage.at<uchar>(refPixel);
                
                // Warp point to get corresponding point in deformed image
                cv::Point2f defPt = warpPoint(refSubsetPt, warpParams);
                cv::Point2f defImgPt(refPoint.x + defPt.x, refPoint.y + defPt.y);
                
                // Check if within deformed image bounds
                if (defImgPt.x < 0 || defImgPt.x >= m_defImage.cols - 1 ||
                    defImgPt.y < 0 || defImgPt.y >= m_defImage.rows - 1) {
                    continue;
                }
                
                // Get deformed intensity (interpolated)
                double defIntensity = interpolate(m_defImage, defImgPt);
                
                // Calculate intensity error
                double error = refIntensity - defIntensity;
                
                // Update error vector
                for (int p = 0; p < m_numParams; p++) {
                    errorVector.at<double>(p) += error * steepestDescentImages[p].at<double>(y + m_subsetRadius, x + m_subsetRadius);
                }
            }
        }
        
        // Calculate parameter update: Δp = H⁻¹ * error
        cv::Mat deltaP = hessianInv * errorVector;
        
        // Update parameters (inverse compositional update)
        // For translation parameters, simple addition works
        warpParams.at<double>(0) += deltaP.at<double>(0);
        warpParams.at<double>(1) += deltaP.at<double>(1);
        
        // For deformation parameters, proper update uses the chain rule
        // This is a simplification - full IC update is more complex
        for (int p = 2; p < m_numParams; p++) {
            warpParams.at<double>(p) += deltaP.at<double>(p);
        }
        
        // Check convergence based on parameter update norm
        double deltaNorm = cv::norm(deltaP);
        if (deltaNorm < m_convergenceThreshold) {
            break;
        }
        
        iter++;
    }
    
    // Final ZNCC calculation
    zncc = computeZNCC(refPoint, warpParams);
    
    return true;
}

double ICGNOptimizer::computeZNCC(const cv::Point& refPoint, const cv::Mat& warpParams) const {
    double sumRef = 0, sumDef = 0;
    double sumRefSq = 0, sumDefSq = 0;
    double sumRefDef = 0;
    int count = 0;
    
    // For each pixel in subset
    for (int y = -m_subsetRadius; y <= m_subsetRadius; y++) {
        for (int x = -m_subsetRadius; x <= m_subsetRadius; x++) {
            // Reference subset coordinates
            cv::Point2f refSubsetPt(x, y);
            cv::Point refPixel = refPoint + cv::Point(x, y);
            
            // Check if within reference image bounds
            if (refPixel.x < 0 || refPixel.x >= m_refImage.cols ||
                refPixel.y < 0 || refPixel.y >= m_refImage.rows) {
                continue;
            }
            
            // Get reference intensity
            double refIntensity = m_refImage.at<uchar>(refPixel);
            
            // Warp point to get corresponding point in deformed image
            cv::Point2f defPt = warpPoint(refSubsetPt, warpParams);
            cv::Point2f defImgPt(refPoint.x + defPt.x, refPoint.y + defPt.y);
            
            // Check if within deformed image bounds
            if (defImgPt.x < 0 || defImgPt.x >= m_defImage.cols - 1 ||
                defImgPt.y < 0 || defImgPt.y >= m_defImage.rows - 1) {
                continue;
            }
            
            // Get deformed intensity (interpolated)
            double defIntensity = interpolate(m_defImage, defImgPt);
            
            // Update sums for ZNCC
            sumRef += refIntensity;
            sumDef += defIntensity;
            sumRefSq += refIntensity * refIntensity;
            sumDefSq += defIntensity * defIntensity;
            sumRefDef += refIntensity * defIntensity;
            count++;
        }
    }
    
    // Calculate ZNCC if we have enough points
    if (count > 0) {
        double meanRef = sumRef / count;
        double meanDef = sumDef / count;
        double varRef = sumRefSq / count - meanRef * meanRef;
        double varDef = sumDefSq / count - meanDef * meanDef;
        double covar = sumRefDef / count - meanRef * meanDef;
        
        if (varRef > 0 && varDef > 0) {
            // Return 1 - ZNCC to convert to minimization problem (0 is perfect match)
            return 1.0 - (covar / std::sqrt(varRef * varDef));
        }
    }
    
    return std::numeric_limits<double>::max(); // Error case
}

void ICGNOptimizer::computeSteepestDescentImages(const cv::Point& refPoint, 
                                                 std::vector<cv::Mat>& steepestDescentImages) const {
    // Calculate image gradients
    cv::Mat gradX, gradY;
    cv::Sobel(m_refImage, gradX, CV_64F, 1, 0, 3);
    cv::Sobel(m_refImage, gradY, CV_64F, 0, 1, 3);
    
    // Initialize steepest descent images
    steepestDescentImages.clear();
    for (int i = 0; i < m_numParams; i++) {
        steepestDescentImages.push_back(cv::Mat::zeros(2 * m_subsetRadius + 1, 2 * m_subsetRadius + 1, CV_64F));
    }
    
    // For each pixel in subset
    for (int y = -m_subsetRadius; y <= m_subsetRadius; y++) {
        for (int x = -m_subsetRadius; x <= m_subsetRadius; x++) {
            cv::Point pixel = refPoint + cv::Point(x, y);
            
            // Check if within image bounds
            if (pixel.x < 0 || pixel.x >= m_refImage.cols ||
                pixel.y < 0 || pixel.y >= m_refImage.rows) {
                continue;
            }
            
            // Get gradients at this pixel
            double dx = gradX.at<double>(pixel);
            double dy = gradY.at<double>(pixel);
            
            // Compute Jacobian matrix
            cv::Mat jacobian;
            computeWarpJacobian(cv::Point2f(x, y), jacobian);
            
            // Compute steepest descent images
            int row = y + m_subsetRadius;
            int col = x + m_subsetRadius;
            
            // First order parameters
            steepestDescentImages[0].at<double>(row, col) = dx; // du
            steepestDescentImages[1].at<double>(row, col) = dy; // dv
            steepestDescentImages[2].at<double>(row, col) = dx * x; // du/dx
            steepestDescentImages[3].at<double>(row, col) = dx * y; // du/dy
            steepestDescentImages[4].at<double>(row, col) = dy * x; // dv/dx
            steepestDescentImages[5].at<double>(row, col) = dy * y; // dv/dy
            
            // Second order parameters (if applicable)
            if (m_order == SECOND_ORDER) {
                steepestDescentImages[6].at<double>(row, col) = dx * x * x / 2.0; // d²u/dx²
                steepestDescentImages[7].at<double>(row, col) = dx * x * y; // d²u/dxdy
                steepestDescentImages[8].at<double>(row, col) = dx * y * y / 2.0; // d²u/dy²
                steepestDescentImages[9].at<double>(row, col) = dy * x * x / 2.0; // d²v/dx²
                steepestDescentImages[10].at<double>(row, col) = dy * x * y; // d²v/dxdy
                steepestDescentImages[11].at<double>(row, col) = dy * y * y / 2.0; // d²v/dy²
            }
        }
    }
}

void ICGNOptimizer::computeHessian(const std::vector<cv::Mat>& steepestDescentImages, 
                                   cv::Mat& hessian) const {
    // Initialize Hessian matrix
    hessian = cv::Mat::zeros(m_numParams, m_numParams, CV_64F);
    
    // For each parameter pair
    for (int i = 0; i < m_numParams; i++) {
        for (int j = i; j < m_numParams; j++) { // Take advantage of symmetry
            double sum = 0;
            
            // Sum over all pixels in subset
            for (int y = 0; y < 2 * m_subsetRadius + 1; y++) {
                for (int x = 0; x < 2 * m_subsetRadius + 1; x++) {
                    sum += steepestDescentImages[i].at<double>(y, x) * steepestDescentImages[j].at<double>(y, x);
                }
            }
            
            // Set Hessian element
            hessian.at<double>(i, j) = sum;
            
            // Set symmetric element
            if (i != j) {
                hessian.at<double>(j, i) = sum;
            }
        }
    }
}

cv::Point2f ICGNOptimizer::warpPoint(const cv::Point2f& pt, const cv::Mat& warpParams) const {
    double x = pt.x;
    double y = pt.y;
    
    // Extract parameters
    double u = warpParams.at<double>(0);
    double v = warpParams.at<double>(1);
    double dudx = warpParams.at<double>(2);
    double dudy = warpParams.at<double>(3);
    double dvdx = warpParams.at<double>(4);
    double dvdy = warpParams.at<double>(5);
    
    // First-order warp
    double warpedX = x + u + dudx * x + dudy * y;
    double warpedY = y + v + dvdx * x + dvdy * y;
    
    // Add second-order terms if using second-order shape function
    if (m_order == SECOND_ORDER && m_numParams >= 12) {
        double d2udx2 = warpParams.at<double>(6);
        double d2udxdy = warpParams.at<double>(7);
        double d2udy2 = warpParams.at<double>(8);
        double d2vdx2 = warpParams.at<double>(9);
        double d2vdxdy = warpParams.at<double>(10);
        double d2vdy2 = warpParams.at<double>(11);
        
        warpedX += 0.5 * d2udx2 * x * x + d2udxdy * x * y + 0.5 * d2udy2 * y * y;
        warpedY += 0.5 * d2vdx2 * x * x + d2vdxdy * x * y + 0.5 * d2vdy2 * y * y;
    }
    
    return cv::Point2f(warpedX, warpedY);
}

void ICGNOptimizer::computeWarpJacobian(const cv::Point2f& pt, cv::Mat& jacobian) const {
    double x = pt.x;
    double y = pt.y;
    
    // For first-order shape function
    if (m_order == FIRST_ORDER) {
        jacobian = (cv::Mat_<double>(2, 6) << 
                   1, 0, x, y, 0, 0,
                   0, 1, 0, 0, x, y);
    }
    // For second-order shape function
    else {
        jacobian = (cv::Mat_<double>(2, 12) << 
                   1, 0, x, y, 0, 0, 0.5*x*x, x*y, 0.5*y*y, 0, 0, 0,
                   0, 1, 0, 0, x, y, 0, 0, 0, 0.5*x*x, x*y, 0.5*y*y);
    }
}

double ICGNOptimizer::interpolate(const cv::Mat& image, const cv::Point2f& pt) const {
    // Bounds check
    if (pt.x < 0 || pt.x >= image.cols - 1 || pt.y < 0 || pt.y >= image.rows - 1) {
        return 0;
    }
    
    // Get integer and fractional parts
    int x1 = static_cast<int>(pt.x);
    int y1 = static_cast<int>(pt.y);
    int x2 = x1 + 1;
    int y2 = y1 + 1;
    
    double fx = pt.x - x1;
    double fy = pt.y - y1;
    
    // Bilinear interpolation
    double val = (1 - fx) * (1 - fy) * image.at<uchar>(y1, x1) +
                fx * (1 - fy) * image.at<uchar>(y1, x2) +
                (1 - fx) * fy * image.at<uchar>(y2, x1) +
                fx * fy * image.at<uchar>(y2, x2);
    
    return val;
}
