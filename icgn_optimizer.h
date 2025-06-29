#ifndef ICGN_OPTIMIZER_H
#define ICGN_OPTIMIZER_H

#include <opencv2/opencv.hpp>
#include <vector>

// Enumeration for shape function order
enum ShapeFunctionOrder {
    FIRST_ORDER = 6,  // 6 parameters: u, v, du/dx, du/dy, dv/dx, dv/dy
    SECOND_ORDER = 12  // 12 parameters: first order + second derivatives
};

// ICGN (Inverse Compositional Gauss-Newton) optimizer class
class ICGNOptimizer {
public:
    ICGNOptimizer(const cv::Mat& refImage, const cv::Mat& defImage,
                  int subsetRadius, ShapeFunctionOrder order,
                  double convergenceThreshold, int maxIterations);
    
    // Finds optimal deformation parameters for a point
    bool optimize(const cv::Point& refPoint, cv::Mat& warpParams, double& zncc) const;
    
    // Initial guess using ZNCC grid search
    bool initialGuess(const cv::Point& refPoint, cv::Mat& warpParams, double& zncc) const;
    
    // Getter for number of parameters
    int getNumParams() const { return m_numParams; }

private:
    const cv::Mat& m_refImage;
    const cv::Mat& m_defImage;
    int m_subsetRadius;
    ShapeFunctionOrder m_order;
    double m_convergenceThreshold;
    int m_maxIterations;
    int m_numParams;
    
    // Computes ZNCC between reference subset and warped current subset
    double computeZNCC(const cv::Point& refPoint, const cv::Mat& warpParams) const;
    
    // Computes steepest descent images
    void computeSteepestDescentImages(const cv::Point& refPoint, 
                                     std::vector<cv::Mat>& steepestDescentImages) const;
    
    // Computes Hessian matrix
    void computeHessian(const std::vector<cv::Mat>& steepestDescentImages, cv::Mat& hessian) const;
    
    // Warps a point using shape function parameters
    cv::Point2f warpPoint(const cv::Point2f& pt, const cv::Mat& warpParams) const;
    
    // Computes warp Jacobian
    void computeWarpJacobian(const cv::Point2f& pt, cv::Mat& jacobian) const;
    
    // Interpolates image intensity at non-integer coordinates
    double interpolate(const cv::Mat& image, const cv::Point2f& pt) const;
};

#endif // ICGN_OPTIMIZER_H
