#ifndef GLOBAL_DIC_H
#define GLOBAL_DIC_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <memory>
#include <unordered_map>

class GlobalDIC {
public:
    // Enumeration for shape function order
    enum ShapeFunctionOrder {
        FIRST_ORDER = 1,  // 6 parameters: u, v, du/dx, du/dy, dv/dx, dv/dy
        SECOND_ORDER = 2  // 12 parameters: first order + second derivatives
    };
    
    // Enumeration for regularization type
    enum RegularizationType {
        TIKHONOV = 0,     // Simple Tikhonov regularization (default)
        DIFFUSION = 1,    // Diffusion-based regularization (better preserves edges)
        TOTAL_VARIATION = 2 // Total variation regularization (preserves discontinuities)
    };
    
    // Result structure to hold displacement and strain fields
    struct Result {
        cv::Mat u;           // x-displacement field
        cv::Mat v;           // y-displacement field
        cv::Mat exx;         // x-direction normal strain
        cv::Mat eyy;         // y-direction normal strain
        cv::Mat exy;         // shear strain
        cv::Mat cc;          // correlation coefficient
        cv::Mat validMask;   // valid points mask
        
        // Confidence metrics
        cv::Mat confidence;  // confidence metric (0-1)
        double meanResidual; // mean residual after convergence
        int iterations;      // number of iterations performed
        
        // Constructor to initialize all matrices
        Result(const cv::Size& size);
    };
    
    // Parameters structure to centralize all parameters
    struct Parameters {
        int nodeSpacing;              // Spacing between nodes in pixels
        int subsetRadius;             // Radius of subset around each node
        double regularizationWeight;  // Weight of regularization term
        RegularizationType regType;   // Type of regularization to use
        double convergenceThreshold;  // Convergence criterion
        int maxIterations;            // Maximum number of iterations
        ShapeFunctionOrder order;     // Shape function order
        bool useMultiScaleApproach;   // Use multi-scale approach
        int numScaleLevels;           // Number of scale levels
        double scaleFactor;           // Scale factor between levels
        bool useParallel;             // Use parallelization
        cv::Size minImageSize;        // 最小允许的图像尺寸
        
        // Constructor with default values
        Parameters();
    };
    
    // Constructor
    GlobalDIC(const Parameters& params = Parameters());
    
    // Main function to perform Global DIC analysis
    Result compute(const cv::Mat& refImage, 
                   const cv::Mat& defImage,
                   const cv::Mat& roi);
    
    // Calculate strain fields from displacement fields
    void calculateStrains(Result& result, bool useWindowedLeastSquares = true, int windowSize = 5);
    
    // Utility function to display results
    void displayResults(const cv::Mat& refImage, const Result& result, 
                        bool showDisplacement = true, bool showStrain = true,
                        bool useEnhancedVisualization = true);
    
    // Evaluate errors if ground truth is available
    void evaluateErrors(const Result& result, 
                        const cv::Mat& trueDispX, const cv::Mat& trueDispY);
                        
    // Get/Set parameters
    Parameters& getParameters() { return m_params; }
    void setParameters(const Parameters& params) { m_params = params; }

private:
    // Parameters
    Parameters m_params;
    
    // Create node grid based on ROI
    void createNodeGrid(const cv::Mat& roi, 
                        std::vector<cv::Point>& nodePoints);
    
    // Calculate global DIC system matrix and vector
    void buildGlobalSystem(const cv::Mat& refImage, 
                          const cv::Mat& defImage,
                          const std::vector<cv::Point>& nodePoints,
                          cv::Mat& systemMatrix,
                          cv::Mat& systemVector,
                          const cv::Mat& nodeDisplacements);
    
    // Add regularization terms to system
    void addRegularization(const std::vector<cv::Point>& nodePoints,
                          cv::Mat& systemMatrix);
    
    // Solve the system efficiently
    bool solveSystem(const cv::Mat& systemMatrix, const cv::Mat& systemVector, 
                    cv::Mat& solution, double& residualNorm);
    
    // Interpolate displacement at arbitrary point using FEM shape functions
    void interpolateDisplacement(const cv::Point& point,
                                const std::vector<cv::Point>& nodePoints,
                                const cv::Mat& nodeDisplacements,
                                double& u, double& v);
    
    // Calculate correlation coefficient between reference and warped image
    double calculateCorrelation(const cv::Mat& refImage, 
                               const cv::Mat& defImage,
                               const std::vector<cv::Point>& nodePoints,
                               const cv::Mat& nodeDisplacements);
    
    // Interpolate intensity at non-integer coordinates
    double interpolate(const cv::Mat& image, const cv::Point2f& pt) const;
    
    // Generate full displacement field from nodal displacements
    void generateDisplacementField(const std::vector<cv::Point>& nodePoints,
                                  const cv::Mat& nodeDisplacements,
                                  Result& result,
                                  const cv::Mat& roi);
                                  
    // Multi-scale implementation
    Result computeMultiScale(const cv::Mat& refImage, 
                           const cv::Mat& defImage,
                           const cv::Mat& roi);
    
    // Compute element shape functions and derivatives
    void computeShapeFunctions(const cv::Point& point, 
                              const std::vector<cv::Point>& elementNodes,
                              std::vector<double>& N,
                              std::vector<double>& dNdx,
                              std::vector<double>& dNdy);
                              
    // Calculate confidence metrics
    void calculateConfidence(Result& result, 
                           const cv::Mat& refImage, 
                           const cv::Mat& defImage);
    
    // Calculate mean residual for each node
    double calculateResidual(const cv::Mat& refImage, 
                           const cv::Mat& defImage,
                           const std::vector<cv::Point>& nodePoints,
                           const cv::Mat& nodeDisplacements);

    cv::Mat createEnhancedVisualization(const cv::Mat& data, const cv::Mat& validMask, 
                                        double minVal, double maxVal, 
                                        const std::string& title, bool addIsolines);
    
    // Cached gradient images
    mutable cv::Mat m_gradX, m_gradY;
    mutable bool m_gradientsComputed;
    
    // Internal cache for shape function values
    using ShapeFunctionCache = std::unordered_map<size_t, std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>>;
    mutable ShapeFunctionCache m_shapeFunctionCache;
};

#endif // GLOBAL_DIC_H