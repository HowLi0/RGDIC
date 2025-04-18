#include "global_dic.h"
#include <iostream>
#include <functional>
#include <omp.h>
#include <chrono>

// Constructor for Results
GlobalDIC::Result::Result(const cv::Size& size) {
    u = cv::Mat::zeros(size, CV_64F);
    v = cv::Mat::zeros(size, CV_64F);
    exx = cv::Mat::zeros(size, CV_64F);
    eyy = cv::Mat::zeros(size, CV_64F);
    exy = cv::Mat::zeros(size, CV_64F);
    cc = cv::Mat::zeros(size, CV_64F);
    validMask = cv::Mat::zeros(size, CV_8U);
    confidence = cv::Mat::zeros(size, CV_64F);
    meanResidual = 0.0;
    iterations = 0;
}

// Default parameters
GlobalDIC::Parameters::Parameters() 
    : nodeSpacing(15),
      subsetRadius(15),
      regularizationWeight(0.1),
      regType(TIKHONOV),
      convergenceThreshold(0.001),
      maxIterations(50),
      order(FIRST_ORDER),
      useMultiScaleApproach(true),
      numScaleLevels(3),
      scaleFactor(0.5),
      useParallel(true),
      minImageSize(cv::Size(16, 16)) // Set minimum image size to 16x16 
      {}

GlobalDIC::GlobalDIC(const Parameters& params)
    : m_params(params), m_gradientsComputed(false) {
}

GlobalDIC::Result GlobalDIC::compute(const cv::Mat& refImage, const cv::Mat& defImage, const cv::Mat& roi) {
    // Reset gradient cache
    m_gradientsComputed = false;
    m_gradX = cv::Mat();
    m_gradY = cv::Mat();
    m_shapeFunctionCache.clear();
    
    // Check if multi-scale approach is enabled
    if (m_params.useMultiScaleApproach) {
        return computeMultiScale(refImage, defImage, roi);
    }
    
    // Initialize result structure
    Result result(roi.size());
    result.validMask = roi.clone();
    
    // Create node grid
    std::vector<cv::Point> nodePoints;
    createNodeGrid(roi, nodePoints);
    
    // Number of nodes and degrees of freedom
    int numNodes = static_cast<int>(nodePoints.size());
    int numDOFs = numNodes * 2;  // 2 DOFs per node (u and v)
    
    std::cout << "Total nodes: " << numNodes << ", Total DOFs: " << numDOFs << std::endl;
    
    // Initial guess for displacements - start with zeros
    cv::Mat nodeDisplacements = cv::Mat::zeros(numDOFs, 1, CV_64F);
    
    // Iterative solution
    double prevResidual = std::numeric_limits<double>::max();
    int iter = 0;
    double residualNorm = 0.0;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    for (iter = 0; iter < m_params.maxIterations; iter++) {
        // Build global system matrix and vector
        cv::Mat systemMatrix = cv::Mat::zeros(numDOFs, numDOFs, CV_64F);
        cv::Mat systemVector = cv::Mat::zeros(numDOFs, 1, CV_64F);
        
        buildGlobalSystem(refImage, defImage, nodePoints, systemMatrix, systemVector, nodeDisplacements);
        
        // Add regularization
        addRegularization(nodePoints, systemMatrix);
        
        // Solve the system
        cv::Mat deltaDisplacements;
        bool solveSuccess = solveSystem(systemMatrix, systemVector, deltaDisplacements, residualNorm);
        
        if (!solveSuccess) {
            std::cout << "Warning: Failed to solve the system at iteration " << iter << std::endl;
            break;
        }
        
        // Update displacement field
        nodeDisplacements += deltaDisplacements;
        
        // Check convergence
        std::cout << "Iteration " << iter 
                  << ", Residual: " << residualNorm 
                  << std::endl;
                  
        if (residualNorm < m_params.convergenceThreshold || 
            std::abs(residualNorm - prevResidual) < m_params.convergenceThreshold / 10) {
            std::cout << "Convergence achieved after " << iter << " iterations." << std::endl;
            break;
        }
        
        prevResidual = residualNorm;
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime = endTime - startTime;
    
    std::cout << "Solved in " << elapsedTime.count() << " seconds after " 
              << iter << " iterations." << std::endl;
    
    // Record iterations performed
    result.iterations = iter;
    
    // Generate full displacement field from nodal displacements
    generateDisplacementField(nodePoints, nodeDisplacements, result, roi);
    
    // Calculate correlation coefficient
    double cc = calculateCorrelation(refImage, defImage, nodePoints, nodeDisplacements);
    std::cout << "Final correlation coefficient: " << cc << std::endl;
    
    // Calculate mean residual
    result.meanResidual = calculateResidual(refImage, defImage, nodePoints, nodeDisplacements);
    
    // Calculate confidence metrics
    calculateConfidence(result, refImage, defImage);
    
    // Calculate strain fields
    calculateStrains(result);
    
    return result;
}

void GlobalDIC::createNodeGrid(const cv::Mat& roi, std::vector<cv::Point>& nodePoints) {
    nodePoints.clear();
    
    // Create a regular grid of nodes within the ROI
    for (int y = m_params.subsetRadius; y < roi.rows - m_params.subsetRadius; y += m_params.nodeSpacing) {
        for (int x = m_params.subsetRadius; x < roi.cols - m_params.subsetRadius; x += m_params.nodeSpacing) {
            if (roi.at<uchar>(y, x) > 0) {
                nodePoints.push_back(cv::Point(x, y));
            }
        }
    }
    
    std::cout << "Created " << nodePoints.size() << " nodes for global DIC." << std::endl;
}

void GlobalDIC::buildGlobalSystem(const cv::Mat& refImage, const cv::Mat& defImage,
                                const std::vector<cv::Point>& nodePoints,
                                cv::Mat& systemMatrix, cv::Mat& systemVector,
                                const cv::Mat& nodeDisplacements) {
    int numNodes = static_cast<int>(nodePoints.size());
    int numDOFs = numNodes * 2;
    
    // Convert images to floating point if they're not already
    cv::Mat refFloat, defFloat;
    if (refImage.type() != CV_64F) {
        refImage.convertTo(refFloat, CV_64F);
    } else {
        refFloat = refImage;
    }
    
    if (defImage.type() != CV_64F) {
        defImage.convertTo(defFloat, CV_64F);
    } else {
        defFloat = defImage;
    }
    
    // Calculate image gradients if not already computed
    if (!m_gradientsComputed) {
        cv::Sobel(refFloat, m_gradX, CV_64F, 1, 0, 3);
        cv::Sobel(refFloat, m_gradY, CV_64F, 0, 1, 3);
        m_gradientsComputed = true;
    }
    
    // Reset the system matrix and vector
    systemMatrix = cv::Mat::zeros(numDOFs, numDOFs, CV_64F);
    systemVector = cv::Mat::zeros(numDOFs, 1, CV_64F);
    
    // Use OpenMP for parallelization if enabled
    #pragma omp parallel for if(m_params.useParallel)
    for (int nodeIdx = 0; nodeIdx < numNodes; nodeIdx++) {
        cv::Point nodePoint = nodePoints[nodeIdx];
        
        // Thread-local variables for accumulation
        std::vector<std::vector<double>> localSystemMatrix(numDOFs, std::vector<double>(numDOFs, 0.0));
        std::vector<double> localSystemVector(numDOFs, 0.0);
        
        // For each pixel in subset around this node
        for (int y = -m_params.subsetRadius; y <= m_params.subsetRadius; y++) {
            for (int x = -m_params.subsetRadius; x <= m_params.subsetRadius; x++) {
                cv::Point pixelPoint = nodePoint + cv::Point(x, y);
                
                // Check if the pixel is within image bounds
                if (pixelPoint.x < 0 || pixelPoint.x >= refFloat.cols ||
                    pixelPoint.y < 0 || pixelPoint.y >= refFloat.rows) {
                    continue;
                }
                
                // Get the reference intensity and gradients
                double refIntensity = refFloat.at<double>(pixelPoint);
                double gx = m_gradX.at<double>(pixelPoint);
                double gy = m_gradY.at<double>(pixelPoint);
                
                // Compute warped point based on current displacement estimates
                double u, v;
                interpolateDisplacement(pixelPoint, nodePoints, nodeDisplacements, u, v);
                cv::Point2f warpedPoint(pixelPoint.x + u, pixelPoint.y + v);
                
                // Check if warped point is within image bounds
                if (warpedPoint.x < 0 || warpedPoint.x >= defFloat.cols - 1 ||
                    warpedPoint.y < 0 || warpedPoint.y >= defFloat.rows - 1) {
                    continue;
                }
                
                // Get deformed intensity using bilinear interpolation
                double defIntensity = interpolate(defFloat, warpedPoint);
                
                // Calculate intensity difference
                double intensityDiff = refIntensity - defIntensity;
                
                // Find which element this pixel belongs to
                // For this implementation, we use a distance-based approach
                std::vector<int> influencingNodes;
                std::vector<double> weights;
                
                // Find all nodes within influence radius
                double influenceRadius = m_params.nodeSpacing * 1.5;
                for (int j = 0; j < numNodes; j++) {
                    cv::Point otherNode = nodePoints[j];
                    double dist = cv::norm(cv::Mat(pixelPoint - otherNode));
                    
                    if (dist <= influenceRadius) {
                        influencingNodes.push_back(j);
                        
                        // Calculate weight based on distance (linear weight)
                        double weight = 1.0 - (dist / influenceRadius);
                        weight = std::max(0.0, std::min(1.0, weight));
                        weights.push_back(weight);
                    }
                }
                
                // Normalize weights
                double weightSum = 0.0;
                for (double w : weights) weightSum += w;
                if (weightSum > 0) {
                    for (double& w : weights) w /= weightSum;
                }
                
                // Update system equations based on this pixel's contribution
                for (size_t i = 0; i < influencingNodes.size(); i++) {
                    int nodeI = influencingNodes[i];
                    double weightI = weights[i];
                    
                    // Contribution to system vector (RHS)
                    localSystemVector[nodeI * 2] += weightI * gx * intensityDiff;
                    localSystemVector[nodeI * 2 + 1] += weightI * gy * intensityDiff;
                    
                    // Contribution to system matrix (LHS)
                    for (size_t j = 0; j < influencingNodes.size(); j++) {
                        int nodeJ = influencingNodes[j];
                        double weightJ = weights[j];
                        
                        localSystemMatrix[nodeI * 2][nodeJ * 2] += weightI * weightJ * gx * gx;
                        localSystemMatrix[nodeI * 2][nodeJ * 2 + 1] += weightI * weightJ * gx * gy;
                        localSystemMatrix[nodeI * 2 + 1][nodeJ * 2] += weightI * weightJ * gy * gx;
                        localSystemMatrix[nodeI * 2 + 1][nodeJ * 2 + 1] += weightI * weightJ * gy * gy;
                    }
                }
            }
        }
        
        // Update global system with thread-local results
        #pragma omp critical
        {
            for (int i = 0; i < numDOFs; i++) {
                systemVector.at<double>(i) += localSystemVector[i];
                for (int j = 0; j < numDOFs; j++) {
                    systemMatrix.at<double>(i, j) += localSystemMatrix[i][j];
                }
            }
        }
    }
}

void GlobalDIC::addRegularization(const std::vector<cv::Point>& nodePoints, cv::Mat& systemMatrix) {
    int numNodes = static_cast<int>(nodePoints.size());
    
    // Choose regularization method based on parameter
    switch (m_params.regType) {
        case TIKHONOV: {
            // Standard Tikhonov regularization
            #pragma omp parallel for if(m_params.useParallel)
            for (int i = 0; i < numNodes; i++) {
                cv::Point nodeI = nodePoints[i];
                
                for (int j = 0; j < numNodes; j++) {
                    if (i == j) continue;
                    
                    cv::Point nodeJ = nodePoints[j];
                    double dist = cv::norm(cv::Mat(nodeI - nodeJ));
                    
                    // Only consider nearby nodes for regularization
                    if (dist <= m_params.nodeSpacing * 1.5) {
                        double weight = m_params.regularizationWeight * exp(-dist / m_params.nodeSpacing);
                        
                        #pragma omp critical
                        {
                            // Add regularization terms that encourage similar displacements for nearby nodes
                            systemMatrix.at<double>(i * 2, j * 2) -= weight;
                            systemMatrix.at<double>(i * 2, i * 2) += weight;
                            systemMatrix.at<double>(i * 2 + 1, j * 2 + 1) -= weight;
                            systemMatrix.at<double>(i * 2 + 1, i * 2 + 1) += weight;
                        }
                    }
                }
            }
            break;
        }
        
        case DIFFUSION: {
            // Diffusion-based regularization (edge-preserving)
            double beta = 0.1; // Edge detection parameter
            
            #pragma omp parallel for if(m_params.useParallel)
            for (int i = 0; i < numNodes; i++) {
                cv::Point nodeI = nodePoints[i];
                
                for (int j = 0; j < numNodes; j++) {
                    if (i == j) continue;
                    
                    cv::Point nodeJ = nodePoints[j];
                    double dist = cv::norm(cv::Mat(nodeI - nodeJ));
                    
                    if (dist <= m_params.nodeSpacing * 1.5) {
                        // Calculate gradient magnitude around these nodes if we have gradient images
                        double gradientMagnitude = 0.0;
                        if (m_gradientsComputed) {
                            double midX = (nodeI.x + nodeJ.x) / 2;
                            double midY = (nodeI.y + nodeJ.y) / 2;
                            if (midX >= 0 && midX < m_gradX.cols && midY >= 0 && midY < m_gradX.rows) {
                                double gx = m_gradX.at<double>(midY, midX);
                                double gy = m_gradY.at<double>(midY, midX);
                                gradientMagnitude = sqrt(gx*gx + gy*gy);
                            }
                        }
                        
                        // Calculate diffusion coefficient
                        double diffCoeff = exp(-(gradientMagnitude * gradientMagnitude) / (beta * beta));
                        
                        // Scale regularization by diffusion coefficient
                        double weight = m_params.regularizationWeight * diffCoeff * exp(-dist / m_params.nodeSpacing);
                        
                        #pragma omp critical
                        {
                            systemMatrix.at<double>(i * 2, j * 2) -= weight;
                            systemMatrix.at<double>(i * 2, i * 2) += weight;
                            systemMatrix.at<double>(i * 2 + 1, j * 2 + 1) -= weight;
                            systemMatrix.at<double>(i * 2 + 1, i * 2 + 1) += weight;
                        }
                    }
                }
            }
            break;
        }
        
        case TOTAL_VARIATION: {
            // Total Variation regularization - approximated version for DIC
            double epsilon = 1e-6; // Small value to avoid division by zero
            
            // This is just an approximation - true TV would require an iterative solution
            #pragma omp parallel for if(m_params.useParallel)
            for (int i = 0; i < numNodes; i++) {
                cv::Point nodeI = nodePoints[i];
                
                for (int j = 0; j < numNodes; j++) {
                    if (i == j) continue;
                    
                    cv::Point nodeJ = nodePoints[j];
                    double dist = cv::norm(cv::Mat(nodeI - nodeJ));
                    
                    if (dist <= m_params.nodeSpacing * 1.5) {
                        // For simplicity, use a constant weight
                        double weight = m_params.regularizationWeight * exp(-dist / m_params.nodeSpacing);
                        
                        #pragma omp critical
                        {
                            systemMatrix.at<double>(i * 2, j * 2) -= weight;
                            systemMatrix.at<double>(i * 2, i * 2) += weight;
                            systemMatrix.at<double>(i * 2 + 1, j * 2 + 1) -= weight;
                            systemMatrix.at<double>(i * 2 + 1, i * 2 + 1) += weight;
                        }
                    }
                }
            }
            break;
        }
    }
    
    // Add small value to diagonal for numerical stability
    for (int i = 0; i < systemMatrix.rows; i++) {
        systemMatrix.at<double>(i, i) += 1e-6;
    }
}

bool GlobalDIC::solveSystem(const cv::Mat& systemMatrix, const cv::Mat& systemVector, 
                          cv::Mat& solution, double& residualNorm) {
    // Try different solvers in order of preference
    bool success = false;
    
    // First try Cholesky - fastest but requires positive definite matrix
    try {
        success = cv::solve(systemMatrix, systemVector, solution, cv::DECOMP_CHOLESKY);
    } catch (const cv::Exception& e) {
        std::cout << "Cholesky decomposition failed: " << e.what() << std::endl;
        success = false;
    }
    
    // If Cholesky failed, try LU decomposition
    if (!success) {
        try {
            success = cv::solve(systemMatrix, systemVector, solution, cv::DECOMP_LU);
        } catch (const cv::Exception& e) {
            std::cout << "LU decomposition failed: " << e.what() << std::endl;
            success = false;
        }
    }
    
    // If LU failed, try SVD (slow but most robust)
    if (!success) {
        try {
            success = cv::solve(systemMatrix, systemVector, solution, cv::DECOMP_SVD);
        } catch (const cv::Exception& e) {
            std::cout << "SVD decomposition failed: " << e.what() << std::endl;
            success = false;
        }
    }
    
    // Calculate residual norm if solution was found
    if (success) {
        cv::Mat residual = systemVector - systemMatrix * solution;
        residualNorm = cv::norm(residual);
    } else {
        residualNorm = std::numeric_limits<double>::max();
    }
    
    return success;
}

void GlobalDIC::interpolateDisplacement(const cv::Point& point,
                                      const std::vector<cv::Point>& nodePoints,
                                      const cv::Mat& nodeDisplacements,
                                      double& u, double& v) {
    u = v = 0.0;
    double totalWeight = 0.0;
    
    // Find the nearest nodes and interpolate using inverse distance weighting
    for (size_t i = 0; i < nodePoints.size(); i++) {
        double dist = cv::norm(cv::Mat(point - nodePoints[i]));
        
        // Only use nodes within a certain radius
        if (dist <= m_params.nodeSpacing * 2.0) {
            // Inverse distance weight with quadratic falloff
            double weight = 1.0 / (dist*dist + 1e-6);
            totalWeight += weight;
            
            u += weight * nodeDisplacements.at<double>(2 * i);
            v += weight * nodeDisplacements.at<double>(2 * i + 1);
        }
    }
    
    if (totalWeight > 0) {
        u /= totalWeight;
        v /= totalWeight;
    }
}

double GlobalDIC::calculateCorrelation(const cv::Mat& refImage, const cv::Mat& defImage,
                                     const std::vector<cv::Point>& nodePoints,
                                     const cv::Mat& nodeDisplacements) {
    double sumRefDef = 0, sumRef = 0, sumDef = 0;
    double sumRefSq = 0, sumDefSq = 0;
    int count = 0;
    
    // Convert images to floating point if they're not already
    cv::Mat refFloat, defFloat;
    if (refImage.type() != CV_64F) {
        refImage.convertTo(refFloat, CV_64F);
    } else {
        refFloat = refImage;
    }
    
    if (defImage.type() != CV_64F) {
        defImage.convertTo(defFloat, CV_64F);
    } else {
        defFloat = defImage;
    }
    
    // For each pixel in the ROI (sample every few pixels for speed)
    int step = 5;
    for (int y = m_params.subsetRadius; y < refImage.rows - m_params.subsetRadius; y += step) {
        for (int x = m_params.subsetRadius; x < refImage.cols - m_params.subsetRadius; x += step) {
            cv::Point pt(x, y);
            
            // Get reference intensity
            double refIntensity = refFloat.at<double>(pt);
            
            // Interpolate displacement at this pixel
            double u, v;
            interpolateDisplacement(pt, nodePoints, nodeDisplacements, u, v);
            
            // Calculate deformed position
            cv::Point2f defPt(x + u, y + v);
            
            // Check if deformed point is within bounds
            if (defPt.x >= 0 && defPt.x < defImage.cols - 1 &&
                defPt.y >= 0 && defPt.y < defImage.rows - 1) {
                
                // Get deformed intensity (interpolated)
                double defIntensity = interpolate(defFloat, defPt);
                
                // Update correlation sums
                sumRef += refIntensity;
                sumDef += defIntensity;
                sumRefSq += refIntensity * refIntensity;
                sumDefSq += defIntensity * defIntensity;
                sumRefDef += refIntensity * defIntensity;
                count++;
            }
        }
    }
    
    // Calculate correlation coefficient (ZNCC)
    if (count > 0) {
        double meanRef = sumRef / count;
        double meanDef = sumDef / count;
        double varRef = sumRefSq / count - meanRef * meanRef;
        double varDef = sumDefSq / count - meanDef * meanDef;
        double covar = sumRefDef / count - meanRef * meanDef;
        
        if (varRef > 0 && varDef > 0) {
            return covar / std::sqrt(varRef * varDef);
        }
    }
    
    return 0.0;
}

void GlobalDIC::generateDisplacementField(const std::vector<cv::Point>& nodePoints,
                                        const cv::Mat& nodeDisplacements,
                                        Result& result,
                                        const cv::Mat& roi) {
    // For each pixel in the ROI
    #pragma omp parallel for if(m_params.useParallel)
    for (int y = 0; y < roi.rows; y++) {
        for (int x = 0; x < roi.cols; x++) {
            if (roi.at<uchar>(y, x) > 0) {
                cv::Point pt(x, y);
                
                // Interpolate displacement at this pixel
                double u, v;
                interpolateDisplacement(pt, nodePoints, nodeDisplacements, u, v);
                
                // Store the interpolated displacements
                result.u.at<double>(y, x) = u;
                result.v.at<double>(y, x) = v;
                result.validMask.at<uchar>(y, x) = 1;
            }
        }
    }
}

void GlobalDIC::calculateStrains(Result& result, bool useWindowedLeastSquares, int windowSize) {
    // Calculate strain fields from displacement gradients
    cv::Mat gradUx, gradUy, gradVx, gradVy;
    
    if (useWindowedLeastSquares) {
        // Use windowed least squares to compute smoother derivatives
        int halfWindow = windowSize / 2;
        
        // Initialize gradient matrices
        gradUx = cv::Mat::zeros(result.u.size(), CV_64F);
        gradUy = cv::Mat::zeros(result.u.size(), CV_64F);
        gradVx = cv::Mat::zeros(result.v.size(), CV_64F);
        gradVy = cv::Mat::zeros(result.v.size(), CV_64F);
        
        #pragma omp parallel for if(m_params.useParallel)
        for (int y = halfWindow; y < result.u.rows - halfWindow; y++) {
            for (int x = halfWindow; x < result.u.cols - halfWindow; x++) {
                if (!result.validMask.at<uchar>(y, x)) continue;
                
                // Prepare data for least squares fit
                std::vector<cv::Point2f> points;
                std::vector<double> uValues, vValues;
                
                for (int j = -halfWindow; j <= halfWindow; j++) {
                    for (int i = -halfWindow; i <= halfWindow; i++) {
                        int ny = y + j;
                        int nx = x + i;
                        
                        if (nx >= 0 && ny >= 0 && nx < result.u.cols && ny < result.u.rows && 
                            result.validMask.at<uchar>(ny, nx)) {
                            points.push_back(cv::Point2f(nx - x, ny - y));
                            uValues.push_back(result.u.at<double>(ny, nx));
                            vValues.push_back(result.v.at<double>(ny, nx));
                        }
                    }
                }
                
                if (points.size() < 6) continue; // Need enough points for a stable fit
                
                // Construct design matrix for least squares
                cv::Mat A(points.size(), 3, CV_64F);
                cv::Mat bu(points.size(), 1, CV_64F);
                cv::Mat bv(points.size(), 1, CV_64F);
                
                for (size_t i = 0; i < points.size(); i++) {
                    A.at<double>(i, 0) = 1.0;
                    A.at<double>(i, 1) = points[i].x;
                    A.at<double>(i, 2) = points[i].y;
                    bu.at<double>(i) = uValues[i];
                    bv.at<double>(i) = vValues[i];
                }
                
                // Solve least squares problems
                cv::Mat xu, xv;
                cv::solve(A, bu, xu, cv::DECOMP_SVD);
                cv::solve(A, bv, xv, cv::DECOMP_SVD);
                
                // Extract gradients from solved parameters
                gradUx.at<double>(y, x) = xu.at<double>(1);
                gradUy.at<double>(y, x) = xu.at<double>(2);
                gradVx.at<double>(y, x) = xv.at<double>(1);
                gradVy.at<double>(y, x) = xv.at<double>(2);
            }
        }
    } else {
        // Use standard Sobel operator
        cv::Sobel(result.u, gradUx, CV_64F, 1, 0, 5);
        cv::Sobel(result.u, gradUy, CV_64F, 0, 1, 5);
        cv::Sobel(result.v, gradVx, CV_64F, 1, 0, 5);
        cv::Sobel(result.v, gradVy, CV_64F, 0, 1, 5);
        
        // Apply a Gaussian filter to smooth the strain fields
        cv::GaussianBlur(gradUx, gradUx, cv::Size(windowSize, windowSize), 0);
        cv::GaussianBlur(gradUy, gradUy, cv::Size(windowSize, windowSize), 0);
        cv::GaussianBlur(gradVx, gradVx, cv::Size(windowSize, windowSize), 0);
        cv::GaussianBlur(gradVy, gradVy, cv::Size(windowSize, windowSize), 0);
    }
    
    // Calculate strain components using engineering strain definitions
    #pragma omp parallel for if(m_params.useParallel)
    for (int y = 0; y < result.u.rows; y++) {
        for (int x = 0; x < result.u.cols; x++) {
            if (result.validMask.at<uchar>(y, x)) {
                // Small strain tensor components
                result.exx.at<double>(y, x) = gradUx.at<double>(y, x);
                result.eyy.at<double>(y, x) = gradVy.at<double>(y, x);
                result.exy.at<double>(y, x) = 0.5 * (gradUy.at<double>(y, x) + gradVx.at<double>(y, x));
            }
        }
    }
}

double GlobalDIC::interpolate(const cv::Mat& image, const cv::Point2f& pt) const {
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
    double val;
    
    if (image.type() == CV_8U) {
        val = (1 - fx) * (1 - fy) * image.at<uchar>(y1, x1) +
              fx * (1 - fy) * image.at<uchar>(y1, x2) +
              (1 - fx) * fy * image.at<uchar>(y2, x1) +
              fx * fy * image.at<uchar>(y2, x2);
    } else if (image.type() == CV_64F) {
        val = (1 - fx) * (1 - fy) * image.at<double>(y1, x1) +
              fx * (1 - fy) * image.at<double>(y1, x2) +
              (1 - fx) * fy * image.at<double>(y2, x1) +
              fx * fy * image.at<double>(y2, x2);
    } else if (image.type() == CV_32F) {
        val = (1 - fx) * (1 - fy) * image.at<float>(y1, x1) +
              fx * (1 - fy) * image.at<float>(y1, x2) +
              (1 - fx) * fy * image.at<float>(y2, x1) +
              fx * fy * image.at<float>(y2, x2);
    } else {
        val = 0.0;  // Unsupported type
    }
    
    return val;
}

void GlobalDIC::displayResults(const cv::Mat& refImage, const Result& result, 
                            bool showDisplacement, bool showStrain,
                            bool useEnhancedVisualization) {
    // Create visualizations
    cv::Mat refRGB;
    cv::cvtColor(refImage, refRGB, cv::COLOR_GRAY2BGR);
    
    if (showDisplacement) {
        // Find min/max values for normalization
        double minU, maxU, minV, maxV;
        cv::minMaxLoc(result.u, &minU, &maxU, nullptr, nullptr, result.validMask);
        cv::minMaxLoc(result.v, &minV, &maxV, nullptr, nullptr, result.validMask);
        
        std::cout << "Displacement ranges:" << std::endl;
        std::cout << "  U: " << minU << " to " << maxU << " pixels" << std::endl;
        std::cout << "  V: " << minV << " to " << maxV << " pixels" << std::endl;
        
        // Create displacement visualizations
        cv::Mat uColor, vColor;
        
        if (useEnhancedVisualization) {
            // Enhanced visualization with isolines
            uColor = createEnhancedVisualization(result.u, result.validMask, minU, maxU, "X Displacement", true);
            vColor = createEnhancedVisualization(result.v, result.validMask, minV, maxV, "Y Displacement", true);
        } else {
            // Standard visualization
            cv::Mat uNorm, vNorm;
            cv::normalize(result.u, uNorm, 0, 255, cv::NORM_MINMAX, CV_8U, result.validMask);
            cv::normalize(result.v, vNorm, 0, 255, cv::NORM_MINMAX, CV_8U, result.validMask);
            
            // Apply color map
            cv::applyColorMap(uNorm, uColor, cv::COLORMAP_JET);
            cv::applyColorMap(vNorm, vColor, cv::COLORMAP_JET);
            
            // Apply valid mask
            cv::Mat validMask3Ch;
            cv::cvtColor(result.validMask, validMask3Ch, cv::COLOR_GRAY2BGR);
            uColor = uColor.mul(validMask3Ch, 1.0/255.0);
            vColor = vColor.mul(validMask3Ch, 1.0/255.0);
        }
        
        // Create displacement field visualization with arrows
        cv::Mat dispField = refRGB.clone();
        
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
        
        // Display displacement results
        cv::imshow("X Displacement", uColor);
        cv::imshow("Y Displacement", vColor);
        cv::imshow("Displacement Field", dispField);
        
        // Calculate displacement magnitude
        cv::Mat dispMag = cv::Mat::zeros(result.u.size(), CV_64F);
        for (int y = 0; y < result.u.rows; y++) {
            for (int x = 0; x < result.u.cols; x++) {
                if (result.validMask.at<uchar>(y, x)) {
                    double u = result.u.at<double>(y, x);
                    double v = result.v.at<double>(y, x);
                    dispMag.at<double>(y, x) = std::sqrt(u*u + v*v);
                }
            }
        }
        
        double minMag, maxMag;
        cv::minMaxLoc(dispMag, &minMag, &maxMag, nullptr, nullptr, result.validMask);
        
        cv::Mat magColor;
        if (useEnhancedVisualization) {
            magColor = createEnhancedVisualization(dispMag, result.validMask, minMag, maxMag, "Displacement Magnitude", true);
        } else {
            cv::Mat magNorm;
            cv::normalize(dispMag, magNorm, 0, 255, cv::NORM_MINMAX, CV_8U, result.validMask);
            cv::applyColorMap(magNorm, magColor, cv::COLORMAP_JET);
            cv::Mat validMask3Ch;
            cv::cvtColor(result.validMask, validMask3Ch, cv::COLOR_GRAY2BGR);
            magColor = magColor.mul(validMask3Ch, 1.0/255.0);
        }
        
        cv::imshow("Displacement Magnitude", magColor);
    }
    
    if (showStrain) {
        // Find min/max values for normalization
        double minExx, maxExx, minEyy, maxEyy, minExy, maxExy;
        cv::minMaxLoc(result.exx, &minExx, &maxExx, nullptr, nullptr, result.validMask);
        cv::minMaxLoc(result.eyy, &minEyy, &maxEyy, nullptr, nullptr, result.validMask);
        cv::minMaxLoc(result.exy, &minExy, &maxExy, nullptr, nullptr, result.validMask);
        
        std::cout << "Strain ranges:" << std::endl;
        std::cout << "  Exx: " << minExx << " to " << maxExx << std::endl;
        std::cout << "  Eyy: " << minEyy << " to " << maxEyy << std::endl;
        std::cout << "  Exy: " << minExy << " to " << maxExy << std::endl;
        
        // Create strain visualizations
        cv::Mat exxColor, eyyColor, exyColor;
        
        if (useEnhancedVisualization) {
            exxColor = createEnhancedVisualization(result.exx, result.validMask, minExx, maxExx, "Exx Strain", true);
            eyyColor = createEnhancedVisualization(result.eyy, result.validMask, minEyy, maxEyy, "Eyy Strain", true);
            exyColor = createEnhancedVisualization(result.exy, result.validMask, minExy, maxExy, "Exy Strain", true);
        } else {
            // Standard visualization
            cv::Mat exxNorm, eyyNorm, exyNorm;
            cv::normalize(result.exx, exxNorm, 0, 255, cv::NORM_MINMAX, CV_8U, result.validMask);
            cv::normalize(result.eyy, eyyNorm, 0, 255, cv::NORM_MINMAX, CV_8U, result.validMask);
            cv::normalize(result.exy, exyNorm, 0, 255, cv::NORM_MINMAX, CV_8U, result.validMask);
            
            // Apply color map
            cv::applyColorMap(exxNorm, exxColor, cv::COLORMAP_JET);
            cv::applyColorMap(eyyNorm, eyyColor, cv::COLORMAP_JET);
            cv::applyColorMap(exyNorm, exyColor, cv::COLORMAP_JET);
            
            // Apply valid mask
            cv::Mat validMask3Ch;
            cv::cvtColor(result.validMask, validMask3Ch, cv::COLOR_GRAY2BGR);
            exxColor = exxColor.mul(validMask3Ch, 1.0/255.0);
            eyyColor = eyyColor.mul(validMask3Ch, 1.0/255.0);
            exyColor = exyColor.mul(validMask3Ch, 1.0/255.0);
        }
        
        // Display strain results
        cv::imshow("Exx Strain", exxColor);
        cv::imshow("Eyy Strain", eyyColor);
        cv::imshow("Exy Strain", exyColor);
        
        // Calculate principal strains
        cv::Mat e1 = cv::Mat::zeros(result.exx.size(), CV_64F);
        cv::Mat e2 = cv::Mat::zeros(result.exx.size(), CV_64F);
        cv::Mat angle = cv::Mat::zeros(result.exx.size(), CV_64F);
        
        for (int y = 0; y < result.exx.rows; y++) {
            for (int x = 0; x < result.exx.cols; x++) {
                if (result.validMask.at<uchar>(y, x)) {
                    double exx = result.exx.at<double>(y, x);
                    double eyy = result.eyy.at<double>(y, x);
                    double exy = result.exy.at<double>(y, x);
                    
                    // Calculate principal strains
                    double eavg = (exx + eyy) / 2.0;
                    double diff = (exx - eyy) / 2.0;
                    double r = std::sqrt(diff*diff + exy*exy);
                    
                    e1.at<double>(y, x) = eavg + r; // Max principal strain
                    e2.at<double>(y, x) = eavg - r; // Min principal strain
                    
                    // Calculate principal angle in degrees
                    double theta = 0.5 * std::atan2(2.0 * exy, exx - eyy);
                    angle.at<double>(y, x) = theta * 180.0 / CV_PI;
                }
            }
        }
        
        // Find min/max values for principal strains
        double minE1, maxE1, minE2, maxE2;
        cv::minMaxLoc(e1, &minE1, &maxE1, nullptr, nullptr, result.validMask);
        cv::minMaxLoc(e2, &minE2, &maxE2, nullptr, nullptr, result.validMask);
        
        std::cout << "Principal Strain ranges:" << std::endl;
        std::cout << "  E1 (Max): " << minE1 << " to " << maxE1 << std::endl;
        std::cout << "  E2 (Min): " << minE2 << " to " << maxE2 << std::endl;
        
        // Visualize principal strains
        cv::Mat e1Color, e2Color;
        
        if (useEnhancedVisualization) {
            e1Color = createEnhancedVisualization(e1, result.validMask, minE1, maxE1, "Max Principal Strain", true);
            e2Color = createEnhancedVisualization(e2, result.validMask, minE2, maxE2, "Min Principal Strain", true);
        } else {
            cv::Mat e1Norm, e2Norm;
            cv::normalize(e1, e1Norm, 0, 255, cv::NORM_MINMAX, CV_8U, result.validMask);
            cv::normalize(e2, e2Norm, 0, 255, cv::NORM_MINMAX, CV_8U, result.validMask);
            
            cv::applyColorMap(e1Norm, e1Color, cv::COLORMAP_JET);
            cv::applyColorMap(e2Norm, e2Color, cv::COLORMAP_JET);
            
            cv::Mat validMask3Ch;
            cv::cvtColor(result.validMask, validMask3Ch, cv::COLOR_GRAY2BGR);
            e1Color = e1Color.mul(validMask3Ch, 1.0/255.0);
            e2Color = e2Color.mul(validMask3Ch, 1.0/255.0);
        }
        
        cv::imshow("Max Principal Strain", e1Color);
        cv::imshow("Min Principal Strain", e2Color);
    }
    
    cv::waitKey(1); // Update display without blocking
}

void GlobalDIC::evaluateErrors(const Result& result, 
                             const cv::Mat& trueDispX, const cv::Mat& trueDispY) {
    // Convert data types if necessary
    cv::Mat u, v, trueU, trueV;
    result.u.convertTo(u, CV_32F);
    result.v.convertTo(v, CV_32F);
    trueDispX.convertTo(trueU, CV_32F);
    trueDispY.convertTo(trueV, CV_32F);

    // Calculate error maps
    cv::Mat errorU, errorV;
    cv::subtract(u, trueU, errorU);
    cv::subtract(v, trueV, errorV);

    // Calculate absolute errors
    cv::Mat absErrorU, absErrorV;
    cv::absdiff(u, trueU, absErrorU);
    cv::absdiff(v, trueV, absErrorV);

    // Convert valid mask to proper type
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

    // Apply valid mask
    cv::Mat validMask3Ch;
    cv::cvtColor(result.validMask, validMask3Ch, cv::COLOR_GRAY2BGR);
    cv::Mat errorUColorMasked, errorVColorMasked;
    cv::bitwise_and(errorUColor, validMask3Ch, errorUColorMasked);
    cv::bitwise_and(errorVColor, validMask3Ch, errorVColorMasked);

    // Display error maps
    cv::imshow("U Error Map", errorUColorMasked);
    cv::imshow("V Error Map", errorVColorMasked);
}

// Multi-scale implementation
GlobalDIC::Result GlobalDIC::computeMultiScale(const cv::Mat& refImage, 
                                             const cv::Mat& defImage,
                                             const cv::Mat& roi) {
    // Create a pyramid of images
    std::vector<cv::Mat> refPyramid, defPyramid, roiPyramid;
    
    refPyramid.push_back(refImage.clone());
    defPyramid.push_back(defImage.clone());
    roiPyramid.push_back(roi.clone());
    
    for (int i = 1; i < m_params.numScaleLevels; i++) {
        cv::Mat refSmaller, defSmaller, roiSmaller;
        
        // 计算新尺寸并确保不小于最小尺寸
        cv::Size newSize;
        cv::Size nextSize;  // 预测下一级的尺寸
        
        newSize.width = std::max(static_cast<int>(refPyramid[i-1].cols * m_params.scaleFactor), m_params.minImageSize.width);
        newSize.height = std::max(static_cast<int>(refPyramid[i-1].rows * m_params.scaleFactor), m_params.minImageSize.height);
        
        // 预测下一级
        nextSize.width = static_cast<int>(newSize.width * m_params.scaleFactor);
        nextSize.height = static_cast<int>(newSize.height * m_params.scaleFactor);
        
        // 检查当前级或下一级是否会达到最小尺寸
        if (newSize.width <= m_params.minImageSize.width || newSize.height <= m_params.minImageSize.height ||
            nextSize.width < m_params.minImageSize.width || nextSize.height < m_params.minImageSize.height) {
            std::cout << "Reached minimum image size (" << m_params.minImageSize.width 
                      << "x" << m_params.minImageSize.height 
                      << ") at level " << i << ". Stopping pyramid construction." << std::endl;
            
            // 确保当前级别的尺寸不小于最小尺寸
            newSize.width = std::max(newSize.width, m_params.minImageSize.width);
            newSize.height = std::max(newSize.height, m_params.minImageSize.height);
            
            // 处理当前级别然后停止
            cv::resize(refPyramid[i-1], refSmaller, newSize, 0, 0, cv::INTER_AREA);
            cv::resize(defPyramid[i-1], defSmaller, newSize, 0, 0, cv::INTER_AREA);
            cv::resize(roiPyramid[i-1], roiSmaller, newSize, 0, 0, cv::INTER_NEAREST);
            
            refPyramid.push_back(refSmaller);
            defPyramid.push_back(defSmaller);
            roiPyramid.push_back(roiSmaller);
            
            break;
        }
        else {
            // 使用新的安全尺寸处理正常情况
            cv::resize(refPyramid[i-1], refSmaller, newSize, 0, 0, cv::INTER_AREA);
            cv::resize(defPyramid[i-1], defSmaller, newSize, 0, 0, cv::INTER_AREA);
            cv::resize(roiPyramid[i-1], roiSmaller, newSize, 0, 0, cv::INTER_NEAREST);
            
            refPyramid.push_back(refSmaller);
            defPyramid.push_back(defSmaller);
            roiPyramid.push_back(roiSmaller);
            
            // 打印当前图像大小信息
            std::cout << "Scale level " << i << " size: " << newSize.width << "x" << newSize.height << std::endl;
        }
    }
    
    // 更新实际的金字塔级别数
    int actualLevels = refPyramid.size();
    if (actualLevels < m_params.numScaleLevels) {
        std::cout << "Note: Using " << actualLevels << " scale levels instead of requested " 
                  << m_params.numScaleLevels << " due to minimum size constraint." << std::endl;
    }
    
    // Start from the coarsest level
    std::reverse(refPyramid.begin(), refPyramid.end());
    std::reverse(defPyramid.begin(), defPyramid.end());
    std::reverse(roiPyramid.begin(), roiPyramid.end());
    
    // Initialize result at coarsest level
    Result result(roiPyramid[0].size());
    result.validMask = roiPyramid[0].clone();
    
    // Store original parameters to restore later
    Parameters origParams = m_params;
    
    // Process each level
    for (int level = 0; level < m_params.numScaleLevels; level++) {
        std::cout << "Processing scale level " << level + 1 << "/" << m_params.numScaleLevels 
                  << " (" << refPyramid[level].cols << "x" << refPyramid[level].rows << ")" << std::endl;
        
        // Adjust subset radius and node spacing for this level
        double scaleFactor = 1.0;
        for (int i = 0; i < level; i++) {
            scaleFactor /= m_params.scaleFactor;
        }
        
        // Update parameters for current level
        m_params.subsetRadius = std::max(5, static_cast<int>(origParams.subsetRadius * scaleFactor));
        m_params.nodeSpacing = std::max(5, static_cast<int>(origParams.nodeSpacing * scaleFactor));
        
        if (level > 0) {
            // Upscale previous result to current level
            cv::Mat upU, upV, upValid;
            cv::resize(result.u, upU, refPyramid[level].size(), 0, 0, cv::INTER_LINEAR);
            cv::resize(result.v, upV, refPyramid[level].size(), 0, 0, cv::INTER_LINEAR);
            cv::resize(result.validMask, upValid, refPyramid[level].size(), 0, 0, cv::INTER_NEAREST);
            
            // Scale displacement values
            upU /= m_params.scaleFactor;
            upV /= m_params.scaleFactor;
            
            // Use as initial guess
            result.u = upU;
            result.v = upV;
            result.validMask = upValid;
        }
        
        // Process current level
        Result levelResult = compute(refPyramid[level], defPyramid[level], roiPyramid[level]);
        
        // Update result
        result.u = levelResult.u;
        result.v = levelResult.v;
        result.validMask = levelResult.validMask;
        result.cc = levelResult.cc;
        result.confidence = levelResult.confidence;
        result.meanResidual = levelResult.meanResidual;
        result.iterations = levelResult.iterations;
    }
    
    // Restore original parameters
    m_params = origParams;
    
    // Calculate strain fields
    calculateStrains(result);
    
    return result;
}


void GlobalDIC::calculateConfidence(Result& result, 
    const cv::Mat& refImage, 
    const cv::Mat& defImage) {
// Initialize confidence matrix
result.confidence = cv::Mat::zeros(result.u.size(), CV_64F);

// Convert images to floating point if needed
cv::Mat refFloat, defFloat;
if (refImage.type() != CV_64F) {
refImage.convertTo(refFloat, CV_64F);
} else {
refFloat = refImage;
}

if (defImage.type() != CV_64F) {
defImage.convertTo(defFloat, CV_64F);
} else {
defFloat = defImage;
}

// Calculate confidence based on local ZNCC and intensity gradients
#pragma omp parallel for if(m_params.useParallel)
for (int y = m_params.subsetRadius; y < result.u.rows - m_params.subsetRadius; y++) {
for (int x = m_params.subsetRadius; x < result.u.cols - m_params.subsetRadius; x++) {
if (!result.validMask.at<uchar>(y, x)) continue;

// Get displacement at this point
double u = result.u.at<double>(y, x);
double v = result.v.at<double>(y, x);

// Local correlation measure
double sumRefDef = 0, sumRef = 0, sumDef = 0;
double sumRefSq = 0, sumDefSq = 0;
int count = 0;

// Calculate local ZNCC for a small window around this point
int windowRadius = std::min(5, m_params.subsetRadius / 2);

for (int j = -windowRadius; j <= windowRadius; j++) {
for (int i = -windowRadius; i <= windowRadius; i++) {
cv::Point refPt(x + i, y + j);
cv::Point2f defPt(x + i + u, y + j + v);

if (refPt.x < 0 || refPt.x >= refFloat.cols || 
refPt.y < 0 || refPt.y >= refFloat.rows ||
defPt.x < 0 || defPt.x >= defFloat.cols - 1 ||
defPt.y < 0 || defPt.y >= defFloat.rows - 1) {
continue;
}

double refIntensity = refFloat.at<double>(refPt);
double defIntensity = interpolate(defFloat, defPt);

sumRef += refIntensity;
sumDef += defIntensity;
sumRefSq += refIntensity * refIntensity;
sumDefSq += defIntensity * defIntensity;
sumRefDef += refIntensity * defIntensity;
count++;
}
}

double zncc = 0.0;
if (count > 0) {
double meanRef = sumRef / count;
double meanDef = sumDef / count;
double varRef = sumRefSq / count - meanRef * meanRef;
double varDef = sumDefSq / count - meanDef * meanDef;
double covar = sumRefDef / count - meanRef * meanDef;

if (varRef > 0 && varDef > 0) {
zncc = covar / std::sqrt(varRef * varDef);
}
}

// Calculate local gradient magnitude
double gradMag = 0.0;
if (m_gradientsComputed) {
double gx = m_gradX.at<double>(y, x);
double gy = m_gradY.at<double>(y, x);
gradMag = std::sqrt(gx*gx + gy*gy);
}

// Calculate confidence as a combination of correlation and gradient magnitude
double corrWeight = 0.7; // Weight for correlation in confidence measure
double gradWeight = 0.3; // Weight for gradient in confidence measure

// Normalize gradient magnitude to [0,1] range
double normGradMag = std::min(1.0, gradMag / 50.0); // 50 is a reasonable max gradient

// Combine correlation and gradient information
result.confidence.at<double>(y, x) = corrWeight * (zncc + 1.0) / 2.0 + gradWeight * normGradMag;
}
}

// Smooth confidence map
cv::GaussianBlur(result.confidence, result.confidence, cv::Size(5, 5), 0);
}

double GlobalDIC::calculateResidual(const cv::Mat& refImage, 
    const cv::Mat& defImage,
    const std::vector<cv::Point>& nodePoints,
    const cv::Mat& nodeDisplacements) {
// Convert images to floating point if needed
cv::Mat refFloat, defFloat;
if (refImage.type() != CV_64F) {
refImage.convertTo(refFloat, CV_64F);
} else {
refFloat = refImage;
}

if (defImage.type() != CV_64F) {
defImage.convertTo(defFloat, CV_64F);
} else {
defFloat = defImage;
}

double totalResidual = 0.0;
int count = 0;

// Sample a subset of pixels to calculate residual
int step = 5; // Sample every 'step' pixels to speed up computation

for (int y = m_params.subsetRadius; y < refFloat.rows - m_params.subsetRadius; y += step) {
for (int x = m_params.subsetRadius; x < refFloat.cols - m_params.subsetRadius; x += step) {
cv::Point pt(x, y);

// Get reference intensity
double refIntensity = refFloat.at<double>(pt);

// Interpolate displacement at this pixel
double u, v;
interpolateDisplacement(pt, nodePoints, nodeDisplacements, u, v);

// Calculate deformed position
cv::Point2f defPt(x + u, y + v);

// Check if deformed point is within bounds
if (defPt.x >= 0 && defPt.x < defFloat.cols - 1 &&
defPt.y >= 0 && defPt.y < defFloat.rows - 1) {

// Get deformed intensity (interpolated)
double defIntensity = interpolate(defFloat, defPt);

// Calculate intensity residual
double diff = refIntensity - defIntensity;
totalResidual += diff * diff;
count++;
}
}
}

// Return mean squared residual
return (count > 0) ? std::sqrt(totalResidual / count) : std::numeric_limits<double>::max();
}

void GlobalDIC::computeShapeFunctions(const cv::Point& point, 
      const std::vector<cv::Point>& elementNodes,
      std::vector<double>& N,
      std::vector<double>& dNdx,
      std::vector<double>& dNdy) {
// Compute a unique hash for this calculation to use in the cache
size_t hash = point.y * 1000000 + point.x;
for (const auto& node : elementNodes) {
hash = hash * 31 + node.y * 1000 + node.x;
}

// Check if we have cached values
auto it = m_shapeFunctionCache.find(hash);
if (it != m_shapeFunctionCache.end()) {
std::tie(N, dNdx, dNdy) = it->second;
return;
}

// Initialize shape function values
N.resize(elementNodes.size(), 0.0);
dNdx.resize(elementNodes.size(), 0.0);
dNdy.resize(elementNodes.size(), 0.0);

// Using inverse distance weighting for simplicity
double totalWeight = 0.0;

for (size_t i = 0; i < elementNodes.size(); i++) {
double dist = cv::norm(cv::Mat(point - elementNodes[i]));

// Avoid division by zero
dist = std::max(dist, 1e-6);

// Calculate weight using inverse distance squared
double weight = 1.0 / (dist * dist);
totalWeight += weight;
N[i] = weight;
}

// Normalize weights
if (totalWeight > 0) {
for (size_t i = 0; i < elementNodes.size(); i++) {
N[i] /= totalWeight;
}
}

// Calculate derivatives of shape functions
// This is a simplification - proper FEM would use more complex shape functions
double h = 1.0; // Small step for finite difference approximation

cv::Point pointPlusX(point.x + h, point.y);
cv::Point pointPlusY(point.x, point.y + h);

std::vector<double> NPlusX, NPlusY;
NPlusX.resize(elementNodes.size(), 0.0);
NPlusY.resize(elementNodes.size(), 0.0);

// Calculate shape function values at offset points
totalWeight = 0.0;
for (size_t i = 0; i < elementNodes.size(); i++) {
double dist = cv::norm(cv::Mat(pointPlusX - elementNodes[i]));
dist = std::max(dist, 1e-6);
double weight = 1.0 / (dist * dist);
totalWeight += weight;
NPlusX[i] = weight;
}

if (totalWeight > 0) {
for (size_t i = 0; i < elementNodes.size(); i++) {
NPlusX[i] /= totalWeight;
}
}

totalWeight = 0.0;
for (size_t i = 0; i < elementNodes.size(); i++) {
double dist = cv::norm(cv::Mat(pointPlusY - elementNodes[i]));
dist = std::max(dist, 1e-6);
double weight = 1.0 / (dist * dist);
totalWeight += weight;
NPlusY[i] = weight;
}

if (totalWeight > 0) {
for (size_t i = 0; i < elementNodes.size(); i++) {
NPlusY[i] /= totalWeight;
}
}

// Calculate derivatives using finite differences
for (size_t i = 0; i < elementNodes.size(); i++) {
dNdx[i] = (NPlusX[i] - N[i]) / h;
dNdy[i] = (NPlusY[i] - N[i]) / h;
}

// Store in cache for future use
m_shapeFunctionCache[hash] = std::make_tuple(N, dNdx, dNdy);
}

cv::Mat GlobalDIC::createEnhancedVisualization(const cv::Mat& data, const cv::Mat& validMask, 
               double minVal, double maxVal, 
               const std::string& title, bool addIsolines) {
// Create a normalized version for color mapping
cv::Mat dataNorm;
cv::normalize(data, dataNorm, 0, 255, cv::NORM_MINMAX, CV_8U, validMask);

// Apply color map
cv::Mat colorMap;
cv::applyColorMap(dataNorm, colorMap, cv::COLORMAP_JET);

// Apply valid mask
cv::Mat background = cv::Mat::zeros(data.size(), CV_8UC3);
colorMap.copyTo(background, validMask);

// Draw isolines if requested
if (addIsolines) {
// Number of contour levels
int numLevels = 10;
double step = (maxVal - minVal) / numLevels;

// Generate contour levels
std::vector<double> levels;
for (int i = 0; i <= numLevels; i++) {
levels.push_back(minVal + i * step);
}

// Create a copy of the data for contour finding
cv::Mat dataFloat;
data.convertTo(dataFloat, CV_32F);

// Draw contours
std::vector<std::vector<cv::Point>> contours;
for (double level : levels) {
cv::Mat binaryMap;
cv::threshold(dataFloat, binaryMap, level, 255, cv::THRESH_BINARY);
binaryMap.convertTo(binaryMap, CV_8U);

std::vector<std::vector<cv::Point>> levelContours;
cv::findContours(binaryMap, levelContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

// Draw the contours
cv::drawContours(background, levelContours, -1, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
}
}

// Add border around the image for the scale bar and title
int topBorder = 40;  // Space for title
int bottomBorder = 70;  // Space for scale bar
int leftRightBorder = 30;

cv::Mat result;
cv::copyMakeBorder(background, result, topBorder, bottomBorder, leftRightBorder, leftRightBorder, 
cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

// Add title
cv::putText(result, title, cv::Point(leftRightBorder, 30), cv::FONT_HERSHEY_SIMPLEX, 
0.8, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

// Add scale bar
int barWidth = background.cols - 60;
int barHeight = 20;
int barX = leftRightBorder + 30;
int barY = result.rows - bottomBorder + 20;

// Create gradient for scale bar
cv::Mat scaleBar(barHeight, barWidth, CV_8UC3);
for (int x = 0; x < barWidth; x++) {
double value = (double)x / barWidth * 255.0;
cv::Mat color;
cv::Mat temp(1, 1, CV_8UC1, cv::Scalar(value));
cv::applyColorMap(temp, color, cv::COLORMAP_JET);
cv::rectangle(scaleBar, cv::Point(x, 0), cv::Point(x, barHeight), color.at<cv::Vec3b>(0, 0), 1);
}

// Place scale bar on the result image
scaleBar.copyTo(result(cv::Rect(barX, barY, barWidth, barHeight)));

// Add min and max values as text
std::stringstream ssMin, ssMax;
ssMin << std::fixed << std::setprecision(4) << minVal;
ssMax << std::fixed << std::setprecision(4) << maxVal;

cv::putText(result, ssMin.str(), cv::Point(barX - 5, barY + barHeight + 15), 
cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
cv::putText(result, ssMax.str(), cv::Point(barX + barWidth - 30, barY + barHeight + 15), 
cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

return result;
}