#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "cuda_rgdic.h"

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --cuda              Use CUDA acceleration (default)" << std::endl;
    std::cout << "  --cpu               Use CPU only" << std::endl;
    std::cout << "  --synthetic         Use synthetic test images (default)" << std::endl;
    std::cout << "  --real <ref> <def>  Use real images" << std::endl;
    std::cout << "  --subset <radius>   Subset radius (default: 15)" << std::endl;
    std::cout << "  --batch <size>      Batch size for CUDA (default: 5000)" << std::endl;
    std::cout << "  --first-order       Use first-order shape function" << std::endl;
    std::cout << "  --second-order      Use second-order shape function (default)" << std::endl;
    std::cout << "  --help              Show this help message" << std::endl;
}

void generateSyntheticImages(cv::Mat& refImg, cv::Mat& defImg, 
                           cv::Mat& trueDispX, cv::Mat& trueDispY,
                           int width = 512, int height = 512) {
    std::cout << "Generating synthetic speckle pattern images..." << std::endl;
    
    // Create reference image with speckle pattern
    refImg = cv::Mat::zeros(height, width, CV_8UC1);
    cv::RNG rng(12345);
    
    // Generate random speckle pattern
    for (int i = 0; i < 5000; i++) {
        int x = rng.uniform(0, width);
        int y = rng.uniform(0, height);
        int radius = rng.uniform(2, 4);
        cv::circle(refImg, cv::Point(x, y), radius, cv::Scalar(255), -1);
    }
    
    // Apply Gaussian blur for more realistic speckles
    cv::GaussianBlur(refImg, refImg, cv::Size(3, 3), 0.8);
    
    // Create true displacement field (sinusoidal displacement)
    trueDispX = cv::Mat::zeros(height, width, CV_32F);
    trueDispY = cv::Mat::zeros(height, width, CV_32F);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Generate displacement field
            double dx = 3.0 * sin(2.0 * CV_PI * y / height);
            double dy = 2.0 * cos(2.0 * CV_PI * x / width);
            
            trueDispX.at<float>(y, x) = dx;
            trueDispY.at<float>(y, x) = dy;
        }
    }
    
    // Create deformed image using the displacement field
    defImg = cv::Mat::zeros(height, width, CV_8UC1);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Get displacement at this point
            float dx = trueDispX.at<float>(y, x);
            float dy = trueDispY.at<float>(y, x);
            
            // Source position in reference image
            float srcX = x - dx;
            float srcY = y - dy;
            
            // Check if within bounds
            if (srcX >= 0 && srcX < width - 1 && srcY >= 0 && srcY < height - 1) {
                // Bilinear interpolation
                int x1 = floor(srcX);
                int y1 = floor(srcY);
                int x2 = x1 + 1;
                int y2 = y1 + 1;
                
                float wx = srcX - x1;
                float wy = srcY - y1;
                
                float val = (1 - wx) * (1 - wy) * refImg.at<uchar>(y1, x1) +
                           wx * (1 - wy) * refImg.at<uchar>(y1, x2) +
                           (1 - wx) * wy * refImg.at<uchar>(y2, x1) +
                           wx * wy * refImg.at<uchar>(y2, x2);
                
                defImg.at<uchar>(y, x) = static_cast<uchar>(val);
            }
        }
    }
    
    // Fill any holes in the deformed image
    cv::Mat mask = (defImg == 0);
    cv::inpaint(defImg, mask, defImg, 5, cv::INPAINT_TELEA);
}

void calculateErrors(const RGDIC::DisplacementResult& result,
                    const cv::Mat& trueDispX, const cv::Mat& trueDispY) {
    cv::Mat uError(result.u.size(), CV_64F);
    cv::Mat vError(result.v.size(), CV_64F);
    
    double totalErrorU = 0.0, totalErrorV = 0.0;
    int validCount = 0;
    
    for (int y = 0; y < result.u.rows; y++) {
        for (int x = 0; x < result.u.cols; x++) {
            if (result.validMask.at<uchar>(y, x)) {
                double errU = std::abs(result.u.at<double>(y, x) - trueDispX.at<float>(y, x));
                double errV = std::abs(result.v.at<double>(y, x) - trueDispY.at<float>(y, x));
                
                uError.at<double>(y, x) = errU;
                vError.at<double>(y, x) = errV;
                
                totalErrorU += errU;
                totalErrorV += errV;
                validCount++;
            }
        }
    }
    
    if (validCount > 0) {
        double meanErrorU = totalErrorU / validCount;
        double meanErrorV = totalErrorV / validCount;
        
        std::cout << "\nAccuracy Analysis:" << std::endl;
        std::cout << "  Mean X displacement error: " << meanErrorU << " pixels" << std::endl;
        std::cout << "  Mean Y displacement error: " << meanErrorV << " pixels" << std::endl;
        std::cout << "  Valid points analyzed: " << validCount << std::endl;
    }
}

int main(int argc, char** argv) {
    // Default parameters
    bool useCuda = true;
    bool useSynthetic = true;
    int subsetRadius = 15;
    int batchSize = 10000;
    bool useFirstOrder = false;
    std::string refImagePath, defImagePath;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--cuda") {
            useCuda = true;
        } else if (arg == "--cpu") {
            useCuda = false;
        } else if (arg == "--synthetic") {
            useSynthetic = true;
        } else if (arg == "--real" && i + 2 < argc) {
            useSynthetic = false;
            refImagePath = argv[++i];
            defImagePath = argv[++i];
        } else if (arg == "--subset" && i + 1 < argc) {
            subsetRadius = std::atoi(argv[++i]);
        } else if (arg == "--batch" && i + 1 < argc) {
            batchSize = std::atoi(argv[++i]);
        } else if (arg == "--first-order") {
            useFirstOrder = true;
        } else if (arg == "--second-order") {
            useFirstOrder = false;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printUsage(argv[0]);
            return -1;
        }
    }
    
    // Print configuration
    std::cout << "CUDA-Accelerated RGDIC Configuration:" << std::endl;
    std::cout << "  Acceleration: " << (useCuda ? "CUDA GPU" : "CPU") << std::endl;
    std::cout << "  Images: " << (useSynthetic ? "Synthetic" : "Real") << std::endl;
    std::cout << "  Subset radius: " << subsetRadius << std::endl;
    std::cout << "  Shape function: " << (useFirstOrder ? "First-order" : "Second-order") << std::endl;
    if (useCuda) {
        std::cout << "  Batch size: " << batchSize << std::endl;
    }
    std::cout << std::endl;
    
    // Load or generate images
    cv::Mat refImage, defImage, trueDispX, trueDispY;
    
    if (useSynthetic) {
        generateSyntheticImages(refImage, defImage, trueDispX, trueDispY);
    } else {
        refImage = cv::imread(refImagePath, cv::IMREAD_GRAYSCALE);
        defImage = cv::imread(defImagePath, cv::IMREAD_GRAYSCALE);
        
        if (refImage.empty() || defImage.empty()) {
            std::cerr << "Error: Could not load images!" << std::endl;
            return -1;
        }
        
        if (refImage.size() != defImage.size()) {
            std::cerr << "Error: Images must have the same size!" << std::endl;
            return -1;
        }
    }
    
    std::cout << "Image size: " << refImage.cols << "x" << refImage.rows << std::endl;
    
    // Create ROI (exclude border regions)
    cv::Mat roi = cv::Mat::ones(refImage.size(), CV_8UC1);
    int borderWidth = subsetRadius + 5;
    cv::rectangle(roi, cv::Point(0, 0), cv::Point(roi.cols-1, roi.rows-1), 0, borderWidth);
    
    int roiPoints = cv::countNonZero(roi);
    std::cout << "ROI points: " << roiPoints << std::endl;
    
    // Create RGDIC object
    ShapeFunctionOrder order = useFirstOrder ? FIRST_ORDER : SECOND_ORDER;
    std::unique_ptr<RGDIC> dic;
    
    if (useCuda) {
        dic = std::make_unique<CudaRGDIC>(subsetRadius, 0.00001, 30, 0.8, 1.0, order, 1, batchSize);
    } else {
        dic = std::make_unique<RGDIC>(subsetRadius, 0.00001, 30, 0.8, 1.0, order, 1);
    }
    
    // Measure execution time
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Run RGDIC algorithm
    std::cout << "\\nRunning RGDIC computation..." << std::endl;
    auto result = dic->compute(refImage, defImage, roi);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(endTime - startTime).count();
    
    // Count valid points
    int validPoints = cv::countNonZero(result.validMask);
    double coverage = 100.0 * validPoints / roiPoints;
    
    std::cout << "\\nResults Summary:" << std::endl;
    std::cout << "  Computation time: " << duration << " seconds" << std::endl;
    std::cout << "  Coverage: " << coverage << "% (" << validPoints << "/" << roiPoints << " points)" << std::endl;
    std::cout << "  Performance: " << (validPoints / duration) << " points/second" << std::endl;
    
    // Print detailed CUDA performance stats if available
    if (useCuda) {
        auto cudaDic = dynamic_cast<CudaRGDIC*>(dic.get());
        if (cudaDic) {
            auto stats = cudaDic->getLastPerformanceStats();
            std::cout << "\\nCUDA Performance Details:" << std::endl;
            std::cout << "  GPU compute time: " << stats.gpuComputeTime << " seconds" << std::endl;
            std::cout << "  Memory transfer time: " << stats.memoryTransferTime << " seconds" << std::endl;
            std::cout << "  CPU processing time: " << stats.cpuProcessingTime << " seconds" << std::endl;
            std::cout << "  GPU utilization: " << (stats.gpuComputeTime / stats.totalTime * 100.0) << "%" << std::endl;
            std::cout << "  Estimated speedup: " << stats.speedup << "x" << std::endl;
        }
    }
    
    // Calculate accuracy if using synthetic images
    if (useSynthetic && !trueDispX.empty() && !trueDispY.empty()) {
        calculateErrors(result, trueDispX, trueDispY);
    }
    
    // Save results
    if (!result.u.empty()) {
        std::cout << "\\nSaving results..." << std::endl;
        
        // Find min/max values for visualization
        double minU, maxU, minV, maxV;
        cv::minMaxLoc(result.u, &minU, &maxU, nullptr, nullptr, result.validMask);
        cv::minMaxLoc(result.v, &minV, &maxV, nullptr, nullptr, result.validMask);
        
        // Create visualizations
        cv::Mat uNorm, vNorm;
        cv::normalize(result.u, uNorm, 0, 255, cv::NORM_MINMAX, CV_8U, result.validMask);
        cv::normalize(result.v, vNorm, 0, 255, cv::NORM_MINMAX, CV_8U, result.validMask);
        
        cv::Mat uColor, vColor;
        cv::applyColorMap(uNorm, uColor, cv::COLORMAP_JET);
        cv::applyColorMap(vNorm, vColor, cv::COLORMAP_JET);
        
        // Apply mask
        cv::Mat mask3Ch;
        cv::cvtColor(result.validMask, mask3Ch, cv::COLOR_GRAY2BGR);
        uColor = uColor.mul(mask3Ch, 1.0/255.0);
        vColor = vColor.mul(mask3Ch, 1.0/255.0);
        
        // Save images
        cv::imwrite("result/computed_disp_x.png", uColor);
        cv::imwrite("result/computed_disp_y.png", vColor);
        cv::imwrite("result/reference.png", refImage);
        cv::imwrite("result/deformed.png", defImage);
        
        std::cout << "Results saved to result/ directory" << std::endl;
    }
    
    std::cout << "\\nComputations completed successfully!" << std::endl;
    return 0;
}
