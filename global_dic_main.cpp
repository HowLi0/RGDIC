#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <chrono>
#include "global_dic.h"

// Global variables for ROI drawing
cv::Mat roiImage;
cv::Mat roi;
std::vector<cv::Point> roiPoints;
bool roiFinished = false;

// Mouse callback function for drawing ROI
void drawROI(int event, int x, int y, int flags, void* userdata) {
    if (roiFinished)
        return;
        
    if (event == cv::EVENT_LBUTTONDOWN) {
        // Add point to the ROI
        roiPoints.push_back(cv::Point(x, y));
        
        // Draw a circle at the clicked point
        cv::circle(roiImage, cv::Point(x, y), 3, cv::Scalar(0, 0, 255), -1);
        
        // If we have at least two points, draw a line between the last two points
        if (roiPoints.size() > 1) {
            cv::line(roiImage, roiPoints[roiPoints.size() - 2], roiPoints[roiPoints.size() - 1], 
                    cv::Scalar(0, 0, 255), 2);
        }
        
        cv::imshow("Draw ROI", roiImage);
    }
    else if (event == cv::EVENT_MOUSEMOVE && !roiPoints.empty()) {
        // Show a temporary line from the last point to the current position
        cv::Mat tempImage = roiImage.clone();
        cv::line(tempImage, roiPoints.back(), cv::Point(x, y), cv::Scalar(0, 0, 255), 2);
        cv::imshow("Draw ROI", tempImage);
    }
}

// Function to generate synthetic speckle pattern images for testing
void generateSyntheticImages(cv::Mat& refImg, cv::Mat& defImg, 
                           cv::Mat& trueDispX, cv::Mat& trueDispY,
                           int width = 500, int height = 500,
                           int patternType = 0) {
    // Create reference image with speckle pattern
    refImg = cv::Mat::zeros(height, width, CV_8UC1);
    cv::RNG rng(12345);
    
    if (patternType == 0) {
        // Random speckle pattern
        for (int i = 0; i < 4000; i++) {
            int x = rng.uniform(0, width);
            int y = rng.uniform(0, height);
            int radius = rng.uniform(2, 4);
            cv::circle(refImg, cv::Point(x, y), radius, cv::Scalar(255), -1);
        }
    } else if (patternType == 1) {
        // Grid pattern with noise
        int gridSize = 10;
        for (int y = 0; y < height; y += gridSize) {
            for (int x = 0; x < width; x += gridSize) {
                int radius = rng.uniform(2, 4);
                int offsetX = rng.uniform(-2, 2);
                int offsetY = rng.uniform(-2, 2);
                cv::circle(refImg, cv::Point(x + offsetX, y + offsetY), radius, cv::Scalar(255), -1);
            }
        }
    } else if (patternType == 2) {
        // Perlin noise-like pattern
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Simple noise function based on sin
                double nx = x * 0.05;
                double ny = y * 0.05;
                double v1 = std::sin(nx) * std::sin(ny) * 127 + 128;
                double v2 = std::sin(nx * 0.1) * std::sin(ny * 0.1) * 127 + 128;
                double v = (v1 + v2) / 2.0;
                refImg.at<uchar>(y, x) = static_cast<uchar>(v);
            }
        }
    }
    
    // Apply Gaussian blur for more realistic speckles
    cv::GaussianBlur(refImg, refImg, cv::Size(3, 3), 0.8);
    
    // Create true displacement field based on pattern type
    trueDispX = cv::Mat::zeros(height, width, CV_32F);
    trueDispY = cv::Mat::zeros(height, width, CV_32F);
    
    if (patternType == 0 || patternType == 1) {
        // Sinusoidal displacement
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Generate displacement field
                double dx = 3.0 * std::sin(2.0 * CV_PI * y / height);
                double dy = 2.0 * std::cos(2.0 * CV_PI * x / width);
                
                trueDispX.at<float>(y, x) = dx;
                trueDispY.at<float>(y, x) = dy;
            }
        }
    } else if (patternType == 2) {
        // Radial displacement from center
        cv::Point2f center(width / 2.0f, height / 2.0f);
        float maxRadius = std::min(width, height) / 4.0f;
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                cv::Point2f pt(x, y);
                cv::Point2f dir = pt - center;
                float dist = cv::norm(dir);
                
                if (dist > 0) {
                    dir /= dist; // Normalize direction vector
                    
                    // Scale displacement based on distance from center
                    float scale = std::max(0.0f, maxRadius - dist) / maxRadius * 5.0f;
                    
                    trueDispX.at<float>(y, x) = dir.x * scale;
                    trueDispY.at<float>(y, x) = dir.y * scale;
                }
            }
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

// Function to let user draw ROI manually
cv::Mat createManualROI(const cv::Mat& image) {
    // Create ROI mask
    cv::Mat manualROI = cv::Mat::zeros(image.size(), CV_8UC1);
    
    // Create an image for drawing
    if (image.channels() == 1) {
        cv::cvtColor(image, roiImage, cv::COLOR_GRAY2BGR);
    } else {
        roiImage = image.clone();
    }
    
    // Set up window and mouse callback
    cv::namedWindow("Draw ROI", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("Draw ROI", drawROI);
    
    // Instructions
    std::cout << "Draw ROI: Click to add points, press Enter to complete, Esc to cancel" << std::endl;
    
    // Wait for ROI drawing to complete
    roiPoints.clear();
    roiFinished = false;
    
    while (true) {
        char key = (char)cv::waitKey(10);
        
        // Enter key completes the ROI
        if (key == 13 && roiPoints.size() >= 3) {
            // Complete the polygon by connecting to the first point
            cv::line(roiImage, roiPoints.back(), roiPoints.front(), cv::Scalar(0, 0, 255), 2);
            cv::imshow("Draw ROI", roiImage);
            
            // Fill the ROI polygon
            std::vector<std::vector<cv::Point>> contours = { roiPoints };
            cv::fillPoly(manualROI, contours, cv::Scalar(255));
            
            roiFinished = true;
            break;
        }
        // Escape key cancels
        else if (key == 27) {
            manualROI = cv::Mat::ones(image.size(), CV_8UC1) * 255;  // Default to full image
            break;
        }
        
        cv::imshow("Draw ROI", roiImage);
    }
    
    cv::destroyWindow("Draw ROI");
    return manualROI;
}

// Function to export displacement and strain data to CSV file
void exportToCSV(const GlobalDIC::Result& result, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    
    // Write header
    file << "x,y,u_displacement,v_displacement,exx_strain,eyy_strain,exy_strain,confidence" << std::endl;
    
    // Write data points
    for (int y = 0; y < result.u.rows; y++) {
        for (int x = 0; x < result.u.cols; x++) {
            if (result.validMask.at<uchar>(y, x)) {
                file << x << "," 
                     << y << "," 
                     << result.u.at<double>(y, x) << "," 
                     << result.v.at<double>(y, x) << ","
                     << result.exx.at<double>(y, x) << ","
                     << result.eyy.at<double>(y, x) << ","
                     << result.exy.at<double>(y, x) << ","
                     << result.confidence.at<double>(y, x) << std::endl;
            }
        }
    }
    
    std::cout << "Displacement and strain data exported to: " << filename << std::endl;
}

int main(int argc, char** argv) {
    // Parse command line arguments
    bool useSyntheticImages = true;
    bool useManualROI = false;
    bool exportData = true;
    int patternType = 0;
    std::string outputDir = "E:/code_C++/RGDIC/";
    bool useMultiScale = true;
    bool useParallel = true;
    
    // Simple command line argument parsing
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--no-synthetic") {
            useSyntheticImages = false;
        } else if (arg == "--manual-roi") {
            useManualROI = true;
        } else if (arg == "--no-export") {
            exportData = false;
        } else if (arg == "--pattern" && i+1 < argc) {
            patternType = std::stoi(argv[++i]);
        } else if (arg == "--output-dir" && i+1 < argc) {
            outputDir = argv[++i];
            if (outputDir.back() != '/') outputDir += '/';
        } else if (arg == "--no-multiscale") {
            useMultiScale = false;
        } else if (arg == "--no-parallel") {
            useParallel = false;
        }
    }
    
    cv::Mat refImage, defImage;
    cv::Mat trueDispX, trueDispY;
    
    if (useSyntheticImages) {
        std::cout << "Generating synthetic speckle pattern images (pattern type: " << patternType << ")..." << std::endl;
        generateSyntheticImages(refImage, defImage, trueDispX, trueDispY, 500, 500, patternType);
        
        // Save the generated images
        cv::imwrite(outputDir + "synthetic_reference.png", refImage);
        cv::imwrite(outputDir + "synthetic_deformed.png", defImage);
        
        // Display images
        cv::imshow("Reference Image", refImage);
        cv::imshow("Deformed Image", defImage);
        cv::waitKey(100); // Brief pause to ensure windows are displayed
    } else {
        // Load real images if provided
        if (argc < 3) {
            std::cout << "Usage for real images: " << argv[0] << " --no-synthetic <reference_image> <deformed_image>" << std::endl;
            return -1;
        }
        
        // Find first non-flag argument
        std::string refPath, defPath;
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg[0] != '-') {
                if (refPath.empty()) {
                    refPath = arg;
                } else if (defPath.empty()) {
                    defPath = arg;
                    break;
                }
            } else if (arg == "--output-dir") {
                i++; // Skip next arg since it's the directory
            }
        }
        
        if (refPath.empty() || defPath.empty()) {
            std::cout << "Error: Reference and deformed image paths not provided!" << std::endl;
            return -1;
        }
        
        refImage = cv::imread(refPath, cv::IMREAD_GRAYSCALE);
        defImage = cv::imread(defPath, cv::IMREAD_GRAYSCALE);
        
        if (refImage.empty() || defImage.empty()) {
            std::cerr << "Error loading images!" << std::endl;
            return -1;
        }
        
        // Display images
        cv::imshow("Reference Image", refImage);
        cv::imshow("Deformed Image", defImage);
        cv::waitKey(100);
    }
    
    // Create ROI
    cv::Mat roi;
    if (useManualROI) {
        // Let user draw ROI
        roi = createManualROI(refImage);
        
        // Display the ROI
        cv::Mat roiViz;
        cv::cvtColor(refImage, roiViz, cv::COLOR_GRAY2BGR);
        roiViz.setTo(cv::Scalar(0, 0, 255), roi);
        cv::addWeighted(roiViz, 0.3, cv::Mat(roiViz.size(), roiViz.type(), cv::Scalar(0)), 0.7, 0, roiViz);
        
        // Convert reference image to color and copy only the ROI area
        cv::Mat colorRef;
        cv::cvtColor(refImage, colorRef, cv::COLOR_GRAY2BGR);
        colorRef.copyTo(roiViz, roi);
        
        cv::imshow("Selected ROI", roiViz);
        cv::waitKey(100);
        cv::imwrite(outputDir + "selected_roi.png", roiViz);
    } else {
        // Create automatic ROI (exclude border regions)
        int borderWidth = 25;
        roi = cv::Mat::ones(refImage.size(), CV_8UC1) * 255;
        cv::rectangle(roi, cv::Point(0, 0), cv::Point(roi.cols-1, roi.rows-1), 0, borderWidth);
    }
    
    // Configure GlobalDIC parameters
    GlobalDIC::Parameters params;
    params.nodeSpacing = 15;
    params.subsetRadius = 15;
    params.regularizationWeight = 0.1;
    params.regType = GlobalDIC::DIFFUSION; // Use diffusion-based regularization
    params.convergenceThreshold = 0.001;
    params.maxIterations = 50;
    params.order = GlobalDIC::FIRST_ORDER;
    params.useMultiScaleApproach = useMultiScale;
    params.numScaleLevels = 3;
    params.scaleFactor = 0.5;
    params.useParallel = useParallel;
    params.minImageSize = cv::Size(16, 16); // 设置最小允许尺寸为16x16
    
    // Create GlobalDIC object with configured parameters
    GlobalDIC dic(params);
    
    std::cout << "Running Global DIC algorithm..." << std::endl;
    std::cout << "Node spacing: " << params.nodeSpacing << " pixels" << std::endl;
    std::cout << "Subset radius: " << params.subsetRadius << " pixels" << std::endl;
    std::cout << "Regularization: " << 
        (params.regType == GlobalDIC::TIKHONOV ? "Tikhonov" : 
         params.regType == GlobalDIC::DIFFUSION ? "Diffusion" : "Total Variation") << 
        " (weight: " << params.regularizationWeight << ")" << std::endl;
    std::cout << "Multi-scale approach: " << (params.useMultiScaleApproach ? "Enabled" : "Disabled") << std::endl;
    std::cout << "Parallelization: " << (params.useParallel ? "Enabled" : "Disabled") << std::endl;
    
    // Measure execution time
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Run Global DIC algorithm
    GlobalDIC::Result result = dic.compute(refImage, defImage, roi);
    
    // Calculate execution time
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime = endTime - startTime;
    
    std::cout << "Global DIC computation completed in " << elapsedTime.count() << " seconds." << std::endl;
    std::cout << "Iterations performed: " << result.iterations << std::endl;
    std::cout << "Mean residual: " << result.meanResidual << std::endl;
    
    // Count valid points
    int validPoints = cv::countNonZero(result.validMask);
    int totalRoiPoints = cv::countNonZero(roi);
    double coverage = 100.0 * validPoints / totalRoiPoints;
    
    std::cout << "Analysis coverage: " << coverage << "% (" 
              << validPoints << " of " << totalRoiPoints << " points)" << std::endl;
    
    // Display results with enhanced visualization
    dic.displayResults(refImage, result, true, true, true);
    
    // Export data if requested
    if (exportData) {
        exportToCSV(result, outputDir + "global_dic_results.csv");
    }
    
    // If we have ground truth, calculate errors
    if (useSyntheticImages) {
        dic.evaluateErrors(result, trueDispX, trueDispY);
    }
    
    std::cout << "Press any key to exit..." << std::endl;
    cv::waitKey(0);
    return 0;
}