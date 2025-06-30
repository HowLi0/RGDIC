#include <iostream>
#include <opencv2/opencv.hpp>
#include "rgdic.h"
#include "cuda_rgdic.h"
#include "common_functions.h"

int main(int argc, char** argv) {
    std::cout << "=== RGDIC CUDA Version ===" << std::endl;
    
    // Configuration flags
    bool useSyntheticImages = true;
    bool useFirstOrderShapeFunction = false; // Set to false to use second-order
    bool useManualROI = true; // Set to true to manually draw ROI
    
    // Ask user to select interpolation method
    int interpolationMethodChoice = 0;
    std::cout << "Select interpolation method:" << std::endl;
    std::cout << "0 - Bilinear interpolation (fast, good for smooth displacement fields)" << std::endl;
    std::cout << "1 - Bicubic interpolation (slower, better quality for complex displacement fields)" << std::endl;
    std::cout << "2 - Inverse Distance Weighting (default, robust for sparse data)" << std::endl;
    std::cout << "Enter choice (0-2): ";
    std::cin >> interpolationMethodChoice;
    
    // Validate input
    if (interpolationMethodChoice < 0 || interpolationMethodChoice > 2) {
        std::cout << "Invalid choice, using default (Inverse Distance Weighting)" << std::endl;
        interpolationMethodChoice = 2;
    }
    
    cv::Mat refImage, defImage;
    cv::Mat trueDispX, trueDispY;
    
    // Load or generate images
    if (useSyntheticImages) {
        std::cout << "Generating synthetic speckle pattern images..." << std::endl;
        generateSyntheticImages(refImage, defImage, trueDispX, trueDispY);
        
        // Save the generated images
        cv::imwrite("reference.png", refImage);
        cv::imwrite("deformed.png", defImage);
        
        // Display images
        cv::imshow("Reference Image", refImage);
        cv::imshow("Deformed Image", defImage);
        cv::waitKey(100); // Brief pause to ensure windows are displayed
        
        // Visualize true displacement fields
        double minX, maxX, minY, maxY;
        cv::minMaxLoc(trueDispX, &minX, &maxX);
        cv::minMaxLoc(trueDispY, &minY, &maxY);
        
        // Create visualizations with scale bars
        cv::Mat trueXViz = visualizeDisplacementWithScaleBar(trueDispX, cv::Mat::ones(trueDispX.size(), CV_8UC1),
                                                          minX, maxX, "True X Displacement");
        cv::Mat trueYViz = visualizeDisplacementWithScaleBar(trueDispY, cv::Mat::ones(trueDispY.size(), CV_8UC1),
                                                          minY, maxY, "True Y Displacement");
        
        cv::imshow("True X Displacement", trueXViz);
        cv::imshow("True Y Displacement", trueYViz);
        
        cv::imwrite("./result/true_disp_x.png", trueXViz);
        cv::imwrite("./result/true_disp_y.png", trueYViz);
    } else {
        // Load real images if provided
        if (argc < 3) {
            std::cout << "Usage: " << argv[0] << " <reference_image> <deformed_image>" << std::endl;
            return -1;
        }
        
        refImage = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
        defImage = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
        
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
        cv::imwrite("./result/selected_roi.png", roiViz);
    } else {
        // Create automatic ROI (exclude border regions)
        int borderWidth = 25;
        roi = cv::Mat::ones(refImage.size(), CV_8UC1);
        cv::rectangle(roi, cv::Point(0, 0), cv::Point(roi.cols-1, roi.rows-1), 0, borderWidth);
    }
    
    // Create CUDA RGDIC object
    ShapeFunctionOrder order = useFirstOrderShapeFunction ? 
                              FIRST_ORDER : SECOND_ORDER;
    
    // Convert interpolation method choice to enum
    InterpolationMethod interpMethod;
    std::string methodName;
    switch (interpolationMethodChoice) {
        case 0:
            interpMethod = BILINEAR_INTERPOLATION;
            methodName = "bilinear";
            break;
        case 1:
            interpMethod = BICUBIC_INTERPOLATION;
            methodName = "bicubic";
            break;
        case 2:
        default:
            interpMethod = INVERSE_DISTANCE_WEIGHTING;
            methodName = "inverse distance weighting";
            break;
    }
    
    std::unique_ptr<RGDIC> dic;
    try {
        dic = std::make_unique<CudaRGDIC>(19, 0.00001, 30, 0.2, 1.0, order, 5, 50000, interpMethod);
        std::cout << "Using CUDA-accelerated RGDIC algorithm with " 
                  << (useFirstOrderShapeFunction ? "first" : "second") 
                  << "-order shape function and "
                  << methodName << " interpolation..." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "CUDA initialization failed: " << e.what() << std::endl;
        std::cerr << "Please ensure CUDA is properly installed and a compatible GPU is available." << std::endl;
        return -1;
    }
    
    // Measure execution time
    double t = (double)cv::getTickCount();
    
    // Run RGDIC algorithm
    auto result = dic->compute(refImage, defImage, roi);
    
    // Calculate execution time
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "CUDA RGDIC computation completed in " << t << " seconds." << std::endl;
    
    // Print CUDA performance statistics
    auto cudaDic = dynamic_cast<CudaRGDIC*>(dic.get());
    if (cudaDic) {
        auto stats = cudaDic->getLastPerformanceStats();
        std::cout << "\nDetailed Performance Statistics:" << std::endl;
        std::cout << "  Total Time: " << stats.totalTime << " seconds" << std::endl;
        std::cout << "  GPU Compute Time: " << stats.gpuComputeTime << " seconds" << std::endl;
        std::cout << "  Memory Transfer Time: " << stats.memoryTransferTime << " seconds" << std::endl;
        std::cout << "  GPU Utilization: " 
                  << (stats.gpuComputeTime / stats.totalTime * 100.0) << "%" << std::endl;
        std::cout << "  Memory Transfer Overhead: " 
                  << (stats.memoryTransferTime / stats.totalTime * 100.0) << "%" << std::endl;
    }
    
    // Count valid points
    int validPoints = cv::countNonZero(result.validMask);
    int totalRoiPoints = cv::countNonZero(roi);
    double coverage = 100.0 * validPoints / totalRoiPoints;
    
    std::cout << "Analysis coverage: " << coverage << "% (" 
              << validPoints << " of " << totalRoiPoints << " points)" << std::endl;
    
    // Export CSV data using dedicated IO stream
    if (cudaDic && cudaDic->hasStrainField()) {
        // Export with strain data
        auto strainField = cudaDic->getLastStrainField();
        exportToCSVWithStrain(result.u, result.v, result.validMask,
                             strainField.exx, strainField.eyy, strainField.exy,
                             result.cc, roi, "./result/displacement_results.csv");
        
        // Save strain field visualizations
        cv::imwrite("./result/strain_exx.png", 
                   visualizeDisplacementWithScaleBar(strainField.exx, strainField.validMask, 
                                                   -0.01, 0.01, "Normal Strain Exx"));
        cv::imwrite("./result/strain_eyy.png", 
                   visualizeDisplacementWithScaleBar(strainField.eyy, strainField.validMask, 
                                                   -0.01, 0.01, "Normal Strain Eyy"));
        cv::imwrite("./result/strain_exy.png", 
                   visualizeDisplacementWithScaleBar(strainField.exy, strainField.validMask, 
                                                   -0.01, 0.01, "Shear Strain Exy"));
    } else {
        // Export without strain data (use empty strain fields)
        cv::Mat emptyStrain = cv::Mat::zeros(result.u.size(), CV_64F);
        cv::Mat emptyMask = cv::Mat::zeros(result.u.size(), CV_8U);
        exportToCSVWithStrain(result.u, result.v, result.validMask,
                             emptyStrain, emptyStrain, emptyStrain,
                             result.cc, roi, "./result/displacement_results.csv");
    }
    
    // Process and save other results (excluding CSV export which is already done)
    processAndSaveResults(refImage, defImage, trueDispX, trueDispY,
                         result.u, result.v, result.validMask, useSyntheticImages);
    
    std::cout << "Press any key to exit..." << std::endl;
    cv::waitKey(0);
    return 0;
}
