#include <iostream>
#include <opencv2/opencv.hpp>
#include "rgdic.h"
#include "common_functions.h"

int main(int argc, char** argv) {
    std::cout << "=== RGDIC CPU Version ===" << std::endl;
    
    // Configuration flags
    bool useSyntheticImages = true;
    bool useFirstOrderShapeFunction = false; // Set to false to use second-order
    bool useManualROI = true; // Set to true to manually draw ROI
    
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
    
    // Create CPU RGDIC object
    ShapeFunctionOrder order = useFirstOrderShapeFunction ? 
                              FIRST_ORDER : SECOND_ORDER;
    
    auto dic = createRGDIC(false, 15, 0.00001, 30, 0.8, 1.0, order, 5);
    std::cout << "Using CPU-based RGDIC algorithm with " 
              << (useFirstOrderShapeFunction ? "first" : "second") 
              << "-order shape function..." << std::endl;
    
    // Measure execution time
    double t = (double)cv::getTickCount();
    
    // Run RGDIC algorithm
    auto result = dic->compute(refImage, defImage, roi);
    
    // Calculate execution time
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << "CPU RGDIC computation completed in " << t << " seconds." << std::endl;
    
    // Count valid points
    int validPoints = cv::countNonZero(result.validMask);
    int totalRoiPoints = cv::countNonZero(roi);
    double coverage = 100.0 * validPoints / totalRoiPoints;
    
    std::cout << "Analysis coverage: " << coverage << "% (" 
              << validPoints << " of " << totalRoiPoints << " points)" << std::endl;
    
    // Process and save all results
    processAndSaveResults(refImage, defImage, trueDispX, trueDispY,
                         result.u, result.v, result.validMask, useSyntheticImages);
    
    std::cout << "Press any key to exit..." << std::endl;
    cv::waitKey(0);
    return 0;
}
