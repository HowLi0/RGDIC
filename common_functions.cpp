#include "common_functions.h"

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
                           int width, int height) {
    // Create reference image with speckle pattern
    refImg = cv::Mat::zeros(height, width, CV_8UC1);
    cv::RNG rng(12345);
    
    // Generate random speckle pattern
    for (int i = 0; i < 4000; i++) {
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
            manualROI = cv::Mat::ones(image.size(), CV_8UC1);  // Default to full image
            break;
        }
        
        cv::imshow("Draw ROI", roiImage);
    }
    
    cv::destroyWindow("Draw ROI");
    return manualROI;
}

// Function to export displacement data to CSV file
void exportToCSV(const cv::Mat& u, const cv::Mat& v, const cv::Mat& validMask, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    
    // Write header
    file << "x,y,u_displacement,v_displacement" << std::endl;
    
    // Write data points
    for (int y = 0; y < u.rows; y++) {
        for (int x = 0; x < u.cols; x++) {
            if (validMask.at<uchar>(y, x)) {
                file << x << "," << y << "," << u.at<double>(y, x) << "," << v.at<double>(y, x) << std::endl;
            }
        }
    }
    
    std::cout << "Displacement data exported to: " << filename << std::endl;
}

// Enhanced CSV export with strain fields and full grid output
void exportToCSVWithStrain(const cv::Mat& u, const cv::Mat& v, const cv::Mat& validMask,
                          const cv::Mat& exx, const cv::Mat& eyy, const cv::Mat& exy,
                          const cv::Mat& zncc, const cv::Mat& roi, const std::string& filename) {
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }
    
    // Write header with new format
    file << "left_x,left_y,right_x,right_y,u,v,exx,eyy,exy,zncc" << std::endl;
    
    int totalPoints = 0;
    int validPoints = 0;
    
    // Write only ROI data points (both valid and invalid within ROI)
    for (int y = 0; y < u.rows; y++) {
        for (int x = 0; x < u.cols; x++) {
            // Only process points that are within ROI
            if (roi.at<uchar>(y, x) > 0) {
                totalPoints++;
                
                // left_x, left_y are the reference coordinates
                double left_x = static_cast<double>(x);
                double left_y = static_cast<double>(y);
                
                bool isValid = validMask.at<uchar>(y, x) > 0;
                
                if (isValid) {
                    validPoints++;
                    
                    // right_x, right_y are the deformed coordinates (left + displacement)
                    double u_disp = u.at<double>(y, x);
                    double v_disp = v.at<double>(y, x);
                    double right_x = left_x + u_disp;
                    double right_y = left_y + v_disp;
                    
                    // Get strain values if available
                    double strain_exx = (exx.empty()) ? 0.0 : exx.at<double>(y, x);
                    double strain_eyy = (eyy.empty()) ? 0.0 : eyy.at<double>(y, x);
                    double strain_exy = (exy.empty()) ? 0.0 : exy.at<double>(y, x);
                    double correlation = (zncc.empty()) ? 0.0 : zncc.at<double>(y, x);
                    
                    // Write valid point data
                    file << std::fixed << std::setprecision(6)
                         << left_x << "," << left_y << "," 
                         << right_x << "," << right_y << ","
                         << u_disp << "," << v_disp << ","
                         << strain_exx << "," << strain_eyy << "," << strain_exy << ","
                         << correlation << std::endl;
                } else {
                    // Write invalid point within ROI with zeros
                    double right_x = left_x;  // No displacement
                    double right_y = left_y;  // No displacement
                    
                    file << std::fixed << std::setprecision(6)
                         << left_x << "," << left_y << "," 
                         << right_x << "," << right_y << ","
                         << "0.0,0.0,0.0,0.0,0.0,0.0" << std::endl;
                }
            }
        }
    }
    
    file.close();
    
    std::cout << "Enhanced displacement and strain data exported to: " << filename << std::endl;
    std::cout << "  Total ROI points: " << totalPoints << std::endl;
    std::cout << "  Valid points: " << validPoints << std::endl;
    std::cout << "  Invalid points: " << (totalPoints - validPoints) << std::endl;
}

// Function to create a color map visualization with a scale bar
cv::Mat visualizeDisplacementWithScaleBar(const cv::Mat& displacement, const cv::Mat& validMask, 
                                      double minVal, double maxVal, 
                                      const std::string& title,
                                      int colorMap) {
    // Normalize displacement for visualization
    cv::Mat dispNorm;
    cv::normalize(displacement, dispNorm, 0, 255, cv::NORM_MINMAX, CV_8U, validMask);
    
    // Apply color map
    cv::Mat colorDisp;
    cv::applyColorMap(dispNorm, colorDisp, colorMap);
    
    // Apply valid mask
    cv::Mat background = cv::Mat::zeros(displacement.size(), CV_8UC3);
    colorDisp.copyTo(background, validMask);
    
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
        cv::applyColorMap(temp, color, colorMap);
        cv::rectangle(scaleBar, cv::Point(x, 0), cv::Point(x, barHeight), color.at<cv::Vec3b>(0, 0), 1);
    }
    
    // Place scale bar on the result image
    scaleBar.copyTo(result(cv::Rect(barX, barY, barWidth, barHeight)));
    
    // Add min and max values as text
    std::stringstream ssMin, ssMax;
    ssMin << std::fixed << std::setprecision(2) << minVal;
    ssMax << std::fixed << std::setprecision(2) << maxVal;
    
    cv::putText(result, ssMin.str(), cv::Point(barX - 5, barY + barHeight + 15), 
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    cv::putText(result, ssMax.str(), cv::Point(barX + barWidth - 10, barY + barHeight + 15), 
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
               
    // Add "pixels" unit text
    cv::putText(result, "[pixels]", cv::Point(barX + barWidth/2 - 20, barY + barHeight + 15), 
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    
    return result;
}

// Function to process and save all results
void processAndSaveResults(const cv::Mat& refImage, const cv::Mat& defImage, 
                          const cv::Mat& trueDispX, const cv::Mat& trueDispY,
                          const cv::Mat& resultU, const cv::Mat& resultV, 
                          const cv::Mat& validMask, bool useSyntheticImages) {
    
    // Find min/max values of computed displacement
    double minU, maxU, minV, maxV;
    cv::minMaxLoc(resultU, &minU, &maxU, nullptr, nullptr, validMask);
    cv::minMaxLoc(resultV, &minV, &maxV, nullptr, nullptr, validMask);
    
    // Create visualizations with scale bars
    cv::Mat uViz = visualizeDisplacementWithScaleBar(resultU, validMask, 
                                                   minU, maxU, "X Displacement");
    cv::Mat vViz = visualizeDisplacementWithScaleBar(resultV, validMask, 
                                                   minV, maxV, "Y Displacement");
    
    // Display results
    cv::imshow("X Displacement", uViz);
    cv::imshow("Y Displacement", vViz);
    
    // Save results to disk
    cv::imwrite("./result/computed_disp_x.png", uViz);
    cv::imwrite("./result/computed_disp_y.png", vViz);
    
    // Note: CSV export is now handled in CudaRGDIC::compute() for enhanced format
    
    // Calculate vector magnitude of displacement (for visualization)
    cv::Mat dispMag = cv::Mat::zeros(resultU.size(), CV_64F);
    for (int y = 0; y < dispMag.rows; y++) {
        for (int x = 0; x < dispMag.cols; x++) {
            if (validMask.at<uchar>(y, x)) {
                double dx = resultU.at<double>(y, x);
                double dy = resultV.at<double>(y, x);
                dispMag.at<double>(y, x) = std::sqrt(dx*dx + dy*dy);
            }
        }
    }
    
    // Find min/max of magnitude
    double minMag, maxMag;
    cv::minMaxLoc(dispMag, &minMag, &maxMag, nullptr, nullptr, validMask);
    
    // Create visualization of magnitude
    cv::Mat magViz = visualizeDisplacementWithScaleBar(dispMag, validMask, 
                                                    minMag, maxMag, "Displacement Magnitude");
    
    cv::imshow("Displacement Magnitude", magViz);
    cv::imwrite("./result/computed_disp_magnitude.png", magViz);
    
    // Create vector field visualization on reference image
    cv::Mat vectorField;
    cv::cvtColor(refImage, vectorField, cv::COLOR_GRAY2BGR);
    
    // Draw displacement vectors (subsampled for clarity)
    int step = 5; // Sample every 5 pixels
    for (int y = 0; y < resultU.rows; y += step) {
        for (int x = 0; x < resultU.cols; x += step) {
            if (validMask.at<uchar>(y, x)) {
                double dx = resultU.at<double>(y, x);
                double dy = resultV.at<double>(y, x);
                
                // Scale for visibility (adjust this as needed)
                double scale = 5.0;
                cv::arrowedLine(vectorField, 
                              cv::Point(x, y), 
                              cv::Point(x + dx * scale, y + dy * scale), 
                              cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
            }
        }
    }
    
    cv::imshow("Vector Field", vectorField);
    cv::imwrite("./result/vector_field.png", vectorField);
    
    // If we have ground truth, calculate errors
    if (useSyntheticImages) {
        // Convert types for calculation
        cv::Mat uError(resultU.size(), CV_64F);
        cv::Mat vError(resultV.size(), CV_64F);
        
        for (int y = 0; y < resultU.rows; y++) {
            for (int x = 0; x < resultU.cols; x++) {
                if (validMask.at<uchar>(y, x)) {
                    uError.at<double>(y, x) = std::abs(resultU.at<double>(y, x) - trueDispX.at<float>(y, x));
                    vError.at<double>(y, x) = std::abs(resultV.at<double>(y, x) - trueDispY.at<float>(y, x));
                }
            }
        }
        
        // Find min/max error
        double minErrU, maxErrU, minErrV, maxErrV;
        cv::minMaxLoc(uError, &minErrU, &maxErrU, nullptr, nullptr, validMask);
        cv::minMaxLoc(vError, &minErrV, &maxErrV, nullptr, nullptr, validMask);
        
        // Create error visualizations
        cv::Mat uErrViz = visualizeDisplacementWithScaleBar(uError, validMask, 
                                                         minErrU, maxErrU, "X Displacement Error");
        cv::Mat vErrViz = visualizeDisplacementWithScaleBar(vError, validMask, 
                                                         minErrV, maxErrV, "Y Displacement Error");
        
        cv::imshow("X Displacement Error", uErrViz);
        cv::imshow("Y Displacement Error", vErrViz);
        
        cv::imwrite("./result/error_disp_x.png", uErrViz);
        cv::imwrite("./result/error_disp_y.png", vErrViz);
        
        // Calculate error statistics
        cv::Scalar meanErrU = cv::mean(uError, validMask);
        cv::Scalar meanErrV = cv::mean(vError, validMask);
        
        std::cout << "Error Statistics:" << std::endl;
        std::cout << "  X Displacement: Mean Error = " << meanErrU[0] 
                  << " pixels, Max Error = " << maxErrU << " pixels" << std::endl;
        std::cout << "  Y Displacement: Mean Error = " << meanErrV[0] 
                  << " pixels, Max Error = " << maxErrV << " pixels" << std::endl;
    }
}

// POI visualization functions implementation

cv::Mat visualizePOICorrespondences(const cv::Mat& refImage, const cv::Mat& defImage,
                                   const POICollection& pois, int maxPOIs) {
    // Create side-by-side visualization
    cv::Mat refColor, defColor;
    if (refImage.channels() == 1) {
        cv::cvtColor(refImage, refColor, cv::COLOR_GRAY2BGR);
    } else {
        refColor = refImage.clone();
    }
    
    if (defImage.channels() == 1) {
        cv::cvtColor(defImage, defColor, cv::COLOR_GRAY2BGR);
    } else {
        defColor = defImage.clone();
    }
    
    // Create combined image
    cv::Mat combined;
    cv::hconcat(refColor, defColor, combined);
    
    // Draw POI correspondences
    int count = 0;
    cv::RNG rng(12345);
    
    for (const auto& poi : pois.pois) {
        if (!poi.isValid() || count >= maxPOIs) break;
        
        // Generate random color for this correspondence
        cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        
        // Draw points
        cv::Point leftPt(static_cast<int>(poi.leftCoord.x), static_cast<int>(poi.leftCoord.y));
        cv::Point rightPt(static_cast<int>(poi.rightCoord.x + refImage.cols), static_cast<int>(poi.rightCoord.y));
        
        cv::circle(combined, leftPt, 3, color, -1);
        cv::circle(combined, rightPt, 3, color, -1);
        
        // Draw correspondence line
        cv::line(combined, leftPt, rightPt, color, 1);
        
        count++;
    }
    
    // Add text annotation
    cv::putText(combined, "Reference Image", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    cv::putText(combined, "Deformed Image", cv::Point(refImage.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    cv::putText(combined, "POIs: " + std::to_string(count), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    return combined;
}

cv::Mat visualizePOIDisplacementField(const cv::Size& imageSize, const POICollection& pois, double scale) {
    // Create visualization image
    cv::Mat vis = cv::Mat::zeros(imageSize, CV_8UC3);
    
    for (const auto& poi : pois.pois) {
        if (!poi.isValid()) continue;
        
        cv::Point start(static_cast<int>(poi.leftCoord.x), static_cast<int>(poi.leftCoord.y));
        cv::Point end(static_cast<int>(poi.leftCoord.x + poi.displacement[0] * scale),
                      static_cast<int>(poi.leftCoord.y + poi.displacement[1] * scale));
        
        // Check bounds
        if (start.x >= 0 && start.x < imageSize.width && start.y >= 0 && start.y < imageSize.height) {
            // Color based on displacement magnitude
            double magnitude = poi.getDisplacementMagnitude();
            cv::Scalar color;
            
            if (magnitude < 1.0) {
                color = cv::Scalar(0, 255, 0); // Green for small displacements
            } else if (magnitude < 5.0) {
                color = cv::Scalar(0, 255, 255); // Yellow for medium displacements
            } else {
                color = cv::Scalar(0, 0, 255); // Red for large displacements
            }
            
            // Draw displacement vector
            cv::arrowedLine(vis, start, end, color, 1, 8, 0, 0.3);
            cv::circle(vis, start, 2, color, -1);
        }
    }
    
    return vis;
}
