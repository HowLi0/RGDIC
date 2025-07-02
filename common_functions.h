#ifndef COMMON_FUNCTIONS_H
#define COMMON_FUNCTIONS_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iomanip>
#include <sstream>
#include "poi.h"

// Global variables for ROI drawing
extern cv::Mat roiImage;
extern cv::Mat roi;
extern std::vector<cv::Point> roiPoints;
extern bool roiFinished;

// Function declarations
void drawROI(int event, int x, int y, int flags, void* userdata);

void generateSyntheticImages(cv::Mat& refImg, cv::Mat& defImg, 
                           cv::Mat& trueDispX, cv::Mat& trueDispY,
                           int width = 500, int height = 500);

cv::Mat createManualROI(const cv::Mat& image);

void exportToCSV(const cv::Mat& u, const cv::Mat& v, const cv::Mat& validMask, const std::string& filename);

// Enhanced CSV export with strain fields and full grid output
void exportToCSVWithStrain(const cv::Mat& u, const cv::Mat& v, const cv::Mat& validMask,
                          const cv::Mat& exx, const cv::Mat& eyy, const cv::Mat& exy,
                          const cv::Mat& zncc, const cv::Mat& roi, const std::string& filename);

cv::Mat visualizeDisplacementWithScaleBar(const cv::Mat& displacement, const cv::Mat& validMask, 
                                      double minVal, double maxVal, 
                                      const std::string& title,
                                      int colorMap = cv::COLORMAP_JET);

void processAndSaveResults(const cv::Mat& refImage, const cv::Mat& defImage, 
                          const cv::Mat& trueDispX, const cv::Mat& trueDispY,
                          const cv::Mat& resultU, const cv::Mat& resultV, 
                          const cv::Mat& validMask, bool useSyntheticImages);

// POI visualization functions
/**
 * @brief Visualize POI correspondences between reference and deformed images
 * @param refImage Reference image
 * @param defImage Deformed image
 * @param pois POI collection
 * @param maxPOIs Maximum number of POIs to visualize (for performance)
 * @return Combined visualization image
 */
cv::Mat visualizePOICorrespondences(const cv::Mat& refImage, const cv::Mat& defImage,
                                   const POICollection& pois, int maxPOIs = 200);

/**
 * @brief Create displacement vector field visualization from POIs
 * @param imageSize Size of the output image
 * @param pois POI collection
 * @param scale Scale factor for displacement vectors
 * @return Displacement vector field image
 */
cv::Mat visualizePOIDisplacementField(const cv::Size& imageSize, const POICollection& pois, 
                                     double scale = 10.0);

#endif // COMMON_FUNCTIONS_H
