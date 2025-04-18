#include "rgdic.h"
#include "global_dic.h"
#include "parallel_utils.h"
#include <iostream>
#include <string>

// 可视化工具函数
cv::Mat visualizeDisplacementWithScaleBar(const cv::Mat& displacement, const cv::Mat& validMask,
                                         double minVal, double maxVal, const std::string& title) {
    // 创建可视化
    cv::Mat normalized;
    cv::normalize(displacement, normalized, 0, 255, cv::NORM_MINMAX, CV_8U, validMask);
    
    cv::Mat colored;
    cv::applyColorMap(normalized, colored, cv::COLORMAP_JET);
    colored.setTo(cv::Scalar(0, 0, 0), ~validMask);
    
    // 添加标题
    cv::Mat result = colored.clone();
    cv::putText(result, title, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 
               0.8, cv::Scalar(255, 255, 255), 2);
    
    // 添加刻度条
    int barWidth = 300;
    int barHeight = 30;
    int margin = 50;
    cv::Rect barRect(result.cols - barWidth - margin, result.rows - barHeight - margin, 
                   barWidth, barHeight);
    
    // 绘制刻度条背景
    cv::rectangle(result, barRect, cv::Scalar(50, 50, 50), -1);
    
    // 绘制彩色刻度
    for (int i = 0; i < barWidth; i++) {
        double ratio = static_cast<double>(i) / barWidth;
        int colorValue = static_cast<int>(ratio * 255);
        cv::Mat colorBar(barHeight, 1, CV_8UC3);
        cv::applyColorMap(cv::Mat(barHeight, 1, CV_8U, cv::Scalar(colorValue)), 
                         colorBar, cv::COLORMAP_JET);
        
        cv::Rect barSection(barRect.x + i, barRect.y, 1, barHeight);
        colorBar.copyTo(result(barSection));
    }
    
    // 添加最小和最大值标签
    std::stringstream minStr, maxStr;
    minStr << std::fixed << std::setprecision(3) << minVal;
    maxStr << std::fixed << std::setprecision(3) << maxVal;
    
    cv::putText(result, minStr.str(), cv::Point(barRect.x - 5, barRect.y + barHeight + 20), 
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
    cv::putText(result, maxStr.str(), cv::Point(barRect.x + barWidth - 40, barRect.y + barHeight + 20), 
               cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
    
    return result;
}

int main(int argc, char** argv) {
    std::cout << "RGDIC与金字塔全局DIC方法比较" << std::endl;
    std::cout << "User: " << "HowLi0" << std::endl;
    std::cout << "Date: " << "2025-04-18" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // 输入图像路径
    std::string refImagePath = "reference.png";
    std::string defImagePath = "deformed.png";
    
    // 加载参考图像和变形图像
    cv::Mat refImage = cv::imread(refImagePath, cv::IMREAD_GRAYSCALE);
    cv::Mat defImage = cv::imread(defImagePath, cv::IMREAD_GRAYSCALE);
    
    if (refImage.empty() || defImage.empty()) {
        std::cerr << "错误: 无法加载图像." << std::endl;
        return -1;
    }
    
    // 转换为浮点类型
    cv::Mat refImageFloat, defImageFloat;
    refImage.convertTo(refImageFloat, CV_64F);
    defImage.convertTo(defImageFloat, CV_64F);
    
    // 创建ROI
    cv::Mat roi = cv::Mat::ones(refImage.size(), CV_8U);
    int borderWidth = 20;
    cv::rectangle(roi, cv::Point(0, 0), cv::Point(roi.cols-1, roi.rows-1), 0, borderWidth);
    
    // 显示输入图像
    cv::imshow("参考图像", refImage);
    cv::imshow("变形图像", defImage);
    cv::waitKey(100);
    
    // 创建性能监控器
    ParallelPerformanceMonitor rgdicMonitor("RGDIC");
    ParallelPerformanceMonitor globalDicMonitor("金字塔全局DIC");
    
    // 运行RGDIC
    std::cout << "运行RGDIC算法..." << std::endl;
    RGDIC rgdic(15, 0.001, 30, 0.8, 1.0, RGDIC::FIRST_ORDER);
    
    rgdicMonitor.start();
    RGDIC::DisplacementResult rgdicResult = rgdic.compute(refImageFloat, defImageFloat, roi);
    rgdicMonitor.stop();
    
    // 运行金字塔全局DIC
    std::cout << "运行金字塔全局DIC算法..." << std::endl;
    PyramidGlobalDIC::Parameters params;
    params.nodeSpacing = 10;
    params.subsetRadius = 15;
    params.regularizationWeight = 0.1;
    params.regType = PyramidGlobalDIC::TIKHONOV;
    params.convergenceThreshold = 0.001;
    params.maxIterations = 30;
    params.order = PyramidGlobalDIC::FIRST_ORDER;
    params.useMultiScaleApproach = true;
    params.numScaleLevels = 3;
    params.scaleFactor = 0.5;
    params.useParallel = true;
    params.numThreads = omp_get_max_threads();
    
    PyramidGlobalDIC globalDIC(params);
    
    globalDicMonitor.start();
    PyramidGlobalDIC::Result globalDicResult = globalDIC.compute(refImageFloat, defImageFloat, roi);
    globalDicMonitor.stop();
    
    // 计算全局DIC结果的应变
    globalDIC.calculateStrains(globalDicResult);
    
    // 显示性能结果
    std::cout << "性能比较:" << std::endl;
    std::cout << "  RGDIC执行时间: " << rgdicMonitor.getLastExecutionTime() << " 秒" << std::endl;
    std::cout << "  金字塔全局DIC执行时间: " << globalDicMonitor.getLastExecutionTime() << " 秒" << std::endl;
    std::cout << "  加速比: " << rgdicMonitor.getLastExecutionTime() / globalDicMonitor.getLastExecutionTime() << std::endl;
    
    // 比较结果的覆盖率
    int rgdicValidPoints = cv::countNonZero(rgdicResult.validMask);
    int globalDicValidPoints = cv::countNonZero(globalDicResult.validMask);
    int totalRoiPoints = cv::countNonZero(roi);
    
    double rgdicCoverage = 100.0 * rgdicValidPoints / totalRoiPoints;
    double globalDicCoverage = 100.0 * globalDicValidPoints / totalRoiPoints;
    
    std::cout << "覆盖率比较:" << std::endl;
    std::cout << "  RGDIC覆盖率: " << rgdicCoverage << "% (" << rgdicValidPoints << " 点)" << std::endl;
    std::cout << "  金字塔全局DIC覆盖率: " << globalDicCoverage << "% (" << globalDicValidPoints << " 点)" << std::endl;
    
    // 找到位移范围
    double minRgdicU, maxRgdicU, minRgdicV, maxRgdicV;
    double minGlobalU, maxGlobalU, minGlobalV, maxGlobalV;
    
    cv::minMaxLoc(rgdicResult.u, &minRgdicU, &maxRgdicU, nullptr, nullptr, rgdicResult.validMask);
    cv::minMaxLoc(rgdicResult.v, &minRgdicV, &maxRgdicV, nullptr, nullptr, rgdicResult.validMask);
    cv::minMaxLoc(globalDicResult.u, &minGlobalU, &maxGlobalU, nullptr, nullptr, globalDicResult.validMask);
    cv::minMaxLoc(globalDicResult.v, &minGlobalV, &maxGlobalV, nullptr, nullptr, globalDicResult.validMask);
    
    // 创建可视化
    cv::Mat rgdicUViz = visualizeDisplacementWithScaleBar(rgdicResult.u, rgdicResult.validMask, 
                                                       minRgdicU, maxRgdicU, "RGDIC X位移");
    cv::Mat rgdicVViz = visualizeDisplacementWithScaleBar(rgdicResult.v, rgdicResult.validMask, 
                                                       minRgdicV, maxRgdicV, "RGDIC Y位移");
    cv::Mat globalUViz = visualizeDisplacementWithScaleBar(globalDicResult.u, globalDicResult.validMask, 
                                                       minGlobalU, maxGlobalU, "全局DIC X位移");
    cv::Mat globalVViz = visualizeDisplacementWithScaleBar(globalDicResult.v, globalDicResult.validMask, 
                                                       minGlobalV, maxGlobalV, "全局DIC Y位移");
    
    // 显示结果
    cv::imshow("RGDIC X位移", rgdicUViz);
    cv::imshow("RGDIC Y位移", rgdicVViz);
    cv::imshow("全局DIC X位移", globalUViz);
    cv::imshow("全局DIC Y位移", globalVViz);
    
    // 创建差异图
    cv::Mat diffU = cv::Mat::zeros(refImage.size(), CV_64F);
    cv::Mat diffV = cv::Mat::zeros(refImage.size(), CV_64F);
    cv::Mat commonMask = rgdicResult.validMask & globalDicResult.validMask;
    
    for (int y = 0; y < diffU.rows; y++) {
        for (int x = 0; x < diffU.cols; x++) {
            if (commonMask.at<uchar>(y, x)) {
                diffU.at<double>(y, x) = std::abs(rgdicResult.u.at<double>(y, x) - globalDicResult.u.at<double>(y, x));
                diffV.at<double>(y, x) = std::abs(rgdicResult.v.at<double>(y, x) - globalDicResult.v.at<double>(y, x));
            }
        }
    }
    
    double maxDiffU, maxDiffV;
    cv::minMaxLoc(diffU, nullptr, &maxDiffU, nullptr, nullptr, commonMask);
    cv::minMaxLoc(diffV, nullptr, &maxDiffV, nullptr, nullptr, commonMask);
    
    cv::Mat diffUViz = visualizeDisplacementWithScaleBar(diffU, commonMask, 0, maxDiffU, "X位移差异");
    cv::Mat diffVViz = visualizeDisplacementWithScaleBar(diffV, commonMask, 0, maxDiffV, "Y位移差异");
    
    cv::imshow("X位移差异", diffUViz);
    cv::imshow("Y位移差异", diffVViz);
    
    // 计算平均差异
    cv::Scalar meanDiffU = cv::mean(diffU, commonMask);
    cv::Scalar meanDiffV = cv::mean(diffV, commonMask);
    
    std::cout << "结果差异比较:" << std::endl;
    std::cout << "  X位移平均差异: " << meanDiffU[0] << " (最大: " << maxDiffU << ")" << std::endl;
    std::cout << "  Y位移平均差异: " << meanDiffV[0] << " (最大: " << maxDiffV << ")" << std::endl;
    
    // 保存比较结果
    cv::imwrite("rgdic_u.png", rgdicUViz);
    cv::imwrite("rgdic_v.png", rgdicVViz);
    cv::imwrite("global_dic_u.png", globalUViz);
    cv::imwrite("global_dic_v.png", globalVViz);
    cv::imwrite("diff_u.png", diffUViz);
    cv::imwrite("diff_v.png", diffVViz);
    
    std::cout << "按任意键退出..." << std::endl;
    cv::waitKey(0);
    
    return 0;
}