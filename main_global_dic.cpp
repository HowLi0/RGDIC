
#include "global_dic.h"
#include <iostream>
#include <string>
#include <chrono>
#include <omp.h>

int main(int argc, char** argv) {
    std::cout << "基于金字塔的全局DIC实现" << std::endl;
    std::cout << "User: " << "HowLi0" << std::endl;
    std::cout << "Date: " << "2025-04-18" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    // 输入图像路径 - 请根据您的实际路径修改
    std::string refImagePath = "E:/code_C++/RGDIC/reference.png";
    std::string defImagePath = "E:/code_C++/RGDIC/deformed.png";
    
    // 加载参考图像和变形图像
    cv::Mat refImage = cv::imread(refImagePath, cv::IMREAD_GRAYSCALE);
    cv::Mat defImage = cv::imread(defImagePath, cv::IMREAD_GRAYSCALE);
    
    // 检查图像是否成功加载
    if (refImage.empty() || defImage.empty()) {
        std::cerr << "错误: 无法加载图像." << std::endl;
        std::cerr << "请检查图像路径: " << refImagePath << ", " << defImagePath << std::endl;
        return -1;
    }
    
    // 转换为浮点类型以提高精度
    cv::Mat refImageFloat, defImageFloat;
    refImage.convertTo(refImageFloat, CV_64F);
    defImage.convertTo(defImageFloat, CV_64F);
    
    // 创建感兴趣区域 (ROI) - 默认使用整个图像
    cv::Mat roi = cv::Mat::ones(refImage.size(), CV_8U);
    
    // 可选: 创建自定义ROI (例如，排除图像边缘)
    int borderWidth = 20;
    roi = cv::Mat::ones(refImage.size(), CV_8U);
    cv::rectangle(roi, cv::Point(0, 0), cv::Point(roi.cols-1, roi.rows-1), 0, borderWidth);
    
    // 设置DIC参数
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
    params.numThreads = omp_get_max_threads() - 1;
    params.useEigenSolver = true;
    params.useSparseMatrix = true;
    params.useFastInterpolation = true;
    params.useSSE = true;
    params.useCaching = true;
    params.initialGuessMethod = PyramidGlobalDIC::FFTCC;
    params.useFFTCC = true;
    params.fftCCSearchRadius = 10;
    
    // 创建金字塔全局DIC实例
    PyramidGlobalDIC dic(params);
    
    // 测量计算时间
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // 执行DIC计算
    std::cout << "开始DIC计算..." << std::endl;
    PyramidGlobalDIC::Result result = dic.compute(refImageFloat, defImageFloat, roi);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime = endTime - startTime;
    
    std::cout << "DIC计算在 " << elapsedTime.count() << " 秒内完成" << std::endl;
    std::cout << "迭代次数: " << result.iterations << std::endl;
    
    // 计算应变场 (如果尚未在计算过程中完成)
    if (result.exx.empty() || result.eyy.empty() || result.exy.empty()) {
        dic.calculateStrains(result, true, 5);
    }
    
    // 显示结果
    dic.displayResults(refImage, result, true, true, true);
    
    // 保存位移场和应变场结果
    std::string outputPrefix = "E:/code_C++/RGDIC/";
    cv::Mat colorU, colorV, colorExx, colorEyy, colorExy;
    
    // 归一化并应用色彩映射到位移场
    double minU, maxU, minV, maxV;
    cv::minMaxLoc(result.u, &minU, &maxU, nullptr, nullptr, result.validMask);
    cv::minMaxLoc(result.v, &minV, &maxV, nullptr, nullptr, result.validMask);
    
    cv::Mat uNorm, vNorm;
    cv::normalize(result.u, uNorm, 0, 255, cv::NORM_MINMAX, CV_8U, result.validMask);
    cv::normalize(result.v, vNorm, 0, 255, cv::NORM_MINMAX, CV_8U, result.validMask);
    
    cv::applyColorMap(uNorm, colorU, cv::COLORMAP_JET);
    cv::applyColorMap(vNorm, colorV, cv::COLORMAP_JET);
    
    // 归一化并应用色彩映射到应变场
    double minExx, maxExx, minEyy, maxEyy, minExy, maxExy;
    cv::minMaxLoc(result.exx, &minExx, &maxExx, nullptr, nullptr, result.validMask);
    cv::minMaxLoc(result.eyy, &minEyy, &maxEyy, nullptr, nullptr, result.validMask);
    cv::minMaxLoc(result.exy, &minExy, &maxExy, nullptr, nullptr, result.validMask);
    
    cv::Mat exxNorm, eyyNorm, exyNorm;
    cv::normalize(result.exx, exxNorm, 0, 255, cv::NORM_MINMAX, CV_8U, result.validMask);
    cv::normalize(result.eyy, eyyNorm, 0, 255, cv::NORM_MINMAX, CV_8U, result.validMask);
    cv::normalize(result.exy, exyNorm, 0, 255, cv::NORM_MINMAX, CV_8U, result.validMask);
    
    cv::applyColorMap(exxNorm, colorExx, cv::COLORMAP_JET);
    cv::applyColorMap(eyyNorm, colorEyy, cv::COLORMAP_JET);
    cv::applyColorMap(exyNorm, colorExy, cv::COLORMAP_JET);
    
    // 创建位移矢量场可视化
    cv::Mat vectorField;
    cv::cvtColor(refImage, vectorField, cv::COLOR_GRAY2BGR);
    
    // 绘制位移矢量 (子采样以增强可视性)
    int step = 10; // 每10个像素采样一次
    for (int y = 0; y < result.u.rows; y += step) {
        for (int x = 0; x < result.u.cols; x += step) {
            if (result.validMask.at<uchar>(y, x)) {
                double dx = result.u.at<double>(y, x);
                double dy = result.v.at<double>(y, x);
                
                // 为可视性调整尺度
                double scale = 5.0;
                cv::arrowedLine(vectorField, 
                              cv::Point(x, y), 
                              cv::Point(x + dx * scale, y + dy * scale), 
                              cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
            }
        }
    }
    
    // 保存结果
    cv::imwrite(outputPrefix + "u_displacement.png", colorU);
    cv::imwrite(outputPrefix + "v_displacement.png", colorV);
    cv::imwrite(outputPrefix + "exx_strain.png", colorExx);
    cv::imwrite(outputPrefix + "eyy_strain.png", colorEyy);
    cv::imwrite(outputPrefix + "exy_strain.png", colorExy);
    cv::imwrite(outputPrefix + "vector_field.png", vectorField);
    
    // 保存原始计算数据 (可选，用于后续分析)
    cv::FileStorage fs(outputPrefix + "dic_results.yml", cv::FileStorage::WRITE);
    fs << "u" << result.u;
    fs << "v" << result.v;
    fs << "exx" << result.exx;
    fs << "eyy" << result.eyy;
    fs << "exy" << result.exy;
    fs << "validMask" << result.validMask;
    fs << "confidence" << result.confidence;
    fs << "meanResidual" << result.meanResidual;
    fs << "iterations" << result.iterations;
    fs.release();
    
    std::cout << "结果已保存到磁盘。" << std::endl;
    std::cout << "位移范围 - U: [" << minU << ", " << maxU << "] - V: [" << minV << ", " << maxV << "]" << std::endl;
    std::cout << "应变范围 - Exx: [" << minExx << ", " << maxExx << "] - Eyy: [" << minEyy << ", " << maxEyy 
              << "] - Exy: [" << minExy << ", " << maxExy << "]" << std::endl;
    std::cout << "按任意键退出..." << std::endl;
    cv::waitKey(0);
    
    return 0;
}