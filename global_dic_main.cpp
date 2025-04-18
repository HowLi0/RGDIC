#include "global_dic.h"
#include <iostream>
#include <string>
#include <chrono>

int main(int argc, char** argv) {
    std::cout << "RGDIC with OpenCorr Pyramid Architecture" << std::endl;
    std::cout << "User: " << "HowLi0" << std::endl;
    std::cout << "Date: " << "2025-04-18" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // 设置输入图像路径
    std::string refImagePath = "E:/code_C++/RGDIC/synthetic_reference.png";
    std::string defImagePath = "E:/code_C++/RGDIC/synthetic_deformed.png";
    
    // 加载参考图像和变形图像
    cv::Mat refImage = cv::imread(refImagePath, cv::IMREAD_GRAYSCALE);
    cv::Mat defImage = cv::imread(defImagePath, cv::IMREAD_GRAYSCALE);
    
    // 检查图像是否成功加载
    if (refImage.empty() || defImage.empty()) {
        std::cerr << "Error: Could not load images." << std::endl;
        std::cerr << "Please check image paths: " << refImagePath << ", " << defImagePath << std::endl;
        return -1;
    }
    
    // 转换为浮点数类型以提高精度
    cv::Mat refImageFloat, defImageFloat;
    refImage.convertTo(refImageFloat, CV_64F);
    defImage.convertTo(defImageFloat, CV_64F);
    
    // 创建感兴趣区域 (ROI) - 默认使用整个图像
    cv::Mat roi = cv::Mat::ones(refImage.size(), CV_8U);
    
    // 可选: 创建自定义 ROI
    // cv::Mat roi = cv::Mat::zeros(refImage.size(), CV_8U);
    // cv::rectangle(roi, cv::Point(100, 100), cv::Point(900, 900), 255, -1);
    
    // 设置 GlobalDIC 参数
    GlobalDIC::Parameters params;
    params.nodeSpacing = 5;             // 节点间距
    params.subsetRadius = 15;            // 子区半径
    params.regularizationWeight = 0.1;   // 正则化权重
    params.regType = GlobalDIC::TIKHONOV; // 正则化类型
    params.convergenceThreshold = 0.001; // 收敛阈值
    params.maxIterations = 50;           // 最大迭代次数
    params.order = GlobalDIC::FIRST_ORDER; // 形函数阶数
    params.useMultiScaleApproach = true; // 使用多尺度方法
    params.numScaleLevels = 3;           // 尺度级别数
    params.scaleFactor = 0.5;            // 尺度因子
    params.useParallel = true;           // 启用并行计算
    params.minImageSize = cv::Size(32, 32); // 最小图像尺寸
    
    // 创建 GlobalDIC 实例
    GlobalDIC dic(params);
    
    // 测量计算时间
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // 执行 DIC 计算
    std::cout << "Starting DIC computation..." << std::endl;
    GlobalDIC::Result result = dic.compute(refImageFloat, defImageFloat, roi);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime = endTime - startTime;
    
    std::cout << "DIC computation completed in " << elapsedTime.count() << " seconds" << std::endl;
    std::cout << "Number of iterations: " << result.iterations << std::endl;
    
    // 计算应变场
    dic.calculateStrains(result);
    
    // 显示位移和应变结果
    dic.displayResults(refImage, result, true, true, true);
    
    // 保存位移场结果
    cv::Mat colorU, colorV;
    double minU, maxU, minV, maxV;
    cv::minMaxLoc(result.u, &minU, &maxU, nullptr, nullptr, result.validMask);
    cv::minMaxLoc(result.v, &minV, &maxV, nullptr, nullptr, result.validMask);
    
    // 归一化并应用色彩映射
    cv::Mat uNorm, vNorm;
    cv::normalize(result.u, uNorm, 0, 255, cv::NORM_MINMAX, CV_8U, result.validMask);
    cv::normalize(result.v, vNorm, 0, 255, cv::NORM_MINMAX, CV_8U, result.validMask);
    
    cv::applyColorMap(uNorm, colorU, cv::COLORMAP_JET);
    cv::applyColorMap(vNorm, colorV, cv::COLORMAP_JET);
    
    // 保存结果
    cv::imwrite("E:/code_C++/RGDIC/u_displacement.png", colorU);
    cv::imwrite("E:/code_C++/RGDIC/v_displacement.png", colorV);
    cv::imwrite("E:/code_C++/RGDIC/exx_strain.png", result.exx);
    cv::imwrite("E:/code_C++/RGDIC/eyy_strain.png", result.eyy);
    cv::imwrite("E:/code_C++/RGDIC/exy_strain.png", result.exy);
    
    std::cout << "Results saved to disk." << std::endl;
    std::cout << "Press any key to exit..." << std::endl;
    cv::waitKey(0);
    
    return 0;
}