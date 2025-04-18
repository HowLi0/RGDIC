#include "global_dic.h"
#include "parallel_utils.h"
#include "pyramid_interface.h"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    std::cout << "全局DIC并行性能测试" << std::endl;
    std::cout << "User: " << "HowLi0" << std::endl;
    std::cout << "Date: " << "2025-04-18" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // 输入图像路径 - 请根据您的实际路径修改
    std::string refImagePath = "reference.png";
    std::string defImagePath = "deformed.png";
    
    // 加载参考图像和变形图像
    cv::Mat refImage = cv::imread(refImagePath, cv::IMREAD_GRAYSCALE);
    cv::Mat defImage = cv::imread(defImagePath, cv::IMREAD_GRAYSCALE);
    
    if (refImage.empty() || defImage.empty()) {
        std::cerr << "错误: 无法加载图像." << std::endl;
        return -1;
    }
    
    // 转换为浮点类型以提高精度
    cv::Mat refImageFloat, defImageFloat;
    refImage.convertTo(refImageFloat, CV_64F);
    defImage.convertTo(defImageFloat, CV_64F);
    
    // 创建ROI
    cv::Mat roi = cv::Mat::ones(refImage.size(), CV_8U);
    
    // 显示输入图像
    cv::imshow("参考图像", refImage);
    cv::imshow("变形图像", defImage);
    cv::waitKey(100);
    
    // 创建金字塔并显示
    std::vector<PyramidInterface::Layer> refPyramid;
    PyramidInterface::createPyramid(refImageFloat, roi, refPyramid, 3, 0.5);
    PyramidInterface::visualizePyramid(refPyramid, "参考图像金字塔");
    cv::waitKey(100);
    
    // 设置DIC参数
    PyramidGlobalDIC::Parameters params;
    params.nodeSpacing = 10;
    params.subsetRadius = 15;
    params.regularizationWeight = 0.1;
    params.regType = PyramidGlobalDIC::TIKHONOV;
    params.convergenceThreshold = 0.001;
    params.maxIterations = 20;
    params.order = PyramidGlobalDIC::FIRST_ORDER;
    params.useMultiScaleApproach = true;
    params.numScaleLevels = 3;
    params.scaleFactor = 0.5;
    params.useParallel = true;
    
    // 创建并行性能监控器
    ParallelPerformanceMonitor monitor("全局DIC计算");
    
    // 定义DIC计算任务，接受线程数作为参数
    auto dicTask = [&](int numThreads) {
        params.numThreads = numThreads;
        PyramidGlobalDIC dic(params);
        dic.compute(refImageFloat, defImageFloat, roi);
    };
    
    // 运行线程扩展测试
    monitor.runThreadScalingTest(dicTask, omp_get_max_threads());
    
    // 使用最优线程数运行完整DIC并显示结果
    int optimalThreads = omp_get_max_threads();
    params.numThreads = optimalThreads;
    PyramidGlobalDIC dic(params);
    
    monitor.start();
    PyramidGlobalDIC::Result result = dic.compute(refImageFloat, defImageFloat, roi);
    monitor.stop();
    
    monitor.printReport();
    
    // 显示结果
    dic.displayResults(refImage, result, true, true, true);
    
    std::cout << "按任意键退出" << std::endl;
    cv::waitKey(0);
    
    return 0;
}