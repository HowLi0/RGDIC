#include "global_dic.h"
#include <iostream>
#include <functional>
#include <omp.h>
#include <chrono>
#include <unordered_map>

// Result构造函数
PyramidGlobalDIC::Result::Result(const cv::Size& size) {
    u = cv::Mat::zeros(size, CV_64F);
    v = cv::Mat::zeros(size, CV_64F);
    exx = cv::Mat::zeros(size, CV_64F);
    eyy = cv::Mat::zeros(size, CV_64F);
    exy = cv::Mat::zeros(size, CV_64F);
    cc = cv::Mat::zeros(size, CV_64F);
    validMask = cv::Mat::zeros(size, CV_8U);
    confidence = cv::Mat::zeros(size, CV_64F);
    meanResidual = 0.0;
    iterations = 0;
}

// 默认参数
PyramidGlobalDIC::Parameters::Parameters() 
        : nodeSpacing(10),
        subsetRadius(15),
        regularizationWeight(0.1),
        regType(TIKHONOV),
        convergenceThreshold(0.001),
        maxIterations(50),
        order(FIRST_ORDER),
        useMultiScaleApproach(true),
        numScaleLevels(3),
        scaleFactor(0.5),
        useParallel(true),
        numThreads(omp_get_max_threads() - 1),
        minImageSize(cv::Size(32, 32)), // 设置最小图像尺寸为32x32
        useEigenSolver(true),           // 使用Eigen高效求解器
        useSparseMatrix(true),          // 使用稀疏矩阵
        useFastInterpolation(true),     // 使用快速插值
        useSSE(true),                   // 使用SSE指令集
        useCaching(true),               // 使用缓存
        initialGuessMethod(FFTCC),      // 使用FFT-CC作为初始猜测
        useFFTCC(true),                 // 启用FFT加速互相关
        fftCCSearchRadius(10)           // FFTCC搜索半径
      {}

PyramidGlobalDIC::PyramidGlobalDIC(const Parameters& params)
: m_params(params), m_gradientsComputed(false) {
    // 初始化FFTW线程
    fftwf_init_threads();
    fftwf_plan_with_nthreads(params.numThreads);
    
    // 设置OpenMP线程数
    if (m_params.useParallel) {
        omp_set_num_threads(m_params.numThreads);
        std::cout << "Using " << m_params.numThreads << " threads for parallel computation." << std::endl;
    }
    
    // 初始化图像插值缓存
    if (m_params.useCaching) {
        m_interpolationCache.clear();
        m_gradientCache.clear();
    }
    
    // 初始化FFTW实例池
    if (m_params.useParallel) {
        m_fftwPool.resize(m_params.numThreads);
    } else {
        m_fftwPool.resize(1); // 单线程模式下只需要一个实例
    }
    
    // 串行初始化FFTW实例池以避免并发问题
    for (int i = 0; i < static_cast<int>(m_fftwPool.size()); i++) {
        m_fftwPool[i] = FFTW::allocate(m_params.subsetRadius, m_params.subsetRadius);
        if (!m_fftwPool[i]) {
            std::cerr << "警告: 无法创建线程 " << i << " 的FFTW实例。将在需要时重试。" << std::endl;
        }
    }
}

PyramidGlobalDIC::~PyramidGlobalDIC() {
    // 清理FFTW实例池
    for (auto& instance : m_fftwPool) {
        if (instance) {
            FFTW::release(instance);
        }
    }
    m_fftwPool.clear();
    
    // 释放FFTW线程资源
    fftwf_cleanup_threads();
}

bool PyramidGlobalDIC::computeFFTCCDisplacement(const cv::Mat& refImage,
    const cv::Mat& defImage,
    const cv::Point& point,
    double& u, double& v,
    double& zncc) {
    // 获取当前线程的FFTW实例
    std::unique_ptr<FFTW>& fftw = getFFTWInstance(omp_get_thread_num());
    
    // Check if the FFTW instance is valid and initialized
    if (!fftw || !fftw->is_initialized) {
        std::cerr << "错误: FFTW实例无效或未初始化" << std::endl;
        return false;
    }
    
    int subset_width = fftw->subset_width;
    int subset_height = fftw->subset_height;
    int subset_radius_x = subset_width / 2;
    int subset_radius_y = subset_height / 2;
    int subset_size = subset_width * subset_height;
    
    // 检查点是否在有效边界内
    if (point.x - subset_radius_x < 0 || point.x + subset_radius_x >= refImage.cols ||
        point.y - subset_radius_y < 0 || point.y + subset_radius_y >= refImage.rows) {
        return false;
    }
    
    // 计算搜索范围 - 确保搜索区域不超出图像边界
    int search_radius_x = std::min(m_params.fftCCSearchRadius, 
                                  std::min(point.x, defImage.cols - point.x - 1));
    int search_radius_y = std::min(m_params.fftCCSearchRadius, 
                                  std::min(point.y, defImage.rows - point.y - 1));
    
    if (search_radius_x <= 0 || search_radius_y <= 0) {
        return false;
    }
    
    try {
        // 提取参考子集
        float ref_mean = 0.0f;
        for (int y = -subset_radius_y; y < subset_radius_y; y++) {
            for (int x = -subset_radius_x; x < subset_radius_x; x++) {
                int idx = (y + subset_radius_y) * subset_width + (x + subset_radius_x);
                if (idx < 0 || idx >= subset_size) continue; // 安全检查
                
                float value = static_cast<float>(refImage.at<double>(point.y + y, point.x + x));
                fftw->ref_subset[idx] = value;
                ref_mean += value;
            }
        }
        ref_mean /= subset_size;
        
        // 提取目标子集
        float tar_mean = 0.0f;
        for (int y = -subset_radius_y; y < subset_radius_y; y++) {
            for (int x = -subset_radius_x; x < subset_radius_x; x++) {
                int idx = (y + subset_radius_y) * subset_width + (x + subset_radius_x);
                if (idx < 0 || idx >= subset_size) continue; // 安全检查
                
                float value = static_cast<float>(defImage.at<double>(point.y + y, point.x + x));
                fftw->tar_subset[idx] = value;
                tar_mean += value;
            }
        }
        tar_mean /= subset_size;
        
        // 计算零均值子集和标准差
        float ref_norm = 0.0f;
        float tar_norm = 0.0f;
        
        for (int i = 0; i < subset_size; i++) {
            fftw->ref_subset[i] -= ref_mean;
            fftw->tar_subset[i] -= tar_mean;
            ref_norm += fftw->ref_subset[i] * fftw->ref_subset[i];
            tar_norm += fftw->tar_subset[i] * fftw->tar_subset[i];
        }
        
        // 检查标准差为零的情况（避免除零）
        if (ref_norm < 1e-10 || tar_norm < 1e-10) {
            return false;
        }
        
        // 执行FFT - 添加错误检查
        fftwf_execute(fftw->ref_plan);
        fftwf_execute(fftw->tar_plan);
        
        // 计算频域互相关
        int freq_size = fftw->subset_height * (fftw->subset_width / 2 + 1);
        for (int n = 0; n < freq_size; n++) {
            fftw->zncc_freq[n][0] = (fftw->ref_freq[n][0] * fftw->tar_freq[n][0]) + 
                                   (fftw->ref_freq[n][1] * fftw->tar_freq[n][1]);
            fftw->zncc_freq[n][1] = (fftw->ref_freq[n][0] * fftw->tar_freq[n][1]) - 
                                   (fftw->ref_freq[n][1] * fftw->tar_freq[n][0]);
        }
        
        // 执行逆FFT获取空域互相关
        fftwf_execute(fftw->zncc_plan);
        
        // 寻找最大互相关
        float max_zncc = -std::numeric_limits<float>::max();
        int max_idx = 0;
        
        for (int i = 0; i < subset_size; i++) {
            if (fftw->zncc[i] > max_zncc) {
                max_zncc = fftw->zncc[i];
                max_idx = i;
            }
        }
        
        // 计算位移
        int disp_x = max_idx % subset_width;
        int disp_y = max_idx / subset_width;
        
        // 将位移转换为相对于子集中心的位移
        if (disp_x > subset_radius_x) {
            disp_x -= subset_width;
        }
        if (disp_y > subset_radius_y) {
            disp_y -= subset_height;
        }
        
        // 设置位移和ZNCC
        u = disp_x;
        v = disp_y;
        zncc = max_zncc / (sqrt(ref_norm * tar_norm) * subset_size);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "FFTCC计算异常: " << e.what() << std::endl;
        return false;
    }
}

// 使用FFTCC计算初始位移场
void PyramidGlobalDIC::calculateInitialGuessFFTCC(const cv::Mat& refImage,
     const cv::Mat& defImage,
     const cv::Mat& roi,
     cv::Mat& initialU,
     cv::Mat& initialV) {
        // 初始化位移场矩阵
        initialU = cv::Mat::zeros(roi.size(), CV_64F);
        initialV = cv::Mat::zeros(roi.size(), CV_64F);

        // 定义采样步长（使用比节点间距更密集的网格可以提高精度）
        int samplingStep = std::max(1, m_params.nodeSpacing / 2);

        // 创建采样点网格
        std::vector<cv::Point> samplingPoints;
        for (int y = m_params.subsetRadius; y < roi.rows - m_params.subsetRadius; y += samplingStep) {
        for (int x = m_params.subsetRadius; x < roi.cols - m_params.subsetRadius; x += samplingStep) {
        if (roi.at<uchar>(y, x) > 0) {
        samplingPoints.push_back(cv::Point(x, y));
        }
        }
        }

        // 用于记录有效计算点的掩码
        cv::Mat validMask = cv::Mat::zeros(roi.size(), CV_8U);

        // 并行计算每个采样点的位移
        std::cout << "使用FFTCC计算初始位移场，采样点数：" << samplingPoints.size() << std::endl;

        #pragma omp parallel for if(m_params.useParallel)
        for (int i = 0; i < (int)samplingPoints.size(); i++) {
        double u, v, zncc;
        bool success = computeFFTCCDisplacement(refImage, defImage, samplingPoints[i], u, v, zncc);

        if (success) {
        #pragma omp critical
        {
        initialU.at<double>(samplingPoints[i]) = u;
        initialV.at<double>(samplingPoints[i]) = v;
        validMask.at<uchar>(samplingPoints[i]) = 1;
        }
        }
        }

        // 填充未计算的区域（使用平均值或邻近插值）
        cv::Mat validU, validV;
        initialU.copyTo(validU, validMask);
        initialV.copyTo(validV, validMask);

        // 使用平均值填充未计算区域
        double meanU = cv::mean(validU, validMask)[0];
        double meanV = cv::mean(validV, validMask)[0];

        for (int y = 0; y < roi.rows; y++) {
        for (int x = 0; x < roi.cols; x++) {
        if (roi.at<uchar>(y, x) > 0 && validMask.at<uchar>(y, x) == 0) {
        initialU.at<double>(y, x) = meanU;
        initialV.at<double>(y, x) = meanV;
        }
        }
        }

        // 应用高斯平滑以减少噪声
        cv::GaussianBlur(initialU, initialU, cv::Size(5, 5), 1.0);
        cv::GaussianBlur(initialV, initialV, cv::Size(5, 5), 1.0);

        std::cout << "FFTCC初始位移场计算完成，平均位移：u=" << meanU << ", v=" << meanV << std::endl;
}


// 使用FFT加速的全局DIC计算
PyramidGlobalDIC::Result PyramidGlobalDIC::compute(const cv::Mat& refImage, const cv::Mat& defImage, const cv::Mat& roi) {
    // 重置梯度缓存
    m_gradientsComputed = false;
    m_gradX = cv::Mat();
    m_gradY = cv::Mat();
    m_shapeFunctionCache.clear();
    
    // 预先计算图像梯度 - 这将改善后续处理速度
    if (m_params.useCaching) {
        precomputeGradients(refImage);
    }
    
    // 检查是否启用多尺度方法
    if (m_params.useMultiScaleApproach) {
        return computeMultiScale(refImage, defImage, roi);
    }
    
    // 初始化结果结构体
    Result result(roi.size());
    result.validMask = roi.clone();
    
    // 创建节点网格
    std::vector<cv::Point> nodePoints;
    createNodeGrid(roi, nodePoints);
    
    // 节点数量和自由度
    int numNodes = static_cast<int>(nodePoints.size());
    int numDOFs = numNodes * 2;  // 每个节点2个自由度 (u和v)
    
    std::cout << "总节点数: " << numNodes << ", 总自由度: " << numDOFs << std::endl;
    
    // 位移初始猜测
    cv::Mat nodeDisplacements = cv::Mat::zeros(numDOFs, 1, CV_64F);
    
    // 如果启用FFTCC初始猜测
    if (m_params.initialGuessMethod == FFTCC) {
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // 使用FFTCC计算初始位移场
        cv::Mat initialU, initialV;
        calculateInitialGuessFFTCC(refImage, defImage, roi, initialU, initialV);
        
        // 将位移场应用到节点
        for (int i = 0; i < numNodes; i++) {
            cv::Point node = nodePoints[i];
            nodeDisplacements.at<double>(i * 2) = initialU.at<double>(node);
            nodeDisplacements.at<double>(i * 2 + 1) = initialV.at<double>(node);
        }
        
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = endTime - startTime;
        std::cout << "FFTCC初始猜测计算耗时: " << duration.count() << " 秒" << std::endl;
    }
    
    // 迭代求解
    double prevResidual = std::numeric_limits<double>::max();
    int iter = 0;
    double residualNorm = 0.0;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // 使用高斯-牛顿优化迭代
    for (iter = 0; iter < m_params.maxIterations; iter++) {
        // 构建全局系统
        if (m_params.useSparseMatrix) {
            // 使用稀疏矩阵实现，大大提高求解速度
            Eigen::SparseMatrix<double> systemMatrix(numDOFs, numDOFs);
            Eigen::VectorXd systemVector(numDOFs);
            systemVector.setZero();
            
            buildGlobalSystemSparse(refImage, defImage, nodePoints, systemMatrix, systemVector, nodeDisplacements);
            
            // 添加正则化
            addRegularizationSparse(nodePoints, systemMatrix);
            
            // 求解系统
            Eigen::VectorXd deltaDisplacements;
            bool solveSuccess = solveSystemSparse(systemMatrix, systemVector, deltaDisplacements, residualNorm);
            
            if (!solveSuccess) {
                std::cout << "警告: 无法在迭代 " << iter << " 求解系统" << std::endl;
                break;
            }
            
            // 更新位移场
            for (int i = 0; i < numDOFs; i++) {
                nodeDisplacements.at<double>(i, 0) += deltaDisplacements(i);
            }
        } else {
            // 使用OpenCV的稠密矩阵实现
            cv::Mat systemMatrix = cv::Mat::zeros(numDOFs, numDOFs, CV_64F);
            cv::Mat systemVector = cv::Mat::zeros(numDOFs, 1, CV_64F);
            
            buildGlobalSystem(refImage, defImage, nodePoints, systemMatrix, systemVector, nodeDisplacements);
            
            // 添加正则化
            addRegularization(nodePoints, systemMatrix);
            
            // 求解系统
            cv::Mat deltaDisplacements;
            bool solveSuccess = solveSystem(systemMatrix, systemVector, deltaDisplacements, residualNorm);
            
            if (!solveSuccess) {
                std::cout << "警告: 无法在迭代 " << iter << " 求解系统" << std::endl;
                break;
            }
            
            // 更新位移场
            nodeDisplacements += deltaDisplacements;
        }
        
        // 输出迭代状态
        std::cout << "迭代 " << iter 
                  << ", 残差: " << residualNorm 
                  << std::endl;
                  
        // 检查收敛
        if (residualNorm < m_params.convergenceThreshold || 
            std::abs(residualNorm - prevResidual) < m_params.convergenceThreshold / 10) {
            std::cout << "经过 " << iter << " 次迭代后达到收敛." << std::endl;
            break;
        }
        
        prevResidual = residualNorm;
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsedTime = endTime - startTime;
    
    std::cout << "在 " << elapsedTime.count() << " 秒后经过 " 
              << iter << " 次迭代求解完成." << std::endl;
    
    // 记录执行的迭代次数
    result.iterations = iter;
    
    // 从节点位移生成完整位移场
    generateDisplacementField(nodePoints, nodeDisplacements, result, roi);
    
    // 计算相关系数
    double cc = calculateCorrelation(refImage, defImage, nodePoints, nodeDisplacements);
    std::cout << "最终相关系数: " << cc << std::endl;
    
    // 计算平均残差
    result.meanResidual = calculateResidual(refImage, defImage, nodePoints, nodeDisplacements);
    
    // 计算置信度指标
    calculateConfidence(result, refImage, defImage);
    
    // 计算应变场
    calculateStrains(result);
    
    return result;
}

void PyramidGlobalDIC::calculateStrains(Result& result, bool useWindowedLeastSquares, int windowSize) {
    // 确保有有效的位移场
    if (result.u.empty() || result.v.empty() || result.validMask.empty()) {
        std::cerr << "无法计算应变场：位移场为空" << std::endl;
        return;
    }
    
    // 使用滑动窗口最小二乘法计算应变
    if (useWindowedLeastSquares) {
        int halfWindow = windowSize / 2;
        
        // 对每个有效像素计算应变
        #pragma omp parallel for if(m_params.useParallel)
        for (int y = halfWindow; y < result.u.rows - halfWindow; y++) {
            for (int x = halfWindow; x < result.u.cols - halfWindow; x++) {
                if (result.validMask.at<uchar>(y, x)) {
                    // 收集窗口内的有效点
                    std::vector<cv::Point2f> points;
                    std::vector<cv::Point2f> displacements;
                    
                    for (int dy = -halfWindow; dy <= halfWindow; dy++) {
                        for (int dx = -halfWindow; dx <= halfWindow; dx++) {
                            int nx = x + dx;
                            int ny = y + dy;
                            
                            if (nx >= 0 && nx < result.u.cols &&
                                ny >= 0 && ny < result.u.rows &&
                                result.validMask.at<uchar>(ny, nx)) {
                                
                                points.push_back(cv::Point2f(dx, dy));
                                displacements.push_back(cv::Point2f(
                                    result.u.at<double>(ny, nx),
                                    result.v.at<double>(ny, nx)
                                ));
                            }
                        }
                    }
                    
                    // 如果有足够的有效点
                    if (points.size() >= 6) { // 至少需要6个点进行最小二乘拟合
                        // 构建最小二乘问题
                        cv::Mat A(points.size(), 3, CV_64F);
                        cv::Mat bx(points.size(), 1, CV_64F);
                        cv::Mat by(points.size(), 1, CV_64F);
                        
                        for (size_t i = 0; i < points.size(); i++) {
                            A.at<double>(i, 0) = 1.0;
                            A.at<double>(i, 1) = points[i].x;
                            A.at<double>(i, 2) = points[i].y;
                            
                            bx.at<double>(i, 0) = displacements[i].x;
                            by.at<double>(i, 0) = displacements[i].y;
                        }
                        
                        // 求解 Ax = b
                        cv::Mat xu, xv;
                        cv::solve(A, bx, xu, cv::DECOMP_SVD);
                        cv::solve(A, by, xv, cv::DECOMP_SVD);
                        
                        // 提取应变分量
                        double dudx = xu.at<double>(1, 0);
                        double dudy = xu.at<double>(2, 0);
                        double dvdx = xv.at<double>(1, 0);
                        double dvdy = xv.at<double>(2, 0);
                        
                        // 计算应变分量
                        double exx = dudx;
                        double eyy = dvdy;
                        double exy = 0.5 * (dudy + dvdx);
                        
                        // 存储结果
                        result.exx.at<double>(y, x) = exx;
                        result.eyy.at<double>(y, x) = eyy;
                        result.exy.at<double>(y, x) = exy;
                    }
                }
            }
        }
    }
    // 使用中心差分计算应变
    else {
        // 使用Sobel算子计算梯度
        cv::Sobel(result.u, result.exx, CV_64F, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
        cv::Sobel(result.v, result.eyy, CV_64F, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
        
        cv::Mat dudy, dvdx;
        cv::Sobel(result.u, dudy, CV_64F, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
        cv::Sobel(result.v, dvdx, CV_64F, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
        
        // 计算剪切应变
        result.exy = 0.5 * (dudy + dvdx);
        
        // 仅保留有效区域中的应变值
        result.exx.setTo(0, ~result.validMask);
        result.eyy.setTo(0, ~result.validMask);
        result.exy.setTo(0, ~result.validMask);
    }
}

void PyramidGlobalDIC::displayResults(const cv::Mat& refImage, const Result& result, 
    bool showDisplacement, bool showStrain, 
    bool useEnhancedVisualization) {
    // 仅在有效区域内可视化结果
    cv::Mat validMask = result.validMask > 0;

    // 可视化位移
    if (showDisplacement) {
    // 找到位移范围
    double minU, maxU, minV, maxV;
    cv::minMaxLoc(result.u, &minU, &maxU, nullptr, nullptr, validMask);
    cv::minMaxLoc(result.v, &minV, &maxV, nullptr, nullptr, validMask);

    // 创建热力图显示
    cv::Mat uNorm, vNorm;
    cv::normalize(result.u, uNorm, 0, 255, cv::NORM_MINMAX, CV_8U, validMask);
    cv::normalize(result.v, vNorm, 0, 255, cv::NORM_MINMAX, CV_8U, validMask);

    cv::Mat uColor, vColor;
    cv::applyColorMap(uNorm, uColor, cv::COLORMAP_JET);
    cv::applyColorMap(vNorm, vColor, cv::COLORMAP_JET);

    // 创建标题文本
    std::stringstream ssU, ssV;
    ssU << "X Displacement (min: " << minU << ", max: " << maxU << ")";
    ssV << "Y Displacement (min: " << minV << ", max: " << maxV << ")";

    // 显示结果
    cv::imshow(ssU.str(), uColor);
    cv::imshow(ssV.str(), vColor);

    // 创建位移矢量场可视化
    cv::Mat vectorField;
    cv::cvtColor(refImage, vectorField, cv::COLOR_GRAY2BGR);

    // 绘制位移矢量（为清晰起见进行子采样）
    int step = 10; // 每隔10个像素采样
    for (int y = 0; y < result.u.rows; y += step) {
    for (int x = 0; x < result.u.cols; x += step) {
    if (validMask.at<uchar>(y, x)) {
    double dx = result.u.at<double>(y, x);
    double dy = result.v.at<double>(y, x);

    // 缩放以增强可见性
    double scale = 5.0;
    cv::arrowedLine(vectorField, 
        cv::Point(x, y), 
        cv::Point(x + dx * scale, y + dy * scale), 
        cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }
    }
    }

    cv::imshow("Displacement Vector Field", vectorField);
    }

    // 可视化应变
    if (showStrain) {
    // 找到应变范围
    double minExx, maxExx, minEyy, maxEyy, minExy, maxExy;
    cv::minMaxLoc(result.exx, &minExx, &maxExx, nullptr, nullptr, validMask);
    cv::minMaxLoc(result.eyy, &minEyy, &maxEyy, nullptr, nullptr, validMask);
    cv::minMaxLoc(result.exy, &minExy, &maxExy, nullptr, nullptr, validMask);

    // 创建热力图显示
    cv::Mat exxNorm, eyyNorm, exyNorm;
    double maxStrainAbs = std::max({std::abs(minExx), std::abs(maxExx), 
            std::abs(minEyy), std::abs(maxEyy), 
            std::abs(minExy), std::abs(maxExy)});

    // 使用对称归一化，使得0应变映射到中间色
    cv::normalize(result.exx, exxNorm, 0, 255, cv::NORM_MINMAX, CV_8U, validMask);
    cv::normalize(result.eyy, eyyNorm, 0, 255, cv::NORM_MINMAX, CV_8U, validMask);
    cv::normalize(result.exy, exyNorm, 0, 255, cv::NORM_MINMAX, CV_8U, validMask);

    cv::Mat exxColor, eyyColor, exyColor;
    cv::applyColorMap(exxNorm, exxColor, cv::COLORMAP_JET);
    cv::applyColorMap(eyyNorm, eyyColor, cv::COLORMAP_JET);
    cv::applyColorMap(exyNorm, exyColor, cv::COLORMAP_JET);

    // 将非有效区域设置为黑色
    exxColor.setTo(cv::Scalar(0, 0, 0), ~validMask);
    eyyColor.setTo(cv::Scalar(0, 0, 0), ~validMask);
    exyColor.setTo(cv::Scalar(0, 0, 0), ~validMask);

    // 创建标题文本
    std::stringstream ssExx, ssEyy, ssExy;
    ssExx << "Normal Strain Exx (min: " << minExx << ", max: " << maxExx << ")";
    ssEyy << "Normal Strain Eyy (min: " << minEyy << ", max: " << maxEyy << ")";
    ssExy << "Shear Strain Exy (min: " << minExy << ", max: " << maxExy << ")";
            
    // 显示结果
    cv::imshow(ssExx.str(), exxColor);
    cv::imshow(ssEyy.str(), eyyColor);
    cv::imshow(ssExy.str(), exyColor);
    }

    // 显示置信度地图
    if (!result.confidence.empty()) {
    cv::Mat confNorm;
    cv::normalize(result.confidence, confNorm, 0, 255, cv::NORM_MINMAX, CV_8U, validMask);

    cv::Mat confColor;
    cv::applyColorMap(confNorm, confColor, cv::COLORMAP_VIRIDIS);
    confColor.setTo(cv::Scalar(0, 0, 0), ~validMask);

    cv::imshow("Confidence Map", confColor);
    }

    cv::waitKey(1); // 更新显示
}

void PyramidGlobalDIC::createNodeGrid(const cv::Mat& roi, std::vector<cv::Point>& nodePoints) {
    nodePoints.clear();
    
    // 在ROI内创建规则节点网格
    for (int y = m_params.subsetRadius; y < roi.rows - m_params.subsetRadius; y += m_params.nodeSpacing) {
        for (int x = m_params.subsetRadius; x < roi.cols - m_params.subsetRadius; x += m_params.nodeSpacing) {
            if (roi.at<uchar>(y, x) > 0) {
                nodePoints.push_back(cv::Point(x, y));
            }
        }
    }
    
    std::cout << "为全局DIC创建了 " << nodePoints.size() << " 个节点." << std::endl;
}

void PyramidGlobalDIC::buildGlobalSystem(const cv::Mat& refImage, const cv::Mat& defImage,
    const std::vector<cv::Point>& nodePoints,
    cv::Mat& systemMatrix, cv::Mat& systemVector,
    const cv::Mat& nodeDisplacements) {
        int numNodes = static_cast<int>(nodePoints.size());
        int numDOFs = numNodes * 2;

        // 如果图像不是浮点类型，则转换为浮点类型
        cv::Mat refFloat, defFloat;
        if (refImage.type() != CV_64F) {
        refImage.convertTo(refFloat, CV_64F);
        } else {
        refFloat = refImage;
        }

        if (defImage.type() != CV_64F) {
        defImage.convertTo(defFloat, CV_64F);
        } else {
        defFloat = defImage;
        }

        // 如果尚未计算，则计算图像梯度
        if (!m_gradientsComputed) {
        cv::Sobel(refFloat, m_gradX, CV_64F, 1, 0, 3);
        cv::Sobel(refFloat, m_gradY, CV_64F, 0, 1, 3);
        m_gradientsComputed = true;
        }

        // 重置系统矩阵和向量
        systemMatrix = cv::Mat::zeros(numDOFs, numDOFs, CV_64F);
        systemVector = cv::Mat::zeros(numDOFs, 1, CV_64F);

        // 如果启用，则使用OpenMP进行并行化
        #pragma omp parallel for if(m_params.useParallel)
        for (int nodeIdx = 0; nodeIdx < numNodes; nodeIdx++) {
        cv::Point nodePoint = nodePoints[nodeIdx];

        // 线程局部变量，用于累积
        std::vector<std::vector<double>> localSystemMatrix(numDOFs, std::vector<double>(numDOFs, 0.0));
        std::vector<double> localSystemVector(numDOFs, 0.0);

        // 遍历该节点周围子集中的每个像素
        for (int y = -m_params.subsetRadius; y <= m_params.subsetRadius; y++) {
        for (int x = -m_params.subsetRadius; x <= m_params.subsetRadius; x++) {
        cv::Point pixelPoint = nodePoint + cv::Point(x, y);

        // 检查像素是否在图像边界内
        if (pixelPoint.x < 0 || pixelPoint.x >= refFloat.cols ||
        pixelPoint.y < 0 || pixelPoint.y >= refFloat.rows) {
        continue;
        }

        // 获取参考强度和梯度
        double refIntensity = refFloat.at<double>(pixelPoint);
        double gx = m_gradX.at<double>(pixelPoint);
        double gy = m_gradY.at<double>(pixelPoint);

        // 基于当前位移估计计算变形点
        double u = 0.0, v = 0.0;
        interpolateDisplacement(pixelPoint, nodePoints, nodeDisplacements, u, v);

        cv::Point2f warpedPoint(pixelPoint.x + u, pixelPoint.y + v);

        // 检查变形点是否在图像边界内
        if (warpedPoint.x < 0 || warpedPoint.x >= defFloat.cols - 1 ||
        warpedPoint.y < 0 || warpedPoint.y >= defFloat.rows - 1) {
        continue;
        }

        // 在变形图像中插值
        double defIntensity = interpolate(defFloat, warpedPoint);

        // 计算残差 r = I_def(x+u,y+v) - I_ref(x,y)
        double residual = defIntensity - refIntensity;

        // 对附近节点的每个自由度...
        for (int influenceNode = 0; influenceNode < numNodes; influenceNode++) {
        // 计算形函数
        std::vector<double> N, dNdx, dNdy;
        computeShapeFunctions(pixelPoint, nodePoints, N, dNdx, dNdy);

        // drdp 是关于每个参数的残差梯度
        double drdpu = gx * N[influenceNode];  // dr/du
        double drdpv = gy * N[influenceNode];  // dr/dv

        // 向系统向量添加此像素的贡献
        int uIdx = influenceNode * 2;        // u DOF index
        int vIdx = influenceNode * 2 + 1;    // v DOF index

        localSystemVector[uIdx] -= drdpu * residual;
        localSystemVector[vIdx] -= drdpv * residual;

        // 向系统矩阵添加此像素的贡献
        for (int influenceNode2 = 0; influenceNode2 < numNodes; influenceNode2++) {
        double N2 = N[influenceNode2];

        int uIdx2 = influenceNode2 * 2;
        int vIdx2 = influenceNode2 * 2 + 1;

        localSystemMatrix[uIdx][uIdx2] += drdpu * gx * N2;
        localSystemMatrix[uIdx][vIdx2] += drdpu * gy * N2;
        localSystemMatrix[vIdx][uIdx2] += drdpv * gx * N2;
        localSystemMatrix[vIdx][vIdx2] += drdpv * gy * N2;
        }
        }
        }
        }

        // 将线程局部结果与全局系统矩阵和向量合并
        #pragma omp critical
        {
        for (int i = 0; i < numDOFs; i++) {
        systemVector.at<double>(i, 0) += localSystemVector[i];

        for (int j = 0; j < numDOFs; j++) {
        systemMatrix.at<double>(i, j) += localSystemMatrix[i][j];
        }
        }
        }
        }
}

void PyramidGlobalDIC::addRegularization(const std::vector<cv::Point>& nodePoints, cv::Mat& systemMatrix) {
        int numNodes = static_cast<int>(nodePoints.size());
        int numDOFs = numNodes * 2;

        // 简单的Tikhonov正则化
        if (m_params.regType == TIKHONOV) {
        for (int i = 0; i < numDOFs; i++) {
        systemMatrix.at<double>(i, i) += m_params.regularizationWeight;
        }
        }
        // 扩散正则化 - 基于位置相近的节点添加联系
        else if (m_params.regType == DIFFUSION) {
        for (int i = 0; i < numNodes; i++) {
        cv::Point p1 = nodePoints[i];
        int i_u = i * 2;
        int i_v = i * 2 + 1;

        for (int j = 0; j < numNodes; j++) {
        cv::Point p2 = nodePoints[j];

        // 计算距离
        double dist = cv::norm(p1 - p2);

        // 只考虑近邻节点
        if (dist > 0 && dist < m_params.nodeSpacing * 2.5) {
        int j_u = j * 2;
        int j_v = j * 2 + 1;

        double weight = m_params.regularizationWeight * (1.0 / dist);

        // 为差分方程添加权重
        systemMatrix.at<double>(i_u, i_u) += weight;
        systemMatrix.at<double>(i_u, j_u) -= weight;
        systemMatrix.at<double>(i_v, i_v) += weight;
        systemMatrix.at<double>(i_v, j_v) -= weight;
        }
        }
        }
        }
        // 总变分正则化 - 此处简化实现
        else if (m_params.regType == TOTAL_VARIATION) {
        // TV正则化的简化实现
        double epsilon = 1e-4;  // 小常数，避免零除

        for (int i = 0; i < numNodes; i++) {
        cv::Point p1 = nodePoints[i];
        int i_u = i * 2;
        int i_v = i * 2 + 1;

        for (int j = 0; j < numNodes; j++) {
        cv::Point p2 = nodePoints[j];

        // 计算距离
        double dist = cv::norm(p1 - p2);

        // 只考虑近邻节点
        if (dist > 0 && dist < m_params.nodeSpacing * 2.5) {
        int j_u = j * 2;
        int j_v = j * 2 + 1;

        double weight = m_params.regularizationWeight / (dist * std::sqrt(epsilon + dist * dist));

        // 为TV正则化添加权重
        systemMatrix.at<double>(i_u, i_u) += weight;
        systemMatrix.at<double>(i_u, j_u) -= weight;
        systemMatrix.at<double>(i_v, i_v) += weight;
        systemMatrix.at<double>(i_v, j_v) -= weight;
        }
        }
        }
        }
}

bool PyramidGlobalDIC::solveSystem(const cv::Mat& systemMatrix, const cv::Mat& systemVector, 
    cv::Mat& solution, double& residualNorm) {
    // 使用OpenCV的求解器求解线性系统
    bool success = cv::solve(systemMatrix, systemVector, solution, cv::DECOMP_SVD);

    // 计算残差范数
    if (success) {
    cv::Mat residual = systemMatrix * solution - systemVector;
    residualNorm = cv::norm(residual);
    } else {
    residualNorm = std::numeric_limits<double>::max();
    }

    return success;
}

double PyramidGlobalDIC::calculateCorrelation(const cv::Mat& refImage, 
  const cv::Mat& defImage,
  const std::vector<cv::Point>& nodePoints,
  const cv::Mat& nodeDisplacements) {
// 转换图像为CV_64F
cv::Mat refFloat, defFloat;
refImage.convertTo(refFloat, CV_64F);
defImage.convertTo(defFloat, CV_64F);

// 初始化变量
double sumRefIntensity = 0.0;
double sumDefIntensity = 0.0;
double sumRefSquared = 0.0;
double sumDefSquared = 0.0;
double sumProduct = 0.0;
int validPixelCount = 0;

// 遍历图像中的所有像素
#pragma omp parallel for reduction(+:sumRefIntensity,sumDefIntensity,sumRefSquared,sumDefSquared,sumProduct,validPixelCount) if(m_params.useParallel)
for (int y = 0; y < refFloat.rows; y++) {
for (int x = 0; x < refFloat.cols; x++) {
cv::Point currentPoint(x, y);

// 计算变形点位置
double u, v;
interpolateDisplacement(currentPoint, nodePoints, nodeDisplacements, u, v);

cv::Point2f warpedPoint(x + u, y + v);

// 检查变形点是否在图像边界内
if (warpedPoint.x < 0 || warpedPoint.x >= defFloat.cols - 1 ||
warpedPoint.y < 0 || warpedPoint.y >= defFloat.rows - 1) {
continue;
}

double refIntensity = refFloat.at<double>(y, x);
double defIntensity = interpolate(defFloat, warpedPoint);

sumRefIntensity += refIntensity;
sumDefIntensity += defIntensity;
sumRefSquared += refIntensity * refIntensity;
sumDefSquared += defIntensity * defIntensity;
sumProduct += refIntensity * defIntensity;
validPixelCount++;
}
}

// 计算ZNCC
double meanRef = sumRefIntensity / validPixelCount;
double meanDef = sumDefIntensity / validPixelCount;

double numerator = sumProduct - meanRef * sumDefIntensity - meanDef * sumRefIntensity + validPixelCount * meanRef * meanDef;
double denominator = std::sqrt((sumRefSquared - validPixelCount * meanRef * meanRef) * 
(sumDefSquared - validPixelCount * meanDef * meanDef));

if (denominator < 1e-10) {
return 0.0;
}

return numerator / denominator;
}

double PyramidGlobalDIC::interpolate(const cv::Mat& image, const cv::Point2f& pt) const {
// 双线性插值
int x = static_cast<int>(pt.x);
int y = static_cast<int>(pt.y);

double dx = pt.x - x;
double dy = pt.y - y;

double value = (1.0 - dx) * (1.0 - dy) * image.at<double>(y, x) +
dx * (1.0 - dy) * image.at<double>(y, x + 1) +
(1.0 - dx) * dy * image.at<double>(y + 1, x) +
dx * dy * image.at<double>(y + 1, x + 1);

return value;
}

void PyramidGlobalDIC::generateDisplacementField(const std::vector<cv::Point>& nodePoints,
     const cv::Mat& nodeDisplacements,
     Result& result,
     const cv::Mat& roi) {
// 遍历图像中的所有像素
#pragma omp parallel for if(m_params.useParallel)
for (int y = 0; y < result.u.rows; y++) {
for (int x = 0; x < result.u.cols; x++) {
if (roi.at<uchar>(y, x) > 0) {
cv::Point currentPoint(x, y);

double u, v;
interpolateDisplacement(currentPoint, nodePoints, nodeDisplacements, u, v);

result.u.at<double>(y, x) = u;
result.v.at<double>(y, x) = v;
result.validMask.at<uchar>(y, x) = 1;
}
}
}
}

void PyramidGlobalDIC::createImagePyramid(const cv::Mat& image, const cv::Mat& mask, std::vector<Layer>& pyramid) {
// 清除旧金字塔
pyramid.clear();

// 确定图像的最小尺寸
int minDim = std::min(image.cols, image.rows);

// 设置金字塔级别
int nLevels = m_params.numScaleLevels;

// 确保我们不会生成太小的图像
int maxLevels = 1;
double currentScale = 1.0;
while (maxLevels < 10) { // 限制最大级别
currentScale *= m_params.scaleFactor;
int width = static_cast<int>(image.cols * currentScale);
int height = static_cast<int>(image.rows * currentScale);
if (width < m_params.minImageSize.width || height < m_params.minImageSize.height) {
break;
}
maxLevels++;
}

nLevels = std::min(nLevels, maxLevels);

// 调整金字塔大小
pyramid.resize(nLevels);

// 设置第一层（原始尺寸）
pyramid[0].img = image.clone();
pyramid[0].mask = mask.clone();
pyramid[0].octave = 0;
pyramid[0].scale = 1.0;
pyramid[0].sigma = 0.0; // 无模糊

// 为每一层创建图像
for (int i = 1; i < nLevels; i++) {
double scale = std::pow(m_params.scaleFactor, static_cast<double>(i));
int newWidth = static_cast<int>(image.cols * scale);
int newHeight = static_cast<int>(image.rows * scale);

// 设置层信息
pyramid[i].octave = i;
pyramid[i].scale = scale;
pyramid[i].sigma = 1.0; // 适用于降尺度的高斯模糊

// 调整图像大小
cv::resize(image, pyramid[i].img, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_AREA);
cv::resize(mask, pyramid[i].mask, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_NEAREST);

// 应用模糊以减少缩小时的锯齿状
cv::GaussianBlur(pyramid[i].img, pyramid[i].img, cv::Size(3, 3), pyramid[i].sigma);
}

std::cout << "创建了 " << pyramid.size() << " 层图像金字塔" << std::endl;
}

// 预计算梯度以用于缓存
void PyramidGlobalDIC::precomputeGradients(const cv::Mat& image) {
    // 确保图像是CV_64F类型
    cv::Mat imageFloat;
    if (image.type() != CV_64F) {
        image.convertTo(imageFloat, CV_64F);
    } else {
        imageFloat = image;
    }
    
    // 计算图像梯度
    cv::Sobel(imageFloat, m_gradX, CV_64F, 1, 0, 3);
    cv::Sobel(imageFloat, m_gradY, CV_64F, 0, 1, 3);
    m_gradientsComputed = true;
    
    // 清空缓存
    m_gradientCache.clear();
    
    std::cout << "预计算图像梯度完成" << std::endl;
}

// 优化的双线性插值函数
inline double PyramidGlobalDIC::fastInterpolate(const cv::Mat& image, const cv::Point2f& pt) const {
    if (m_params.useCaching) {
        // 创建缓存键 (四舍五入到0.1的精度以减少缓存大小并保持精度)
        int64_t key = (int64_t)(std::round(pt.x * 10) * 100000 + std::round(pt.y * 10));
        
        // 检查缓存
        auto it = m_interpolationCache.find(key);
        if (it != m_interpolationCache.end()) {
            return it->second;
        }
    }
    
    // 标准双线性插值
    int x = static_cast<int>(pt.x);
    int y = static_cast<int>(pt.y);
    
    double dx = pt.x - x;
    double dy = pt.y - y;
    
    // 快速插值实现
    if (m_params.useFastInterpolation && m_params.useSSE) {
        // 访问边界检查
        if (x < 0 || x >= image.cols-1 || y < 0 || y >= image.rows-1) {
            return 0.0;
        }
        
        // 直接访问图像数据以加速
        const double* row0 = image.ptr<double>(y);
        const double* row1 = image.ptr<double>(y+1);
        
        double a = row0[x];
        double b = row0[x+1];
        double c = row1[x];
        double d = row1[x+1];
        
        // 使用双线性插值公式
        double value = a * (1.0 - dx) * (1.0 - dy) +
                      b * dx * (1.0 - dy) +
                      c * (1.0 - dx) * dy +
                      d * dx * dy;
        
        // 缓存结果
        if (m_params.useCaching) {
            int64_t key = (int64_t)(std::round(pt.x * 10) * 100000 + std::round(pt.y * 10));
            m_interpolationCache[key] = value;
        }
        
        return value;
    } else {
        // 标准插值
        double value = (1.0 - dx) * (1.0 - dy) * image.at<double>(y, x) +
                      dx * (1.0 - dy) * image.at<double>(y, x + 1) +
                      (1.0 - dx) * dy * image.at<double>(y + 1, x) +
                      dx * dy * image.at<double>(y + 1, x + 1);
        
        // 缓存结果
        if (m_params.useCaching) {
            int64_t key = (int64_t)(std::round(pt.x * 10) * 100000 + std::round(pt.y * 10));
            m_interpolationCache[key] = value;
        }
        
        return value;
    }
}

// 获取梯度值，带缓存优化
inline void PyramidGlobalDIC::getGradientValue(const cv::Point& point, double& gx, double& gy) const {
    if (m_params.useCaching) {
        // 创建缓存键
        int64_t key = (int64_t)(point.x * 100000 + point.y);
        
        // 检查缓存
        auto it = m_gradientCache.find(key);
        if (it != m_gradientCache.end()) {
            gx = it->second.first;
            gy = it->second.second;
            return;
        }
    }
    
    // 访问边界检查
    if (point.x < 0 || point.x >= m_gradX.cols || 
        point.y < 0 || point.y >= m_gradX.rows) {
        gx = gy = 0.0;
        return;
    }
    
    // 获取梯度值
    gx = m_gradX.at<double>(point);
    gy = m_gradY.at<double>(point);
    
    // 缓存结果
    if (m_params.useCaching) {
        int64_t key = (int64_t)(point.x * 100000 + point.y);
        m_gradientCache[key] = std::make_pair(gx, gy);
    }
}

// 使用稀疏矩阵的全局系统构建（更高效）
void PyramidGlobalDIC::buildGlobalSystemSparse(const cv::Mat& refImage, const cv::Mat& defImage,
                                     const std::vector<cv::Point>& nodePoints,
                                     Eigen::SparseMatrix<double>& systemMatrix,
                                     Eigen::VectorXd& systemVector,
                                     const cv::Mat& nodeDisplacements) {
    int numNodes = static_cast<int>(nodePoints.size());
    int numDOFs = numNodes * 2;
    
    // 确保图像是浮点类型
    cv::Mat refFloat, defFloat;
    if (refImage.type() != CV_64F) {
        refImage.convertTo(refFloat, CV_64F);
    } else {
        refFloat = refImage;
    }
    
    if (defImage.type() != CV_64F) {
        defImage.convertTo(defFloat, CV_64F);
    } else {
        defFloat = defImage;
    }
    
    // 计算图像梯度（如果尚未计算）
    if (!m_gradientsComputed) {
        precomputeGradients(refFloat);
    }
    
    // 初始化矩阵构建器
    std::vector<Eigen::Triplet<double>> coefficients;
    coefficients.reserve(numDOFs * numDOFs); // 预留空间
    
    // 重置系统向量
    systemVector.setZero();
    
    // 使用OpenMP进行并行化
    std::vector<std::vector<Eigen::Triplet<double>>> localCoefficients(m_params.numThreads);
    std::vector<Eigen::VectorXd> localVectors(m_params.numThreads, Eigen::VectorXd::Zero(numDOFs));
    
    #pragma omp parallel for if(m_params.useParallel)
    for (int nodeIdx = 0; nodeIdx < numNodes; nodeIdx++) {
        cv::Point nodePoint = nodePoints[nodeIdx];
        int threadId = omp_get_thread_num();
        
        // 为每个线程预分配内存
        if (localCoefficients[threadId].capacity() == 0) {
            localCoefficients[threadId].reserve(numDOFs * 20); // 每个节点约20个非零元素
        }
        
        // 遍历该节点周围子集中的每个像素
        for (int y = -m_params.subsetRadius; y <= m_params.subsetRadius; y++) {
            for (int x = -m_params.subsetRadius; x <= m_params.subsetRadius; x++) {
                cv::Point pixelPoint = nodePoint + cv::Point(x, y);
                
                // 检查像素是否在图像边界内
                if (pixelPoint.x < 0 || pixelPoint.x >= refFloat.cols ||
                    pixelPoint.y < 0 || pixelPoint.y >= refFloat.rows) {
                    continue;
                }
                
                // 获取参考强度和梯度
                double refIntensity = refFloat.at<double>(pixelPoint);
                double gx, gy;
                getGradientValue(pixelPoint, gx, gy);
                
                // 基于当前位移估计计算变形点
                double u = 0.0, v = 0.0;
                interpolateDisplacement(pixelPoint, nodePoints, nodeDisplacements, u, v);
                
                cv::Point2f warpedPoint(pixelPoint.x + u, pixelPoint.y + v);
                
                // 检查变形点是否在图像边界内
                if (warpedPoint.x < 0 || warpedPoint.x >= defFloat.cols - 1 ||
                    warpedPoint.y < 0 || warpedPoint.y >= defFloat.rows - 1) {
                    continue;
                }
                
                // 在变形图像中插值
                double defIntensity = fastInterpolate(defFloat, warpedPoint);
                
                // 计算残差 r = I_def(x+u,y+v) - I_ref(x,y)
                double residual = defIntensity - refIntensity;
                
                // 计算形函数
                std::vector<double> N, dNdx, dNdy;
                computeShapeFunctions(pixelPoint, nodePoints, N, dNdx, dNdy);
                
                // 对附近节点的每个自由度...
                for (int i = 0; i < numNodes; i++) {
                    // drdp 是关于每个参数的残差梯度
                    double drdpu = gx * N[i];  // dr/du
                    double drdpv = gy * N[i];  // dr/dv
                    
                    // 计算索引
                    int uIdx = i * 2;        // u DOF index
                    int vIdx = i * 2 + 1;    // v DOF index
                    
                    // 向系统向量添加此像素的贡献
                    localVectors[threadId](uIdx) -= drdpu * residual;
                    localVectors[threadId](vIdx) -= drdpv * residual;
                    
                    // 向系统矩阵添加此像素的贡献（仅非零元素）
                    for (int j = 0; j < numNodes; j++) {
                        if (std::abs(N[j]) < 1e-10) continue; // 跳过接近零的贡献
                        
                        double drdpu_j = gx * N[j];
                        double drdpv_j = gy * N[j];
                        
                        int uIdx_j = j * 2;
                        int vIdx_j = j * 2 + 1;
                        
                        // 仅添加显著的贡献
                        if (std::abs(drdpu * drdpu_j) > 1e-10) {
                            localCoefficients[threadId].push_back(Eigen::Triplet<double>(uIdx, uIdx_j, drdpu * drdpu_j));
                        }
                        if (std::abs(drdpu * drdpv_j) > 1e-10) {
                            localCoefficients[threadId].push_back(Eigen::Triplet<double>(uIdx, vIdx_j, drdpu * drdpv_j));
                        }
                        if (std::abs(drdpv * drdpu_j) > 1e-10) {
                            localCoefficients[threadId].push_back(Eigen::Triplet<double>(vIdx, uIdx_j, drdpv * drdpu_j));
                        }
                        if (std::abs(drdpv * drdpv_j) > 1e-10) {
                            localCoefficients[threadId].push_back(Eigen::Triplet<double>(vIdx, vIdx_j, drdpv * drdpv_j));
                        }
                    }
                }
            }
        }
    }
    
    // 合并线程局部结果
    for (int t = 0; t < m_params.numThreads; t++) {
        systemVector += localVectors[t];
        coefficients.insert(coefficients.end(), localCoefficients[t].begin(), localCoefficients[t].end());
    }
    
    // 构建稀疏矩阵
    systemMatrix.setFromTriplets(coefficients.begin(), coefficients.end());
    
    // 进行矩阵压缩以提高求解速度
    systemMatrix.makeCompressed();
}

// 向稀疏矩阵添加正则化
void PyramidGlobalDIC::addRegularizationSparse(const std::vector<cv::Point>& nodePoints,
                                     Eigen::SparseMatrix<double>& systemMatrix) {
    int numNodes = static_cast<int>(nodePoints.size());
    int numDOFs = numNodes * 2;
    
    // 使用稀疏矩阵进行正则化
    std::vector<Eigen::Triplet<double>> regularizationCoeffs;
    regularizationCoeffs.reserve(numDOFs);
    
    // 简单的Tikhonov正则化
    if (m_params.regType == TIKHONOV) {
        for (int i = 0; i < numDOFs; i++) {
            regularizationCoeffs.push_back(Eigen::Triplet<double>(i, i, m_params.regularizationWeight));
        }
    }
    // 扩散正则化 - 基于位置相近的节点添加联系
    else if (m_params.regType == DIFFUSION) {
        for (int i = 0; i < numNodes; i++) {
            cv::Point p1 = nodePoints[i];
            int i_u = i * 2;
            int i_v = i * 2 + 1;
            
            // 首先添加对角元素
            regularizationCoeffs.push_back(Eigen::Triplet<double>(i_u, i_u, m_params.regularizationWeight));
            regularizationCoeffs.push_back(Eigen::Triplet<double>(i_v, i_v, m_params.regularizationWeight));
            
            // 然后找邻近节点并添加
            for (int j = 0; j < numNodes; j++) {
                if (i == j) continue;
                
                cv::Point p2 = nodePoints[j];
                
                // 计算距离
                double dist = cv::norm(p1 - p2);
                
                // 只考虑近邻节点
                if (dist > 0 && dist < m_params.nodeSpacing * 2.5) {
                    int j_u = j * 2;
                    int j_v = j * 2 + 1;
                    
                    double weight = m_params.regularizationWeight * (1.0 / dist);
                    
                    // 添加非对角元素
                    regularizationCoeffs.push_back(Eigen::Triplet<double>(i_u, j_u, -weight));
                    regularizationCoeffs.push_back(Eigen::Triplet<double>(i_v, j_v, -weight));
                }
            }
        }
    }
    // 总变分正则化 - 此处简化实现
    else if (m_params.regType == TOTAL_VARIATION) {
        double epsilon = 1e-4;  // 小常数，避免零除
        
        for (int i = 0; i < numNodes; i++) {
            cv::Point p1 = nodePoints[i];
            int i_u = i * 2;
            int i_v = i * 2 + 1;
            
            // 首先添加对角元素
            regularizationCoeffs.push_back(Eigen::Triplet<double>(i_u, i_u, m_params.regularizationWeight));
            regularizationCoeffs.push_back(Eigen::Triplet<double>(i_v, i_v, m_params.regularizationWeight));
            
            for (int j = 0; j < numNodes; j++) {
                if (i == j) continue;
                
                cv::Point p2 = nodePoints[j];
                
                // 计算距离
                double dist = cv::norm(p1 - p2);
                
                // 只考虑近邻节点
                if (dist > 0 && dist < m_params.nodeSpacing * 2.5) {
                    int j_u = j * 2;
                    int j_v = j * 2 + 1;
                    
                    double weight = m_params.regularizationWeight / (dist * std::sqrt(epsilon + dist * dist));
                    
                    // 添加非对角元素
                    regularizationCoeffs.push_back(Eigen::Triplet<double>(i_u, j_u, -weight));
                    regularizationCoeffs.push_back(Eigen::Triplet<double>(i_v, j_v, -weight));
                }
            }
        }
    }
    
    // 将正则化项添加到系统矩阵
    Eigen::SparseMatrix<double> regularizationMatrix(numDOFs, numDOFs);
    regularizationMatrix.setFromTriplets(regularizationCoeffs.begin(), regularizationCoeffs.end());
    
    // 添加到原系统矩阵
    systemMatrix += regularizationMatrix;
}

// 求解稀疏系统
bool PyramidGlobalDIC::solveSystemSparse(const Eigen::SparseMatrix<double>& systemMatrix,
                                const Eigen::VectorXd& systemVector,
                                Eigen::VectorXd& solution,
                                double& residualNorm) {
    // 初始化解向量
    solution.resize(systemVector.size());
    
    // 使用共轭梯度法求解器 - 比直接SVD更快
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> solver;
    solver.setMaxIterations(1000);
    solver.setTolerance(1e-6);
    
    // 配置解算器
    solver.compute(systemMatrix);
    
    if (solver.info() != Eigen::Success) {
        std::cerr << "矩阵分解失败!" << std::endl;
        residualNorm = std::numeric_limits<double>::max();
        return false;
    }
    
    // 求解线性系统
    solution = solver.solve(systemVector);
    
    if (solver.info() != Eigen::Success) {
        std::cerr << "求解线性系统失败!" << std::endl;
        residualNorm = std::numeric_limits<double>::max();
        return false;
    }
    
    // 计算残差范数
    Eigen::VectorXd residual = systemMatrix * solution - systemVector;
    residualNorm = residual.norm();
    
    return true;
}

// 使用OpenMP并行优化的位移插值函数
void PyramidGlobalDIC::interpolateDisplacement(const cv::Point& point,
                                           const std::vector<cv::Point>& nodePoints,
                                           const cv::Mat& nodeDisplacements,
                                           double& u, double& v) {
    // 缓存键
    size_t cacheKey = 0;
    if (m_params.useCaching) {
        cacheKey = std::hash<int>{}(point.x) ^ (std::hash<int>{}(point.y) << 1);
        
        // 检查位移缓存
        auto it = m_displacementCache.find(cacheKey);
        if (it != m_displacementCache.end()) {
            u = it->second.first;
            v = it->second.second;
            return;
        }
    }
    
    // 初始化位移
    u = 0.0;
    v = 0.0;
    
    // 计算形函数
    std::vector<double> N, dNdx, dNdy;
    computeShapeFunctions(point, nodePoints, N, dNdx, dNdy);
    
    // 使用形函数插值位移
    for (size_t i = 0; i < nodePoints.size(); i++) {
        int uIdx = i * 2;
        int vIdx = i * 2 + 1;
        
        u += N[i] * nodeDisplacements.at<double>(uIdx);
        v += N[i] * nodeDisplacements.at<double>(vIdx);
    }
    
    // 缓存结果
    if (m_params.useCaching) {
        m_displacementCache[cacheKey] = std::make_pair(u, v);
    }
}

// 优化的形函数计算
void PyramidGlobalDIC::computeShapeFunctions(const cv::Point& point, 
                                       const std::vector<cv::Point>& nodePoints,
                                       std::vector<double>& N,
                                       std::vector<double>& dNdx,
                                       std::vector<double>& dNdy) {
    size_t numNodes = nodePoints.size();
    
    // 使用缓存来避免重复计算
    if (m_params.useCaching) {
        // 创建缓存键 - 一个简单的哈希
        size_t cacheKey = std::hash<int>{}(point.x) ^ (std::hash<int>{}(point.y) << 1);
        
        // 检查缓存中是否已存在
        auto it = m_shapeFunctionCache.find(cacheKey);
        if (it != m_shapeFunctionCache.end()) {
            N = std::get<0>(it->second);
            dNdx = std::get<1>(it->second);
            dNdy = std::get<2>(it->second);
            return;
        }
    }
    
    // 准备输出向量
    N.resize(numNodes, 0.0);
    dNdx.resize(numNodes, 0.0);
    dNdy.resize(numNodes, 0.0);
    
    // 寻找最近的节点（K最近邻）
    const int K = 8; // 使用最近的8个节点
    std::vector<size_t> nearestIndices;
    std::vector<double> distances;
    
    // 查找K最近邻
    findKNearestNodes(point, nodePoints, K, nearestIndices, distances);
    
    double totalWeight = 0.0;
    
    // 仅对最近的K个节点计算权重（而非所有节点）
    for (size_t i = 0; i < nearestIndices.size(); i++) {
        size_t idx = nearestIndices[i];
        double distance = distances[i];
        
        // 避免除以零
        if (distance < 1e-10) {
            // 点位于节点上，完全使用该节点
            std::fill(N.begin(), N.end(), 0.0);
            N[idx] = 1.0;
            totalWeight = 1.0;
            break;
        }
        
        // 使用反距离加权
        double weight = 1.0 / (distance * distance);
        N[idx] = weight;
        totalWeight += weight;
    }
    
    // 归一化权重
    for (size_t i = 0; i < nearestIndices.size(); i++) {
        size_t idx = nearestIndices[i];
        double distance = distances[i];
        
        if (totalWeight > 0) {
            N[idx] /= totalWeight;
        }
        
        // 计算形函数导数 (使用有限差分近似)
        if (distance > 1e-10) {
            double dx = point.x - nodePoints[idx].x;
            double dy = point.y - nodePoints[idx].y;
            double distance3 = distance * distance * distance;
            
            dNdx[idx] = -2.0 * dx / (distance3 * totalWeight);
            dNdy[idx] = -2.0 * dy / (distance3 * totalWeight);
        }
    }
    
    // 缓存计算结果
    if (m_params.useCaching) {
        size_t cacheKey = std::hash<int>{}(point.x) ^ (std::hash<int>{}(point.y) << 1);
        m_shapeFunctionCache[cacheKey] = std::make_tuple(N, dNdx, dNdy);
    }
}

// 查找K个最近的节点
void PyramidGlobalDIC::findKNearestNodes(const cv::Point& point, 
                                   const std::vector<cv::Point>& nodePoints,
                                   int K,
                                   std::vector<size_t>& indices,
                                   std::vector<double>& distances) {
    // 确保K不超过节点总数
    K = std::min(K, static_cast<int>(nodePoints.size()));
    
    // 准备结果
    indices.clear();
    distances.clear();
    indices.reserve(K);
    distances.reserve(K);
    
    // 创建一个按距离排序的队列
    std::priority_queue<std::pair<double, size_t>> pq;
    
    // 计算到所有节点的距离并保留K个最小值
    for (size_t i = 0; i < nodePoints.size(); i++) {
        double dx = point.x - nodePoints[i].x;
        double dy = point.y - nodePoints[i].y;
        double distance = std::sqrt(dx*dx + dy*dy);
        
        if (pq.size() < K) {
            pq.push(std::make_pair(distance, i));
        } else if (distance < pq.top().first) {
            pq.pop();
            pq.push(std::make_pair(distance, i));
        }
    }
    
    // 提取最近的K个节点
    while (!pq.empty()) {
        indices.push_back(pq.top().second);
        distances.push_back(pq.top().first);
        pq.pop();
    }
    
    // 按距离升序排序
    std::vector<size_t> sortedIndices(indices.size());
    std::vector<double> sortedDistances(distances.size());
    for (size_t i = 0; i < indices.size(); i++) {
        sortedIndices[indices.size() - 1 - i] = indices[i];
        sortedDistances[distances.size() - 1 - i] = distances[i];
    }
    
    indices = sortedIndices;
    distances = sortedDistances;
}

// 使用FFT加速的多尺度全局DIC实现
PyramidGlobalDIC::Result PyramidGlobalDIC::computeMultiScale(const cv::Mat& refImage, 
    const cv::Mat& defImage,
    const cv::Mat& roi) {
        // 准备结果结构体
        Result finalResult(roi.size());
        finalResult.validMask = roi.clone();

        // 创建金字塔
        std::vector<Layer> refPyramid, defPyramid;

        // 转换图像为CV_64F（如果需要）
        cv::Mat refFloat, defFloat;
        refImage.convertTo(refFloat, CV_64F);
        defImage.convertTo(defFloat, CV_64F);

        // 创建图像金字塔并选择合适的层数
        createImagePyramid(refFloat, roi, refPyramid);
        createImagePyramid(defFloat, roi, defPyramid);

        // 从最粗糙级别（金字塔顶部）开始处理
        int totalIterations = 0;
        cv::Mat prevLevelU, prevLevelV;

        for (int level = refPyramid.size() - 1; level >= 0; level--) {
        std::cout << "处理金字塔级别 " << level 
        << " (scale=" << refPyramid[level].scale << ")" << std::endl;

        // 重置缓存，避免跨层次的缓存冲突
        if (m_params.useCaching) {
        m_interpolationCache.clear();
        m_gradientCache.clear();
        m_displacementCache.clear();
        m_shapeFunctionCache.clear();
        }

        // 当前级别的图像和掩码
        cv::Mat currentRefImg = refPyramid[level].img;
        cv::Mat currentDefImg = defPyramid[level].img;
        cv::Mat currentMask = refPyramid[level].mask;

        // 当前级别的结果
        Result levelResult(currentMask.size());
        levelResult.validMask = currentMask.clone();

        // 创建节点网格 - 对于粗粒度层次，可以使用更大的节点间距
        std::vector<cv::Point> nodePoints;
        // 根据金字塔级别调整节点间距
        int levelNodeSpacing = std::max(4, m_params.nodeSpacing / (1 << (refPyramid.size() - 1 - level)));

        // 创建节点网格
        for (int y = m_params.subsetRadius; y < currentMask.rows - m_params.subsetRadius; y += levelNodeSpacing) {
        for (int x = m_params.subsetRadius; x < currentMask.cols - m_params.subsetRadius; x += levelNodeSpacing) {
        if (currentMask.at<uchar>(y, x) > 0) {
        nodePoints.push_back(cv::Point(x, y));
        }
        }
        }

        // 节点数量和自由度
        int numNodes = static_cast<int>(nodePoints.size());
        int numDOFs = numNodes * 2;

        std::cout << "级别 " << level << " 的节点数: " << numNodes << std::endl;

        // 初始化当前级别的节点位移
        cv::Mat nodeDisplacements = cv::Mat::zeros(numDOFs, 1, CV_64F);

        // 如果是最粗糙级别，使用FFTCC获取初始猜测
        if (level == refPyramid.size() - 1 && m_params.initialGuessMethod == FFTCC) {
        // 使用FFTCC计算初始位移场
        cv::Mat initialU, initialV;
        calculateInitialGuessFFTCC(currentRefImg, currentDefImg, currentMask, initialU, initialV);

        // 将位移场应用到节点
        for (int i = 0; i < numNodes; i++) {
        cv::Point node = nodePoints[i];
        nodeDisplacements.at<double>(i * 2) = initialU.at<double>(node);
        nodeDisplacements.at<double>(i * 2 + 1) = initialV.at<double>(node);
        }
        }
        // 如果不是最粗糙级别，则使用上一级别的结果作为初始猜测
        else if (level < refPyramid.size() - 1 && !prevLevelU.empty() && !prevLevelV.empty()) {
        // 将上一级别的位移映射到当前级别
        double scaleRatio = refPyramid[level].scale / refPyramid[level+1].scale;

        // 在当前级别创建上一级别位移的插值版本
        cv::Mat scaledU, scaledV;
        cv::resize(prevLevelU, scaledU, currentRefImg.size());
        cv::resize(prevLevelV, scaledV, currentRefImg.size());

        // 调整位移值以匹配当前尺度
        scaledU *= scaleRatio;
        scaledV *= scaleRatio;

        // 将位移应用到节点
        for (int i = 0; i < numNodes; i++) {
        cv::Point node = nodePoints[i];
        nodeDisplacements.at<double>(i * 2) = scaledU.at<double>(node);
        nodeDisplacements.at<double>(i * 2 + 1) = scaledV.at<double>(node);
        }
        }

        // 使用稀疏矩阵求解当前级别
        if (m_params.useSparseMatrix && m_params.useEigenSolver) {
        // 迭代求解当前级别
        double prevResidual = std::numeric_limits<double>::max();
        int iter = 0;
        double residualNorm = 0.0;

        // 重置梯度缓存
        m_gradientsComputed = false;
        precomputeGradients(currentRefImg);

        for (iter = 0; iter < m_params.maxIterations; iter++) {
        // 构建全局系统矩阵和向量
        Eigen::SparseMatrix<double> systemMatrix(numDOFs, numDOFs);
        Eigen::VectorXd systemVector(numDOFs);
        systemVector.setZero();

        buildGlobalSystemSparse(currentRefImg, currentDefImg, nodePoints, 
        systemMatrix, systemVector, nodeDisplacements);

        // 添加正则化
        addRegularizationSparse(nodePoints, systemMatrix);

        // 求解系统
        Eigen::VectorXd deltaDisplacements;
        bool solveSuccess = solveSystemSparse(systemMatrix, systemVector, 
        deltaDisplacements, residualNorm);

        if (!solveSuccess) {
        std::cout << "级别 " << level 
        << " 迭代 " << iter 
        << " 无法求解系统" << std::endl;
        break;
        }

        // 更新位移
        for (int i = 0; i < numDOFs; i++) {
        nodeDisplacements.at<double>(i, 0) += deltaDisplacements(i);
        }

        // 监控收敛状况
        std::cout << "级别 " << level 
        << " 迭代 " << iter 
        << " 残差: " << residualNorm 
        << std::endl;

        // 检查收敛条件
        if (residualNorm < m_params.convergenceThreshold || 
        std::abs(residualNorm - prevResidual) < m_params.convergenceThreshold / 10) {
        std::cout << "级别 " << level 
        << " 迭代 " << iter 
        << " 达到收敛" << std::endl;
        break;
        }

        prevResidual = residualNorm;
        }

        totalIterations += iter;
        } else {
        // 使用OpenCV的稠密矩阵求解器 (略)...
        }

        // 从节点位移生成完整位移场
        generateDisplacementField(nodePoints, nodeDisplacements, levelResult, currentMask);

        // 存储当前级别的位移场用于下一级别
        prevLevelU = levelResult.u.clone();
        prevLevelV = levelResult.v.clone();

        // 如果是最精细级别，则存储为最终结果
        if (level == 0) {
        finalResult = levelResult;
        finalResult.iterations = totalIterations;

        // 计算置信度和残差
        calculateConfidence(finalResult, refImage, defImage);
        finalResult.meanResidual = calculateResidual(refImage, defImage, nodePoints, nodeDisplacements);

        // 计算应变场
        calculateStrains(finalResult);
        }
        }

        return finalResult;
}

void PyramidGlobalDIC::calculateConfidence(Result& result, 
  const cv::Mat& refImage, 
  const cv::Mat& defImage) {
// 将图像转换为CV_64F
cv::Mat refFloat, defFloat;
refImage.convertTo(refFloat, CV_64F);
defImage.convertTo(defFloat, CV_64F);

// 在每个像素计算局部相关系数
#pragma omp parallel for if(m_params.useParallel)
for (int y = m_params.subsetRadius; y < result.u.rows - m_params.subsetRadius; y++) {
for (int x = m_params.subsetRadius; x < result.u.cols - m_params.subsetRadius; x++) {
if (result.validMask.at<uchar>(y, x)) {
double u = result.u.at<double>(y, x);
double v = result.v.at<double>(y, x);

// 局部区域计算ZNCC
double sumRefIntensity = 0.0;
double sumDefIntensity = 0.0;
double sumRefSquared = 0.0;
double sumDefSquared = 0.0;
double sumProduct = 0.0;
int validPixelCount = 0;

for (int dy = -m_params.subsetRadius/2; dy <= m_params.subsetRadius/2; dy++) {
for (int dx = -m_params.subsetRadius/2; dx <= m_params.subsetRadius/2; dx++) {
cv::Point refPoint(x + dx, y + dy);
cv::Point2f defPoint(x + dx + u, y + dy + v);

// 检查边界
if (defPoint.x >= 0 && defPoint.x < defFloat.cols - 1 &&
defPoint.y >= 0 && defPoint.y < defFloat.rows - 1) {

double refIntensity = refFloat.at<double>(refPoint);
double defIntensity = interpolate(defFloat, defPoint);

sumRefIntensity += refIntensity;
sumDefIntensity += defIntensity;
sumRefSquared += refIntensity * refIntensity;
sumDefSquared += defIntensity * defIntensity;
sumProduct += refIntensity * defIntensity;
validPixelCount++;
}
}
}

if (validPixelCount > 0) {
double meanRef = sumRefIntensity / validPixelCount;
double meanDef = sumDefIntensity / validPixelCount;

double numerator = sumProduct - meanRef * sumDefIntensity - meanDef * sumRefIntensity + validPixelCount * meanRef * meanDef;
double denominator = std::sqrt((sumRefSquared - validPixelCount * meanRef * meanRef) * 
               (sumDefSquared - validPixelCount * meanDef * meanDef));

if (denominator > 1e-10) {
double zncc = numerator / denominator;
result.cc.at<double>(y, x) = zncc;
// 将ZNCC映射到[0,1]的置信度，其中1表示完美匹配
result.confidence.at<double>(y, x) = (zncc + 1.0) / 2.0;
}
}
}
}
}
}

double PyramidGlobalDIC::calculateResidual(const cv::Mat& refImage, 
  const cv::Mat& defImage,
  const std::vector<cv::Point>& nodePoints,
  const cv::Mat& nodeDisplacements) {
// 将图像转换为CV_64F
cv::Mat refFloat, defFloat;
refImage.convertTo(refFloat, CV_64F);
defImage.convertTo(defFloat, CV_64F);

double totalResidual = 0.0;
int validPixelCount = 0;

// 遍历图像中的所有像素
#pragma omp parallel for reduction(+:totalResidual,validPixelCount) if(m_params.useParallel)
for (int y = 0; y < refFloat.rows; y++) {
for (int x = 0; x < refFloat.cols; x++) {
cv::Point currentPoint(x, y);

// 计算变形点位置
double u, v;
interpolateDisplacement(currentPoint, nodePoints, nodeDisplacements, u, v);

cv::Point2f warpedPoint(x + u, y + v);

// 检查变形点是否在图像边界内
if (warpedPoint.x < 0 || warpedPoint.x >= defFloat.cols - 1 ||
warpedPoint.y < 0 || warpedPoint.y >= defFloat.rows - 1) {
continue;
}

double refIntensity = refFloat.at<double>(y, x);
double defIntensity = interpolate(defFloat, warpedPoint);

double residual = defIntensity - refIntensity;
totalResidual += residual * residual;
validPixelCount++;
}
}

return (validPixelCount > 0) ? std::sqrt(totalResidual / validPixelCount) : 0.0;
}// 计算应变场


std::unique_ptr<PyramidGlobalDIC::FFTW> &PyramidGlobalDIC::getFFTWInstance(int threadId)
{
    // 确保线程ID在有效范围内
    threadId = std::max(0, std::min(threadId, static_cast<int>(m_fftwPool.size()) - 1));
    
    // 如果该线程ID对应的FFTW实例还未创建，则创建它
    if (!m_fftwPool[threadId] || !m_fftwPool[threadId]->is_initialized) {
        // 使用互斥锁确保线程安全
        static std::mutex fftw_mutex;
        std::lock_guard<std::mutex> lock(fftw_mutex);
        
        // 双重检查锁定模式，避免其他线程可能已经创建了该实例
        if (!m_fftwPool[threadId] || !m_fftwPool[threadId]->is_initialized) {
            // 创建新实例
            auto instance = FFTW::allocate(m_params.subsetRadius, m_params.subsetRadius);
            
            // 检查实例是否创建成功
            if (!instance) {
                std::cerr << "错误: 无法为线程 " << threadId 
                          << " 创建FFTW实例. 子集大小: " 
                          << m_params.subsetRadius * 2 << "x" 
                          << m_params.subsetRadius * 2 << std::endl;
                
                // 创建一个应急实例，尝试使用较小的子集大小
                int reducedRadius = std::max(4, m_params.subsetRadius / 2);
                std::cout << "尝试使用较小的子集大小: " 
                          << reducedRadius * 2 << "x" 
                          << reducedRadius * 2 << std::endl;
                
                instance = FFTW::allocate(reducedRadius, reducedRadius);
                
                // 如果还是失败，使用最小可能的尺寸
                if (!instance) {
                    std::cerr << "错误: 备用FFTW实例创建也失败了." << std::endl;
                    // 创建一个空实例，后续的使用会检查is_initialized标志
                    instance = std::make_unique<FFTW>();
                }
            }
            
            // 替换可能存在的旧实例
            if (m_fftwPool[threadId]) {
                FFTW::release(m_fftwPool[threadId]);
            }
            m_fftwPool[threadId] = std::move(instance);
        }
    }
    
    return m_fftwPool[threadId];
}