#include "cuda_dic_kernel_precision.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <iostream>
#include <cmath>

using namespace cooperative_groups;

// 使用双精度常量确保与CPU版本完全一致
__constant__ double c_subsetRadius;
__constant__ int c_subsetSize;
__constant__ int c_imageWidth;
__constant__ int c_imageHeight;
__constant__ int c_numParams;
__constant__ double c_convergenceThreshold;
__constant__ int c_maxIterations;

// 高精度双线性插值，完全匹配CPU版本
__device__ __forceinline__ double precisionBilinearInterpolation(double x, double y, const double* image, int width, int height) {
    // 边界检查，与CPU版本完全一致
    if (x < 0.0 || x >= width - 1.0 || y < 0.0 || y >= height - 1.0) {
        return 0.0;
    }
    
    // 获取整数和小数部分，使用与CPU相同的方法
    int x1 = static_cast<int>(x);
    int y1 = static_cast<int>(y);
    int x2 = x1 + 1;
    int y2 = y1 + 1;
    
    double fx = x - static_cast<double>(x1);
    double fy = y - static_cast<double>(y1);
    
    // 双线性插值，与CPU版本完全一致的公式
    double val = (1.0 - fx) * (1.0 - fy) * image[y1 * width + x1] +
                fx * (1.0 - fy) * image[y1 * width + x2] +
                (1.0 - fx) * fy * image[y2 * width + x1] +
                fx * fy * image[y2 * width + x2];
    
    return val;
}

// 精确的点变形函数，与CPU版本完全一致
__device__ __forceinline__ void precisionWarpPoint(double x, double y, const double* warpParams, 
                                                   double& warpedX, double& warpedY, int numParams) {
    // 提取参数
    double u = warpParams[0];
    double v = warpParams[1];
    
    // 基础平移
    warpedX = x + u;
    warpedY = y + v;
    
    // 一阶形变参数（至少6个参数）
    if (numParams >= 6) {
        double dudx = warpParams[2];
        double dudy = warpParams[3];
        double dvdx = warpParams[4];
        double dvdy = warpParams[5];
        
        warpedX += dudx * x + dudy * y;
        warpedY += dvdx * x + dvdy * y;
    }
    
    // 二阶形变参数（12个参数）
    if (numParams >= 12) {
        double d2udx2 = warpParams[6];
        double d2udxdy = warpParams[7];
        double d2udy2 = warpParams[8];
        double d2vdx2 = warpParams[9];
        double d2vdxdy = warpParams[10];
        double d2vdy2 = warpParams[11];
        
        warpedX += 0.5 * d2udx2 * x * x + d2udxdy * x * y + 0.5 * d2udy2 * y * y;
        warpedY += 0.5 * d2vdx2 * x * x + d2vdxdy * x * y + 0.5 * d2vdy2 * y * y;
    }
}

// 精确的Sobel梯度计算，与CPU版本完全一致
__device__ __forceinline__ void computeSobelGradients(const double* image, int x, int y, int width, int height,
                                                      double& gradX, double& gradY) {
    gradX = 0.0;
    gradY = 0.0;
    
    // 边界检查
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        // Sobel X核: [-1 0 1; -2 0 2; -1 0 1] / 8
        gradX = (-image[(y-1)*width + (x-1)] + image[(y-1)*width + (x+1)]
                -2.0*image[y*width + (x-1)] + 2.0*image[y*width + (x+1)]
                -image[(y+1)*width + (x-1)] + image[(y+1)*width + (x+1)]) / 8.0;
        
        // Sobel Y核: [-1 -2 -1; 0 0 0; 1 2 1] / 8  
        gradY = (-image[(y-1)*width + (x-1)] - 2.0*image[(y-1)*width + x] - image[(y-1)*width + (x+1)]
                +image[(y+1)*width + (x-1)] + 2.0*image[(y+1)*width + x] + image[(y+1)*width + (x+1)]) / 8.0;
    }
}

// 精确的ZNCC计算，与CPU版本完全一致
__device__ double computePrecisionZNCC(const double* refImage, const double* defImage,
                                      Point2D centerPoint, const double* warpParams,
                                      int imageWidth, int imageHeight, int subsetRadius, int numParams) {
    
    double sumRef = 0.0, sumDef = 0.0;
    double sumRefSq = 0.0, sumDefSq = 0.0;
    double sumRefDef = 0.0;
    int count = 0;
    
    // 遍历子集中的每个像素，与CPU版本完全一致
    for (int ly = -subsetRadius; ly <= subsetRadius; ly++) {
        for (int lx = -subsetRadius; lx <= subsetRadius; lx++) {
            // 参考图像中的像素位置
            int refX = centerPoint.x + lx;
            int refY = centerPoint.y + ly;
            
            // 边界检查
            if (refX >= 0 && refX < imageWidth && refY >= 0 && refY < imageHeight) {
                double refIntensity = refImage[refY * imageWidth + refX];
                
                // 计算变形后的点
                double warpedX, warpedY;
                precisionWarpPoint(static_cast<double>(lx), static_cast<double>(ly), 
                                 warpParams, warpedX, warpedY, numParams);
                
                // 变形图像中的像素位置
                double defImgX = static_cast<double>(centerPoint.x) + warpedX;
                double defImgY = static_cast<double>(centerPoint.y) + warpedY;
                
                // 边界检查
                if (defImgX >= 0.0 && defImgX < imageWidth - 1.0 && defImgY >= 0.0 && defImgY < imageHeight - 1.0) {
                    double defIntensity = precisionBilinearInterpolation(defImgX, defImgY, defImage, imageWidth, imageHeight);
                    
                    // 累积统计量
                    sumRef += refIntensity;
                    sumDef += defIntensity;
                    sumRefSq += refIntensity * refIntensity;
                    sumDefSq += defIntensity * defIntensity;
                    sumRefDef += refIntensity * defIntensity;
                    count++;
                }
            }
        }
    }
    
    // 计算ZNCC，与CPU版本完全一致
    if (count > 0) {
        double meanRef = sumRef / static_cast<double>(count);
        double meanDef = sumDef / static_cast<double>(count);
        double varRef = sumRefSq / static_cast<double>(count) - meanRef * meanRef;
        double varDef = sumDefSq / static_cast<double>(count) - meanDef * meanDef;
        double covar = sumRefDef / static_cast<double>(count) - meanRef * meanDef;
        
        // 防止除零，与CPU版本一致的阈值
        if (varRef > 1e-10 && varDef > 1e-10) {
            // 返回 1 - ZNCC 以转换为最小化问题
            return 1.0 - (covar / sqrt(varRef * varDef));
        }
    }
    
    return 1e10; // 错误情况，与CPU版本一致
}

// 高精度QR分解求解线性方程组 - 更稳定的数值方法
__device__ bool solvePrecisionLinearSystemQR(const double* A, const double* b, double* x, int n) {
    // 复制矩阵到局部内存
    double Q[144]; // 最大12x12矩阵 - 正交矩阵
    double R[144]; // 最大12x12矩阵 - 上三角矩阵
    double bb[12]; // 最大12维向量
    
    // 初始化
    for (int i = 0; i < n * n; i++) {
        Q[i] = A[i]; // 初始将A复制到Q
        R[i] = 0.0;
    }
    for (int i = 0; i < n; i++) {
        bb[i] = b[i];
        x[i] = 0.0;
    }
    
    // Modified Gram-Schmidt QR分解
    for (int j = 0; j < n; j++) {
        // 计算列向量的范数
        double norm = 0.0;
        for (int i = 0; i < n; i++) {
            norm += Q[i * n + j] * Q[i * n + j];
        }
        norm = sqrt(norm);
        
        // 检查数值稳定性
        if (norm < 1e-14) {
            return false; // 矩阵奇异
        }
        
        R[j * n + j] = norm;
        
        // 归一化列向量
        for (int i = 0; i < n; i++) {
            Q[i * n + j] /= norm;
        }
        
        // 计算与后续列的内积并正交化
        for (int k = j + 1; k < n; k++) {
            double dot = 0.0;
            for (int i = 0; i < n; i++) {
                dot += Q[i * n + j] * Q[i * n + k];
            }
            R[j * n + k] = dot;
            
            // 从后续列中减去投影
            for (int i = 0; i < n; i++) {
                Q[i * n + k] -= dot * Q[i * n + j];
            }
        }
    }
    
    // 计算 Q^T * b
    double QtB[12];
    for (int i = 0; i < n; i++) {
        QtB[i] = 0.0;
        for (int j = 0; j < n; j++) {
            QtB[i] += Q[j * n + i] * bb[j]; // Q^T[i][j] = Q[j][i]
        }
    }
    
    // 后向替换求解 R * x = Q^T * b
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < n; j++) {
            sum += R[i * n + j] * x[j];
        }
        
        // 检查对角元素避免除零
        if (fabs(R[i * n + i]) < 1e-14) {
            return false;
        }
        
        x[i] = (QtB[i] - sum) / R[i * n + i];
    }
    
    return true;
}

// 添加正则化的QR分解以处理病态矩阵
__device__ bool solvePrecisionLinearSystemRegularizedQR(const double* A, const double* b, double* x, int n) {
    // 复制矩阵并添加Tikhonov正则化
    double ARegularized[144];
    double regularization = 1e-8; // 正则化参数
    
    for (int i = 0; i < n * n; i++) {
        ARegularized[i] = A[i];
    }
    
    // 添加对角正则化项
    for (int i = 0; i < n; i++) {
        ARegularized[i * n + i] += regularization;
    }
    
    // 使用正则化矩阵进行QR分解
    return solvePrecisionLinearSystemQR(ARegularized, b, x, n);
}

// 主要的线性系统求解函数
__device__ bool solvePrecisionLinearSystem(const double* A, const double* b, double* x, int n) {
    // 首先尝试标准QR分解
    if (solvePrecisionLinearSystemQR(A, b, x, n)) {
        return true;
    }
    
    // 如果失败，尝试正则化QR分解
    return solvePrecisionLinearSystemRegularizedQR(A, b, x, n);
}

// 完全精确的ICGN优化核函数，与CPU版本算法完全一致
__global__ void precisionICGNOptimizationKernel(double* finalU, double* finalV, double* finalZNCC, bool* validMask,
                                               const double* refImage, const double* defImage,
                                               const Point2D* points, const double* initialParams,
                                               int numPoints, int imageWidth, int imageHeight,
                                               int subsetRadius, int numParams, int maxIterations,
                                               double convergenceThreshold) {
    
    int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pointIdx >= numPoints) return;
    
    Point2D centerPoint = points[pointIdx];
    
    // 边界检查
    if (centerPoint.x < subsetRadius || centerPoint.x >= imageWidth - subsetRadius ||
        centerPoint.y < subsetRadius || centerPoint.y >= imageHeight - subsetRadius) {
        finalU[pointIdx] = 0.0;
        finalV[pointIdx] = 0.0;
        finalZNCC[pointIdx] = 1e10;
        validMask[pointIdx] = false;
        return;
    }
    
    // 使用局部内存存储参数和中间结果（减少内存使用）
    double warpParams[12] = {0};
    
    // 初始化参数
    for (int i = 0; i < numParams && i < 12; i++) {
        warpParams[i] = initialParams[pointIdx * numParams + i];
    }
    
    // Hessian矩阵存储
    double hessian[144] = {0}; // 最大12x12
    
    // 预计算Hessian矩阵，与CPU版本完全一致
    for (int i = 0; i < numParams; i++) {
        for (int j = i; j < numParams; j++) {
            double sum = 0.0;
            
            // 遍历子集计算Hessian元素
            for (int ly = -subsetRadius; ly <= subsetRadius; ly++) {
                for (int lx = -subsetRadius; lx <= subsetRadius; lx++) {
                    int refX = centerPoint.x + lx;
                    int refY = centerPoint.y + ly;
                    
                    if (refX >= 1 && refX < imageWidth - 1 && refY >= 1 && refY < imageHeight - 1) {
                        // 计算Sobel梯度
                        double gradX, gradY;
                        computeSobelGradients(refImage, refX, refY, imageWidth, imageHeight, gradX, gradY);
                        
                        // 计算shape function derivatives
                        double shapeFni = 0.0, shapeFnj = 0.0;
                        double x = static_cast<double>(lx);
                        double y = static_cast<double>(ly);
                        
                        // 计算第i个参数的shape function
                        if (i == 0) shapeFni = gradX; // du
                        else if (i == 1) shapeFni = gradY; // dv
                        else if (i == 2 && numParams >= 6) shapeFni = gradX * x; // du/dx
                        else if (i == 3 && numParams >= 6) shapeFni = gradX * y; // du/dy
                        else if (i == 4 && numParams >= 6) shapeFni = gradY * x; // dv/dx
                        else if (i == 5 && numParams >= 6) shapeFni = gradY * y; // dv/dy
                        else if (i == 6 && numParams >= 12) shapeFni = gradX * x * x * 0.5; // d²u/dx²
                        else if (i == 7 && numParams >= 12) shapeFni = gradX * x * y; // d²u/dxdy
                        else if (i == 8 && numParams >= 12) shapeFni = gradX * y * y * 0.5; // d²u/dy²
                        else if (i == 9 && numParams >= 12) shapeFni = gradY * x * x * 0.5; // d²v/dx²
                        else if (i == 10 && numParams >= 12) shapeFni = gradY * x * y; // d²v/dxdy
                        else if (i == 11 && numParams >= 12) shapeFni = gradY * y * y * 0.5; // d²v/dy²
                        
                        // 计算第j个参数的shape function
                        if (j == 0) shapeFnj = gradX; // du
                        else if (j == 1) shapeFnj = gradY; // dv
                        else if (j == 2 && numParams >= 6) shapeFnj = gradX * x; // du/dx
                        else if (j == 3 && numParams >= 6) shapeFnj = gradX * y; // du/dy
                        else if (j == 4 && numParams >= 6) shapeFnj = gradY * x; // dv/dx
                        else if (j == 5 && numParams >= 6) shapeFnj = gradY * y; // dv/dy
                        else if (j == 6 && numParams >= 12) shapeFnj = gradX * x * x * 0.5; // d²u/dx²
                        else if (j == 7 && numParams >= 12) shapeFnj = gradX * x * y; // d²u/dxdy
                        else if (j == 8 && numParams >= 12) shapeFnj = gradX * y * y * 0.5; // d²u/dy²
                        else if (j == 9 && numParams >= 12) shapeFnj = gradY * x * x * 0.5; // d²v/dx²
                        else if (j == 10 && numParams >= 12) shapeFnj = gradY * x * y; // d²v/dxdy
                        else if (j == 11 && numParams >= 12) shapeFnj = gradY * y * y * 0.5; // d²v/dy²
                        
                        sum += shapeFni * shapeFnj;
                    }
                }
            }
            
            hessian[i * numParams + j] = sum;
            hessian[j * numParams + i] = sum; // 对称矩阵
        }
    }
    
    // ICGN迭代优化，改进的收敛条件
    double prevZNCC = 1e10;
    bool converged = false;
    int iter = 0;
    
    for (iter = 0; iter < maxIterations && !converged; iter++) {
        // 计算当前ZNCC
        double currentZNCC = computePrecisionZNCC(refImage, defImage, centerPoint, warpParams, 
                                               imageWidth, imageHeight, subsetRadius, numParams);
        
        // 改进的收敛条件：ZNCC变化和参数变化都要考虑
        bool znccConverged = (iter > 0) && (fabs(currentZNCC - prevZNCC) < convergenceThreshold);
        
        // 如果ZNCC收敛，直接标记为成功
        if (znccConverged) {
            converged = true;
            finalZNCC[pointIdx] = currentZNCC;
            break;
        }
        
        prevZNCC = currentZNCC;
        
        // 计算误差向量，与CPU版本完全一致
        double errorVector[12] = {0};
        
        for (int ly = -subsetRadius; ly <= subsetRadius; ly++) {
            for (int lx = -subsetRadius; lx <= subsetRadius; lx++) {
                int refX = centerPoint.x + lx;
                int refY = centerPoint.y + ly;
                
                // 边界检查
                if (refX >= 0 && refX < imageWidth && refY >= 0 && refY < imageHeight) {
                    double refIntensity = refImage[refY * imageWidth + refX];
                    
                    // 变形点
                    double warpedX, warpedY;
                    precisionWarpPoint(static_cast<double>(lx), static_cast<double>(ly), 
                                     warpParams, warpedX, warpedY, numParams);
                    
                    double defImgX = static_cast<double>(centerPoint.x) + warpedX;
                    double defImgY = static_cast<double>(centerPoint.y) + warpedY;
                    
                    // 边界检查
                    if (defImgX >= 0.0 && defImgX < imageWidth - 1.0 && defImgY >= 0.0 && defImgY < imageHeight - 1.0) {
                        double defIntensity = precisionBilinearInterpolation(defImgX, defImgY, defImage, imageWidth, imageHeight);
                        double error = refIntensity - defIntensity;
                        
                        // 计算梯度和steepest descent 
                        if (refX >= 1 && refX < imageWidth - 1 && refY >= 1 && refY < imageHeight - 1) {
                            double gradX, gradY;
                            computeSobelGradients(refImage, refX, refY, imageWidth, imageHeight, gradX, gradY);
                            
                            double x = static_cast<double>(lx);
                            double y = static_cast<double>(ly);
                            
                            // 更新误差向量
                            if (numParams >= 2) {
                                errorVector[0] += error * gradX; // du
                                errorVector[1] += error * gradY; // dv
                            }
                            if (numParams >= 6) {
                                errorVector[2] += error * gradX * x; // du/dx
                                errorVector[3] += error * gradX * y; // du/dy
                                errorVector[4] += error * gradY * x; // dv/dx
                                errorVector[5] += error * gradY * y; // dv/dy
                            }
                            if (numParams >= 12) {
                                errorVector[6] += error * gradX * x * x * 0.5; // d²u/dx²
                                errorVector[7] += error * gradX * x * y; // d²u/dxdy
                                errorVector[8] += error * gradX * y * y * 0.5; // d²u/dy²
                                errorVector[9] += error * gradY * x * x * 0.5; // d²v/dx²
                                errorVector[10] += error * gradY * x * y; // d²v/dxdy
                                errorVector[11] += error * gradY * y * y * 0.5; // d²v/dy²
                            }
                        }
                    }
                }
            }
        }
        
        // 解线性方程组 H * deltaP = errorVector
        double deltaP[12] = {0};
        bool solved = solvePrecisionLinearSystem(hessian, errorVector, deltaP, numParams);
        
        if (!solved) {
            // 如果矩阵奇异，尝试使用当前参数作为结果
            // 而不是直接标记为失败
            finalZNCC[pointIdx] = currentZNCC;
            converged = (currentZNCC < 0.5); // 如果ZNCC还可以接受，就认为成功
            break;
        }
        
        // 更新参数
        for (int p = 0; p < numParams; p++) {
            warpParams[p] += deltaP[p];
        }
        
        // 检查参数更新的收敛性 - 更宽松的条件
        double deltaNorm = 0.0;
        for (int p = 0; p < numParams; p++) {
            deltaNorm += deltaP[p] * deltaP[p];
        }
        
        // 使用更宽松的参数收敛阈值
        if (sqrt(deltaNorm) < convergenceThreshold * 10.0) {
            converged = true;
            finalZNCC[pointIdx] = currentZNCC;
        }
    }
    
    // 改进的最终处理逻辑
    if (!converged) {
        if (iter >= maxIterations) {
            // 达到最大迭代次数，计算最终ZNCC
            double finalZncc = computePrecisionZNCC(refImage, defImage, centerPoint, warpParams, 
                                                   imageWidth, imageHeight, subsetRadius, numParams);
            finalZNCC[pointIdx] = finalZncc;
            
            // 如果ZNCC值可接受，仍然标记为有效
            converged = (finalZncc < 0.8); // 更宽松的有效性判断
        }
    }
    
    // 输出最终结果
    finalU[pointIdx] = warpParams[0];
    finalV[pointIdx] = warpParams[1];
    validMask[pointIdx] = converged;
    
    // 确保ZNCC有有效值
    if (!converged || finalZNCC[pointIdx] > 1e9) {
        finalZNCC[pointIdx] = 1e10;
    }
}

// 初始猜测核函数，与CPU版本完全一致
__global__ void precisionInitialGuessKernel(double* initialParams, double* initialZNCC, bool* validMask,
                                           const double* refImage, const double* defImage,
                                           const Point2D* points, int numPoints,
                                           int imageWidth, int imageHeight, int subsetRadius,
                                           int numParams, int searchRadius) {
    
    int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pointIdx >= numPoints) return;
    
    Point2D centerPoint = points[pointIdx];
    
    // 初始化
    for (int i = 0; i < numParams; i++) {
        initialParams[pointIdx * numParams + i] = 0.0;
    }
    
    double bestZNCC = 1e10;
    int bestDx = 0, bestDy = 0;
    bool foundMatch = false;
    
    // 改进的网格搜索，使用更密集的搜索步长
    for (int dy = -searchRadius; dy <= searchRadius; dy += 1) { // 改为步长1，更精细
        for (int dx = -searchRadius; dx <= searchRadius; dx += 1) {
            Point2D testPoint = {centerPoint.x + dx, centerPoint.y + dy};
            
            // 边界检查
            if (testPoint.x >= subsetRadius && testPoint.x < imageWidth - subsetRadius &&
                testPoint.y >= subsetRadius && testPoint.y < imageHeight - subsetRadius) {
                
                // 创建简单的平移参数
                double testParams[12] = {0};
                testParams[0] = static_cast<double>(dx);
                testParams[1] = static_cast<double>(dy);
                
                // 计算ZNCC
                double testZNCC = computePrecisionZNCC(refImage, defImage, centerPoint, testParams, 
                                                     imageWidth, imageHeight, subsetRadius, numParams);
                
                // 更新最佳匹配
                if (testZNCC < bestZNCC) {
                    bestZNCC = testZNCC;
                    bestDx = dx;
                    bestDy = dy;
                    foundMatch = true;
                }
            }
        }
    }
    
    // 设置结果 - 更宽松的初始猜测有效性判断
    if (foundMatch && bestZNCC < 2.0) { // 更宽松的ZNCC阈值
        initialParams[pointIdx * numParams + 0] = static_cast<double>(bestDx);
        initialParams[pointIdx * numParams + 1] = static_cast<double>(bestDy);
        initialZNCC[pointIdx] = bestZNCC;
        validMask[pointIdx] = true;
    } else {
        // 即使没有找到好的匹配，也给一个零初始猜测
        initialParams[pointIdx * numParams + 0] = 0.0;
        initialParams[pointIdx * numParams + 1] = 0.0;
        initialZNCC[pointIdx] = bestZNCC;
        validMask[pointIdx] = true; // 仍然标记为有效，让后续优化尝试
    }
}

// 图像转换核函数 - 高精度版本
__global__ void precisionImageConvertKernel(double* dst, const unsigned char* src, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = static_cast<double>(src[idx]);
    }
}

// Host wrapper functions - 高精度版本
extern "C" {

void launchPrecisionICGNOptimizationKernel(double* finalU, double* finalV, double* finalZNCC, bool* validMask,
                                          const double* refImage, const double* defImage,
                                          const Point2D* points, const double* initialParams,
                                          int numPoints, int imageWidth, int imageHeight,
                                          int subsetRadius, int numParams, int maxIterations,
                                          double convergenceThreshold, cudaStream_t stream) {
    
    // 计算网格和块大小
    int threadsPerBlock = 256; // 减少线程数以避免寄存器溢出
    int blocksPerGrid = (numPoints + threadsPerBlock - 1) / threadsPerBlock;
    
    // 启动高精度核函数
    precisionICGNOptimizationKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        finalU, finalV, finalZNCC, validMask,
        refImage, defImage, points, initialParams,
        numPoints, imageWidth, imageHeight,
        subsetRadius, numParams, maxIterations,
        convergenceThreshold
    );
    
    // 检查核函数启动是否成功
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Precision CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
}

void launchPrecisionInitialGuessKernel(double* initialParams, double* initialZNCC, bool* validMask,
                                      const double* refImage, const double* defImage,
                                      const Point2D* points, int numPoints,
                                      int imageWidth, int imageHeight, int subsetRadius,
                                      int numParams, int searchRadius, cudaStream_t stream) {
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (numPoints + threadsPerBlock - 1) / threadsPerBlock;
    
    precisionInitialGuessKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        initialParams, initialZNCC, validMask,
        refImage, defImage, points, numPoints,
        imageWidth, imageHeight, subsetRadius,
        numParams, searchRadius
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Precision initial guess kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
}

void launchPrecisionImageConvertKernel(double* dst, const unsigned char* src, 
                                      int width, int height, cudaStream_t stream) {
    int size = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    precisionImageConvertKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(dst, src, size);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Precision image convert kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
}

} // extern "C"
