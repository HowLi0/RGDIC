#ifndef PYRAMID_INTERFACE_H
#define PYRAMID_INTERFACE_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

class PyramidInterface {
public:
    // 金字塔层结构
    struct Layer {
        cv::Mat img;         // 此金字塔级别的图像
        cv::Mat mask;        // 此金字塔级别的掩码
        int octave;          // 当前阶段
        double scale;        // 相对于原始图像的比例因子
        double sigma;        // 此层的高斯模糊sigma值
    };
    
    // 构造函数
    PyramidInterface() {}
    
    // 创建图像金字塔
    static void createPyramid(const cv::Mat& image, const cv::Mat& mask, 
                             std::vector<Layer>& pyramid, 
                             int numLevels = 3, 
                             double scaleFactor = 0.5,
                             const cv::Size& minSize = cv::Size(32, 32)) {
        // 清除旧金字塔
        pyramid.clear();
        
        // 确定最多可以有多少层
        int maxLevels = 1;
        double currentScale = 1.0;
        while (maxLevels < 10) { // 限制最大级别
            currentScale *= scaleFactor;
            int width = static_cast<int>(image.cols * currentScale);
            int height = static_cast<int>(image.rows * currentScale);
            if (width < minSize.width || height < minSize.height) {
                break;
            }
            maxLevels++;
        }
        
        numLevels = std::min(numLevels, maxLevels);
        
        // 调整金字塔大小
        pyramid.resize(numLevels);
        
        // 设置第一层（原始尺寸）
        pyramid[0].img = image.clone();
        pyramid[0].mask = mask.clone();
        pyramid[0].octave = 0;
        pyramid[0].scale = 1.0;
        pyramid[0].sigma = 0.0; // 无模糊
        
        // 为每一层创建图像
        for (int i = 1; i < numLevels; i++) {
            double scale = std::pow(scaleFactor, static_cast<double>(i));
            int newWidth = static_cast<int>(image.cols * scale);
            int newHeight = static_cast<int>(image.rows * scale);
            
            // 设置层信息
            pyramid[i].octave = i;
            pyramid[i].scale = scale;
            pyramid[i].sigma = 0.9 * scale; // 适用于降尺度的高斯模糊
            
            // 调整图像大小
            cv::resize(image, pyramid[i].img, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_AREA);
            cv::resize(mask, pyramid[i].mask, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_NEAREST);
            
            // 应用模糊以减少缩小时的锯齿状
            cv::GaussianBlur(pyramid[i].img, pyramid[i].img, cv::Size(0, 0), pyramid[i].sigma);
        }
    }
    
    // 从一个金字塔级别上采样到另一个级别
    static void upsample(const cv::Mat& source, cv::Mat& destination, 
                        double sourceScale, double destinationScale) {
        if (sourceScale >= destinationScale) {
            std::cerr << "错误: 上采样需要源比例小于目标比例" << std::endl;
            return;
        }
        
        double scaleFactor = destinationScale / sourceScale;
        cv::resize(source, destination, cv::Size(), scaleFactor, scaleFactor, cv::INTER_LINEAR);
    }
    
    // 从一个金字塔级别下采样到另一个级别
    static void downsample(const cv::Mat& source, cv::Mat& destination, 
                          double sourceScale, double destinationScale) {
        if (sourceScale <= destinationScale) {
            std::cerr << "错误: 下采样需要源比例大于目标比例" << std::endl;
            return;
        }
        
        double scaleFactor = destinationScale / sourceScale;
        cv::resize(source, destination, cv::Size(), scaleFactor, scaleFactor, cv::INTER_AREA);
    }
    
    // 可视化金字塔
    static void visualizePyramid(const std::vector<Layer>& pyramid, const std::string& windowName) {
        if (pyramid.empty()) return;
        
        // 计算画布尺寸
        int maxWidth = pyramid[0].img.cols;
        int totalHeight = 0;
        int spacing = 10; // 层间间距
        
        for (const auto& layer : pyramid) {
            totalHeight += layer.img.rows + spacing;
        }
        
        // 创建画布
        cv::Mat canvas = cv::Mat::zeros(totalHeight, maxWidth, CV_8UC3);
        int yOffset = 0;
        
        for (const auto& layer : pyramid) {
            cv::Mat colorImg;
            
            // 确保图像是3通道
            if (layer.img.channels() == 1) {
                cv::cvtColor(layer.img, colorImg, cv::COLOR_GRAY2BGR);
            } else {
                colorImg = layer.img.clone();
            }
            
            // 根据需要调整尺寸以匹配画布宽度
            cv::Mat resizedImg;
            double scaleFactor = static_cast<double>(maxWidth) / layer.img.cols;
            cv::resize(colorImg, resizedImg, cv::Size(), scaleFactor, scaleFactor);
            
            // 复制到画布
            cv::Mat ROI = canvas(cv::Rect(0, yOffset, resizedImg.cols, resizedImg.rows));
            resizedImg.copyTo(ROI);
            
            // 添加级别信息文本
            std::stringstream ss;
            ss << "Level " << layer.octave << " (Scale: " << layer.scale << ")";
            cv::putText(canvas, ss.str(), cv::Point(10, yOffset + 20), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            
            yOffset += resizedImg.rows + spacing;
        }
        
        cv::imshow(windowName, canvas);
    }
};

#endif // PYRAMID_INTERFACE_H