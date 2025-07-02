#ifndef POI_H
#define POI_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <fstream>

namespace rgdic {

// POI (Point of Interest) 类 - 表示DIC分析中的感兴趣点
class POI {
public:
    // 坐标信息
    cv::Point2f leftCoord;     // 左图（参考图像）坐标
    cv::Point2f rightCoord;    // 右图（变形图像）坐标
    
    // 位移信息
    cv::Vec2f displacement;    // 位移向量 (u, v)
    
    // 质量信息
    double correlation;        // 相关系数 (ZNCC)
    bool valid;               // 是否为有效点
    
    // 应变信息（可选）
    struct StrainInfo {
        double exx, eyy, exy;  // 正应变和剪应变分量
        bool computed;         // 是否已计算应变
        
        StrainInfo() : exx(0), eyy(0), exy(0), computed(false) {}
    } strain;
    
    // 构造函数
    POI();
    POI(const cv::Point2f& left, const cv::Point2f& right, 
        const cv::Vec2f& disp, double corr, bool isValid = true);
    
    // 计算右图坐标（基于左图坐标和位移）
    void updateRightCoord();
    
    // 验证POI的有效性
    bool isValid() const { return valid; }
    
    // 序列化/反序列化
    std::string toString() const;
    static POI fromString(const std::string& str);
};

// POI集合管理类
class POICollection {
private:
    std::vector<POI> m_pois;
    cv::Size m_imageSize;
    std::string m_description;
    
public:
    POICollection();
    POICollection(const cv::Size& imageSize, const std::string& desc = "");
    
    // 基本操作
    void addPOI(const POI& poi);
    void removePOI(size_t index);
    void clear();
    
    // 访问器
    size_t size() const { return m_pois.size(); }
    bool empty() const { return m_pois.empty(); }
    const POI& operator[](size_t index) const { return m_pois[index]; }
    POI& operator[](size_t index) { return m_pois[index]; }
    
    // 迭代器支持
    std::vector<POI>::iterator begin() { return m_pois.begin(); }
    std::vector<POI>::iterator end() { return m_pois.end(); }
    std::vector<POI>::const_iterator begin() const { return m_pois.begin(); }
    std::vector<POI>::const_iterator end() const { return m_pois.end(); }
    
    // 过滤和查询
    POICollection filterByCorrelation(double threshold) const;
    POICollection filterByRegion(const cv::Rect& region) const;
    std::vector<POI> getValidPOIs() const;
    
    // 统计信息
    double getMeanCorrelation() const;
    cv::Vec2f getMeanDisplacement() const;
    size_t getValidCount() const;
    
    // 数据导出
    bool exportToCSV(const std::string& filename) const;
    bool exportToPOIFormat(const std::string& filename) const;
    bool exportToMatlab(const std::string& filename) const;
    
    // 数据导入
    bool importFromCSV(const std::string& filename);
    bool importFromPOIFormat(const std::string& filename);
    
    // 与矩阵格式转换
    void convertToMatrices(cv::Mat& u, cv::Mat& v, cv::Mat& cc, cv::Mat& validMask) const;
    void convertFromMatrices(const cv::Mat& u, const cv::Mat& v, 
                           const cv::Mat& cc, const cv::Mat& validMask);
                           
    // 属性访问
    const cv::Size& getImageSize() const { return m_imageSize; }
    void setImageSize(const cv::Size& size) { m_imageSize = size; }
    const std::string& getDescription() const { return m_description; }
    void setDescription(const std::string& desc) { m_description = desc; }
};

} // namespace rgdic

#endif // POI_H