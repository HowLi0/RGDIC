#ifndef DIC_PROCESSOR_H
#define DIC_PROCESSOR_H

#include "poi.h"
#include "icgn_optimizer.h"
#include "neighbor_utils.h"
#include <opencv2/opencv.hpp>
#include <memory>
#include <functional>

/**
 * @brief Modular DIC Processor using POI design
 * 
 * This class provides a modular interface for Digital Image Correlation
 * computation using the Point of Interest (POI) design pattern.
 */
class DICProcessor {
public:
    // Configuration structure
    struct Config {
        int subsetRadius;
        double convergenceThreshold;
        int maxIterations;
        double ccThreshold;
        double deltaDispThreshold;
        ShapeFunctionOrder order;
        int neighborStep;
        bool enableQualityControl;
        bool enableParallelProcessing;
        
        Config() : subsetRadius(15), convergenceThreshold(0.00001), maxIterations(30),
                  ccThreshold(0.8), deltaDispThreshold(1.0), order(SECOND_ORDER),
                  neighborStep(5), enableQualityControl(true), enableParallelProcessing(true) {}
    };
    
    // Processing statistics
    struct Statistics {
        size_t totalPoints;
        size_t processedPoints;
        size_t validPoints;
        size_t convergedPoints;
        double averageZNCC;
        double processingTime;
        double averageIterations;
        
        // Quality metrics
        cv::Scalar meanDisplacement;
        cv::Scalar stdDisplacement;
        double coverageRatio;  // valid points / total points
        
        Statistics() : totalPoints(0), processedPoints(0), validPoints(0), convergedPoints(0),
                      averageZNCC(0.0), processingTime(0.0), averageIterations(0.0),
                      meanDisplacement(cv::Scalar(0, 0)), stdDisplacement(cv::Scalar(0, 0)), coverageRatio(0.0) {}
    };
    
    // Progress callback function type
    using ProgressCallback = std::function<void(size_t current, size_t total, const std::string& message)>;
    
    // Constructor
    DICProcessor(const Config& config = Config());
    
    // Destructor
    virtual ~DICProcessor() = default;
    
    // === Configuration ===
    void setConfig(const Config& config) { config_ = config; }
    Config getConfig() const { return config_; }
    
    void setProgressCallback(ProgressCallback callback) { progressCallback_ = callback; }
    
    // === Main Processing Interface ===
    
    /**
     * @brief Process POIs using DIC analysis
     * @param refImage Reference image
     * @param defImage Deformed image
     * @param poiManager POI manager containing points to process
     * @return Processing statistics
     */
    virtual Statistics processPOIs(const cv::Mat& refImage, 
                                  const cv::Mat& defImage,
                                  POIManager& poiManager);
    
    /**
     * @brief Process a single POI
     * @param refImage Reference image
     * @param defImage Deformed image  
     * @param poi POI to process
     * @return True if processing succeeded
     */
    virtual bool processSinglePOI(const cv::Mat& refImage,
                                 const cv::Mat& defImage,
                                 std::shared_ptr<POI> poi);
    
    // === ROI-based Processing ===
    
    /**
     * @brief Process entire ROI with automatic POI generation
     * @param refImage Reference image
     * @param defImage Deformed image
     * @param roi Region of interest mask
     * @return POI manager with processed results
     */
    std::unique_ptr<POIManager> processROI(const cv::Mat& refImage,
                                          const cv::Mat& defImage, 
                                          const cv::Mat& roi);
    
    // === Seed Point Processing ===
    
    /**
     * @brief Find and process seed point
     * @param refImage Reference image
     * @param defImage Deformed image
     * @param roi Region of interest
     * @return Processed seed POI or nullptr if not found
     */
    std::shared_ptr<POI> findAndProcessSeedPoint(const cv::Mat& refImage,
                                                const cv::Mat& defImage,
                                                const cv::Mat& roi);
    
    // === Quality Control ===
    
    /**
     * @brief Apply quality control filters to POI manager
     * @param poiManager POI manager to filter
     */
    void applyQualityControl(POIManager& poiManager);
    
    /**
     * @brief Validate POI using displacement continuity
     * @param poi POI to validate
     * @param neighbors Neighbor POIs for continuity check
     * @return True if POI passes validation
     */
    bool validatePOI(std::shared_ptr<POI> poi, 
                    const std::vector<std::shared_ptr<POI>>& neighbors);
    
    // === Statistics and Analysis ===
    
    /**
     * @brief Calculate processing statistics from POI manager
     * @param poiManager POI manager to analyze
     * @return Calculated statistics
     */
    Statistics calculateStatistics(const POIManager& poiManager);
    
    /**
     * @brief Get last processing statistics
     * @return Last computed statistics
     */
    Statistics getLastStatistics() const { return lastStats_; }
    
    // === Utility Methods ===
    
    /**
     * @brief Generate initial guess for POI based on neighbors
     * @param poi POI for which to generate initial guess
     * @param processedPOIs Already processed POIs for interpolation
     * @return Initial displacement guess
     */
    cv::Vec2f generateInitialGuess(std::shared_ptr<POI> poi,
                                  const std::vector<std::shared_ptr<POI>>& processedPOIs);
    
    /**
     * @brief Calculate signed distance array for seed point selection
     * @param roi Region of interest
     * @return Signed distance array
     */
    cv::Mat calculateSDA(const cv::Mat& roi);
    
    /**
     * @brief Find optimal seed point using SDA
     * @param roi Region of interest
     * @param sda Signed distance array
     * @return Seed point coordinates
     */
    cv::Point findSeedPoint(const cv::Mat& roi, const cv::Mat& sda);

protected:
    // Configuration
    Config config_;
    
    // Optimizer instance
    std::unique_ptr<ICGNOptimizer> optimizer_;
    
    // Neighbor utilities
    NeighborUtils neighborUtils_;
    
    // Statistics
    Statistics lastStats_;
    
    // Progress callback
    ProgressCallback progressCallback_;
    
    // === Internal Methods ===
    
    /**
     * @brief Initialize optimizer with current images
     * @param refImage Reference image
     * @param defImage Deformed image
     */
    void initializeOptimizer(const cv::Mat& refImage, const cv::Mat& defImage);
    
    /**
     * @brief Update progress if callback is set
     * @param current Current progress
     * @param total Total work
     * @param message Progress message
     */
    void updateProgress(size_t current, size_t total, const std::string& message);
    
    /**
     * @brief Filter outliers based on displacement jumps
     * @param poiManager POI manager to filter
     */
    void filterDisplacementOutliers(POIManager& poiManager);
    
    /**
     * @brief Setup processing order for reliability-guided analysis
     * @param poiManager POI manager containing points to order
     * @param seedPOI Seed point to start from
     * @return Ordered list of POI indices
     */
    std::vector<size_t> setupProcessingOrder(const POIManager& poiManager,
                                            std::shared_ptr<POI> seedPOI);
};

/**
 * @brief CPU-based DIC Processor implementation
 */
class CPUDICProcessor : public DICProcessor {
public:
    CPUDICProcessor(const Config& config = Config()) : DICProcessor(config) {}
    
    // Override processing methods for CPU-specific optimizations
    Statistics processPOIs(const cv::Mat& refImage, 
                          const cv::Mat& defImage,
                          POIManager& poiManager) override;
    
private:
    // CPU-specific parallel processing
    void processPointsBatch(const cv::Mat& refImage,
                           const cv::Mat& defImage,
                           POIManager& poiManager,
                           const std::vector<size_t>& indices,
                           size_t startIdx, size_t endIdx);
};

/**
 * @brief Factory class for creating DIC processors
 */
class DICProcessorFactory {
public:
    enum ProcessorType {
        CPU_PROCESSOR,
        CUDA_PROCESSOR  // For future CUDA integration
    };
    
    /**
     * @brief Create DIC processor of specified type
     * @param type Processor type
     * @param config Configuration
     * @return Unique pointer to created processor
     */
    static std::unique_ptr<DICProcessor> create(ProcessorType type, 
                                               const DICProcessor::Config& config = DICProcessor::Config());
    
    /**
     * @brief Create optimal processor for current system
     * @param config Configuration
     * @return Unique pointer to created processor
     */
    static std::unique_ptr<DICProcessor> createOptimal(const DICProcessor::Config& config = DICProcessor::Config());
};

#endif // DIC_PROCESSOR_H