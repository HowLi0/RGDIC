#ifndef RGDIC_POI_ADAPTER_H
#define RGDIC_POI_ADAPTER_H

#include "rgdic.h"
#include "poi.h"
#include "dic_processor.h"
#include "poi_io.h"

/**
 * @brief Adapter class that bridges POI design with existing RGDIC interface
 * 
 * This class provides backward compatibility while enabling the use of the new
 * modular POI-based design. It wraps the POI system to maintain the existing
 * RGDIC interface for minimal disruption to existing code.
 */
class RGDICPOIAdapter : public RGDIC {
public:
    // Constructor with same parameters as original RGDIC
    RGDICPOIAdapter(int subsetRadius = 15, 
                   double convergenceThreshold = 0.00001,
                   int maxIterations = 30,
                   double ccThreshold = 0.8,
                   double deltaDispThreshold = 1.0,
                   ShapeFunctionOrder order = SECOND_ORDER,
                   int neighborStep = 5);
    
    // Destructor
    virtual ~RGDICPOIAdapter() = default;
    
    // Override main compute function to use POI-based processing
    DisplacementResult compute(const cv::Mat& refImage, 
                              const cv::Mat& defImage,
                              const cv::Mat& roi) override;
    
    // === POI Access Methods ===
    
    /**
     * @brief Get POI manager from last computation
     * @return Pointer to POI manager or nullptr if no computation performed
     */
    std::shared_ptr<POIManager> getPOIManager() const { return poiManager_; }
    
    /**
     * @brief Get DIC processor used for computation
     * @return Pointer to DIC processor
     */
    std::shared_ptr<DICProcessor> getDICProcessor() const { return dicProcessor_; }
    
    /**
     * @brief Get I/O interface for POI data export/import
     * @return Pointer to I/O interface
     */
    std::shared_ptr<POIIOInterface> getIOInterface() const { return ioInterface_; }
    
    // === Enhanced Export Methods ===
    
    /**
     * @brief Export results in enhanced CSV format with full POI data
     * @param filename Output filename
     * @param includeStrainData Include strain field data if available
     * @return True if export succeeded
     */
    bool exportEnhancedCSV(const std::string& filename, bool includeStrainData = true);
    
    /**
     * @brief Export results in JSON format
     * @param filename Output filename
     * @return True if export succeeded
     */
    bool exportJSON(const std::string& filename);
    
    /**
     * @brief Export processing statistics
     * @param filename Output filename
     * @return True if export succeeded
     */
    bool exportStatistics(const std::string& filename);
    
    // === Import Methods for POI Data ===
    
    /**
     * @brief Import POI data from file and convert to DisplacementResult
     * @param filename Input filename
     * @param imageSize Size for output displacement fields
     * @return DisplacementResult compatible with existing code
     */
    DisplacementResult importFromPOIFile(const std::string& filename, const cv::Size& imageSize);
    
    // === Configuration and Statistics ===
    
    /**
     * @brief Set progress callback for processing updates
     * @param callback Function to call with progress updates
     */
    void setProgressCallback(std::function<void(size_t, size_t, const std::string&)> callback);
    
    /**
     * @brief Get processing statistics from last computation
     * @return Processing statistics
     */
    DICProcessor::Statistics getProcessingStatistics() const;
    
    /**
     * @brief Enable/disable parallel processing
     * @param enable True to enable parallel processing
     */
    void setParallelProcessing(bool enable);
    
    /**
     * @brief Enable/disable quality control
     * @param enable True to enable quality control
     */
    void setQualityControl(bool enable);

protected:
    // POI-based processing components
    std::shared_ptr<DICProcessor> dicProcessor_;
    std::shared_ptr<POIManager> poiManager_;
    std::shared_ptr<POIIOInterface> ioInterface_;
    
    // Configuration
    DICProcessor::Config processorConfig_;
    
    // Progress callback
    std::function<void(size_t, size_t, const std::string&)> progressCallback_;
    
    // Convert POI results to traditional DisplacementResult format
    DisplacementResult convertToDisplacementResult(const cv::Size& imageSize);
    
    // Setup processor configuration from RGDIC parameters
    void setupProcessorConfig();
    
    // Initialize POI system components
    void initializePOISystem();
};

/**
 * @brief Factory function to create RGDIC instances with POI support
 * 
 * This function provides a drop-in replacement for the existing createRGDIC
 * factory function, automatically using the POI-based adapter for enhanced
 * functionality while maintaining backward compatibility.
 */
std::unique_ptr<RGDIC> createRGDICWithPOI(bool useCuda = false,
                                          int subsetRadius = 15,
                                          double convergenceThreshold = 0.00001,
                                          int maxIterations = 30,
                                          double ccThreshold = 0.8,
                                          double deltaDispThreshold = 1.0,
                                          ShapeFunctionOrder order = SECOND_ORDER,
                                          int neighborStep = 5);

/**
 * @brief Enhanced RGDIC factory that automatically chooses optimal implementation
 * 
 * This function analyzes the system and automatically selects the best
 * implementation (CPU POI-based, CUDA POI-based, or legacy) based on
 * available hardware and data size.
 */
class RGDICFactory {
public:
    enum ImplementationType {
        AUTO_SELECT,      // Automatically choose optimal implementation
        POI_CPU,          // POI-based CPU implementation
        POI_CUDA,         // POI-based CUDA implementation (future)
        LEGACY_CPU,       // Original CPU implementation
        LEGACY_CUDA       // Original CUDA implementation
    };
    
    struct FactoryConfig {
        ImplementationType type;
        bool enablePOI;
        bool enableEnhancedIO;
        bool enableStatistics;
        bool enableParallelProcessing;
        
        FactoryConfig() : type(AUTO_SELECT), enablePOI(true), enableEnhancedIO(true),
                         enableStatistics(true), enableParallelProcessing(true) {}
    };
    
    /**
     * @brief Create RGDIC instance with specified configuration
     * @param config Factory configuration
     * @param rgdicParams RGDIC algorithm parameters
     * @return Unique pointer to RGDIC instance
     */
    static std::unique_ptr<RGDIC> create(const FactoryConfig& config,
                                        int subsetRadius = 15,
                                        double convergenceThreshold = 0.00001,
                                        int maxIterations = 30,
                                        double ccThreshold = 0.8,
                                        double deltaDispThreshold = 1.0,
                                        ShapeFunctionOrder order = SECOND_ORDER,
                                        int neighborStep = 5);
    
    /**
     * @brief Get recommended implementation for given parameters
     * @param imageSize Size of images to process
     * @param roiSize Size of region of interest
     * @param pointCount Estimated number of points to process
     * @return Recommended implementation type
     */
    static ImplementationType getRecommendedImplementation(const cv::Size& imageSize,
                                                         const cv::Size& roiSize,
                                                         size_t pointCount);
    
    /**
     * @brief Check if CUDA is available and suitable for POI processing
     * @return True if CUDA POI processing is available
     */
    static bool isCudaPOIAvailable();
    
    /**
     * @brief Get system capabilities for DIC processing
     * @return String describing system capabilities
     */
    static std::string getSystemCapabilities();
};

/**
 * @brief Migration helper for existing RGDIC code
 * 
 * This class provides utilities to help migrate existing RGDIC code
 * to the new POI-based system with minimal changes.
 */
class RGDICMigrationHelper {
public:
    /**
     * @brief Convert existing main function to use POI adapter
     * @param argc Command line argument count
     * @param argv Command line arguments
     * @param usePOI Whether to use POI-based processing
     * @return Exit code
     */
    static int convertedMain(int argc, char** argv, bool usePOI = true);
    
    /**
     * @brief Create POI-compatible RGDIC configuration from legacy parameters
     * @param subsetRadius Subset radius
     * @param convergenceThreshold Convergence threshold
     * @param maxIterations Maximum iterations
     * @param ccThreshold Correlation coefficient threshold
     * @param deltaDispThreshold Displacement jump threshold
     * @param order Shape function order
     * @param neighborStep Neighbor step size
     * @return DIC processor configuration
     */
    static DICProcessor::Config createConfigFromLegacyParams(
        int subsetRadius = 15,
        double convergenceThreshold = 0.00001,
        int maxIterations = 30,
        double ccThreshold = 0.8,
        double deltaDispThreshold = 1.0,
        ShapeFunctionOrder order = SECOND_ORDER,
        int neighborStep = 5);
    
    /**
     * @brief Export comparison between legacy and POI results
     * @param legacyResult Legacy RGDIC result
     * @param poiResult POI-based result
     * @param filename Output filename
     * @return True if export succeeded
     */
    static bool exportResultComparison(const RGDIC::DisplacementResult& legacyResult,
                                      const RGDIC::DisplacementResult& poiResult,
                                      const std::string& filename);
    
    /**
     * @brief Validate POI results against legacy results
     * @param legacyResult Legacy RGDIC result
     * @param poiResult POI-based result
     * @param tolerance Comparison tolerance
     * @return Validation statistics
     */
    static DICProcessor::Statistics validatePOIResults(
        const RGDIC::DisplacementResult& legacyResult,
        const RGDIC::DisplacementResult& poiResult,
        double tolerance = 1e-6);
};

#endif // RGDIC_POI_ADAPTER_H