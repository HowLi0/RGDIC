#ifndef POI_IO_H
#define POI_IO_H

#include "poi.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

/**
 * @brief I/O interface for POI-based DIC results
 * 
 * This class provides modular input/output capabilities for POI data,
 * supporting various formats and backward compatibility with existing RGDIC results.
 */
class POIIOInterface {
public:
    // Export format options
    enum ExportFormat {
        CSV_FORMAT,
        JSON_FORMAT,
        HDF5_FORMAT,  // For future implementation
        MAT_FORMAT    // OpenCV Mat format for backward compatibility
    };
    
    // Import format options
    enum ImportFormat {
        RGDIC_RESULT,  // Import from RGDIC DisplacementResult
        CSV_FILE,      // Import from CSV file
        JSON_FILE      // Import from JSON file
    };
    
    // Export options structure
    struct ExportOptions {
        bool includeInvalid;
        bool includeMetadata;
        bool includeStatistics;
        int precision;
        std::string delimiter;
        std::string encoding;
        
        ExportOptions() : includeInvalid(false), includeMetadata(true), includeStatistics(true),
                         precision(6), delimiter(","), encoding("utf-8") {}
    };
    
    // Import options structure
    struct ImportOptions {
        bool validateData;
        bool setDefaultValues;
        double defaultZNCC;
        bool defaultValid;
        
        ImportOptions() : validateData(true), setDefaultValues(true), defaultZNCC(0.0), defaultValid(false) {}
    };
    
    // Metadata structure
    struct Metadata {
        std::string timestamp;
        std::string software_version;
        std::string description;
        cv::Size imageSize;
        int subsetRadius;
        double convergenceThreshold;
        int maxIterations;
        std::string shapeFunctionOrder;
        
        Metadata() : imageSize(0, 0), subsetRadius(0), convergenceThreshold(0.0), maxIterations(0) {}
    };
    
    // Constructor
    POIIOInterface() = default;
    
    // Destructor
    virtual ~POIIOInterface() = default;
    
    // === Export Methods ===
    
    /**
     * @brief Export POI manager to file
     * @param poiManager POI manager to export
     * @param filename Output filename
     * @param format Export format
     * @param options Export options
     * @return True if export succeeded
     */
    virtual bool exportPOIs(const POIManager& poiManager,
                           const std::string& filename,
                           ExportFormat format = CSV_FORMAT,
                           const ExportOptions& options = ExportOptions());
    
    /**
     * @brief Export to CSV format
     * @param poiManager POI manager to export
     * @param filename CSV filename
     * @param options Export options
     * @return True if export succeeded
     */
    bool exportToCSV(const POIManager& poiManager,
                    const std::string& filename,
                    const ExportOptions& options = ExportOptions());
    
    /**
     * @brief Export to JSON format
     * @param poiManager POI manager to export
     * @param filename JSON filename
     * @param options Export options
     * @return True if export succeeded
     */
    bool exportToJSON(const POIManager& poiManager,
                     const std::string& filename,
                     const ExportOptions& options = ExportOptions());
    
    /**
     * @brief Export valid POIs only
     * @param poiManager POI manager to export
     * @param filename Output filename
     * @param format Export format
     * @return True if export succeeded
     */
    bool exportValidPOIs(const POIManager& poiManager,
                        const std::string& filename,
                        ExportFormat format = CSV_FORMAT);
    
    // === Import Methods ===
    
    /**
     * @brief Import POIs from file
     * @param filename Input filename
     * @param format Import format
     * @param options Import options
     * @return Unique pointer to loaded POI manager
     */
    virtual std::unique_ptr<POIManager> importPOIs(const std::string& filename,
                                                   ImportFormat format,
                                                   const ImportOptions& options = ImportOptions());
    
    /**
     * @brief Import from CSV file
     * @param filename CSV filename
     * @param options Import options
     * @return Unique pointer to loaded POI manager
     */
    std::unique_ptr<POIManager> importFromCSV(const std::string& filename,
                                             const ImportOptions& options = ImportOptions());
    
    /**
     * @brief Import from JSON file
     * @param filename JSON filename
     * @param options Import options
     * @return Unique pointer to loaded POI manager
     */
    std::unique_ptr<POIManager> importFromJSON(const std::string& filename,
                                              const ImportOptions& options = ImportOptions());
    
    /**
     * @brief Import from RGDIC result (backward compatibility)
     * @param u X displacement field
     * @param v Y displacement field
     * @param cc Correlation coefficient field
     * @param validMask Valid points mask
     * @param roi Region of interest
     * @param step Point spacing
     * @return Unique pointer to POI manager
     */
    std::unique_ptr<POIManager> importFromRGDICResult(const cv::Mat& u, const cv::Mat& v,
                                                      const cv::Mat& cc, const cv::Mat& validMask,
                                                      const cv::Mat& roi, int step = 5);
    
    // === Conversion Methods ===
    
    /**
     * @brief Convert POI manager to RGDIC result format (backward compatibility)
     * @param poiManager POI manager to convert
     * @param u Output X displacement field
     * @param v Output Y displacement field
     * @param cc Output correlation coefficient field
     * @param validMask Output valid points mask
     * @param imageSize Size of output fields
     */
    void convertToRGDICResult(const POIManager& poiManager,
                             cv::Mat& u, cv::Mat& v, cv::Mat& cc, cv::Mat& validMask,
                             const cv::Size& imageSize);
    
    // === Metadata Methods ===
    
    /**
     * @brief Set metadata for export
     * @param metadata Metadata to include in exports
     */
    void setMetadata(const Metadata& metadata) { metadata_ = metadata; }
    
    /**
     * @brief Get current metadata
     * @return Current metadata
     */
    Metadata getMetadata() const { return metadata_; }
    
    // === Utility Methods ===
    
    /**
     * @brief Validate file format compatibility
     * @param filename Filename to check
     * @param format Expected format
     * @return True if compatible
     */
    static bool validateFormat(const std::string& filename, ExportFormat format);
    
    /**
     * @brief Get file extension for format
     * @param format Export format
     * @return File extension (with dot)
     */
    static std::string getFormatExtension(ExportFormat format);
    
    /**
     * @brief Create backup of existing file
     * @param filename Original filename
     * @return Backup filename if successful, empty if failed
     */
    static std::string createBackup(const std::string& filename);

protected:
    Metadata metadata_;
    
    // Helper methods for JSON export/import
    std::string poiToJSON(const POI& poi, int indent = 0) const;
    std::shared_ptr<POI> poiFromJSON(const std::string& jsonStr) const;
    
    // Helper methods for CSV processing
    std::vector<std::string> parseCSVLine(const std::string& line, const std::string& delimiter) const;
    std::string escapeCSVField(const std::string& field) const;
    
    // Validation helpers
    bool validatePOIData(const POI& poi) const;
    bool validateFilename(const std::string& filename) const;
};

/**
 * @brief Enhanced I/O interface with strain field support
 */
class POIIOEnhanced : public POIIOInterface {
public:
    // Enhanced export options
    struct EnhancedExportOptions : public ExportOptions {
        bool includeStrainFields;
        bool includeShapeParameters;
        bool includeNeighborInfo;
        bool compressOutput;
        
        EnhancedExportOptions() : ExportOptions(), includeStrainFields(true), includeShapeParameters(false),
                                 includeNeighborInfo(false), compressOutput(false) {}
    };
    
    // Constructor
    POIIOEnhanced() = default;
    
    // Override export methods to include strain data
    bool exportPOIs(const POIManager& poiManager,
                   const std::string& filename,
                   ExportFormat format = CSV_FORMAT,
                   const ExportOptions& options = ExportOptions()) override;
    
    /**
     * @brief Export with strain field data
     * @param poiManager POI manager with strain data
     * @param filename Output filename
     * @param format Export format
     * @param options Enhanced export options
     * @return True if export succeeded
     */
    bool exportWithStrainData(const POIManager& poiManager,
                             const std::string& filename,
                             ExportFormat format = CSV_FORMAT,
                             const EnhancedExportOptions& options = EnhancedExportOptions());
    
    /**
     * @brief Export processing statistics (simplified version)
     * @param filename Output filename
     * @param stats_data Map of statistics data
     * @return True if export succeeded
     */
    bool exportStatistics(const std::string& filename,
                         const std::map<std::string, double>& stats_data);
    
    // Enhanced import with strain data support
    std::unique_ptr<POIManager> importPOIs(const std::string& filename,
                                          ImportFormat format,
                                          const ImportOptions& options = ImportOptions()) override;

private:
    // Enhanced CSV header generation
    std::string generateEnhancedCSVHeader(const EnhancedExportOptions& options) const;
    
    // Enhanced POI to CSV row conversion
    std::string poiToEnhancedCSVRow(const POI& poi, const EnhancedExportOptions& options) const;
};

/**
 * @brief Factory for creating I/O interfaces
 */
class POIIOFactory {
public:
    enum IOType {
        STANDARD_IO,
        ENHANCED_IO
    };
    
    /**
     * @brief Create I/O interface
     * @param type I/O interface type
     * @return Unique pointer to I/O interface
     */
    static std::unique_ptr<POIIOInterface> create(IOType type = STANDARD_IO);
};

// Forward declarations
class DICProcessor;

#endif // POI_IO_H