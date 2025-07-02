#include "poi_io.h"
#include "dic_processor.h"  // For statistics
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>
#include <ctime>
#include <chrono>

// === POIIOInterface Implementation ===

bool POIIOInterface::exportPOIs(const POIManager& poiManager,
                               const std::string& filename,
                               ExportFormat format,
                               const ExportOptions& options) {
    if (!validateFilename(filename)) {
        return false;
    }
    
    switch (format) {
        case CSV_FORMAT:
            return exportToCSV(poiManager, filename, options);
        case JSON_FORMAT:
            return exportToJSON(poiManager, filename, options);
        case HDF5_FORMAT:
            // TODO: Implement HDF5 export
            throw std::runtime_error("HDF5 format not yet implemented");
        case MAT_FORMAT:
            // TODO: Implement OpenCV Mat export
            throw std::runtime_error("MAT format not yet implemented");
        default:
            return false;
    }
}

bool POIIOInterface::exportToCSV(const POIManager& poiManager,
                                const std::string& filename,
                                const ExportOptions& options) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    // Set precision
    file << std::fixed << std::setprecision(options.precision);
    
    // Write header
    file << POI::getCSVHeader() << std::endl;
    
    // Write data
    for (size_t i = 0; i < poiManager.size(); ++i) {
        auto poi = poiManager.getPOI(i);
        
        // Skip invalid POIs unless explicitly requested
        if (!options.includeInvalid && !poi->isValid()) {
            continue;
        }
        
        file << poi->toCSVRow() << std::endl;
    }
    
    // Write metadata as comments if requested
    if (options.includeMetadata) {
        file << std::endl;
        file << "# Metadata" << std::endl;
        file << "# Timestamp: " << metadata_.timestamp << std::endl;
        file << "# Software: " << metadata_.software_version << std::endl;
        file << "# Description: " << metadata_.description << std::endl;
        file << "# Image Size: " << metadata_.imageSize.width << "x" << metadata_.imageSize.height << std::endl;
        file << "# Subset Radius: " << metadata_.subsetRadius << std::endl;
        file << "# Convergence Threshold: " << metadata_.convergenceThreshold << std::endl;
        file << "# Max Iterations: " << metadata_.maxIterations << std::endl;
        file << "# Shape Function: " << metadata_.shapeFunctionOrder << std::endl;
    }
    
    // Write statistics if requested
    if (options.includeStatistics) {
        file << std::endl;
        file << "# Statistics" << std::endl;
        file << "# Total POIs: " << poiManager.size() << std::endl;
        file << "# Valid POIs: " << poiManager.getValidCount() << std::endl;
        file << "# Converged POIs: " << poiManager.getConvergedCount() << std::endl;
        file << "# Average ZNCC: " << poiManager.getAverageZNCC() << std::endl;
        
        cv::Scalar meanDisp = poiManager.getMeanDisplacement();
        cv::Scalar stdDisp = poiManager.getStdDisplacement();
        file << "# Mean Displacement: " << meanDisp[0] << ", " << meanDisp[1] << std::endl;
        file << "# Std Displacement: " << stdDisp[0] << ", " << stdDisp[1] << std::endl;
    }
    
    file.close();
    return true;
}

bool POIIOInterface::exportToJSON(const POIManager& poiManager,
                                 const std::string& filename,
                                 const ExportOptions& options) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    file << "{" << std::endl;
    
    // Write metadata if requested
    if (options.includeMetadata) {
        file << "  \"metadata\": {" << std::endl;
        file << "    \"timestamp\": \"" << metadata_.timestamp << "\"," << std::endl;
        file << "    \"software_version\": \"" << metadata_.software_version << "\"," << std::endl;
        file << "    \"description\": \"" << metadata_.description << "\"," << std::endl;
        file << "    \"image_size\": {" << std::endl;
        file << "      \"width\": " << metadata_.imageSize.width << "," << std::endl;
        file << "      \"height\": " << metadata_.imageSize.height << std::endl;
        file << "    }," << std::endl;
        file << "    \"subset_radius\": " << metadata_.subsetRadius << "," << std::endl;
        file << "    \"convergence_threshold\": " << metadata_.convergenceThreshold << "," << std::endl;
        file << "    \"max_iterations\": " << metadata_.maxIterations << "," << std::endl;
        file << "    \"shape_function_order\": \"" << metadata_.shapeFunctionOrder << "\"" << std::endl;
        file << "  }," << std::endl;
    }
    
    // Write statistics if requested
    if (options.includeStatistics) {
        file << "  \"statistics\": {" << std::endl;
        file << "    \"total_pois\": " << poiManager.size() << "," << std::endl;
        file << "    \"valid_pois\": " << poiManager.getValidCount() << "," << std::endl;
        file << "    \"converged_pois\": " << poiManager.getConvergedCount() << "," << std::endl;
        file << "    \"average_zncc\": " << std::fixed << std::setprecision(options.precision) 
             << poiManager.getAverageZNCC() << "," << std::endl;
        
        cv::Scalar meanDisp = poiManager.getMeanDisplacement();
        cv::Scalar stdDisp = poiManager.getStdDisplacement();
        file << "    \"mean_displacement\": [" << meanDisp[0] << ", " << meanDisp[1] << "]," << std::endl;
        file << "    \"std_displacement\": [" << stdDisp[0] << ", " << stdDisp[1] << "]" << std::endl;
        file << "  }," << std::endl;
    }
    
    // Write POI data
    file << "  \"pois\": [" << std::endl;
    
    bool first = true;
    for (size_t i = 0; i < poiManager.size(); ++i) {
        auto poi = poiManager.getPOI(i);
        
        // Skip invalid POIs unless explicitly requested
        if (!options.includeInvalid && !poi->isValid()) {
            continue;
        }
        
        if (!first) {
            file << "," << std::endl;
        }
        first = false;
        
        file << poiToJSON(*poi, 4);
    }
    
    file << std::endl << "  ]" << std::endl;
    file << "}" << std::endl;
    
    file.close();
    return true;
}

bool POIIOInterface::exportValidPOIs(const POIManager& poiManager,
                                    const std::string& filename,
                                    ExportFormat format) {
    ExportOptions options;
    options.includeInvalid = false;
    return exportPOIs(poiManager, filename, format, options);
}

std::unique_ptr<POIManager> POIIOInterface::importPOIs(const std::string& filename,
                                                      ImportFormat format,
                                                      const ImportOptions& options) {
    switch (format) {
        case CSV_FILE:
            return importFromCSV(filename, options);
        case JSON_FILE:
            return importFromJSON(filename, options);
        case RGDIC_RESULT:
            throw std::invalid_argument("Use importFromRGDICResult for RGDIC result format");
        default:
            return nullptr;
    }
}

std::unique_ptr<POIManager> POIIOInterface::importFromCSV(const std::string& filename,
                                                         const ImportOptions& options) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return std::unique_ptr<POIManager>(nullptr);
    }
    
    std::unique_ptr<POIManager> poiManager(new POIManager());
    std::string line;
    bool headerRead = false;
    
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }
        
        // Skip header line
        if (!headerRead) {
            headerRead = true;
            continue;
        }
        
        // Parse CSV line
        auto fields = parseCSVLine(line, ",");
        if (fields.size() < 10) {  // Minimum expected fields
            continue;
        }
        
        try {
            // Parse coordinates and values
            double ref_x = std::stod(fields[0]);
            double ref_y = std::stod(fields[1]);
            
            auto poi = std::make_shared<POI>(ref_x, ref_y);
            
            // Deformed coordinates
            if (!fields[2].empty() && !fields[3].empty()) {
                double def_x = std::stod(fields[2]);
                double def_y = std::stod(fields[3]);
                poi->setDeformedCoords(def_x, def_y);
            }
            
            // Displacement
            if (!fields[4].empty() && !fields[5].empty()) {
                double u = std::stod(fields[4]);
                double v = std::stod(fields[5]);
                poi->setDisplacement(u, v);
            }
            
            // Strain components
            if (!fields[6].empty() && !fields[7].empty() && !fields[8].empty()) {
                double exx = std::stod(fields[6]);
                double eyy = std::stod(fields[7]);
                double exy = std::stod(fields[8]);
                poi->setStrainComponents(exx, eyy, exy);
            }
            
            // ZNCC
            if (!fields[9].empty()) {
                double zncc = std::stod(fields[9]);
                poi->setZNCC(zncc);
            } else if (options.setDefaultValues) {
                poi->setZNCC(options.defaultZNCC);
            }
            
            // Validate POI if requested
            if (options.validateData && !validatePOIData(*poi)) {
                continue;
            }
            
            // Set as valid by default or use option
            poi->setValid(options.setDefaultValues ? options.defaultValid : true);
            poi->setConverged(poi->isValid());
            
            poiManager->addPOI(poi);
            
        } catch (const std::exception&) {
            // Skip invalid lines
            continue;
        }
    }
    
    file.close();
    return poiManager;
}

std::unique_ptr<POIManager> POIIOInterface::importFromJSON(const std::string& filename,
                                                          const ImportOptions& options) {
    // Simple JSON parser for POI data
    // In a full implementation, would use a proper JSON library
    std::ifstream file(filename);
    if (!file.is_open()) {
        return std::unique_ptr<POIManager>(nullptr);
    }
    
    // For now, return empty manager
    // TODO: Implement full JSON parsing
    std::unique_ptr<POIManager> poiManager(new POIManager());
    
    file.close();
    return poiManager;
}

std::unique_ptr<POIManager> POIIOInterface::importFromRGDICResult(const cv::Mat& u, const cv::Mat& v,
                                                                 const cv::Mat& cc, const cv::Mat& validMask,
                                                                 const cv::Mat& roi, int step) {
    std::unique_ptr<POIManager> poiManager(new POIManager());
    poiManager->importFromRGDICResult(u, v, cc, validMask, roi, step);
    return poiManager;
}

void POIIOInterface::convertToRGDICResult(const POIManager& poiManager,
                                         cv::Mat& u, cv::Mat& v, cv::Mat& cc, cv::Mat& validMask,
                                         const cv::Size& imageSize) {
    // Initialize output matrices
    u = cv::Mat::zeros(imageSize, CV_64F);
    v = cv::Mat::zeros(imageSize, CV_64F);
    cc = cv::Mat::zeros(imageSize, CV_64F);
    validMask = cv::Mat::zeros(imageSize, CV_8U);
    
    // Fill matrices with POI data
    for (size_t i = 0; i < poiManager.size(); ++i) {
        auto poi = poiManager.getPOI(i);
        
        if (!poi->isValid()) continue;
        
        int x = static_cast<int>(std::round(poi->getReferenceX()));
        int y = static_cast<int>(std::round(poi->getReferenceY()));
        
        // Check bounds
        if (x >= 0 && x < imageSize.width && y >= 0 && y < imageSize.height) {
            if (poi->hasDisplacement()) {
                u.at<double>(y, x) = poi->getDisplacementU();
                v.at<double>(y, x) = poi->getDisplacementV();
            }
            
            if (poi->hasZNCC()) {
                cc.at<double>(y, x) = poi->getZNCC();
            }
            
            validMask.at<uchar>(y, x) = 255;
        }
    }
}

bool POIIOInterface::validateFormat(const std::string& filename, ExportFormat format) {
    std::string expectedExt = getFormatExtension(format);
    std::string actualExt = filename.substr(filename.find_last_of("."));
    
    return actualExt == expectedExt;
}

std::string POIIOInterface::getFormatExtension(ExportFormat format) {
    switch (format) {
        case CSV_FORMAT: return ".csv";
        case JSON_FORMAT: return ".json";
        case HDF5_FORMAT: return ".h5";
        case MAT_FORMAT: return ".xml";  // OpenCV FileStorage format
        default: return "";
    }
}

std::string POIIOInterface::createBackup(const std::string& filename) {
    std::ifstream src(filename, std::ios::binary);
    if (!src) {
        return "";  // File doesn't exist, no backup needed
    }
    
    // Generate backup filename with timestamp
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time_t);
    
    std::ostringstream backup_name;
    backup_name << filename << ".backup." << std::put_time(&tm, "%Y%m%d_%H%M%S");
    
    std::ofstream dst(backup_name.str(), std::ios::binary);
    if (!dst) {
        return "";
    }
    
    dst << src.rdbuf();
    src.close();
    dst.close();
    
    return backup_name.str();
}

std::string POIIOInterface::poiToJSON(const POI& poi, int indent) const {
    std::string indentStr(indent, ' ');
    std::ostringstream oss;
    
    oss << indentStr << "{" << std::endl;
    oss << indentStr << "  \"reference\": [" << poi.getReferenceX() << ", " << poi.getReferenceY() << "]," << std::endl;
    
    if (poi.hasDeformedCoords()) {
        oss << indentStr << "  \"deformed\": [" << poi.getDeformedX() << ", " << poi.getDeformedY() << "]," << std::endl;
    }
    
    if (poi.hasDisplacement()) {
        oss << indentStr << "  \"displacement\": [" << poi.getDisplacementU() << ", " << poi.getDisplacementV() << "]," << std::endl;
    }
    
    if (poi.hasZNCC()) {
        oss << indentStr << "  \"zncc\": " << poi.getZNCC() << "," << std::endl;
    }
    
    if (poi.hasStrain()) {
        oss << indentStr << "  \"strain\": {" << std::endl;
        oss << indentStr << "    \"exx\": " << poi.getStrainExx() << "," << std::endl;
        oss << indentStr << "    \"eyy\": " << poi.getStrainEyy() << "," << std::endl;
        oss << indentStr << "    \"exy\": " << poi.getStrainExy() << std::endl;
        oss << indentStr << "  }," << std::endl;
    }
    
    oss << indentStr << "  \"valid\": " << (poi.isValid() ? "true" : "false") << "," << std::endl;
    oss << indentStr << "  \"converged\": " << (poi.isConverged() ? "true" : "false") << std::endl;
    oss << indentStr << "}";
    
    return oss.str();
}

std::shared_ptr<POI> POIIOInterface::poiFromJSON(const std::string& jsonStr) const {
    // Simple JSON parser for POI data
    // TODO: Implement full JSON parsing
    return std::make_shared<POI>(0.0, 0.0);
}

std::vector<std::string> POIIOInterface::parseCSVLine(const std::string& line, const std::string& delimiter) const {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    
    while (std::getline(ss, field, delimiter[0])) {
        // Remove leading/trailing whitespace
        field.erase(0, field.find_first_not_of(" \t"));
        field.erase(field.find_last_not_of(" \t") + 1);
        fields.push_back(field);
    }
    
    return fields;
}

std::string POIIOInterface::escapeCSVField(const std::string& field) const {
    if (field.find(',') != std::string::npos || 
        field.find('"') != std::string::npos || 
        field.find('\n') != std::string::npos) {
        std::string escaped = "\"";
        for (char c : field) {
            if (c == '"') escaped += "\"\"";
            else escaped += c;
        }
        escaped += "\"";
        return escaped;
    }
    return field;
}

bool POIIOInterface::validatePOIData(const POI& poi) const {
    // Basic validation
    if (!std::isfinite(poi.getReferenceX()) || !std::isfinite(poi.getReferenceY())) {
        return false;
    }
    
    if (poi.hasDisplacement()) {
        if (!std::isfinite(poi.getDisplacementU()) || !std::isfinite(poi.getDisplacementV())) {
            return false;
        }
        
        // Check for reasonable displacement bounds
        if (std::abs(poi.getDisplacementU()) > 1000.0 || std::abs(poi.getDisplacementV()) > 1000.0) {
            return false;
        }
    }
    
    if (poi.hasZNCC()) {
        double zncc = poi.getZNCC();
        if (!std::isfinite(zncc) || zncc < -1.0 || zncc > 1.0) {
            return false;
        }
    }
    
    return true;
}

bool POIIOInterface::validateFilename(const std::string& filename) const {
    return !filename.empty() && filename.find_first_of("<>:\"|?*") == std::string::npos;
}

// === POIIOEnhanced Implementation ===

bool POIIOEnhanced::exportPOIs(const POIManager& poiManager,
                               const std::string& filename,
                               ExportFormat format,
                               const ExportOptions& options) {
    // Cast to enhanced options for additional features
    EnhancedExportOptions enhancedOptions;
    enhancedOptions.includeInvalid = options.includeInvalid;
    enhancedOptions.includeMetadata = options.includeMetadata;
    enhancedOptions.includeStatistics = options.includeStatistics;
    enhancedOptions.precision = options.precision;
    enhancedOptions.delimiter = options.delimiter;
    enhancedOptions.encoding = options.encoding;
    
    return exportWithStrainData(poiManager, filename, format, enhancedOptions);
}

bool POIIOEnhanced::exportWithStrainData(const POIManager& poiManager,
                                         const std::string& filename,
                                         ExportFormat format,
                                         const EnhancedExportOptions& options) {
    if (format != CSV_FORMAT) {
        // For non-CSV formats, use base implementation
        return POIIOInterface::exportPOIs(poiManager, filename, format, options);
    }
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    file << std::fixed << std::setprecision(options.precision);
    
    // Write enhanced header
    file << generateEnhancedCSVHeader(options) << std::endl;
    
    // Write data
    for (size_t i = 0; i < poiManager.size(); ++i) {
        auto poi = poiManager.getPOI(i);
        
        if (!options.includeInvalid && !poi->isValid()) {
            continue;
        }
        
        file << poiToEnhancedCSVRow(*poi, options) << std::endl;
    }
    
    file.close();
    return true;
}

bool POIIOEnhanced::exportStatistics(const std::string& filename,
                                     const std::map<std::string, double>& stats_data) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    file << "# DIC Processing Statistics" << std::endl;
    file << "timestamp," << metadata_.timestamp << std::endl;
    
    for (const auto& pair : stats_data) {
        file << pair.first << "," << std::fixed << std::setprecision(6) << pair.second << std::endl;
    }
    
    file.close();
    return true;
}

std::unique_ptr<POIManager> POIIOEnhanced::importPOIs(const std::string& filename,
                                                      ImportFormat format,
                                                      const ImportOptions& options) {
    // Use base implementation for now
    return POIIOInterface::importPOIs(filename, format, options);
}

std::string POIIOEnhanced::generateEnhancedCSVHeader(const EnhancedExportOptions& options) const {
    std::string header = POI::getCSVHeader();
    
    if (options.includeShapeParameters) {
        header += ",shape_params";
    }
    
    if (options.includeNeighborInfo) {
        header += ",neighbor_count";
    }
    
    return header;
}

std::string POIIOEnhanced::poiToEnhancedCSVRow(const POI& poi, const EnhancedExportOptions& options) const {
    std::string row = poi.toCSVRow();
    
    if (options.includeShapeParameters && poi.hasShapeParameters()) {
        cv::Mat params = poi.getShapeParameters();
        row += ",\"[";
        for (int i = 0; i < params.rows; ++i) {
            if (i > 0) row += ",";
            row += std::to_string(params.at<double>(i, 0));
        }
        row += "]\"";
    } else if (options.includeShapeParameters) {
        row += ",";
    }
    
    if (options.includeNeighborInfo) {
        row += "," + std::to_string(poi.getNeighborCount());
    }
    
    return row;
}

// === POIIOFactory Implementation ===

std::unique_ptr<POIIOInterface> POIIOFactory::create(IOType type) {
    switch (type) {
        case STANDARD_IO:
            return std::unique_ptr<POIIOInterface>(new POIIOInterface());
        case ENHANCED_IO:
            return std::unique_ptr<POIIOInterface>(new POIIOEnhanced());
        default:
            return std::unique_ptr<POIIOInterface>(new POIIOInterface());
    }
}