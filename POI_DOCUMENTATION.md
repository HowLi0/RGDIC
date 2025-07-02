# POI (Point of Interest) Support in RGDIC

The RGDIC library now supports Point of Interest (POI) representation alongside the traditional matrix-based displacement fields. This provides enhanced flexibility for working with sparse data, better export capabilities, and improved integration with external tools.

## Overview

POI support is implemented through:

1. **POI Class**: Represents individual points with coordinates, displacement, correlation, and optional strain information
2. **POICollection Class**: Manages collections of POIs with filtering, statistics, and export capabilities  
3. **Enhanced DisplacementResult**: Supports both matrix and POI formats with conversion functions
4. **Visualization Functions**: Display POI correspondences and statistics

## Basic Usage

### 1. Standard RGDIC Analysis with POI Conversion

```cpp
#include "rgdic.h"
#include "poi.h"
#include "common_functions.h"

int main() {
    // Load images and create ROI
    cv::Mat refImage = cv::imread("reference.png", cv::IMREAD_GRAYSCALE);
    cv::Mat defImage = cv::imread("deformed.png", cv::IMREAD_GRAYSCALE);
    cv::Mat roi = createManualROI(refImage);
    
    // Create RGDIC instance and compute displacement
    auto dic = std::make_unique<RGDIC>();
    auto result = dic->compute(refImage, defImage, roi);
    
    // Convert to POI format
    result.enablePOIs(true);
    result.convertMatrixToPOIs();
    
    // Work with POI data
    std::cout << "Total POIs: " << result.pois.size() << std::endl;
    std::cout << "Valid POIs: " << result.pois.getValidCount() << std::endl;
    std::cout << "Mean correlation: " << result.pois.getMeanCorrelation() << std::endl;
    
    return 0;
}
```

### 2. POI Filtering and Analysis

```cpp
// Filter high-quality POIs
auto highQualityPOIs = result.pois.filterByCorrelation(0.9);
std::cout << "High quality POIs: " << highQualityPOIs.size() << std::endl;

// Filter POIs in specific region
cv::Rect region(100, 100, 200, 200);
auto regionalPOIs = result.pois.filterByRegion(region);
std::cout << "POIs in region: " << regionalPOIs.size() << std::endl;

// Get statistics
cv::Vec2f meanDisplacement = result.pois.getMeanDisplacement();
std::cout << "Mean displacement: (" << meanDisplacement[0] 
          << ", " << meanDisplacement[1] << ")" << std::endl;
```

### 3. Data Export

```cpp
// Export to different formats
result.pois.exportToCSV("results.csv");                    // Standard CSV
result.pois.exportToPOIFormat("results.poi");              // Custom POI format
result.pois.exportToMatlab("results.m");                   // MATLAB format

// Or use the DisplacementResult export functions
result.exportToCSV("displacement_results.csv");
result.exportToPOIFormat("displacement_results.poi");
```

### 4. Visualization

```cpp
// Display POI results on images
dic->displayPOIResults(refImage, defImage, result);

// Show correspondences between reference and deformed images
dic->displayPOICorrespondences(refImage, defImage, result.pois, 100);

// Display detailed statistics
dic->displayPOIStatistics(result.pois);
```

### 5. Working with Individual POIs

```cpp
// Create individual POI
rgdic::POI poi;
poi.leftCoord = cv::Point2f(100.5f, 200.3f);
poi.displacement = cv::Vec2f(2.1f, -1.5f);
poi.updateRightCoord();  // Calculates rightCoord from leftCoord + displacement
poi.correlation = 0.95;
poi.valid = true;

// Add strain information (optional)
poi.strain.exx = 0.001;
poi.strain.eyy = -0.0005;
poi.strain.exy = 0.0002;
poi.strain.computed = true;

// Serialize POI to string
std::string serialized = poi.toString();
std::cout << "POI data: " << serialized << std::endl;

// Deserialize from string
rgdic::POI reconstructed = rgdic::POI::fromString(serialized);
```

### 6. Collection Management

```cpp
// Create collection with metadata
rgdic::POICollection collection(cv::Size(512, 512), "Experimental Data Set A");

// Add POIs
collection.addPOI(poi);

// Bulk operations
for (const auto& poi : collection) {
    if (poi.valid && poi.correlation > 0.8) {
        // Process high-quality POI
        std::cout << "High quality POI at (" << poi.leftCoord.x 
                  << ", " << poi.leftCoord.y << ")" << std::endl;
    }
}

// Convert collection back to matrices if needed
cv::Mat u, v, cc, validMask;
collection.convertToMatrices(u, v, cc, validMask);
```

## Data Formats

### CSV Export Format

The CSV export includes the following columns:
- `left_x`, `left_y`: Reference image coordinates
- `right_x`, `right_y`: Deformed image coordinates  
- `u`, `v`: Displacement components
- `zncc`: Correlation coefficient
- `valid`: Validity flag (1 = valid, 0 = invalid)
- `exx`, `eyy`, `exy`: Strain components (if computed)

### POI Format

The custom POI format is a text file with:
- Header with metadata (image size, description, statistics)
- One POI per line with comma-separated values
- Comments start with '#'

### MATLAB Format

Exports arrays that can be directly loaded in MATLAB:
- `left_coords`: Nx2 array of reference coordinates
- `displacements`: Nx2 array of displacement vectors
- `correlations`: Nx1 array of correlation coefficients

## Backward Compatibility

All existing RGDIC functionality remains unchanged. POI support is additive:

- Existing `DisplacementResult.u`, `.v`, `.cc`, `.validMask` fields are preserved
- POI functionality is enabled explicitly via `enablePOIs(true)`
- Matrix and POI formats can be converted bidirectionally
- All existing visualization and export functions continue to work

## Performance Considerations

- POI format is more memory-efficient for sparse data
- Matrix format is more efficient for dense, regular grids
- Conversion between formats has minimal overhead
- CUDA support automatically works with POI-enabled results

## Integration with External Tools

POI data can be easily imported into:
- MATLAB/Octave (via .m export)
- Python (via CSV import with pandas)
- Excel/LibreOffice (via CSV)
- Custom analysis tools (via POI format)

The enhanced export capabilities make RGDIC results more accessible to the broader digital image correlation community.