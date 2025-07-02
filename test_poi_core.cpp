#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>

// Minimal POI structure for testing without OpenCV linking
struct TestPOI {
    float left_x, left_y;
    float right_x, right_y;
    float u, v;
    double correlation;
    bool valid;
    
    TestPOI(float lx = 0, float ly = 0, float dx = 0, float dy = 0, double corr = 0, bool v = false)
        : left_x(lx), left_y(ly), u(dx), v(dy), correlation(corr), valid(v) {
        right_x = left_x + u;
        right_y = left_y + v;
    }
    
    std::string toString() const {
        std::ostringstream oss;
        oss << left_x << "," << left_y << "," << right_x << "," << right_y << ","
            << u << "," << v << "," << correlation << "," << (valid ? 1 : 0);
        return oss.str();
    }
};

class TestPOICollection {
private:
    std::vector<TestPOI> pois;
    
public:
    void add(const TestPOI& poi) { pois.push_back(poi); }
    size_t size() const { return pois.size(); }
    
    size_t getValidCount() const {
        size_t count = 0;
        for (const auto& poi : pois) {
            if (poi.valid) count++;
        }
        return count;
    }
    
    double getMeanCorrelation() const {
        if (pois.empty()) return 0.0;
        double sum = 0.0;
        size_t count = 0;
        for (const auto& poi : pois) {
            if (poi.valid) {
                sum += poi.correlation;
                count++;
            }
        }
        return count > 0 ? sum / count : 0.0;
    }
    
    std::vector<TestPOI> filterByCorrelation(double threshold) const {
        std::vector<TestPOI> filtered;
        for (const auto& poi : pois) {
            if (poi.valid && poi.correlation >= threshold) {
                filtered.push_back(poi);
            }
        }
        return filtered;
    }
    
    bool exportToCSV(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) return false;
        
        file << "left_x,left_y,right_x,right_y,u,v,zncc,valid" << std::endl;
        for (const auto& poi : pois) {
            file << poi.toString() << std::endl;
        }
        return true;
    }
    
    bool importFromCSV(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) return false;
        
        std::string line;
        bool first = true;
        while (std::getline(file, line)) {
            if (first) { first = false; continue; } // Skip header
            if (line.empty()) continue;
            
            std::istringstream iss(line);
            std::string token;
            std::vector<std::string> tokens;
            
            while (std::getline(iss, token, ',')) {
                tokens.push_back(token);
            }
            
            if (tokens.size() >= 8) {
                TestPOI poi;
                poi.left_x = std::stof(tokens[0]);
                poi.left_y = std::stof(tokens[1]);
                poi.right_x = std::stof(tokens[2]);
                poi.right_y = std::stof(tokens[3]);
                poi.u = std::stof(tokens[4]);
                poi.v = std::stof(tokens[5]);
                poi.correlation = std::stod(tokens[6]);
                poi.valid = (std::stoi(tokens[7]) != 0);
                add(poi);
            }
        }
        return true;
    }
};

int main() {
    std::cout << "=== POI Core Functionality Test ===" << std::endl;
    
    // Test 1: Basic POI creation
    TestPOI poi1(10.5f, 20.3f, 1.6f, 1.5f, 0.95, true);
    
    std::cout << "Test POI created:" << std::endl;
    std::cout << "  Left: (" << poi1.left_x << ", " << poi1.left_y << ")" << std::endl;
    std::cout << "  Right: (" << poi1.right_x << ", " << poi1.right_y << ")" << std::endl;
    std::cout << "  Displacement: (" << poi1.u << ", " << poi1.v << ")" << std::endl;
    std::cout << "  Correlation: " << poi1.correlation << std::endl;
    std::cout << "  Serialized: " << poi1.toString() << std::endl;
    
    // Test 2: Collection operations
    TestPOICollection collection;
    
    // Add test data
    for (int i = 0; i < 15; ++i) {
        TestPOI poi(i * 2.0f, i * 1.5f, i * 0.1f, i * 0.05f, 
                   0.7 + (i % 10) * 0.03, (i % 3 != 0));
        collection.add(poi);
    }
    
    std::cout << "\nCollection Statistics:" << std::endl;
    std::cout << "  Total POIs: " << collection.size() << std::endl;
    std::cout << "  Valid POIs: " << collection.getValidCount() << std::endl;
    std::cout << "  Mean correlation: " << collection.getMeanCorrelation() << std::endl;
    
    // Test 3: Filtering
    auto filtered = collection.filterByCorrelation(0.9);
    std::cout << "  High quality POIs (>= 0.9): " << filtered.size() << std::endl;
    
    // Test 4: Export/Import
    bool exported = collection.exportToCSV("test_core.csv");
    std::cout << "\nExport test: " << (exported ? "Success" : "Failed") << std::endl;
    
    if (exported) {
        // Verify the file was created
        std::ifstream testFile("test_core.csv");
        if (testFile.is_open()) {
            std::cout << "  File created successfully" << std::endl;
            
            // Read first few lines to verify content
            std::string line;
            int lineCount = 0;
            while (std::getline(testFile, line) && lineCount < 3) {
                if (lineCount == 0) {
                    std::cout << "  Header: " << line << std::endl;
                } else if (lineCount == 1) {
                    std::cout << "  First data line: " << line << std::endl;
                }
                lineCount++;
            }
            testFile.close();
            
            // Test import
            TestPOICollection imported;
            bool imported_success = imported.importFromCSV("test_core.csv");
            std::cout << "\nImport test: " << (imported_success ? "Success" : "Failed") << std::endl;
            
            if (imported_success) {
                std::cout << "  Imported POIs: " << imported.size() << std::endl;
                std::cout << "  Original POIs: " << collection.size() << std::endl;
                std::cout << "  Valid imported: " << imported.getValidCount() << std::endl;
                
                // Verify data integrity
                double origMean = collection.getMeanCorrelation();
                double importMean = imported.getMeanCorrelation();
                bool correlationMatch = std::abs(origMean - importMean) < 1e-6;
                std::cout << "  Correlation preservation: " << (correlationMatch ? "Success" : "Failed") << std::endl;
                std::cout << "    Original mean: " << origMean << std::endl;
                std::cout << "    Imported mean: " << importMean << std::endl;
            }
        } else {
            std::cout << "  Error: Could not open exported file" << std::endl;
        }
    }
    
    std::cout << "\n=== Core Test Complete ===" << std::endl;
    
    return 0;
}