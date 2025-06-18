#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <thread>
#include "../radar_handler.hpp"

// Test utilities
void printTestHeader(const std::string &test_name)
{
    std::cout << "\n=== " << test_name << " ===" << std::endl;
}

void printTestResult(const std::string &test_name, bool passed)
{
    std::cout << test_name << ": " << (passed ? "PASSED ✓" : "FAILED ✗") << std::endl;
}

// Test basic radar handler functionality
bool testRadarHandlerBasics()
{
    printTestHeader("Radar Handler Basic Functionality");

    RadarHandler radar(5000); // Small buffer for testing

    // Test initial state
    bool test1 = !radar.isConnected() && !radar.isScanning();
    printTestResult("Initial state", test1);

    // Test configuration
    radar.setMinQuality(15);
    radar.setRangeFilter(200.0f, 5000.0f);

    bool test2 = (radar.getMinQuality() == 15);
    auto range_filter = radar.getRangeFilter();
    bool test3 = (range_filter.first == 200.0f && range_filter.second == 5000.0f);

    printTestResult("Configuration", test2 && test3);

    // Test data access with empty buffer
    auto points = radar.getRecentPoints();
    bool test4 = points.empty();
    bool test5 = (radar.getPointCount() == 0);

    printTestResult("Empty buffer access", test4 && test5);

    return test1 && test2 && test3 && test4 && test5;
}

// Test radar connection (gracefully handle if no hardware)
bool testRadarConnection()
{
    printTestHeader("Radar Connection Test");

    RadarHandler radar;

    // Try to connect (this may fail if no hardware)
    bool connected = radar.connect("/dev/ttyUSB0", 115200);

    if (connected)
    {
        std::cout << "Hardware LIDAR detected and connected" << std::endl;

        // Test scanning
        bool scan_started = radar.startScanning();
        printTestResult("Start scanning", scan_started);

        if (scan_started)
        {
            std::cout << "\nCollecting data for 5 seconds..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(5));

            auto points = radar.getRecentPoints();
            std::cout << "Collected " << points.size() << " points" << std::endl;
            std::cout << "Completed " << radar.getRotationCount() << " rotations" << std::endl;

            radar.stopScanning();
        }

        radar.disconnect();
        return scan_started;
    }
    else
    {
        std::cout << "No hardware LIDAR detected (this is OK for testing)" << std::endl;
        return true; // Not a failure in test environment
    }
}

void savePointsToCSV(const std::vector<RadarPoint> &points, const std::string &filename)
{
    std::ofstream file(filename);
    file << "forward_distance,lateral_distance,range_mm,bearing_deg,quality,timestamp_ms\n";

    for (const auto &point : points)
    {
        auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                point.timestamp.time_since_epoch())
                                .count();

        file << std::fixed << std::setprecision(2)
             << point.forward_distance << ","
             << point.lateral_distance << ","
             << point.range_mm << ","
             << point.bearing_deg << ","
             << point.quality << ","
             << timestamp_ms << "\n";
    }

    std::cout << "Saved " << points.size() << " points to " << filename << std::endl;
}

void printPointStatistics(const std::vector<RadarPoint> &points)
{
    if (points.empty())
    {
        std::cout << "No points to analyze" << std::endl;
        return;
    }

    float min_range = points[0].range_mm, max_range = points[0].range_mm;
    float min_bearing = points[0].bearing_deg, max_bearing = points[0].bearing_deg;
    float sum_range = 0, sum_quality = 0;
    int min_quality = points[0].quality, max_quality = points[0].quality;

    for (const auto &point : points)
    {
        min_range = std::min(min_range, point.range_mm);
        max_range = std::max(max_range, point.range_mm);
        min_bearing = std::min(min_bearing, point.bearing_deg);
        max_bearing = std::max(max_bearing, point.bearing_deg);
        min_quality = std::min(min_quality, point.quality);
        max_quality = std::max(max_quality, point.quality);
        sum_range += point.range_mm;
        sum_quality += point.quality;
    }

    std::cout << "\n--- Point Statistics ---" << std::endl;
    std::cout << "Total points: " << points.size() << std::endl;
    std::cout << "Range: " << min_range << " - " << max_range << " mm (avg: "
              << (sum_range / points.size()) << ")" << std::endl;
    std::cout << "Bearing: " << min_bearing << "° - " << max_bearing << "°" << std::endl;
    std::cout << "Quality: " << min_quality << " - " << max_quality << " (avg: "
              << (sum_quality / points.size()) << ")" << std::endl;
}

void visualizePointDistribution(const std::vector<RadarPoint> &points)
{
    if (points.empty())
        return;

    // Create a simple text-based polar plot
    const int SECTORS = 36; // 10-degree sectors
    const int RANGES = 10;  // Range bins

    std::vector<std::vector<int>> grid(SECTORS, std::vector<int>(RANGES, 0));

    for (const auto &point : points)
    {
        int sector = static_cast<int>(point.bearing_deg / 10.0f) % SECTORS;
        int range_bin = std::min(RANGES - 1, static_cast<int>(point.range_mm / 600.0f)); // 600mm per bin
        if (sector >= 0 && sector < SECTORS && range_bin >= 0 && range_bin < RANGES)
        {
            grid[sector][range_bin]++;
        }
    }

    std::cout << "\n--- Point Distribution (Polar Grid) ---" << std::endl;
    std::cout << "Sectors: 10° each, Ranges: 600mm each" << std::endl;
    std::cout << "    ";
    for (int r = 0; r < RANGES; r++)
    {
        std::cout << std::setw(4) << (r * 600) << "mm";
    }
    std::cout << std::endl;

    for (int s = 0; s < SECTORS; s++)
    {
        std::cout << std::setw(3) << (s * 10) << "° ";
        for (int r = 0; r < RANGES; r++)
        {
            if (grid[s][r] == 0)
            {
                std::cout << "   .";
            }
            else if (grid[s][r] < 10)
            {
                std::cout << "   " << grid[s][r];
            }
            else
            {
                std::cout << std::setw(4) << grid[s][r];
            }
        }
        std::cout << std::endl;
    }
}

// Enhanced radar connection test with detailed diagnostics:
bool testRadarConnectionDetailed()
{
    printTestHeader("Detailed Radar Connection Test");

    RadarHandler radar;

    // Try to connect (this may fail if no hardware)
    bool connected = radar.connect("/dev/ttyUSB0", 115200);

    if (connected)
    {
        std::cout << "✓ Hardware LIDAR detected and connected" << std::endl;

        // Test scanning
        bool scan_started = radar.startScanning();
        printTestResult("Start scanning", scan_started);

        if (scan_started)
        {
            std::cout << "Collecting data..." << std::endl;

            // Monitor data collection in real-time
            for (int i = 0; i < 6; ++i)
            { // 6 iterations of 0.5 seconds each
                std::this_thread::sleep_for(std::chrono::milliseconds(500));

                auto points = radar.getRecentPoints();
                std::cout << "After " << (i + 1) * 0.5 << "s: "
                          << points.size() << " points, "
                          << radar.getRotationCount() << " rotations" << std::endl;

                // Check if we're getting any points at all
                if (i == 1 && points.empty())
                {
                    std::cout << "⚠ Warning: No points collected after 1 second" << std::endl;
                }
            }

            auto final_points = radar.getRecentPoints();
            std::cout << "\nFinal collection: " << final_points.size() << " points" << std::endl;
            std::cout << "Total rotations: " << radar.getRotationCount() << std::endl;
            std::cout << "Average points per rotation: " << radar.getAveragePointsPerRotation() << std::endl;

            if (!final_points.empty())
            {
                printPointStatistics(final_points);
                visualizePointDistribution(final_points);
                savePointsToCSV(final_points, "collected_points.csv");

                // Test square detection with real data
                std::cout << "\n--- Testing Square Detection on Real Data ---" << std::endl;
                auto square_result = radar.detectSquareBoundary(100.0f, 500);

                if (square_result.valid)
                {
                    std::cout << "✓ Square detected!" << std::endl;
                    std::cout << "  Center: (" << square_result.square.center_forward
                              << ", " << square_result.square.center_lateral << ")" << std::endl;
                    std::cout << "  Orientation: " << square_result.square.orientation_deg << "°" << std::endl;
                    std::cout << "  Inliers: " << square_result.num_inliers << std::endl;
                    std::cout << "  Edge distances: [" << square_result.edge_distances[0]
                              << ", " << square_result.edge_distances[1]
                              << ", " << square_result.edge_distances[2]
                              << ", " << square_result.edge_distances[3] << "]" << std::endl;
                }
                else
                {
                    std::cout << "✗ No square detected in real data" << std::endl;
                }
            }
            else
            {
                std::cout << "❌ No points collected - check LIDAR connection and permissions" << std::endl;
            }

            radar.stopScanning();
        }

        radar.disconnect();
        return scan_started && !radar.getRecentPoints().empty();
    }
    else
    {
        std::cout << "⚠ No hardware LIDAR detected (this is OK for testing)" << std::endl;
        return true; // Not a failure in test environment
    }
}

// Replace the original testRadarConnection with this detailed version in main():
// all_passed &= testRadarConnectionDetailed();

// The core function - detectSquareBoundary is not tested here because of the absence of real 4x4m field.
// To use detectSquareBoundary, simply "SquareDetectionResult result = radar.detectSquareBoundary()" should work,
// it will use most recent points in its point_buffer_ to fit a square using RANSAC algorithm and return a SquareDetectionResult object.
int main()
{
    std::cout << "=== Radar Handler Test Suite ===" << std::endl;
    std::cout << "Testing RadarHandler functionality..." << std::endl;

    bool all_passed = true;

    all_passed &= testRadarHandlerBasics();
    all_passed &= testRadarConnection();
    all_passed &= testRadarConnectionDetailed();

    std::cout << "\n=== Test Summary ===" << std::endl;
    if (all_passed)
    {
        std::cout << "All tests passed!" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "Some tests failed." << std::endl;
        return 1;
    }
}
