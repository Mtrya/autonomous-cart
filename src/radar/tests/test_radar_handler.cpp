#include <iostream>
#include <vector>
#include <random>
#include <chrono>
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
            std::cout << "Collecting data for 3 seconds..." << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(3));

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
