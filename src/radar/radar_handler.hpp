#ifndef RADAR_HANDLER_HPP
#define RADAR_HANDLER_HPP

#include <deque>
#include <thread>
#include <atomic>
#include <chrono>
#include <mutex>
#include <vector>
#include <array>
#include <memory>
#include <string>
#include <cmath>

#include "sl_lidar.h"
#include "sl_lidar_driver.h"

struct RadarPoint
{
    float range_mm;         // distance from radar in millimeters
    float bearing_deg;      // angles in degrees
    float forward_distance; // component in radar's forward direction, forward(+)/backward(-)
    float lateral_distance; // component perpendicular to forward, right(+)/left(-)
    int quality;
    std::chrono::steady_clock::time_point timestamp;

    RadarPoint(float range = 0.0f, float bearing = 0.0f, int qual = 0)
        : range_mm(range), bearing_deg(bearing), quality(qual)
    {
        float bearing_rad = bearing * M_PI / 180.0f;
        forward_distance = range_mm * cos(bearing_rad);
        lateral_distance = range_mm * sin(bearing_rad);
        timestamp = std::chrono::steady_clock::now();
    }
};

struct Square
{
    float center_forward;
    float center_lateral;
    float orientation_deg;

    Square(float cf = 0, float cl = 0, float angle = 0)
        : center_forward(cf), center_lateral(cl), orientation_deg(angle) {}
};

struct SquareDetectionResult
{
    Square square;
    std::vector<RadarPoint> inliers;
    std::vector<RadarPoint> interior_outliers; // outliers inside square (objects)
    size_t num_inliers;
    float fitness_score;

    // Distance from radar to each square [front, right, back, left] in mm
    std::array<float, 4> edge_distances;

    bool valid;

    SquareDetectionResult() : num_inliers(0), fitness_score(0.0f), valid(false) {}
};

struct Object
{ // Object = Target or Obstacle
    float center_forward;
    float center_lateral;
    float radius;    // approximate radius
    int point_count; // number of points in this obstacle
    float density;   // points per unit area (quality metric)

    Object(float cf = 0, float cl = 0, float r = 0, int count = 0)
        : center_forward(cf), center_lateral(cl), radius(r), point_count(count)
    {
        density = count > 0 ? count / (M_PI * r * r + 1e-6f) : 0.0f;
    }
};

struct ObjectDetectionResult
{
    std::vector<Object> objects;
    int num_objects;
    float object_coverage;
    bool valid;

    ObjectDetectionResult() : num_objects(0), object_coverage(0.0f), valid(false) {}
};

struct ComprehensiveDetectionResult
{
    SquareDetectionResult square_result;
    ObjectDetectionResult object_result;
    bool pipeline_success;

    ComprehensiveDetectionResult() : pipeline_success(false) {}
};

class RadarHandler
{
public:
    RadarHandler(size_t max_buffer_size = 1024); // 1024 = approximately 1s
    ~RadarHandler();                             // ensures clean shutdown

    // Connection management
    bool connect(const std::string &serial_port = "/dev/ttyUSB0", sl_u32 baudrate = 115200);
    void disconnect();
    bool isConnected() const { return is_connected_; }

    // Scanning control
    bool startScanning();
    void stopScanning();
    bool isScanning() const { return is_scanning_; }

    // Data access
    std::vector<RadarPoint> getRecentPoints(size_t max_points = 0) const; // 0 for all points in queue
    std::vector<RadarPoint> getPointsInRange(float min_range_mm, float max_range_mm) const;
    std::vector<RadarPoint> getPointsInSector(float min_bearing_deg, float max_bearing_deg) const;

    // Statistics
    size_t getPointCount() const;
    size_t getRotationCount() const { return rotation_count_; }
    double getAveragePointsPerRotation() const;
    void clearPoints();

    // Quality filtering
    void setMinQuality(int min_quality) { min_quality_ = min_quality; }
    void setRangeFilter(float min_range_mm, float max_range_mm);
    int getMinQuality() const { return min_quality_; }
    std::pair<float, float> getRangeFilter() const { return {min_range_mm_, max_range_mm_}; }

    // Core Detection Methods
    SquareDetectionResult detectSquareBoundary(float inlier_threshold = 128.0f, int max_iterations = 1000);
    ObjectDetectionResult detectObjects(const std::vector<RadarPoint> &interior_outliers,
                                        int num_objects = 4, float min_object_radius = 200.0f);
    ComprehensiveDetectionResult runDetectionPipeline(int num_objects = 4, float square_inlier_threshold = 100.0f);

private:
    // LIDAR hardware
    sl::ILidarDriver *driver_;
    sl::IChannel *channel_;
    std::atomic<bool> is_connected_;
    std::atomic<bool> is_scanning_;

    // Data storage
    std::deque<RadarPoint> point_buffer_;
    size_t max_buffer_size_;
    mutable std::mutex buffer_mutex_;
    sl_lidar_response_measurement_node_hq_t scan_buffer_[8192]; // grabScanDataHq function in SLAMTEC LIDAR API requires a pre-allocated array

    // Background scanning
    std::thread scan_thread_;
    std::atomic<bool> should_stop_scanning_;

    // Statistics tracking
    std::atomic<size_t> rotation_count_;
    bool last_sync_state_;

    // Filtering parameters
    int min_quality_;
    float min_range_mm_;
    float max_range_mm_;

    // Internal methods
    void scanLoop();
    void addPoint(const sl_lidar_response_measurement_node_hq_t &node);
    bool isPointValid(const sl_lidar_response_measurement_node_hq_t &node) const;
    void pruneOldPoints();

    // Square Detection
    static constexpr float SIDE_LENGTH = 4000.0f;
    static constexpr float HALF_SIZE = 2000.0f; // 4m / 2 = 2000 mm
    static constexpr int OFFSET = 5;            // affects point selection strategy
    float detection_inlier_threshold_;
    int detection_max_iterations_;
    Square buildSquareFromTwoPoints(const RadarPoint &p1, const RadarPoint &p2);
    float pointToSquareEdgeDistance(const RadarPoint &point, const Square &square);
    std::vector<RadarPoint> findSquareInliers(const std::vector<RadarPoint> &points,
                                              const Square &square, float threshold);
    std::array<float, 4> calculateRadarToEdgeDistances(const Square &square);

    // Object Detection
    bool isPointInsideSquare(const RadarPoint &point, const Square &square);
    std::vector<std::vector<RadarPoint>> kMeansCluster(const std::vector<RadarPoint> &points, int k);
    Object computeObjectFromCluster(const std::vector<RadarPoint> &cluster);
};

#endif