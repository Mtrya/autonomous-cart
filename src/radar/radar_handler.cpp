#include "radar_handler.hpp"
#include <iostream>
#include <algorithm>
#include <random>

// >>> Constructor/Destructor >>>
RadarHandler::RadarHandler(size_t max_buffer_size)
    : driver_(nullptr), channel_(nullptr), is_connected_(false), is_scanning_(false),
      max_buffer_size_(max_buffer_size), rotation_count_(0), last_sync_state_(false),
      should_stop_scanning_(false), min_quality_(10), min_range_mm_(150.0f), max_range_mm_(6000.0f),
      detection_inlier_threshold_(100.0f), detection_max_iterations_(1000)
{
    std::cout << "RadarHandler initialized with buffer size: " << max_buffer_size_ << std::endl;
}

RadarHandler::~RadarHandler()
{
    disconnect();
}
// <<< End Constructor/Destructor <<<

// >>> Connection Management >>>
bool RadarHandler::connect(const std::string &serial_port, sl_u32 baudrate)
{
    if (is_connected_)
    {
        std::cout << "Already connected to LIDAR" << std::endl;
        return true;
    }

    std::cout << "Connecting to LIDAR on " << serial_port << " @ " << baudrate << std::endl;

    // Create LIDAR driver
    driver_ = *sl::createLidarDriver();
    if (!driver_)
    {
        std::cerr << "Failed to create LIDAR driver" << std::endl;
        return false;
    }

    // Create serial channel and connect
    channel_ = *sl::createSerialPortChannel(serial_port.c_str(), baudrate);
    sl_result op_result = driver_->connect(channel_);

    if (SL_IS_FAIL(op_result))
    {
        std::cerr << "Failed to connect to LIDAR: 0x" << std::hex << op_result << std::endl;
        delete driver_;
        driver_ = nullptr;
        return false;
    }

    // Get device info
    sl_lidar_response_device_info_t devinfo;
    op_result = driver_->getDeviceInfo(devinfo);
    if (SL_IS_OK(op_result))
    {
        std::cout << "LIDAR Connected - Firmware: " << (devinfo.firmware_version >> 8)
                  << "." << (devinfo.firmware_version & 0xFF) << std::endl;
    }

    // Check health
    sl_lidar_response_device_health_t healthinfo;
    op_result = driver_->getHealth(healthinfo);
    if (SL_IS_FAIL(op_result) || healthinfo.status == SL_LIDAR_STATUS_ERROR)
    {
        std::cerr << "LIDAR health check failed" << std::endl;
        delete driver_;
        driver_ = nullptr;
        return false;
    }

    is_connected_ = true;
    std::cout << "LIDAR connection successful" << std::endl;
    return true;
}

void RadarHandler::disconnect()
{
    if (is_scanning_)
    {
        stopScanning();
    }
    if (is_connected_ && driver_)
    {
        driver_->stop();
        driver_->setMotorSpeed(0);
        delete driver_;
        driver_ = nullptr;
        is_connected_ = false;
        std::cout << "LIDAR disconnected" << std::endl;
    }
}
// <<< End Connection Management <<<

// >>> Scanning Control >>>
bool RadarHandler::startScanning()
{
    if (!is_connected_)
    {
        std::cerr << "Cannot start scanning: LIDAR not connected" << std::endl;
        return false;
    }

    if (is_scanning_)
    {
        std::cout << "Already scanning" << std::endl;
        return true;
    }

    // Start motor and scanning
    driver_->setMotorSpeed();
    sl_result op_result = driver_->startScan(0, 1);
    if (SL_IS_FAIL(op_result))
    {
        std::cerr << "Failed to start scanning: 0x" << std::hex << op_result << std::endl;
        return false;
    }

    should_stop_scanning_ = false;
    is_scanning_ = true;
    scan_thread_ = std::thread(&RadarHandler::scanLoop, this);

    std::cout << "LIDAR scanning started" << std::endl;
    return true;
}

void RadarHandler::stopScanning()
{
    if (!is_scanning_)
    {
        return;
    }
    std::cout << "Stopping LIDAR scanning..." << std::endl;

    should_stop_scanning_ = true;
    is_scanning_ = false;

    // Wait for thread to finish
    if (scan_thread_.joinable())
    {
        scan_thread_.join();
    }

    if (driver_)
    {
        driver_->stop();
        driver_->setMotorSpeed(0);
    }
    std::cout << "LIDAR scanning stopped" << std::endl;
}
// <<< End Scanning Control <<<

// >>> Data Access >>>
std::vector<RadarPoint> RadarHandler::getRecentPoints(size_t max_points) const
{
    std::lock_guard<std::mutex> lock(buffer_mutex_);

    if (max_points == 0 || max_points >= point_buffer_.size())
    {
        return std::vector<RadarPoint>(point_buffer_.begin(), point_buffer_.end());
    }

    auto start_it = point_buffer_.end() - max_points;
    return std::vector<RadarPoint>(start_it, point_buffer_.end());
}

std::vector<RadarPoint> RadarHandler::getPointsInRange(float min_range_mm, float max_range_mm) const
{
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    std::vector<RadarPoint> filtered_points;

    for (const auto &point : point_buffer_)
    {
        if (point.range_mm >= min_range_mm && point.range_mm <= max_range_mm)
        {
            filtered_points.push_back(point);
        }
    }

    return filtered_points;
}

std::vector<RadarPoint> RadarHandler::getPointsInSector(float min_bearing_deg, float max_bearing_deg) const
{
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    std::vector<RadarPoint> filtered_points;

    // Normalize angles to [0, 360)
    while (min_bearing_deg < 0)
        min_bearing_deg += 360;
    while (max_bearing_deg < 0)
        max_bearing_deg += 360;
    while (min_bearing_deg >= 360)
        min_bearing_deg -= 360;
    while (max_bearing_deg >= 360)
        max_bearing_deg -= 360;

    for (const auto &point : point_buffer_)
    {
        float bearing = point.bearing_deg;
        while (bearing < 0)
            bearing += 360;
        while (bearing >= 360)
            bearing -= 360;

        // Handle sector crossing 0°
        if (min_bearing_deg <= max_bearing_deg)
        {
            if (bearing >= min_bearing_deg && bearing <= max_bearing_deg)
            {
                filtered_points.push_back(point);
            }
        }
        else
        {
            if (bearing >= min_bearing_deg || bearing <= max_bearing_deg)
            {
                filtered_points.push_back(point);
            }
        }
    }

    return filtered_points;
}
// <<< End Data Access <<<

// >>> Statistics >>>
size_t RadarHandler::getPointCount() const
{
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    return point_buffer_.size();
}

double RadarHandler::getAveragePointsPerRotation() const
{
    if (rotation_count_ == 0)
        return 0.0;
    return static_cast<double>(getPointCount()) / static_cast<double>(rotation_count_);
}

void RadarHandler::clearPoints()
{
    std::lock_guard<std::mutex> lock(buffer_mutex_);
    point_buffer_.clear();
    rotation_count_ = 0;
    std::cout << "Point buffer cleared" << std::endl;
}
// <<< End Statistics <<<

// >>> Quality Filtering >>>
void RadarHandler::setRangeFilter(float min_range_mm, float max_range_mm)
{
    if (min_range_mm >= max_range_mm)
    {
        std::cerr << "Invalid range filter: min >= max" << std::endl;
        return;
    }

    min_range_mm_ = min_range_mm;
    max_range_mm_ = max_range_mm;
    std::cout << "Range filter set: " << min_range_mm_ << " - " << max_range_mm_ << " mm" << std::endl;
}
// <<< End Quality Filtering <<<

// >>> Internal Methods >>>
void RadarHandler::scanLoop()
{
    std::cout << "Scan loop started" << std::endl;

    while (!should_stop_scanning_ && is_connected_)
    {
        size_t count = 8192;

        sl_result op_result = driver_->grabScanDataHq(scan_buffer_, count);
        if (SL_IS_OK(op_result))
        {
            driver_->ascendScanData(scan_buffer_, count);

            for (size_t i = 0; i < count; ++i)
            {
                if (isPointValid(scan_buffer_[i]))
                {
                    addPoint(scan_buffer_[i]);
                }
            }
        }
        else
        {
            std::cerr << "Failed to grab scan data: 0x" << std::hex << op_result << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); // brief pause before retrying
        }
    }

    std::cout << "Scan loop ended" << std::endl;
}

void RadarHandler::addPoint(const sl_lidar_response_measurement_node_hq_t &node)
{
    // Track rotations using sync bit
    bool current_sync = (node.flag & SL_LIDAR_RESP_HQ_FLAG_SYNCBIT) != 0;
    if (current_sync && !last_sync_state_)
    {
        rotation_count_++;
    }
    last_sync_state_ = current_sync;

    // Convert LIDAR data to RadarPoint
    float angle_deg = (node.angle_z_q14 * 90.0f) / 16384.0f;
    float distance_mm = node.dist_mm_q2 / 4.0f;
    int quality = node.quality >> SL_LIDAR_RESP_MEASUREMENT_QUALITY_SHIFT;

    RadarPoint point(distance_mm, angle_deg, quality);

    // Thread-safe addition to buffer
    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        point_buffer_.push_back(point);

        // Maintain buffer size limit
        while (point_buffer_.size() > max_buffer_size_)
        {
            point_buffer_.pop_front();
        }
    }
}

bool RadarHandler::isPointValid(const sl_lidar_response_measurement_node_hq_t &node) const
{
    float distance_mm = node.dist_mm_q2 / 4.0f;
    int quality = node.quality >> SL_LIDAR_RESP_MEASUREMENT_QUALITY_SHIFT;

    // Apply quality and range filters
    return (quality >= min_quality_ &&
            distance_mm >= min_range_mm_ &&
            distance_mm <= max_range_mm_);
}
// <<< Internal Methods <<<

// >>> Core Detection Methods >>>
Square RadarHandler::buildSquareFromTwoPoints(const RadarPoint &p1, const RadarPoint &p2)
{
    // Note: two points should be adjacent in deque to ensure they are likely on one edge
    float edge_dx = p2.forward_distance - p1.forward_distance;
    float edge_dy = p2.lateral_distance - p1.lateral_distance;
    float edge_length = sqrt(edge_dx * edge_dx + edge_dy * edge_dy);
    edge_dx /= edge_length;
    edge_dy /= edge_length;

    float edge_mid_forward = (p1.forward_distance + p2.forward_distance) / 2.0f;
    float edge_mid_lateral = (p1.lateral_distance + p2.lateral_distance) / 2.0f;

    float perp_dx = -edge_dy;
    float perp_dy = edge_dx;
    float dot_product = -perp_dx * edge_mid_forward - perp_dy * edge_mid_lateral;
    if (dot_product < 0)
    {
        // Flip perpendicular to point toward radar
        perp_dx = -perp_dx;
        perp_dy = -perp_dy;
    }

    float center_forward = edge_mid_forward + perp_dx * HALF_SIZE;
    float center_lateral = edge_mid_lateral + perp_dy * HALF_SIZE;

    float orientation_deg = atan2(edge_dy, edge_dx) * 180.0f / M_PI;
    return Square{center_forward, center_lateral, orientation_deg};
}

float RadarHandler::pointToSquareEdgeDistance(const RadarPoint &point, const Square &square)
{
    // Transform point to square-centered coordinate system
    float cos_theta = cos(-square.orientation_deg * M_PI / 180.0f);
    float sin_theta = sin(-square.orientation_deg * M_PI / 180.0f);

    float dx = point.forward_distance - square.center_forward;
    float dy = point.lateral_distance - square.center_lateral;

    float local_x = dx * cos_theta - dy * sin_theta;
    float local_y = dx * sin_theta + dy * cos_theta;

    // Perpendicular distance to each edge line
    float dist_to_right = std::abs(local_x - HALF_SIZE); // Right edge (x = +2000)
    float dist_to_left = std::abs(local_x + HALF_SIZE);  // Left edge (x = -2000)
    float dist_to_front = std::abs(local_y - HALF_SIZE); // Front edge (y = +2000)
    float dist_to_back = std::abs(local_y + HALF_SIZE);  // Back edge (y = -2000)

    // Return distance to closest edge
    return std::min({dist_to_right, dist_to_left, dist_to_front, dist_to_back});
}

std::vector<RadarPoint> RadarHandler::findSquareInliers(const std::vector<RadarPoint> &points, const Square &square, float threshold)
{
    std::vector<RadarPoint> inliers;
    for (const auto &point : points)
    {
        if (pointToSquareEdgeDistance(point, square) <= threshold)
        {
            inliers.push_back(point);
        }
    }
    return inliers;
}

std::array<float, 4> RadarHandler::calculateRadarToEdgeDistances(const Square &square)
{
    // Transform radar position (0,0) to square-centered coordinate system
    float cos_theta = cos(-square.orientation_deg * M_PI / 180.0f);
    float sin_theta = sin(-square.orientation_deg * M_PI / 180.0f);

    float dx = -square.center_forward; // Radar to square center
    float dy = -square.center_lateral;

    float local_x = dx * cos_theta - dy * sin_theta;
    float local_y = dx * sin_theta + dy * cos_theta;

    // Distances from radar to each edge: [front, right, back, left]
    return {
        HALF_SIZE - local_y, // Front edge (y = +2000)
        HALF_SIZE - local_x, // Right edge (x = +2000)
        HALF_SIZE + local_y, // Back edge (y = -2000)
        HALF_SIZE + local_x  // Left edge (x = -2000)
    };
}

SquareDetectionResult RadarHandler::detectSquareBoundary(float inlier_threshold, int max_iterations)
{
    std::vector<RadarPoint> points = getRecentPoints();

    if (points.size() < 2)
    {
        std::cout << "Insufficient points for square detection: " << points.size() << std::endl;
        return SquareDetectionResult();
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, points.size() - 1 - OFFSET);

    Square best_square;
    size_t best_inlier_count = 0;
    std::vector<RadarPoint> best_inliers;

    for (int iter = 0; iter < max_iterations; ++iter)
    {
        int idx1 = dis(gen);
        int idx2 = idx1 + OFFSET; // ensure semi-adjacent points for "on-same-edge" assumption

        const RadarPoint &p1 = points[idx1];
        const RadarPoint &p2 = points[idx2];

        Square candidate_square = buildSquareFromTwoPoints(p1, p2);

        std::vector<RadarPoint> inliers = findSquareInliers(points, candidate_square, inlier_threshold);

        if (inliers.size() > best_inlier_count)
        {
            best_inlier_count = inliers.size();
            best_square = candidate_square;
            best_inliers = inliers;
        }
    }
    SquareDetectionResult result;
    if (best_inlier_count >= 8)
    { // Minimum points for valid square (2 per edge)
        result.square = best_square;
        result.inliers = best_inliers;
        result.num_inliers = best_inlier_count;
        result.fitness_score = static_cast<float>(best_inlier_count) / points.size();
        result.edge_distances = calculateRadarToEdgeDistances(best_square);
        result.valid = true;

        std::cout << "Square detected: " << best_inlier_count << " inliers, center("
                  << best_square.center_forward << ", " << best_square.center_lateral
                  << "), angle=" << best_square.orientation_deg << "°" << std::endl;
    }
    else
    {
        std::cout << "No valid square found (max inliers: " << best_inlier_count << ")" << std::endl;
    }

    return result;
}