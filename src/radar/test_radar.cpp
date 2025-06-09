#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <signal.h>

#include "sl_lidar.h"
#include "sl_lidar_driver.h"

using namespace sl;
using namespace cv;

struct Point2D
{
    float x, y;
    Point2D(float x = 0, float y = 0) : x(x), y(y) {}
};

class RadarPointCloudCollector
{
private:
    std::vector<Point2D> accumulated_points;
    int circles_completed = 0;
    int target_circles = 5;
    bool last_sync_state = false;

    static constexpr int IMAGE_SIZE = 800;
    static constexpr float MAX_RANGE_MM = 6000.0f;
    static constexpr float SCALE = IMAGE_SIZE / (2.0f * MAX_RANGE_MM);

public:
    RadarPointCloudCollector(int circles_per_image = 50) : target_circles(circles_per_image) {}

    void processScanNode(const sl_lidar_response_measurement_node_hq_t &node)
    {
        bool current_sync = (node.flag & SL_LIDAR_RESP_HQ_FLAG_SYNCBIT) != 0;

        if (current_sync && !last_sync_state)
        {
            circles_completed++;
            std::cout << "Completed rotation: " << circles_completed << std::endl;

            if (circles_completed >= target_circles)
            {
                savePointCloudAsImage();
                resetAccumulation();
            }
        }
        last_sync_state = current_sync;

        // Convert polar to Cartesian coordinates
        float angle_deg = (node.angle_z_q14 * 90.0f) / 16384.0f;
        float distance_mm = node.dist_mm_q2 / 4.0f;
        int quality = node.quality >> SL_LIDAR_RESP_MEASUREMENT_QUALITY_SHIFT;

        // Filter out low quality measurements and invalid distances
        if (quality > 10 && distance_mm > 150.0f && distance_mm < MAX_RANGE_MM)
        {
            float angle_rad = angle_deg * M_PI / 180.0f;
            float x = distance_mm * cos(angle_rad);
            float y = distance_mm * sin(angle_rad);

            accumulated_points.emplace_back(x, y);
        }
    }

    void savePointCloudAsImage()
    {
        std::cout << "Saving point cloud with " << accumulated_points.size() << " points..." << std::endl;

        Mat image = Mat::zeros(IMAGE_SIZE, IMAGE_SIZE, CV_8UC3);

        for (const auto &point : accumulated_points)
        {
            int img_x = static_cast<int>(point.x * SCALE + IMAGE_SIZE / 2);
            int img_y = static_cast<int>(-point.y * SCALE + IMAGE_SIZE / 2); // Flip Y

            if (img_x >= 0 && img_x < IMAGE_SIZE && img_y >= 0 && img_y < IMAGE_SIZE)
            {
                // Draw point as small circle (white for obstacles)
                circle(image, Point(img_x, img_y), 1, Scalar(255, 255, 255), -1);
            }
        }

        // Draw center origin
        circle(image, Point(IMAGE_SIZE / 2, IMAGE_SIZE / 2), 5, Scalar(0, 0, 255), -1);

        // Draw range circles for reference
        for (int range_m = 1; range_m <= 6; range_m++)
        {
            int radius = static_cast<int>(range_m * 1000.0f * SCALE);
            circle(image, Point(IMAGE_SIZE / 2, IMAGE_SIZE / 2), radius, Scalar(64, 64, 64), 1);
        }

        // Save image with timestamp
        std::string filename = "pointcloud_" + getCurrentTimestamp() + ".png";
        imwrite(filename, image);
        std::cout << "Saved: " << filename << std::endl;
    }

private:
    void resetAccumulation()
    {
        accumulated_points.clear();
        circles_completed = 0;
    }

    std::string getCurrentTimestamp()
    {
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        char buffer[64];
        std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &tm);
        return std::string(buffer);
    }
};

// Global variables for signal handling
bool ctrl_c_pressed = false;
void ctrlc(int)
{
    ctrl_c_pressed = true;
}

int main(int argc, const char *argv[])
{
    const char *serial_port = "/dev/ttyUSB0";
    sl_u32 baudrate = 115200;

    if (argc >= 2)
    {
        serial_port = argv[1];
    }
    if (argc >= 3)
    {
        baudrate = strtoul(argv[2], NULL, 10);
    }

    std::cout << "Radar Point Cloud Collector" << std::endl;
    std::cout << "Connecting to: " << serial_port << " @ " << baudrate << std::endl;

    // Create LIDAR driver
    ILidarDriver *drv = *createLidarDriver();
    if (!drv)
    {
        std::cerr << "Failed to create LIDAR driver" << std::endl;
        return -1;
    }

    // Create serial channel and connect
    IChannel *channel = *createSerialPortChannel(serial_port, baudrate);
    sl_result op_result = drv->connect(channel);

    if (SL_IS_FAIL(op_result))
    {
        std::cerr << "Failed to connect to LIDAR: " << std::hex << op_result << std::endl;
        delete drv;
        return -1;
    }

    // Get device info
    sl_lidar_response_device_info_t devinfo;
    op_result = drv->getDeviceInfo(devinfo);
    if (SL_IS_OK(op_result))
    {
        std::cout << "LIDAR Connected - Firmware: " << (devinfo.firmware_version >> 8)
                  << "." << (devinfo.firmware_version & 0xFF) << std::endl;
    }

    // Check health
    sl_lidar_response_device_health_t healthinfo;
    op_result = drv->getHealth(healthinfo);
    if (SL_IS_FAIL(op_result) || healthinfo.status == SL_LIDAR_STATUS_ERROR)
    {
        std::cerr << "LIDAR health check failed" << std::endl;
        delete drv;
        return -1;
    }

    // Setup signal handler
    signal(SIGINT, ctrlc);

    // Start scanning
    drv->setMotorSpeed();
    drv->startScan(0, 1);

    // Create point cloud collector
    RadarPointCloudCollector collector(5);

    std::cout << "Starting point cloud collection..." << std::endl;

    while (!ctrl_c_pressed)
    {
        sl_lidar_response_measurement_node_hq_t nodes[8192];
        size_t count = sizeof(nodes) / sizeof(nodes[0]);

        op_result = drv->grabScanDataHq(nodes, count);
        if (SL_IS_OK(op_result))
        {
            drv->ascendScanData(nodes, count);

            // Process each measurement node
            for (size_t i = 0; i < count; ++i)
            {
                collector.processScanNode(nodes[i]);
            }
        }
        else
        {
            std::cerr << "Failed to grab scan data: " << std::hex << op_result << std::endl;
        }
    }

    // Cleanup
    std::cout << "\nShutting down..." << std::endl;
    drv->stop();
    drv->setMotorSpeed(0);
    delete drv;

    return 0;
}