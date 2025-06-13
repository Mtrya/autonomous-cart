#include "../camera_handler.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    std::cout << "=== Simple Camera Test ===" << std::endl;

    // Test 1: Initialization
    std::cout << "Testing camera initialization..." << std::endl;
    CameraHandler camera;

    if (!camera.initCamera(0))
    {
        std::cerr << "❌ Camera initialization failed!" << std::endl;
        return -1;
    }
    std::cout << "✅ Camera initialized" << std::endl;

    // Test 2: Grab frames and show them
    std::cout << "Testing frame grabbing (press ESC to exit)..." << std::endl;

    for (int i = 0; i < 100; i++)
    { // Test 100 frames
        cv::Mat frame = camera.grabFrame();

        if (frame.empty())
        {
            std::cerr << "❌ Failed to grab frame " << i << std::endl;
            return -1;
        }

        cv::imshow("Camera Test", frame);

        int key = cv::waitKey(30);
        if (key == 27)
            break; // ESC to exit early

        if (i % 10 == 0)
        {
            std::cout << "Frame " << i << ": " << frame.cols << "x" << frame.rows << std::endl;
        }
    }

    std::cout << "✅ Frame grabbing test completed!" << std::endl;
    cv::destroyAllWindows();

    return 0;
}