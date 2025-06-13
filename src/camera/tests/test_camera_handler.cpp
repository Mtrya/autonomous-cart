#include "../camera_handler.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>

// Draw detections on image and save
void saveDetectionResults(const cv::Mat &frame, const DetectionResult &detections,
                          const std::string &filename)
{
    cv::Mat output_frame = frame.clone();

    for (const auto &det : detections)
    {
        // Draw bounding box
        cv::rectangle(output_frame, det.box, cv::Scalar(0, 255, 0), 2);

        // Prepare label
        std::ostringstream label_stream;
        label_stream << det.class_name << " " << std::fixed << std::setprecision(2) << det.confidence;
        std::string label = label_stream.str();

        // Draw label background
        int baseline;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::Point label_pos(det.box.x, det.box.y - 10);

        cv::rectangle(output_frame, label_pos,
                      cv::Point(label_pos.x + text_size.width, label_pos.y - text_size.height - baseline),
                      cv::Scalar(0, 255, 0), cv::FILLED);

        // Draw label text
        cv::putText(output_frame, label, cv::Point(label_pos.x, label_pos.y - baseline),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    cv::imwrite(filename, output_frame);
    std::cout << "📸 Saved " << filename << " with " << detections.size() << " detections" << std::endl;
}

int main()
{
    std::cout << "=== Testing Camera Handler ===" << std::endl;

    // Initialize camera handler
    CameraHandler camera;

    if (!camera.initCamera(0, 640, 480, 30, "/home/optics/autonomous_cart/models/yolov8n.onnx", 0.5f, 0.3f))
    {
        std::cerr << "Failed to initialize camera handler" << std::endl;
        return -1;
    }

    std::cout << "\nStarting detection test..." << std::endl;

    int test_frames = 50;
    int save_every = 10;
    std::vector<double> inference_times;

    for (int frame_count = 1; frame_count <= test_frames; frame_count++)
    {
        auto cycle_start = std::chrono::high_resolution_clock::now();

        // 1. Grab frame
        cv::Mat frame = camera.grabFrame();
        if (frame.empty())
        {
            std::cerr << "Failed to grab frame " << frame_count << std::endl;
            continue;
        }

        // 2. Run detection - the magic happens here!
        auto inference_start = std::chrono::high_resolution_clock::now();
        DetectionResult detections = camera.runModel(frame);
        auto inference_end = std::chrono::high_resolution_clock::now();

        // Calculate timing
        auto inference_time = std::chrono::duration_cast<std::chrono::microseconds>(
                                  inference_end - inference_start)
                                  .count() /
                              1000.0;
        inference_times.push_back(inference_time);

        auto cycle_end = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(
                              cycle_end - cycle_start)
                              .count() /
                          1000.0;

        // Save detection results periodically
        if (frame_count % save_every == 0)
        {
            std::ostringstream filename;
            filename << "detection_640x480_frame_" << std::setfill('0') << std::setw(3) << frame_count << ".jpg";
            saveDetectionResults(frame, detections, filename.str());
        }

        // Progress report
        std::cout << "Frame " << frame_count << "/" << test_frames
                  << " | Inference: " << std::fixed << std::setprecision(1) << inference_time << "ms"
                  << " | Total: " << std::fixed << std::setprecision(1) << total_time << "ms"
                  << " | FPS: " << std::fixed << std::setprecision(1) << (1000.0 / total_time)
                  << " | Detections: " << detections.size() << std::endl;
    }

    // Calculate performance statistics
    if (!inference_times.empty())
    {
        double total = 0, min_time = inference_times[0], max_time = inference_times[0];
        for (double time : inference_times)
        {
            total += time;
            min_time = std::min(min_time, time);
            max_time = std::max(max_time, time);
        }

        double avg_inference = total / inference_times.size();
        double avg_fps = 1000.0 / avg_inference; // Approximate FPS based on inference only

        std::cout << "\n=== Camera Handler Performance ===" << std::endl;
        std::cout << "Average Inference Time: " << std::fixed << std::setprecision(2) << avg_inference << " ms" << std::endl;
        std::cout << "Min Inference Time: " << std::fixed << std::setprecision(2) << min_time << " ms" << std::endl;
        std::cout << "Max Inference Time: " << std::fixed << std::setprecision(2) << max_time << " ms" << std::endl;
        std::cout << "Theoretical Max FPS: " << std::fixed << std::setprecision(1) << avg_fps << std::endl;

        std::cout << "\n🎯 Performance Assessment:" << std::endl;
        if (avg_fps >= 40.0)
        {
            std::cout << "✅ EXCELLENT: Ready for 40+ FPS autonomous navigation!" << std::endl;
        }
        else if (avg_fps >= 30.0)
        {
            std::cout << "🔥 VERY GOOD: " << avg_fps << " FPS - autonomous ready!" << std::endl;
        }
        else
        {
            std::cout << "⚠️  OPTIMIZATION NEEDED: " << avg_fps << " FPS" << std::endl;
        }
    }

    std::cout << "\nCamera Handler test completed!" << std::endl;
    return 0;
}