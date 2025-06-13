#ifndef CAMERA_HANDLER_HPP
#define CAMERA_HANDLER_HPP

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <memory>
#include <vector>
#include "detection.hpp"

class CameraHandler
{
public:
    CameraHandler();
    ~CameraHandler();

    // Initialize camera and YOLO model
    bool initCamera(int camera_id = 0);                            // initialize with default settings
    bool initCamera(int camera_id, int width, int height, int fps, // initialize with custom settings
                    const std::string &model_path = "/home/optics/autonomous_cart/models/yolov8n.onnx",
                    float confidence_threshold = 0.5f, float nms_threshold = 0.3f);

    // Core functionalities
    cv::Mat grabFrame();                            // grab a single frame
    DetectionResult runModel(const cv::Mat &frame); // run YOLO model on frame

    // Utility functions
    bool isReady() const { return camera_ready && model_ready && cap.isOpened(); } // check if camera is ready
    int getWidth() const { return frame_width; }
    int getHeight() const { return frame_height; }
    double getFPS() const { return target_fps; }
    float getConfidenceThreshold() const { return confidence_threshold_; }
    float getNMSThreshold() const { return nms_threshold_; }
    std::string getModelPath() const { return model_path_; }

    void release(); // cleanup reources

private:
    // Camera properties
    cv::VideoCapture cap;
    bool camera_ready;
    int frame_width;
    int frame_height;
    double target_fps;

    // YOLO model components
    std::unique_ptr<Ort::Env> ort_env_;
    std::unique_ptr<Ort::Session> ort_session_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;
    bool model_ready;

    // YOlO model parameters
    std::string model_path_;
    float confidence_threshold_;
    float nms_threshold_;
    std::vector<std::string> class_names_;

    // Model input/output info
    std::vector<const char *> input_names_;
    std::vector<const char *> output_names_;
    std::vector<int64_t> input_shape_;

    // Internal helper methods
    void configureCameraProperties();
    bool initializeModel(const std::string &model_path);
    void initializeClassNames();
    std::vector<float> preprocessFrame(const cv::Mat &frmae);
    DetectionResult postprocessOutput(float *output_data,
                                      const std::vector<int64_t> &output_shape,
                                      int img_width, int img_height);
};

#endif