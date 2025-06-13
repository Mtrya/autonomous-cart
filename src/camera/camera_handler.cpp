#include "camera_handler.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <thread>

// >>> Initialization Functions >>>
CameraHandler::CameraHandler()
    : camera_ready(false), model_ready(false), frame_width(640), frame_height(480), target_fps(30.0),
      confidence_threshold_(0.5f), nms_threshold_(0.3f)
{
    initializeClassNames();
}
CameraHandler::~CameraHandler() { release(); }

bool CameraHandler::initCamera(int camera_id)
{
    return initCamera(camera_id, 640, 480, 30, "/home/optics/autonomous_cart/models/yolov8n.onnx", 0.5f, 0.3f);
}

bool CameraHandler::initCamera(int camera_id, int width, int height, int fps,
                               const std::string &model_path, float confidence_threshold, float nms_threshold)
{
    frame_width = width;
    frame_height = height;
    target_fps = static_cast<double>(fps);
    model_path_ = model_path;
    confidence_threshold_ = confidence_threshold;
    nms_threshold_ = nms_threshold;

    // 1. Initialize camera hardware
    cap.open(camera_id);

    if (!cap.isOpened())
    {
        std::cerr << "Error: Cannot open camera " << camera_id << std::endl;
        return false;
    }

    configureCameraProperties();
    camera_ready = true;

    // 2. Initialize YOLO model
    if (!initializeModel(model_path))
    {
        std::cerr << "Error: Failed to load YOLO model" << std::endl;
        camera_ready = false;
        return false;
    }

    model_ready = true;

    std::cout << "Camera Handler initialized successfully!" << std::endl;
    std::cout << "   Camera: " << frame_width << "x" << frame_height << " @ " << target_fps << " FPS" << std::endl;
    std::cout << "   Model: " << model_path_ << std::endl;
    std::cout << "   Confidence: " << confidence_threshold_ << ", NMS: " << nms_threshold_ << std::endl;

    return true;
}
// <<< End Initialization Functions <<<

// >>> Internal Helper Functions >>>
void CameraHandler::release()
{
    if (cap.isOpened())
    {
        cap.release();
        std::cout << "Camera resources released." << std::endl;
    }
    camera_ready = false;
}

bool CameraHandler::initializeModel(const std::string &model_path)
{
    try
    {
        ort_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "camera_yolo");

        // Configure session options for performance
        session_options_ = std::make_unique<Ort::SessionOptions>();
        session_options_->SetIntraOpNumThreads(std::thread::hardware_concurrency());
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options_->SetExecutionMode(ExecutionMode::ORT_PARALLEL);
        session_options_->EnableCpuMemArena();
        session_options_->EnableMemPattern();

        // Load Model
        ort_session_ = std::make_unique<Ort::Session>(*ort_env_, model_path.c_str(), *session_options_);

        memory_info_ = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

        input_names_ = {"images"};
        output_names_ = {"output0"};

        // Default input shape for 640x480 resolution: [1,3,480,640]
        input_shape_ = {1, 3, frame_height, frame_width};

        initializeClassNames();

        std::cout << "   Model loaded with input shape: [1, 3, "
                  << frame_height << ", " << frame_width << "]" << std::endl;

        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "ONNX Runtime initialization failed: " << e.what() << std::endl;
        return false;
    }
}

void CameraHandler::initializeClassNames()
{
    class_names_ = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
}

void CameraHandler::configureCameraProperties()
{
    // Configure properties
    cap.set(cv::CAP_PROP_FRAME_WIDTH, frame_width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, frame_height);
    cap.set(cv::CAP_PROP_FPS, target_fps);

    // Additional optimizations suggested by Claude. Need testing
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25); // Reduce auto-exposure for consistency
}
// <<< End Internal Helper Functions

// >>> Core Functions <<<
cv::Mat CameraHandler::grabFrame()
{
    if (!camera_ready || !cap.isOpened())
    {
        std::cerr << "Warning: Camera not ready, returning empty frame" << std::endl;
        return cv::Mat();
    }

    cv::Mat frame;
    bool success = cap.read(frame);

    if (!success || frame.empty())
    {
        std::cerr << "Warning: Failed to capture frame" << std::endl;
        return cv::Mat();
    }
    return frame;
}

std::vector<Detection> CameraHandler::runModel(const cv::Mat &frame)
{
    if (!model_ready || frame.empty())
    {
        std::cerr << "Warning: Model not ready or empty frame" << std::endl;
        return DetectionResult();
    }

    // 1. Preprocess frame (no letterboxing needed for 640x480)
    std::vector<float> input_tensor_values = preprocessFrame(frame);

    // 2. Create input tensor
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        *memory_info_, input_tensor_values.data(), input_tensor_values.size(),
        input_shape_.data(), input_shape_.size());

    // 3. Run inference
    auto output_tensors = ort_session_->Run(Ort::RunOptions{nullptr},
                                            input_names_.data(), &input_tensor, 1, output_names_.data(), 1);

    // 4. Postprocess outputs
    float *output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    return postprocessOutput(output_data, output_shape, frame.cols, frame.rows);
}

std::vector<float> CameraHandler::preprocessFrame(const cv::Mat &frmae)
{
    // TODO
}

DetectionResult CameraHandler::postprocessOutput(float *output_data,
                                                 const std::vector<int64_t> &output_shape,
                                                 int img_width, int img_height)
{
    // TODO
}
// <<< End Core Functions <<<