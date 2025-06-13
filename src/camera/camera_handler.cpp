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

std::vector<float> CameraHandler::preprocessFrame(const cv::Mat &frame)
{
    std::vector<float> input_tensor_values;
    input_tensor_values.resize(3 * frame_height * frame_width);

    // Convert BGR to RGB and normalize to [0,1]
    cv::Mat rgb_frame;
    cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
    rgb_frame.convertTo(rgb_frame, CV_32F, 1.0 / 255.0);

    // Convert HWC to CHW format
    std::vector<cv::Mat> channels(3);
    cv::split(rgb_frame, channels);

    int channel_size = frame_height * frame_width;
    for (int c = 0; c < 3; c++)
    {
        std::memcpy(input_tensor_values.data() + c * channel_size, channels[c].data,
                    channel_size * sizeof(float));
    }

    return input_tensor_values;
}

DetectionResult CameraHandler::postprocessOutput(float *output_data,
                                                 const std::vector<int64_t> &output_shape,
                                                 int img_width, int img_height)
{
    DetectionResult detections;

    // YOLOv8 output format: [batch, 84, num_detections] where 84 = 4 box coords + 80 classes
    int num_classes = output_shape[1] - 4;
    int num_detections = output_shape[2];

    for (int i = 0; i < num_detections; i++)
    {
        // Extract box coordinates (center_x, center_y, width, height)
        float cx = output_data[0 * num_detections + i];
        float cy = output_data[1 * num_detections + i];
        float w = output_data[2 * num_detections + i];
        float h = output_data[3 * num_detections + i];

        // Find class with highest confidence
        float max_conf = 0;
        int class_id = 0;
        for (int c = 4; c < output_shape[1]; c++)
        {
            float conf = output_data[c * num_detections + i];
            if (conf > max_conf)
            {
                max_conf = conf;
                class_id = c - 4;
            }
        }

        if (max_conf > confidence_threshold_)
        {
            // Convert center coordinates to corner coordinates
            float x1 = cx - w / 2;
            float y1 = cy - h / 2;
            float x2 = cx + w / 2;
            float y2 = cy + h / 2;

            // Clamp to image boundaries
            x1 = std::max(0.0f, std::min(x1, (float)img_width));
            y1 = std::max(0.0f, std::min(y1, (float)img_height));
            x2 = std::max(0.0f, std::min(x2, (float)img_width));
            y2 = std::max(0.0f, std::min(y2, (float)img_height));

            cv::Rect box(cv::Point(x1, y1), cv::Point(x2, y2));
            std::string class_name = (class_id < class_names_.size()) ? class_names_[class_id] : "unknown";

            detections.emplace_back(box, max_conf, class_id, class_name);
        }
    }

    // Apply Non-Maximum Suppression
    std::vector<int> indices;
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    for (const auto &det : detections)
    {
        boxes.push_back(det.box);
        scores.push_back(det.confidence);
    }

    cv::dnn::NMSBoxes(boxes, scores, confidence_threshold_, nms_threshold_, indices);

    DetectionResult final_detections;
    for (int idx : indices)
    {
        final_detections.push_back(detections[idx]);
    }

    return final_detections;
}
// <<< End Core Functions <<<