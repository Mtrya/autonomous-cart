#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <thread>
#include <cstring>

const int INPUT_WIDTH = 640;
const int INPUT_HEIGHT = 640;
const float CONFIDENCE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.3;

const std::vector<std::string> CLASS_NAMES = {
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

struct Detection
{
    cv::Rect box;
    float confidence;
    int class_id;
};

struct LetterboxInfo
{
    float scale;
    float pad_x;
    float pad_y;
};

bool isModelFP16(Ort::Session &session)
{
    auto input_type_info = session.GetInputTypeInfo(0);
    auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    return tensor_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
}

// Helper function to convert float32 to float16
std::vector<uint16_t> convertToFP16(const std::vector<float> &fp32_data)
{
    std::vector<uint16_t> fp16_data(fp32_data.size());

    for (size_t i = 0; i < fp32_data.size(); ++i)
    {
        float value = fp32_data[i];

        // Handle special cases
        if (std::isnan(value))
        {
            fp16_data[i] = 0x7E00; // NaN in FP16
            continue;
        }
        if (std::isinf(value))
        {
            fp16_data[i] = value > 0 ? 0x7C00 : 0xFC00; // +/-Inf in FP16
            continue;
        }

        // Convert to FP16
        uint32_t f32_bits = *reinterpret_cast<const uint32_t *>(&value);
        uint32_t sign = (f32_bits >> 31) & 0x1;
        uint32_t exp = (f32_bits >> 23) & 0xFF;
        uint32_t mant = f32_bits & 0x7FFFFF;

        uint16_t fp16_value = 0;

        if (exp == 0)
        {
            // Zero or denormalized
            fp16_value = static_cast<uint16_t>(sign << 15);
        }
        else if (exp == 0xFF)
        {
            // Infinity or NaN
            fp16_value = static_cast<uint16_t>((sign << 15) | 0x7C00 | (mant ? 0x200 : 0));
        }
        else
        {
            // Normalized number
            int32_t new_exp = static_cast<int32_t>(exp) - 127 + 15;
            if (new_exp <= 0)
            {
                // Underflow to zero
                fp16_value = static_cast<uint16_t>(sign << 15);
            }
            else if (new_exp >= 31)
            {
                // Overflow to infinity
                fp16_value = static_cast<uint16_t>((sign << 15) | 0x7C00);
            }
            else
            {
                // Normal case
                uint32_t new_mant = mant >> 13;
                fp16_value = static_cast<uint16_t>((sign << 15) | (new_exp << 10) | new_mant);
            }
        }

        fp16_data[i] = fp16_value;
    }

    return fp16_data;
}

// Letterbox preprocessing, returns scaling info
LetterboxInfo letterbox_with_info(const cv::Mat &img, cv::Mat &output, cv::Size new_shape)
{
    cv::Size shape = img.size();
    float r = std::min((float)new_shape.height / (float)shape.height,
                       (float)new_shape.width / (float)shape.width);
    cv::Size new_unpad((int)std::round((float)shape.width * r),
                       (int)std::round((float)shape.height * r));
    cv::Mat resized;
    if (shape != new_unpad)
    {
        cv::resize(img, resized, new_unpad, cv::INTER_LINEAR);
    }
    else
    {
        resized = img.clone();
    }
    int dw = new_shape.width - new_unpad.width;
    int dh = new_shape.height - new_unpad.height;

    float pad_x = dw / 2.0f;
    float pad_y = dh / 2.0f;

    cv::copyMakeBorder(resized, output, pad_y, pad_y, pad_x, pad_x,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return {r, pad_x, pad_y};
}

// Convert OpenCV Mat to ONNX tensor
std::vector<float> mat_to_tensor(const cv::Mat &img)
{
    std::vector<float> input_tensor_values;
    input_tensor_values.resize(3 * INPUT_HEIGHT * INPUT_WIDTH);

    // Convert BGR to RGB and normalize to [0,1] in one pass
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    // HWC to CHW format
    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);

    int channel_size = INPUT_HEIGHT * INPUT_WIDTH;
    for (int c = 0; c < 3; c++)
    {
        std::memcpy(input_tensor_values.data() + c * channel_size,
                    channels[c].data, channel_size * sizeof(float));
    }

    return input_tensor_values;
}

// YOLOv8 output parsing for [batch,84,8400] format
std::vector<Detection> parse_yolo_output(float *output, const std::vector<int64_t> &output_shape,
                                         float img_width, float img_height,
                                         const LetterboxInfo &letterbox_info)
{
    std::vector<Detection> detections;

    // YOLOv8 output format: [batch, 84, 8400] where 84 = 4 box coords + 80 classes
    int num_classes = output_shape[1] - 4; // should be 80
    int num_detections = output_shape[2];  // should be 8400

    std::cout << "Output shape: [" << output_shape[0] << ", " << output_shape[1]
              << ", " << output_shape[2] << "]" << std::endl;
    std::cout << "Parsing " << num_detections << " detections with " << num_classes << " classes" << std::endl;

    for (int i = 0; i < num_detections; i++)
    {
        // Extract box coordinates (center_x, center_y, width, height)
        float cx = output[0 * num_detections + i];
        float cy = output[1 * num_detections + i];
        float w = output[2 * num_detections + i];
        float h = output[3 * num_detections + i];

        // Find class with highest confidence (classes start at channel 4)
        float max_conf = 0;
        int class_id = 0;
        for (int c = 4; c < output_shape[1]; c++)
        {
            float conf = output[c * num_detections + i];
            if (conf > max_conf)
            {
                max_conf = conf;
                class_id = c - 4;
            }
        }

        if (max_conf > CONFIDENCE_THRESHOLD)
        {
            // Convert from letterbox coordinates back to original image coordinates
            float x1 = (cx - w / 2 - letterbox_info.pad_x) / letterbox_info.scale;
            float y1 = (cy - h / 2 - letterbox_info.pad_y) / letterbox_info.scale;
            float x2 = (cx + w / 2 - letterbox_info.pad_x) / letterbox_info.scale;
            float y2 = (cy + h / 2 - letterbox_info.pad_y) / letterbox_info.scale;

            // Clamp to image boundaries
            x1 = std::max(0.0f, std::min(x1, img_width));
            y1 = std::max(0.0f, std::min(y1, img_height));
            x2 = std::max(0.0f, std::min(x2, img_width));
            y2 = std::max(0.0f, std::min(y2, img_height));

            if (x2 > x1 && y2 > y1)
            {
                Detection det;
                det.box = cv::Rect(x1, y1, x2 - x1, y2 - y1); // Fixed: width = x2-x1, height = y2-y1
                det.confidence = max_conf;
                det.class_id = class_id;
                detections.push_back(det);
            }
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

    cv::dnn::NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);

    std::vector<Detection> final_detections;
    for (int idx : indices)
    {
        final_detections.push_back(detections[idx]);
    }

    std::cout << "Raw detections: " << detections.size() << " -> After NMS: " << final_detections.size() << std::endl;

    return final_detections;
}

// Draw detections on image (for saved frames)
void draw_detections(cv::Mat &img, const std::vector<Detection> &detections)
{
    std::cout << "\n=== Drawing " << detections.size() << " detections on "
              << img.cols << "x" << img.rows << " image ===" << std::endl;

    for (size_t i = 0; i < detections.size(); i++)
    {
        const auto &det = detections[i];

        // Debug: Print detection details
        std::cout << "Detection " << i << ": " << CLASS_NAMES[det.class_id]
                  << " conf=" << det.confidence
                  << " box=(" << det.box.x << "," << det.box.y
                  << "," << det.box.width << "," << det.box.height << ")" << std::endl;

        // Validate bounding box
        if (det.box.width <= 0 || det.box.height <= 0)
        {
            std::cout << "  ❌ Invalid box size: " << det.box.width << "x" << det.box.height << std::endl;
            continue;
        }

        if (det.box.x < 0 || det.box.y < 0 ||
            det.box.x + det.box.width > img.cols ||
            det.box.y + det.box.height > img.rows)
        {
            std::cout << "  ❌ Box out of bounds: (" << det.box.x << "," << det.box.y
                      << ") + (" << det.box.width << "," << det.box.height
                      << ") vs image " << img.cols << "x" << img.rows << std::endl;
            continue;
        }

        // Draw with multiple visual indicators for debugging

        // 1. Main rectangle (thick green)
        cv::rectangle(img, det.box, cv::Scalar(0, 255, 0), 3);

        // 2. Corner markers (red circles)
        cv::circle(img, cv::Point(det.box.x, det.box.y), 5, cv::Scalar(0, 0, 255), -1);
        cv::circle(img, cv::Point(det.box.x + det.box.width, det.box.y + det.box.height), 5, cv::Scalar(0, 0, 255), -1);

        // 3. Center point (blue circle)
        cv::Point center(det.box.x + det.box.width / 2, det.box.y + det.box.height / 2);
        cv::circle(img, center, 3, cv::Scalar(255, 0, 0), -1);

        // 4. Label with background
        std::string label = CLASS_NAMES[det.class_id] + " " +
                            std::to_string(det.confidence).substr(0, 4);

        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);

        cv::Point label_pos(det.box.x, std::max(det.box.y - 10, text_size.height));

        // Background rectangle for text
        cv::rectangle(img,
                      cv::Point(label_pos.x - 2, label_pos.y - text_size.height - 2),
                      cv::Point(label_pos.x + text_size.width + 2, label_pos.y + baseline + 2),
                      cv::Scalar(0, 255, 0), cv::FILLED);

        // Text
        cv::putText(img, label, label_pos, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);

        std::cout << "  ✅ Drew detection successfully" << std::endl;
    }

    std::cout << "=== Drawing complete ===" << std::endl;
}

int main()
{
    std::cout << "=== Enhanced YOLO Integration Test (FP16 Optimized) ===" << std::endl;

    try
    {
        // Initialize ONNX Runtime with optimizations
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo_camera");
        Ort::SessionOptions session_options;

        session_options.SetIntraOpNumThreads(std::thread::hardware_concurrency());
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
        session_options.EnableCpuMemArena();
        session_options.EnableMemPattern();

        session_options.AddConfigEntry("session.disable_prepacking", "1");
        session_options.AddConfigEntry("session.force_spinning_stop", "1");

        // For CPU execution with FP16 models, sometimes we need to allow type conversions
        try
        {
            session_options.AddConfigEntry("session.allow_released_onnx_opset_only", "0");
        }
        catch (...)
        {
            // Some ONNX Runtime versions might not support this config
        }

        // Try fp16 model first, fallback to fp32
        std::string model_path = "../models/yolov8n_fp16.onnx";
        std::ifstream fp16_file(model_path);
        if (!fp16_file.good())
        {
            std::cout << "fp16 model not found, using fp32 model" << std::endl;
            model_path = "../models/yolov8n.onnx";
        }
        else
        {
            std::cout << "Using fp16 quantized model for better performance.";
        }

        // Load YOLO model
        std::cout << "Loading YOLO model: " << model_path << std::endl;
        Ort::Session session(env, model_path.c_str(), session_options);
        std::cout << "Model loaded successfully" << std::endl;

        // Check if model expects FP16 input
        bool model_is_fp16 = isModelFP16(session);
        std::cout << "Model input type: " << (model_is_fp16 ? "FP16" : "FP32") << std::endl;

        // Get model input/output info
        auto input_info = session.GetInputTypeInfo(0);
        auto output_info = session.GetOutputTypeInfo(0);
        std::cout << "Model input name: " << session.GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions()).get() << std::endl;
        std::cout << "Model output name: " << session.GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions()).get() << std::endl;

        // Initialize camera
        std::cout << "Initializing camera.." << std::endl;
        cv::VideoCapture cap(0);
        if (!cap.isOpened())
        {
            std::cerr << "Error: Cannot open camera" << std::endl;
            return -1;
        }

        // Set camera properties for optimal performance
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(cv::CAP_PROP_FPS, 30);
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        std::cout << "Camera initialized" << std::endl;

        // Prepare ONNX input/output
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<const char *> input_names = {"images"};
        std::vector<const char *> output_names = {"output0"};
        std::vector<int64_t> input_shape = {1, 3, INPUT_HEIGHT, INPUT_WIDTH};

        cv::Mat frame, processed_frame;
        std::vector<double> fps_history;
        std::vector<double> inference_times;
        int frame_count = 0;
        const int max_frames = 300; // Run for 300 frames
        int save_every = 50;        // Save every 50th frame

        std::cout << "\n Starting real-time detection..." << std::endl;
        std::cout << "Will process " << max_frames << " frames and exit automatically.";

        auto test_start = std::chrono::high_resolution_clock::now();
        while (frame_count < max_frames)
        {
            auto cycle_start = std::chrono::high_resolution_clock::now();

            // 1. Capture frame
            bool ret = cap.read(frame);
            if (!ret)
            {
                std::cerr << "Failed to capture frame " << frame_count << std::endl;
                break;
            }

            // 2. Preprocess with letterbox info tracking
            auto preprocess_start = std::chrono::high_resolution_clock::now();
            LetterboxInfo letterbox_info = letterbox_with_info(frame, processed_frame, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
            std::vector<float> input_tensor_values = mat_to_tensor(processed_frame);
            auto preprocess_end = std::chrono::high_resolution_clock::now();

            // 3. Model inference
            auto inference_start = std::chrono::high_resolution_clock::now();
            Ort::Value input_tensor;
            if (model_is_fp16)
            {
                // Convert FP32 input to FP16
                std::vector<uint16_t> fp16_input = convertToFP16(input_tensor_values);

                input_tensor = Ort::Value::CreateTensor<Ort::Float16_t>(
                    memory_info,
                    reinterpret_cast<Ort::Float16_t *>(fp16_input.data()),
                    fp16_input.size(),
                    input_shape.data(),
                    input_shape.size());
            }
            else
            {
                // Use FP32 as before
                input_tensor = Ort::Value::CreateTensor<float>(
                    memory_info,
                    input_tensor_values.data(),
                    input_tensor_values.size(),
                    input_shape.data(),
                    input_shape.size());
            }
            auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                              input_names.data(), &input_tensor, 1,
                                              output_names.data(), 1);
            auto inference_end = std::chrono::high_resolution_clock::now();

            // 4. Postprocess
            auto postprocess_start = std::chrono::high_resolution_clock::now();
            float *output_data = output_tensors[0].GetTensorMutableData<float>();
            auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

            std::vector<Detection> detections = parse_yolo_output(output_data, output_shape,
                                                                  frame.cols, frame.rows, letterbox_info);
            auto postprocess_end = std::chrono::high_resolution_clock::now();

            auto cycle_end = std::chrono::high_resolution_clock::now();

            // Calculate timing
            auto preprocess_time = std::chrono::duration_cast<std::chrono::microseconds>(preprocess_end - preprocess_start).count() / 1000.0;
            auto inference_time = std::chrono::duration_cast<std::chrono::microseconds>(inference_end - inference_start).count() / 1000.0;
            auto postprocess_time = std::chrono::duration_cast<std::chrono::microseconds>(postprocess_end - postprocess_start).count() / 1000.0;
            auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(cycle_end - cycle_start).count() / 1000.0;

            double fps = 1000.0 / total_time;
            fps_history.push_back(fps);
            inference_times.push_back(inference_time);

            frame_count++;

            // Save frame with detections every save_every frames
            if (frame_count % save_every == 0)
            {
                cv::Mat output_frame = frame.clone();
                draw_detections(output_frame, detections);
                std::string filename = "detection_frame_" + std::to_string(frame_count) + ".jpg";
                cv::imwrite(filename, output_frame);
                std::cout << "📸 Saved " << filename << " with " << detections.size() << " detections" << std::endl;
            }

            // Progress report every 30 frames
            if (frame_count % 30 == 0)
            {
                // Calculate recent average (last 30 frames)
                size_t start_idx = fps_history.size() > 30 ? fps_history.size() - 30 : 0;
                double recent_avg_fps = 0;
                double recent_avg_inference = 0;
                int recent_count = fps_history.size() - start_idx;

                for (size_t i = start_idx; i < fps_history.size(); i++)
                {
                    recent_avg_fps += fps_history[i];
                    recent_avg_inference += inference_times[i];
                }
                recent_avg_fps /= recent_count;
                recent_avg_inference /= recent_count;

                std::cout << "Frame " << frame_count << "/" << max_frames
                          << " | FPS: " << std::to_string(recent_avg_fps).substr(0, 5)
                          << " | Inference: " << std::to_string(recent_avg_inference).substr(0, 4) << "ms"
                          << " | Detections: " << detections.size() << std::endl;
            }
        }

        auto test_end = std::chrono::high_resolution_clock::now();
        auto total_test_time = std::chrono::duration_cast<std::chrono::milliseconds>(test_end - test_start);

        // Calculate comprehensive statistics
        if (!fps_history.empty())
        {
            double total_fps = 0;
            double total_inference = 0;
            double min_fps = fps_history[0];
            double max_fps = fps_history[0];
            double min_inference = inference_times[0];
            double max_inference = inference_times[0];

            for (size_t i = 0; i < fps_history.size(); i++)
            {
                total_fps += fps_history[i];
                total_inference += inference_times[i];
                min_fps = std::min(min_fps, fps_history[i]);
                max_fps = std::max(max_fps, fps_history[i]);
                min_inference = std::min(min_inference, inference_times[i]);
                max_inference = std::max(max_inference, inference_times[i]);
            }

            double avg_fps = total_fps / fps_history.size();
            double avg_inference = total_inference / inference_times.size();
            double overall_fps = (double)frame_count / (total_test_time.count() / 1000.0);

            std::cout << "\n=== ENHANCED PERFORMANCE RESULTS ===" << std::endl;
            std::cout << "Test Duration: " << total_test_time.count() << " ms" << std::endl;
            std::cout << "Total Frames Processed: " << frame_count << std::endl;
            std::cout << "Overall Throughput FPS: " << overall_fps << std::endl;
            std::cout << "\nPer-Frame Analysis:" << std::endl;
            std::cout << "  Average FPS: " << avg_fps << std::endl;
            std::cout << "  Min FPS: " << min_fps << std::endl;
            std::cout << "  Max FPS: " << max_fps << std::endl;
            std::cout << "\nInference Performance:" << std::endl;
            std::cout << "  Average Inference Time: " << avg_inference << " ms" << std::endl;
            std::cout << "  Min Inference Time: " << min_inference << " ms" << std::endl;
            std::cout << "  Max Inference Time: " << max_inference << " ms" << std::endl;

            // Performance assessment with 40fps target
            std::cout << "\n=== 40 FPS TARGET ASSESSMENT ===" << std::endl;
            if (avg_fps >= 40.0)
            {
                std::cout << "🎯 TARGET ACHIEVED: " << avg_fps << " FPS >= 40 FPS target!" << std::endl;
                std::cout << "   Perfect for autonomous cart real-time navigation!" << std::endl;
            }
            else if (avg_fps >= 35.0)
            {
                std::cout << "🔥 VERY CLOSE: " << avg_fps << " FPS (only " << (40.0 - avg_fps) << " FPS away)" << std::endl;
                std::cout << "   Consider trying YOLOv8s with FP16 quantization" << std::endl;
            }
            else if (avg_fps >= 30.0)
            {
                std::cout << "⚡ GOOD PROGRESS: " << avg_fps << " FPS vs original ~30 FPS" << std::endl;
                std::cout << "   " << (avg_fps > 30 ? "FP16 optimization working!" : "Try YOLOv8s with FP16") << std::endl;
            }
            else
            {
                std::cout << "⚠️  NEEDS OPTIMIZATION: " << avg_fps << " FPS < 30 FPS baseline" << std::endl;
                std::cout << "   Check model loading and verify FP16 quantization" << std::endl;
            }

            // Save detailed results
            std::ofstream results_file("enhanced_performance_results.txt");
            results_file << "Enhanced YOLO Real-Time Performance Test Results\n";
            results_file << "===============================================\n";
            results_file << "Model: " << model_path << "\n";
            results_file << "Target FPS: 40\n";
            results_file << "Achieved FPS: " << avg_fps << "\n";
            results_file << "Target Achievement: " << (avg_fps >= 40 ? "YES" : "NO") << "\n";
            results_file << "Performance Gap: " << (40.0 - avg_fps) << " FPS\n";
            results_file << "Test Duration: " << total_test_time.count() << " ms\n";
            results_file << "Total Frames: " << frame_count << "\n";
            results_file << "Overall FPS: " << overall_fps << "\n";
            results_file << "Average FPS: " << avg_fps << "\n";
            results_file << "Min FPS: " << min_fps << "\n";
            results_file << "Max FPS: " << max_fps << "\n";
            results_file << "Average Inference: " << avg_inference << " ms\n";
            results_file.close();

            std::cout << "\nDetailed results saved to enhanced_performance_results.txt" << std::endl;
        }

        cap.release();
    }
    catch (const std::exception &e)
    {
        std::cerr << "\n"
                  << e.what() << '\n';
        return -1;
    }

    std::cout << "Integration test completed" << std::endl;
    return 0;
}