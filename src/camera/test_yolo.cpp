#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <cmath>
#include <fstream>

// YOLO configuration
const int INPUT_WIDTH = 640;
const int INPUT_HEIGHT = 640;
const float CONFIDENCE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.6;

// COCO class names (first 80 classes)
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
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
};

struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
};

// Letterbox preprocessing (maintains aspect ratio)
cv::Mat letterbox(const cv::Mat& img, cv::Size new_shape, cv::Scalar color = cv::Scalar(114, 114, 114)) {
    cv::Size shape = img.size();
    float r = std::min((float)new_shape.height / (float)shape.height, 
                       (float)new_shape.width / (float)shape.width);
    
    cv::Size new_unpad((int)std::round((float)shape.width * r), (int)std::round((float)shape.height * r));
    cv::Mat resized;
    if (shape != new_unpad) {
        cv::resize(img, resized, new_unpad, 0, 0, cv::INTER_LINEAR);
    } else {
        resized = img.clone();
    }
    
    int dw = new_shape.width - new_unpad.width;
    int dh = new_shape.height - new_unpad.height;
    
    dw /= 2;
    dh /= 2;
    
    cv::Mat letterboxed;
    cv::copyMakeBorder(resized, letterboxed, dh, dh, dw, dw, cv::BORDER_CONSTANT, color);
    
    return letterboxed;
}

// Convert OpenCV Mat to ONNX tensor
std::vector<float> mat_to_tensor(const cv::Mat& img) {
    std::vector<float> input_tensor_values;
    input_tensor_values.resize(3 * INPUT_HEIGHT * INPUT_WIDTH);
    
    // Convert BGR to RGB and normalize to [0,1]
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);
    
    // HWC to CHW format
    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);
    
    int channel_size = INPUT_HEIGHT * INPUT_WIDTH;
    for (int c = 0; c < 3; c++) {
        std::memcpy(input_tensor_values.data() + c * channel_size, 
                   channels[c].data, channel_size * sizeof(float));
    }
    
    return input_tensor_values;
}

// Parse YOLO output and apply NMS
std::vector<Detection> parse_yolo_output(float* output, int rows, float img_width, float img_height) {
    std::vector<Detection> detections;
    
    // YOLOv8 output format: [batch, 84, 8400] where 84 = 4 box coords + 80 classes
    for (int i = 0; i < rows; i++) {
        float* row = output + i * 84;
        
        // Get bounding box coordinates (center_x, center_y, width, height)
        float cx = row[0];
        float cy = row[1];
        float w = row[2];
        float h = row[3];
        
        // Find class with highest confidence
        float max_conf = 0;
        int class_id = 0;
        for (int j = 4; j < 84; j++) {
            if (row[j] > max_conf) {
                max_conf = row[j];
                class_id = j - 4;
            }
        }
        
        if (max_conf > CONFIDENCE_THRESHOLD) {
            // Convert to corner coordinates and scale back to original image
            float x1 = (cx - w/2) * img_width / INPUT_WIDTH;
            float y1 = (cy - h/2) * img_height / INPUT_HEIGHT;
            float x2 = (cx + w/2) * img_width / INPUT_WIDTH;
            float y2 = (cy + h/2) * img_height / INPUT_HEIGHT;
            
            Detection det;
            det.box = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
            det.confidence = max_conf;
            det.class_id = class_id;
            detections.push_back(det);
        }
    }
    
    // Apply Non-Maximum Suppression
    std::vector<int> indices;
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    
    for (const auto& det : detections) {
        boxes.push_back(det.box);
        scores.push_back(det.confidence);
    }
    
    cv::dnn::NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);
    
    std::vector<Detection> final_detections;
    for (int idx : indices) {
        final_detections.push_back(detections[idx]);
    }
    
    return final_detections;
}

// Draw detections on image (for saved frames)
void draw_detections(cv::Mat& img, const std::vector<Detection>& detections) {
    for (const auto& det : detections) {
        // Draw bounding box
        cv::rectangle(img, det.box, cv::Scalar(0, 255, 0), 2);
        
        // Draw label
        std::string label = CLASS_NAMES[det.class_id] + " " + 
                           std::to_string(det.confidence).substr(0, 4);
        
        int baseline;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        
        cv::Point label_pos(det.box.x, det.box.y - 10);
        cv::rectangle(img, label_pos, 
                     cv::Point(label_pos.x + text_size.width, label_pos.y - text_size.height - baseline),
                     cv::Scalar(0, 255, 0), cv::FILLED);
        
        cv::putText(img, label, cv::Point(label_pos.x, label_pos.y - baseline), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
}

int main() {
    std::cout << "=== Headless Real-Time YOLO Integration Test ===" << std::endl;
    
    try {
        // Initialize ONNX Runtime
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo_camera");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(10);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // Load YOLO model
        std::string model_path = "../models/yolov8n.onnx";
        std::cout << "Loading YOLO model..." << std::endl;
        Ort::Session session(env, model_path.c_str(), session_options);
        std::cout << "✅ Model loaded successfully" << std::endl;
        
        // Initialize camera
        std::cout << "Initializing camera..." << std::endl;
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            std::cerr << "❌ Error: Cannot open camera" << std::endl;
            return -1;
        }
        
        // Set camera properties
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(cv::CAP_PROP_FPS, 30);
        std::cout << "✅ Camera initialized" << std::endl;
        
        // Prepare ONNX input/output
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<const char*> input_names = {"images"};
        std::vector<const char*> output_names = {"output0"};
        std::vector<int64_t> input_shape = {1, 3, INPUT_HEIGHT, INPUT_WIDTH};
        
        cv::Mat frame, processed_frame;
        std::vector<double> fps_history;
        std::vector<double> inference_times;
        int frame_count = 0;
        const int max_frames = 300; // Run for 300 frames (about 10 seconds at 30fps)
        int save_every = 50; // Save every 50th frame
        
        std::cout << "\n🚀 Starting headless real-time detection..." << std::endl;
        std::cout << "Will process " << max_frames << " frames and exit automatically" << std::endl;
        std::cout << "Saving every " << save_every << "th frame to verify detection works" << std::endl;
        
        auto test_start = std::chrono::high_resolution_clock::now();
        
        while (frame_count < max_frames) {
            auto cycle_start = std::chrono::high_resolution_clock::now();
            
            // 1. CAPTURE FRAME
            auto capture_start = std::chrono::high_resolution_clock::now();
            bool ret = cap.read(frame);
            if (!ret) {
                std::cerr << "❌ Failed to capture frame " << frame_count << std::endl;
                break;
            }
            auto capture_end = std::chrono::high_resolution_clock::now();
            
            // 2. PREPROCESS
            auto preprocess_start = std::chrono::high_resolution_clock::now();
            processed_frame = letterbox(frame, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
            std::vector<float> input_tensor_values = mat_to_tensor(processed_frame);
            auto preprocess_end = std::chrono::high_resolution_clock::now();
            
            // 3. INFERENCE
            auto inference_start = std::chrono::high_resolution_clock::now();
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, input_tensor_values.data(), input_tensor_values.size(),
                input_shape.data(), input_shape.size());
            
            auto output_tensors = session.Run(Ort::RunOptions{nullptr}, 
                                            input_names.data(), &input_tensor, 1,
                                            output_names.data(), 1);
            auto inference_end = std::chrono::high_resolution_clock::now();
            
            // 4. POSTPROCESS
            auto postprocess_start = std::chrono::high_resolution_clock::now();
            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
            int rows = output_shape[2]; // Should be 8400 for YOLOv8
            
            std::vector<Detection> detections = parse_yolo_output(output_data, rows, frame.cols, frame.rows);
            auto postprocess_end = std::chrono::high_resolution_clock::now();
            
            auto cycle_end = std::chrono::high_resolution_clock::now();
            
            // Calculate timing
            auto capture_time = std::chrono::duration_cast<std::chrono::microseconds>(capture_end - capture_start).count() / 1000.0;
            auto preprocess_time = std::chrono::duration_cast<std::chrono::microseconds>(preprocess_end - preprocess_start).count() / 1000.0;
            auto inference_time = std::chrono::duration_cast<std::chrono::microseconds>(inference_end - inference_start).count() / 1000.0;
            auto postprocess_time = std::chrono::duration_cast<std::chrono::microseconds>(postprocess_end - postprocess_start).count() / 1000.0;
            auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(cycle_end - cycle_start).count() / 1000.0;
            
            double fps = 1000.0 / total_time;
            fps_history.push_back(fps);
            inference_times.push_back(inference_time);
            
            frame_count++;
            
            // Save frame with detections every save_every frames
            if (frame_count % save_every == 0) {
                cv::Mat output_frame = frame.clone();
                draw_detections(output_frame, detections);
                std::string filename = "detection_frame_" + std::to_string(frame_count) + ".jpg";
                cv::imwrite(filename, output_frame);
                std::cout << "Saved " << filename << " with " << detections.size() << " detections" << std::endl;
            }
            
            // Progress report every 30 frames
            if (frame_count % 30 == 0) {
                // Calculate recent average (last 30 frames)
                size_t start_idx = fps_history.size() > 30 ? fps_history.size() - 30 : 0;
                double recent_avg_fps = 0;
                double recent_avg_inference = 0;
                int recent_count = fps_history.size() - start_idx;
                
                for (size_t i = start_idx; i < fps_history.size(); i++) {
                    recent_avg_fps += fps_history[i];
                    recent_avg_inference += inference_times[i];
                }
                recent_avg_fps /= recent_count;
                recent_avg_inference /= recent_count;
                
                std::cout << "Frame " << frame_count << "/" << max_frames 
                         << " | Recent Avg FPS: " << std::to_string(recent_avg_fps).substr(0, 5)
                         << " | Avg Inference: " << std::to_string(recent_avg_inference).substr(0, 4) << "ms"
                         << " | Detections: " << detections.size() << std::endl;
            }
        }
        
        auto test_end = std::chrono::high_resolution_clock::now();
        auto total_test_time = std::chrono::duration_cast<std::chrono::milliseconds>(test_end - test_start);
        
        // Calculate comprehensive statistics
        if (!fps_history.empty()) {
            double total_fps = 0;
            double total_inference = 0;
            double min_fps = fps_history[0];
            double max_fps = fps_history[0];
            double min_inference = inference_times[0];
            double max_inference = inference_times[0];
            
            for (size_t i = 0; i < fps_history.size(); i++) {
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
            
            std::cout << "\n=== COMPREHENSIVE PERFORMANCE RESULTS ===" << std::endl;
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
            std::cout << "  Inference-only theoretical FPS: " << (1000.0 / avg_inference) << std::endl;
            
            // Write detailed results to file
            std::ofstream results_file("performance_results.txt");
            results_file << "YOLO Real-Time Performance Test Results\n";
            results_file << "======================================\n";
            results_file << "Test Duration: " << total_test_time.count() << " ms\n";
            results_file << "Total Frames: " << frame_count << "\n";
            results_file << "Overall FPS: " << overall_fps << "\n";
            results_file << "Average FPS: " << avg_fps << "\n";
            results_file << "Min FPS: " << min_fps << "\n";
            results_file << "Max FPS: " << max_fps << "\n";
            results_file << "Average Inference: " << avg_inference << " ms\n";
            results_file << "Min Inference: " << min_inference << " ms\n";
            results_file << "Max Inference: " << max_inference << " ms\n";
            results_file.close();
            
            std::cout << "\nDetailed results saved to performance_results.txt" << std::endl;
            
            // Performance assessment
            std::cout << "\n=== PERFORMANCE ASSESSMENT ===" << std::endl;
            if (avg_fps >= 20.0) {
                std::cout << "✅ EXCELLENT: Real-time performance achieved (>= 20 FPS)" << std::endl;
                std::cout << "   Your autonomous cart will work smoothly!" << std::endl;
            } else if (avg_fps >= 15.0) {
                std::cout << "✅ VERY GOOD: Near real-time performance (15-20 FPS)" << std::endl;
                std::cout << "   Suitable for most autonomous navigation tasks" << std::endl;
            } else if (avg_fps >= 10.0) {
                std::cout << "✅ GOOD: Acceptable for autonomous use (10-15 FPS)" << std::endl;
                std::cout << "   May need optimization for fast-moving scenarios" << std::endl;
            } else if (avg_fps >= 5.0) {
                std::cout << "⚠️  MARGINAL: Slow but usable (5-10 FPS)" << std::endl;
                std::cout << "   Consider optimizations or simpler model" << std::endl;
            } else {
                std::cout << "❌ TOO SLOW: Not suitable for real-time use (< 5 FPS)" << std::endl;
                std::cout << "   Needs significant optimization" << std::endl;
            }
        }
        
        cap.release();
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return -1;
    }
    
    std::cout << "\n🎉 Headless integration test completed!" << std::endl;
    std::cout << "Check the saved detection_frame_*.jpg files to verify detection works correctly." << std::endl;
    return 0;
}