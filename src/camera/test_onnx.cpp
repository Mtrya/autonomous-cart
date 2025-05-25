#include <iostream>
#include <vector>
#include <chrono>
#include <onnxruntime_cxx_api.h>

int main() {
    std::cout << "=== YOLO ONNX Model Loading Test ===" << std::endl;
    
    try {
        // Initialize ONNX Runtime environment
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo_test");
        
        // Session options
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(12);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // Load model
        std::string model_path = "../models/yolov8n.onnx";
        std::cout << "Loading model: " << model_path << std::endl;
        
        auto start_load = std::chrono::high_resolution_clock::now();
        Ort::Session session(env, model_path.c_str(), session_options);
        auto end_load = std::chrono::high_resolution_clock::now();
        
        auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_load - start_load);
        std::cout << "✅ Model loaded successfully in " << load_time.count() << " ms" << std::endl;
        
        // Get model input/output info
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Input info
        size_t num_input_nodes = session.GetInputCount();
        std::cout << "\nInput nodes: " << num_input_nodes << std::endl;
        
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = session.GetInputNameAllocated(i, allocator);
            auto input_type_info = session.GetInputTypeInfo(i);
            auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
            auto input_shape = input_tensor_info.GetShape();
            
            std::cout << "  Input " << i << ": " << input_name.get() << " - Shape: [";
            for (size_t j = 0; j < input_shape.size(); j++) {
                std::cout << input_shape[j];
                if (j < input_shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        
        // Output info
        size_t num_output_nodes = session.GetOutputCount();
        std::cout << "\nOutput nodes: " << num_output_nodes << std::endl;
        
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = session.GetOutputNameAllocated(i, allocator);
            auto output_type_info = session.GetOutputTypeInfo(i);
            auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
            auto output_shape = output_tensor_info.GetShape();
            
            std::cout << "  Output " << i << ": " << output_name.get() << " - Shape: [";
            for (size_t j = 0; j < output_shape.size(); j++) {
                std::cout << output_shape[j];
                if (j < output_shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        
        // Create dummy input tensor (typical YOLOv8 input: 1x3x640x640)
        std::vector<int64_t> input_shape = {1, 3, 640, 640};
        size_t input_tensor_size = 1 * 3 * 640 * 640;
        std::vector<float> input_tensor_values(input_tensor_size, 0.5f); // Fill with 0.5
        
        // Create input tensor
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_size,
            input_shape.data(), input_shape.size());
        
        std::cout << "\n✅ Created dummy input tensor: " << input_tensor_size << " elements" << std::endl;
        
        // Prepare input/output names
        std::vector<const char*> input_names = {"images"};  // YOLOv8 standard input name
        std::vector<const char*> output_names = {"output0"}; // YOLOv8 standard output name
        
        // Run inference on dummy data (measure performance)
        std::cout << "\nRunning inference on dummy data..." << std::endl;
        
        const int num_inferences = 600;
        std::vector<double> inference_times;
        
        for (int i = 0; i < num_inferences; i++) {
            auto start_inference = std::chrono::high_resolution_clock::now();
            
            auto output_tensors = session.Run(Ort::RunOptions{nullptr}, 
                                            input_names.data(), &input_tensor, 1,
                                            output_names.data(), 1);
            
            auto end_inference = std::chrono::high_resolution_clock::now();
            auto inference_time = std::chrono::duration_cast<std::chrono::microseconds>(end_inference - start_inference);
            inference_times.push_back(inference_time.count() / 1000.0); // Convert to milliseconds
            
            std::cout << "  Inference " << (i+1) << ": " << inference_times.back() << " ms" << std::endl;
        }
        
        // Calculate statistics
        double total_time = 0;
        double min_time = inference_times[0];
        double max_time = inference_times[0];
        
        for (double time : inference_times) {
            total_time += time;
            min_time = std::min(min_time, time);
            max_time = std::max(max_time, time);
        }
        
        double avg_time = total_time / num_inferences;
        double theoretical_fps = 1000.0 / avg_time;
        
        std::cout << "\n=== Inference Performance ===" << std::endl;
        std::cout << "Average inference time: " << avg_time << " ms" << std::endl;
        std::cout << "Min inference time: " << min_time << " ms" << std::endl;
        std::cout << "Max inference time: " << max_time << " ms" << std::endl;
        std::cout << "Theoretical max FPS: " << theoretical_fps << " FPS" << std::endl;
        
        // Performance assessment
        if (theoretical_fps >= 30.0) {
            std::cout << "✅ YOLO Performance: EXCELLENT (>= 30 FPS)" << std::endl;
        } else if (theoretical_fps >= 15.0) {
            std::cout << "✅ YOLO Performance: GOOD (15-30 FPS)" << std::endl;
        } else if (theoretical_fps >= 10.0) {
            std::cout << "⚠️  YOLO Performance: ACCEPTABLE (10-15 FPS)" << std::endl;
        } else {
            std::cout << "❌ YOLO Performance: POOR (< 10 FPS)" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return -1;
    }
    
    std::cout << "\n🎉 YOLO model loading and inference test completed!" << std::endl;
    return 0;
}