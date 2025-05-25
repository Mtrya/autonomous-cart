#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

int main() {
    std::cout << "=== Library Linking Test ===" << std::endl;
    
    // Test OpenCV
    try {
        std::cout << "OpenCV Version: " << CV_VERSION << std::endl;
        cv::Mat test_image = cv::Mat::zeros(100, 100, CV_8UC3);
        std::cout << "✅ OpenCV: Successfully created test image (" 
                  << test_image.rows << "x" << test_image.cols << ")" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "❌ OpenCV Error: " << e.what() << std::endl;
        return -1;
    }
    
    // Test ONNX Runtime
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        std::cout << "✅ ONNX Runtime: Successfully created environment" << std::endl;
        
        // Get available providers
        auto providers = Ort::GetAvailableProviders();
        std::cout << "Available ONNX providers: ";
        for (const auto& provider : providers) {
            std::cout << provider << " ";
        }
        std::cout << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "❌ ONNX Runtime Error: " << e.what() << std::endl;
        return -1;
    }
    
    std::cout << "🎉 Both libraries working correctly!" << std::endl;
    return 0;
}