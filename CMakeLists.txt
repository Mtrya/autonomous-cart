cmake_minimum_required(VERSION 3.16)
project(autonomous_cart)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_ROOT_PATH /usr/local)
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include)
set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_ROOT_PATH}/lib/libonnxruntime.so)

add_executable(test_libs src/camera/test_libs.cpp)

target_include_directories(test_libs PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

target_link_libraries(test_libs
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARIES}
)

# Create the YOLO loading test executable
add_executable(test_onnx src/camera/test_onnx.cpp)

target_include_directories(test_onnx PRIVATE 
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

target_link_libraries(test_onnx
    ${ONNXRUNTIME_LIBRARIES}
)

# Create the full integration test executable
add_executable(test_yolo src/camera/test_yolo.cpp)

target_include_directories(test_yolo PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

target_link_libraries(test_yolo 
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARIES}
)




# Test lidar
# Add to your existing CMakeLists.txt
find_package(OpenCV REQUIRED)

# RPLIDAR SDK paths
set(RPLIDAR_SDK_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/radar/rplidar_sdk/sdk")
set(RPLIDAR_LIB_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/radar/rplidar_sdk/output/Linux/Release")

# Include directories
include_directories(${RPLIDAR_SDK_PATH}/include)
include_directories(${OpenCV_INCLUDE_DIRS})

# Create test_radar executable
add_executable(test_radar src/radar/test_radar.cpp)

# Link against the pre-compiled RPLIDAR library
target_link_libraries(test_radar 
    ${RPLIDAR_LIB_PATH}/libsl_lidar_sdk.a
    ${OpenCV_LIBS} 
    pthread
)

# Add any necessary compile definitions
target_compile_definitions(test_radar PRIVATE -D_REENTRANT)

