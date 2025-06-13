cmake_minimum_required(VERSION 3.16)
project(CameraTest)

set(CMAKE_CXX_STANDARD 17)

set(ONNXRUNTIME_ROOT_PATH /usr/local)
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include)
set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_ROOT_PATH}/lib/libonnxruntime.so)

find_package(OpenCV REQUIRED)

add_executable(test_camera_handler 
    test_camera_handler.cpp
    ../camera_handler.cpp
)

target_include_directories(test_camera_handler PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

target_link_libraries(test_camera_handler
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARIES}
)
