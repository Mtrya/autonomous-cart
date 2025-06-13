cmake_minimum_required(VERSION 3.16)
project(CameraTest)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_executable(test_camera_handler 
    test_camera_handler.cpp
    ../camera_handler.cpp
)

target_link_libraries(test_camera_handler ${OpenCV_LIBS})
