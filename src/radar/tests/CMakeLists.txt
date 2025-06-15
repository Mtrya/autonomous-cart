cmake_minimum_required(VERSION 3.10)
project(RadarHandlerTests)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../rplidar_sdk/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../rplidar_sdk/sdk/include
)


# Source files
set(RADAR_HANDLER_SOURCES
    ../radar_handler.cpp
)

set(TEST_SOURCES
    test_radar_handler.cpp
)

# SLAMTEC LIDAR SDK library path
set(RPLIDAR_SDK_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../rplidar_sdk)

# Find the SLAMTEC LIDAR SDK library (correct name: libsl_lidar_sdk.a)
find_library(SL_LIDAR_SDK_LIB
    NAMES sl_lidar_sdk
    PATHS ${RPLIDAR_SDK_PATH}/output/Linux/Release
    NO_DEFAULT_PATH
)

if(NOT SL_LIDAR_SDK_LIB)
    # Try direct path
    set(SL_LIDAR_SDK_LIB ${RPLIDAR_SDK_PATH}/output/Linux/Release/libsl_lidar_sdk.a)
    
    if(EXISTS ${SL_LIDAR_SDK_LIB})
        message(STATUS "Found SLAMTEC LIDAR SDK library: ${SL_LIDAR_SDK_LIB}")
    else()
        message(FATAL_ERROR "SLAMTEC LIDAR SDK library not found at: ${SL_LIDAR_SDK_LIB}")
        message(STATUS "Expected library location: ${RPLIDAR_SDK_PATH}/output/Linux/Release/libsl_lidar_sdk.a")
        message(STATUS "Navigate to ${RPLIDAR_SDK_PATH} and run: make")
    endif()
else()
    message(STATUS "Found SLAMTEC LIDAR SDK library: ${SL_LIDAR_SDK_LIB}")
endif()

# Create test executable
add_executable(test_radar_handler 
    ${TEST_SOURCES}
    ${RADAR_HANDLER_SOURCES}
)

# Link libraries
target_link_libraries(test_radar_handler
    ${OpenCV_LIBS}
    ${SL_LIDAR_SDK_LIB}
    Threads::Threads
    pthread
)

# Compiler flags
target_compile_options(test_radar_handler PRIVATE)


# Add custom target to run tests
add_custom_target(run_tests
    COMMAND test_radar_handler
    DEPENDS test_radar_handler
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running radar handler tests"
)
