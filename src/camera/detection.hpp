#ifndef DETECTION_HPP
#define DETECTION_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct Detection
{
    cv::Rect box;
    float confidence;
    int class_id;
    std::string class_name;

    Detection() : confidence(0.0f), class_id(-1) {}

    Detection(const cv::Rect &bbox, float conf, int id, const std::string &name)
        : box(bbox), confidence(conf), class_id(id), class_name(name) {}
};

// Collection of detections for a single frame
using DetectionResult = std::vector<Detection>;

#endif