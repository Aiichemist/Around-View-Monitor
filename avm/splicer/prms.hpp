#ifndef PRMS_H
#define PRMS_H
#include <iostream>
#include <map>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>


static const  char* camera_names[4] = {
    "front", "left", "back", "right"
};

static const  char* camera_flip_mir[4] = {
    "n", "r-", "m", "r+"
};


static const  int shift_w = 45;
static const  int shift_h = 90;

static const  int cali_map_w  = 660;
static const  int cali_map_h  = 780;

static const  int inn_shift_w = 0;
static const  int inn_shift_h = 0;

static const  int total_w = cali_map_w + 2 * shift_w;
static const  int total_h = cali_map_h + 2 * shift_h;

static const  int xl = shift_w + 240 + inn_shift_w;
static const  int xr = total_w - xl;
static const  int yt = shift_h + 270 + inn_shift_h;
static const  int yb = total_h - yt;

static std::map<std::string, cv::Size> project_shapes = {
    {"front",  cv::Size(total_w, yt)},
    {"back",   cv::Size(total_w, yt)},
    {"left",   cv::Size(total_h, xl)},
    {"right",  cv::Size(total_h, xl)},
};

static std::map<std::string, std::vector<cv::Point2f>> project_keypoints = {
    {"front", {cv::Point2f(shift_w + 120, shift_h),
              cv::Point2f(shift_w + 540, shift_h),
              cv::Point2f(shift_w + 120, shift_h + 240),
              cv::Point2f(shift_w + 540, shift_h + 240)}},

    {"back", {cv::Point2f(shift_w + 120, shift_h),
              cv::Point2f(shift_w + 540, shift_h),
              cv::Point2f(shift_w + 120, shift_h + 240),
              cv::Point2f(shift_w + 540, shift_h + 240)}},

    {"left", {cv::Point2f(shift_h + 120, shift_w),
              cv::Point2f(shift_h + 660, shift_w),
              cv::Point2f(shift_h + 120, shift_w + 180),
              cv::Point2f(shift_h + 660, shift_w + 180)}},

    {"right", {cv::Point2f(shift_h + 120, shift_w),
              cv::Point2f(shift_h + 660, shift_w),
              cv::Point2f(shift_h + 120, shift_w + 180),
              cv::Point2f(shift_h + 660, shift_w + 180)}}
};

#endif