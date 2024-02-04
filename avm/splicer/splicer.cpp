#include "splicer.h"
#include <stdexcept>
#include <bits/stdc++.h>
#include <iostream>
#include <map>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "common.h"
#include <thread>
#include <omp.h>

using namespace std;

#define AWB_LUN_BANLANCE_ENALE    0
#define duo_xiancheng 1

void unsord(cv::Mat &src, CameraPrms prms[], int i)
{
    auto& prm = prms[i];
    //cv::Mat& src = origin_dir_img[i];

    undist_by_remap(src, src, prm);
    cv::warpPerspective(src, src, prm.project_matrix, project_shapes[prm.name]);

    if (camera_flip_mir[i] == "r+") {
        cv::rotate(src, src, cv::ROTATE_90_CLOCKWISE);
    }
    else if (camera_flip_mir[i] == "r-") {
        cv::rotate(src, src, cv::ROTATE_90_COUNTERCLOCKWISE);
    }
    else if (camera_flip_mir[i] == "m") {
        cv::rotate(src, src, cv::ROTATE_180);
    }
    //display_mat(src, "project");
    //cv::imwrite(prms.name + "_undist.png", src);
    //undist_dir_img[i] = src.clone();
}

cv::Mat trans(const cv::Mat images[4])
{
    cv::Mat car_img;
    cv::Mat origin_dir_img[4];
    cv::Mat undist_dir_img[4];
    cv::Mat merge_weights_img[4];
    cv::Mat out_put_img;
    float *w_ptr[4];
    CameraPrms prms[4];
    cv::Mat error = cv::imread("./yaml/weight.png");


    //1.read image and read weights
    car_img = cv::imread("./images/car.png");
    cv::resize(car_img, car_img, cv::Size(xr - xl, yb - yt));
    out_put_img = cv::Mat(cv::Size(total_w, total_h), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat weights = cv::imread("./yaml/weights.png", -1);

    if (weights.channels() != 4) {
        std::cerr << "imread weights failed " << weights.channels() << "\r\n";
        return error;
    }
#pragma omp parallel for num_threads(8)
    for (int i = 0; i < 4; ++i) {
        merge_weights_img[i] = cv::Mat(weights.size(), CV_32FC1, cv::Scalar(0, 0, 0));
        w_ptr[i] = (float *)merge_weights_img[i].data;
    }
    //read weights of corner
    int pixel_index = 0;

#pragma omp parallel for num_threads(8)
    for (int h = 0; h < weights.rows; ++h) {
        uchar* uc_pixel = weights.data + h * weights.step;
        #pragma omp parallel for num_threads(8)
        for (int w = 0; w < weights.cols; ++w) {
            w_ptr[0][pixel_index] = uc_pixel[0] / 255.0f;
            w_ptr[1][pixel_index] = uc_pixel[1] / 255.0f;
            w_ptr[2][pixel_index] = uc_pixel[2] / 255.0f;
            w_ptr[3][pixel_index] = uc_pixel[3] / 255.0f;
            uc_pixel += 4;
            ++pixel_index;
        }
    }

#ifdef DEBUG
    for (int i = 0; i < 4; ++i) {
        //0 左下 1 右上 2 左上 3 左下
        display_mat(merge_weights_img[i], "w");
    }
#endif

    //1. read calibration prms
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < 4; ++i) {
        auto& prm = prms[i];
        prm.name = camera_names[i];
        auto ok = read_prms("./yaml/" + prm.name + ".yaml", prm);
        if (!ok) {
            return error;
        }
    }

     //2.lum equalization and awb for four channel image
     
    std::vector<cv::Mat*> srcs;
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < 4; ++i) {
        auto& prm = prms[i];
        origin_dir_img[i] = images[i];
        srcs.push_back(&origin_dir_img[i]);
    }
    

#if AWB_LUN_BANLANCE_ENALE
    awb_and_lum_banlance(srcs);
#endif


    if (duo_xiancheng)
    {
        thread th_1;
        th_1 = thread(unsord, ref(origin_dir_img[0]), ref(prms), 0);
        thread th_2;
        th_2 = thread(unsord, ref(origin_dir_img[1]), ref(prms), 1);
        thread th_3;
        th_3 = thread(unsord, ref(origin_dir_img[2]), ref(prms), 2);
        thread th_4;
        th_4 = thread(unsord, ref(origin_dir_img[3]), ref(prms), 3);
        th_1.join();
        th_2.join();
        th_3.join();
        th_4.join();


        undist_dir_img[0] = origin_dir_img[0].clone();
        undist_dir_img[1] = origin_dir_img[1].clone();
        undist_dir_img[2] = origin_dir_img[2].clone();
        undist_dir_img[3] = origin_dir_img[3].clone();

    }


    else {
        //3. undistort image
        #pragma omp parallel for num_threads(8)
        for (int i = 0; i < 4; ++i) {
            auto& prm = prms[i];
            cv::Mat& src = origin_dir_img[i];

            undist_by_remap(src, src, prm);
            cv::warpPerspective(src, src, prm.project_matrix, project_shapes[prm.name]);

            if (camera_flip_mir[i] == "r+") {
                cv::rotate(src, src, cv::ROTATE_90_CLOCKWISE);
            }
            else if (camera_flip_mir[i] == "r-") {
                cv::rotate(src, src, cv::ROTATE_90_COUNTERCLOCKWISE);
            }
            else if (camera_flip_mir[i] == "m") {
                cv::rotate(src, src, cv::ROTATE_180);
            }
            //display_mat(src, "project");
            //cv::imwrite(prms.name + "_undist.png", src);
            undist_dir_img[i] = src.clone();
        }
    }
    //4.start combine
    car_img.copyTo(out_put_img(cv::Rect(xl, yt, car_img.cols, car_img.rows)));
    //4.1 out_put_img center copy 
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < 4; ++i) {
        cv::Rect roi;
        bool is_cali_roi = false;
        if (std::string(camera_names[i]) == "front") {
            roi = cv::Rect(xl, 0, xr - xl, yt);
            //std::cout << "\nfront" << roi;
            undist_dir_img[i](roi).copyTo(out_put_img(roi));
        } else if (std::string(camera_names[i]) == "left") {
            roi = cv::Rect(0, yt, xl, yb - yt);
            //std::cout << "\nleft" << roi << out_put_img.size();
            undist_dir_img[i](roi).copyTo(out_put_img(roi));
        } else if (std::string(camera_names[i]) == "right") {
            roi = cv::Rect(0, yt, xl, yb - yt);
            //std::cout << "\nright" << roi << out_put_img.size();
            undist_dir_img[i](roi).copyTo(out_put_img(cv::Rect(xr, yt, total_w - xr, yb - yt)));
        } else if (std::string(camera_names[i]) == "back") {
            roi = cv::Rect(xl, 0, xr - xl, yt);
            //std::cout << "\nright" << roi << out_put_img.size();
            undist_dir_img[i](roi).copyTo(out_put_img(cv::Rect(xl, yb, xr - xl, yt)));
        } 
    }
    //4.2the four corner merge
    //w: 0 左下 1 右上 2 左上 3 左下
    //image: "front", "left", "back", "right"
    cv::Rect roi;
    //左上
    roi = cv::Rect(0, 0, xl, yt);
    merge_image(undist_dir_img[0](roi), undist_dir_img[1](roi), merge_weights_img[2], out_put_img(roi));
    //右上
    roi = cv::Rect(xr, 0, xl, yt);
    merge_image(undist_dir_img[0](roi), undist_dir_img[3](cv::Rect(0, 0, xl, yt)), merge_weights_img[1], out_put_img(cv::Rect(xr, 0, xl, yt)));
    //左下
    roi = cv::Rect(0, yb, xl, yt);
    merge_image(undist_dir_img[2](cv::Rect(0, 0, xl, yt)), undist_dir_img[1](roi), merge_weights_img[0], out_put_img(roi));
    //右下
    roi = cv::Rect(xr, 0, xl, yt);
    merge_image(undist_dir_img[2](roi), undist_dir_img[3](cv::Rect(0, yb, xl, yt)), merge_weights_img[3], out_put_img(cv::Rect(xr, yb, xl, yt)));

    //cv::imwrite("ADAS_EYES_360_VIEW.png", out_put_img);
    return out_put_img;

#ifdef DEBUG   
    cv::resize(out_put_img, out_put_img, cv::Size(out_put_img.size()/2)),
    display_mat(out_put_img, "out_put_img");
#endif

}

int main(int argc, char* argv[])
{
    int count;
    printf("The command line has %d arguments:", argc - 1);
    for (count = 1; count < argc; count++)
    {printf("%d: %s", count, argv[count] );}
    cv::VideoCapture cap_fornt;
    cv::VideoCapture cap_back;
    cv::VideoCapture cap_left;
    cv::VideoCapture cap_right;
    cap_fornt.open(argv[1]);
    cap_back.open(argv[4]);
    cap_left.open(argv[2]);
    cap_right.open(argv[3]);
    if (!cap_fornt.isOpened() && cap_back.isOpened() && cap_left.isOpened() && cap_right.isOpened())
        return 0;
    cv::Mat frame_left,
            frame_right,
            frame_front,
            frame_back;

    while (1)
    {
        clock_t start, end;
        start = clock();

        cap_fornt >> frame_front;
        cap_back >> frame_back;
        cap_left >> frame_left;
        cap_right >> frame_right;

        cv::Mat images[4] = { frame_front ,frame_back, frame_front, frame_right };
        cv::Mat out_put_img = trans(images);
        end = clock();   //结束时间
        std::cout << "time = " << double(end - start) / CLOCKS_PER_SEC << "s" << std::endl;
        display_mat(out_put_img, "out_put_img");

    }
}
    
