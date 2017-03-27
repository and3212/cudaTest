/*
 * Author: Liam Lawrence
 * Date: 3.27.17
 *
 * CUDA test code -- main file
 */

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/cuda_devptrs.hpp>
#include <opencv2/cudaarithm.hpp>
#include <iostream>

extern "C" void gpuInRange(const GpuMat& src, Scalar& lowerb, Scalar& upperb, GpuMat& dst, Stream& stream = Stream::Null());

int main() {
    // CPU Mats and access to your camera
    cv::Mat latestFrame,
            finalImage;
    cv::VideoCapture cap(0);

    // GPU Mats using CUDA
    cv::cuda::GpuMat gpuFrame,
                     hsvFrame,   // should the last two be PtrStepSz<> or GpuMat?
                     threshFrame;

    // HSV ranges
    int H[2] = {58, 62},
        S[2] = {253, 255},
        V[2] = {120, 185};


    // Main loop
    for(;;) {

        // Reads in the latest frame and loads it into the GPU
        cap >> latestFrame;
        gpuFrame.upload(latestFrame);

        // Converts the color space from BGR to HSV and thresholds the image
        cv::cuda::cvtColor(gpuFrame, hsvFrame, CV_BGR2HSV);
        gpuInRange(hsvFrame, cv::Scalar(H[0], S[0], V[0]), cv::Scalar(H[1], S[1], V[1]), threshFrame);

        // Downloads the Mat back onto the CPU and displays it
        threshFrame.download(finalImage);

        imshow("Latest Frame", latestFrame);
        imshow("Final Frame", finalImage);

        // Adds a little bit of delay
        // Hit ESC to kill the program
        if(cv::waitKey(1) == 27)
            break;
    }
}
