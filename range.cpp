/*
 * Author: Liam Lawrence
 * Date: 3.27.17
 *
 * CUDA test code -- C++ header
 */

#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/stream_accessor.hpp>

using namespace cv;
using namespace cv::cuda;

extern "C" void gpuInRange_caller(const PtrStepSz<uchar3>& src, Scalar lowerb, Scalar upperb, PtrStep<uchar3> dst, cudaStream_t stream);

extern "C" void gpuInRange(const GpuMat& src, Scalar& lowerb, Scalar& upperb, GpuMat& dst, Stream& stream = Stream::Null()) {
    dst.create(src.size(), CV_8UC1);                    // Sets dst to have the same size, but be a one channel image
    gpuInRange_caller(src, lowerb, upperb, dst, s);     // Calls function
}
