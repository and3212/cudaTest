/*
 * Author: Liam Lawrence
 * Date: 3.27.17
 *
 * CUDA test code -- cuda function
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/cuda_devptrs.hpp>

using namespace cv;
using namespace cv::cuda;

__global__ void gpuInRange(const PtrStepSz<uchar3> src, int lh, int ls, int lv, int uh, int us, int uv, PtrStep<uchar3> dst) {

    // Iterates through pixels
	int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    // If we haven't run out of pixels yet
    if(x < src.cols && y < src.rows) {
        // Thresholds the three channels
        uchar3 v = src(y,x);
        if(v.z >= lh && v.z <= uh &&
           v.y >= ls && v.y <= us &&
           v.x >= lv && v.x <= uv)
            dst(y,x) = 255; // set to a white pixel
        else
            dst(y,x) = 0;   // set to a black pixel
    }
}

extern "C" void gpuInRange_caller(const PtrStepSz<uchar3>& src, Scalar lowerb, Scalar upperb, PtrStep<uchar3> dst) {
    // Set up memory for iterating through pixels
    dim3 block(32, 8);
    dim3 grid((src.cols + block.x - 1)/block.x,(src.rows + block.y - 1)/block.y);

    /* vvv Test these to see if they are faster vvv
     * const int m = 32;
     * is src.cols or int numCols = src.cols faster if called more than once?
     * const dim3 gridSize(ceil((float)numCols / m), ceil((float)numRows / m), 1);
     * const dim3 blockSize(m, m, 1);
     */

    // Runs the CUDA function
    gpuInRange<<<grid, block>>>(src, lowerb[0], lowerb[1], lowerb[2], upperb[0], upperb[1], upperb[2], dst);
}
