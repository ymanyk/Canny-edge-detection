#include <vector>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <cmath>
#include <chrono> // For measuring time

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// For suppressing info logs in debug mode
#include <opencv2/core/utils/logger.hpp>

using namespace cv;
using namespace cv::cuda;

#define NTHREAD 2
#define M_PI 3.14159265358979323846


// Define CUDA error checking macro
#define CUDA_CHECK(call) { \
    cudaError_t cudaError = call; \
    if (cudaError != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(cudaError) << " at Line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

bool isDayTime(Mat source)
{
    Scalar s = mean(source);
    return !(s[0] < 30 || (s[1] < 33 && s[2] < 30));
}

Mat filterColors(Mat source, bool isDayTime)
{
    Mat hsv, whiteMask, whiteImage, yellowMask, yellowImage, whiteYellow;

    // White mask
    std::vector<int> lowerWhite = { 130, 130, 130 };
    std::vector<int> upperWhite = { 255, 255, 255 };
    inRange(source, lowerWhite, upperWhite, whiteMask);
    bitwise_and(source, source, whiteImage, whiteMask);

    // Yellow mask
    cvtColor(source, hsv, COLOR_BGR2HSV);
    std::vector<int> lowerYellow = { 20, 100, 110 };
    std::vector<int> upperYellow = { 30, 180, 240 };
    inRange(hsv, lowerYellow, upperYellow, yellowMask);
    bitwise_and(source, source, yellowImage, yellowMask);

    // Blend yellow and white together
    addWeighted(whiteImage, 1.0, yellowImage, 1.0, 0.0, whiteYellow);

    // Add gray filter if image is not taken during the day
    if (!isDayTime)
    {
        Mat grayMask, grayImage, dst;
        std::vector<int> lowerGray = { 80, 80, 80 };
        std::vector<int> upperGray = { 130, 130, 130 };
        inRange(source, lowerGray, upperGray, grayMask);
        bitwise_and(source, source, grayImage, grayMask);

        addWeighted(grayImage, 1.0, whiteYellow, 1.0, 0.0, dst);
        return dst;
    }

    // Return white and yellow mask if image is taken during the day
    return whiteYellow;
}

Mat applyGrayscale(Mat source)
{
    Mat dst;
    cvtColor(source, dst, COLOR_BGR2GRAY);
    return dst;
}

__global__ void gaussianBlurCUDA(const uchar* src, uchar* dst, int width, int height, float* kernel, int kernelSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int offset = y * width + x;
        float sum = 0.0f;
        float sumWeights = 0.0f;

        for (int i = 0; i < kernelSize; i++)
        {
            for (int j = 0; j < kernelSize; j++)
            {
                int offsetX = x + j - kernelSize / 2;
                int offsetY = y + i - kernelSize / 2;

                if (offsetX >= 0 && offsetX < width && offsetY >= 0 && offsetY < height)
                {
                    float weight = kernel[i * kernelSize + j];
                    sum += static_cast<float>(src[offsetY * width + offsetX]) * weight;
                    sumWeights += weight;
                }
            }
        }

        if (sumWeights != 0.0f)
        {
            dst[offset] = static_cast<uchar>(sum / sumWeights);
        }
    }
}

cv::Mat applyGaussianBlur(const cv::Mat& source, int kernelSize)
{
    int width = source.cols;
    int height = source.rows;
    int imageSize = width * height * sizeof(uchar);

    // Define Gaussian kernel
    float gaussianKernel[] = { 1, 2, 1, 2, 4, 2, 1, 2, 1 };
    float* d_kernel;

    // Allocate memory for the kernel on the GPU
    cudaMalloc((void**)&d_kernel, sizeof(float) * kernelSize * kernelSize);
    cudaMemcpy(d_kernel, gaussianKernel, sizeof(float) * kernelSize * kernelSize, cudaMemcpyHostToDevice);

    uchar* d_src;
    uchar* d_dst;

    // Allocate memory for the source and destination images on the GPU
    cudaMalloc((void**)&d_src, imageSize);
    cudaMalloc((void**)&d_dst, imageSize);

    // Copy the source image to the GPU
    cudaMemcpy(d_src, source.data, imageSize, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // Call the Gaussian blur kernel function
    gaussianBlurCUDA << <grid, block >> > (d_src, d_dst, width, height, d_kernel, kernelSize);
    cudaDeviceSynchronize();

    // Create the result image
    cv::Mat result(height, width, CV_8U);
    cudaMemcpy(result.data, d_dst, imageSize, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_kernel);
    cudaFree(d_src);
    cudaFree(d_dst);

    return result;
}

__global__ void cannyCUDA(const short* grad_x, const short* grad_y, uchar* dst, int width, int height, double low_thresh, double high_thresh)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1)
    {
        int offset = y * width + x;

        int dx = grad_x[offset];
        int dy = grad_y[offset];

        float gradientValue = sqrt(static_cast<float>(dx * dx + dy * dy));

        uchar* pixel = dst + offset;

        if (gradientValue >= high_thresh)
        {
            *pixel = 255;
        }
        else if (gradientValue >= low_thresh && gradientValue < high_thresh)
        {
            if (grad_x[offset + 1] >= high_thresh ||
                grad_x[offset - 1] >= high_thresh ||
                grad_x[offset + width] >= high_thresh ||
                grad_x[offset - width] >= high_thresh ||
                grad_x[offset + width + 1] >= high_thresh ||
                grad_x[offset - width - 1] >= high_thresh ||
                grad_x[offset + width - 1] >= high_thresh ||
                grad_x[offset - width + 1] >= high_thresh ||
                grad_y[offset + 1] >= high_thresh ||
                grad_y[offset - 1] >= high_thresh ||
                grad_y[offset + width] >= high_thresh ||
                grad_y[offset - width] >= high_thresh ||
                grad_y[offset + width + 1] >= high_thresh ||
                grad_y[offset - width - 1] >= high_thresh ||
                grad_y[offset + width - 1] >= high_thresh ||
                grad_y[offset - width + 1] >= high_thresh)
            {
                *pixel = 255;
            }
        }
    }
}

cv::Mat applyCanny(const cv::Mat& source, double low_thresh = 50, double high_thresh = 100, int aperture_size = 3, bool L2gradient = false)
{
    // Record the start time
    auto startTime = std::chrono::high_resolution_clock::now();

    Mat dst;

    // Calculate gradients (Sobel operators)
    Mat grad_x, grad_y;
    Sobel(source, grad_x, CV_16S, 1, 0, aperture_size);
    Sobel(source, grad_y, CV_16S, 0, 1, aperture_size);

    // Allocate GPU memory
    short* d_grad_x;
    short* d_grad_y;
    uchar* d_dst;
    int size = source.cols * source.rows;
    cudaMalloc((void**)&d_grad_x, size * sizeof(short));
    cudaMalloc((void**)&d_grad_y, size * sizeof(short));
    cudaMalloc((void**)&d_dst, size * sizeof(uchar));

    // Copy Sobel results to GPU
    cudaMemcpy(d_grad_x, grad_x.data, size * sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_y, grad_y.data, size * sizeof(short), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 block(16, 16);
    dim3 grid((source.cols + block.x - 1) / block.x, (source.rows + block.y - 1) / block.y);

    // Call the Canny kernel function
    cannyCUDA << <grid, block >> > (d_grad_x, d_grad_y, d_dst, source.cols, source.rows, low_thresh, high_thresh);
    cudaDeviceSynchronize();

    // Copy results back to host
    dst.create(source.size(), CV_8U);
    cudaMemcpy(dst.data, d_dst, size * sizeof(uchar), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_grad_x);
    cudaFree(d_grad_y);
    cudaFree(d_dst);

    // Record the end time
    auto endTime = std::chrono::high_resolution_clock::now();
    // Calculate and output the execution time in seconds
    auto executionTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000000.0;
    std::cout << "Canny Edge Detection completed in " << std::setprecision(7) << std::fixed << executionTime << " seconds." << std::endl;

    return dst;
}

void processFrame(Mat& frame)
{
    // Record the start time
    auto startTime = std::chrono::high_resolution_clock::now();

    // Convert the frame to grayscale
    Mat gray = applyGrayscale(frame);

    // Apply Gaussian blur
    Mat gBlur = applyGaussianBlur(gray, 3);

    // Apply Canny edge detection
    Mat edges = applyCanny(gBlur);

    // Display the Canny edge detection result
    imshow("Canny Edge Detection", edges);

    // Record the end time
    auto endTime = std::chrono::high_resolution_clock::now();
    // Calculate and output the execution time in seconds
    auto executionTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000000.0;
    std::cout << "Frame processed in " << std::setprecision(7) << std::fixed << executionTime << " seconds." << std::endl;
}

int main(int argc, char* argv[])
{
    // Control OpenCV logs
    utils::logging::setLogLevel(utils::logging::LOG_LEVEL_ERROR);

    // Define the number of threads
    cv::setNumThreads(NTHREAD);


    if (argc != 2) {
        std::cout << "Usage: ./exe path-to-video-or-image" << std::endl;
        return -1;
    }

    std::string inputPath = argv[1];
    bool isVideo = inputPath.find(".mp4") != std::string::npos;

    if (isVideo) {
        // Open the video file
        VideoCapture cap(inputPath);
        if (!cap.isOpened()) {
            std::cout << "Failed to open video file!" << std::endl;
            return -1;
        }

        while (true) {
            Mat frame;
            cap >> frame;

            if (frame.empty()) {
                break; // End of video
            }

            // Process the current frame
            processFrame(frame);

            if (waitKey(1) == 27) break; // Press 'ESC' to exit
        }
    }
    else {
        // Load the image
        Mat image = imread(inputPath);
        if (image.empty()) {
            std::cout << "Failed to load image!" << std::endl;
            return -1;
        }

        // Process the image
        processFrame(image);

        // Wait for a key press to close the display window
        waitKey(0);
    }

    return 0;
}
