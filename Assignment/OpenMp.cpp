#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <omp.h>

#define NTHREAD 2
using namespace std;
using namespace cv;

void applyCannyParallel(Mat& src, Mat& dst, double low_thresh, double high_thresh, int aperture_size, bool L2gradient)
{
#pragma omp parallel for
    for (int i = 0; i < src.rows; i++)
    {
        cv::Canny(src.row(i), dst.row(i), low_thresh, high_thresh, aperture_size, L2gradient);
    }
}

void parallelGaussianBlur(Mat& src, Mat& dst)
{
#pragma omp parallel for
    for (int i = 0; i < src.rows; i++)
    {
        GaussianBlur(src.row(i), dst.row(i), Size(5, 5), 0);
    }
}

int main()
{
    auto totalTimeStart = std::chrono::high_resolution_clock::now();

    std::string inputPath = "C:\\Users\\User\\source\\repos\\Assignment\\images\\video1.mp4";

    double grayScaleTime = 0.0;
    double gaussianBlurTime = 0.0;
    double cannyEdgeTime = 0.0;
    double cannyEdgeDetectionTime = 0.0;

    namedWindow("Original", WINDOW_NORMAL);
    namedWindow("Lane Detected", WINDOW_NORMAL);

    VideoCapture cap(inputPath);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open video file." << std::endl;
        return -1;
    }

    std::cout << "Processing video file..." << std::endl;

    Mat frame;
    while (cap.read(frame))
    {
        imshow("Original", frame);

        auto grayScaleStart = std::chrono::high_resolution_clock::now();
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        auto grayScaleEnd = std::chrono::high_resolution_clock::now();
        grayScaleTime += std::chrono::duration_cast<std::chrono::milliseconds>(grayScaleEnd - grayScaleStart).count() / 1000000.0;

        auto gaussianBlurStart = std::chrono::high_resolution_clock::now();
        Mat blurred(gray.size(), CV_8UC1);
        parallelGaussianBlur(gray, blurred);
        auto gaussianBlurEnd = std::chrono::high_resolution_clock::now();
        gaussianBlurTime += std::chrono::duration_cast<std::chrono::milliseconds>(gaussianBlurEnd - gaussianBlurStart).count() / 1000000.0;

        auto cannyEdgeDetectionStart = std::chrono::high_resolution_clock::now();

        double lowThreshold = 50;
        double highThreshold = 150;
        int apertureSize = 3;
        bool L2gradient = false;
        Mat edges;
        Canny(blurred, edges, lowThreshold, highThreshold, apertureSize, L2gradient);

        auto cannyEdgeDetectionEnd = std::chrono::high_resolution_clock::now();
        cannyEdgeDetectionTime += std::chrono::duration_cast<std::chrono::milliseconds>(cannyEdgeDetectionEnd - cannyEdgeDetectionStart).count() / 1000000.0;

        cannyEdgeTime += cannyEdgeDetectionTime;

        imshow("Lane Detected", edges);

        if (waitKey(30) == 27)
            break;
    }

    cap.release();
    std::cout << "=========================================================================" << std::endl;
    std::cout << "Grayscale conversion completed in " << grayScaleTime << " seconds." << std::endl;
    std::cout << "Gaussian blur completed in " << gaussianBlurTime << " seconds." << std::endl;
    std::cout << "Canny edge detection completed in " << cannyEdgeDetectionTime << " seconds." << std::endl;

    auto totalTimeEnd = std::chrono::high_resolution_clock::now();
    auto totalExecutionTime = std::chrono::duration_cast<std::chrono::milliseconds>(totalTimeEnd - totalTimeStart).count() / 1000000.0;
    std::cout << "Total processing time: " << totalExecutionTime << " seconds." << std::endl;
    std::cout << "=========================================================================" << std::endl;
    waitKey(0);
    destroyAllWindows();

    return 0;
}
