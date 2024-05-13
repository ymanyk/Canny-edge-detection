#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace std;
using namespace cv;

int main()
{
    auto totalTimeStart = std::chrono::high_resolution_clock::now();

    std::string inputPath = "C:\\Users\\User\\source\\repos\\Assignment\\images\\image5.jpeg";

    double grayScaleTime = 0.0;
    double gaussianBlurTime = 0.0;
    double cannyEdgeDetectionTime = 0.0;

    namedWindow("Original", WINDOW_NORMAL);
    namedWindow("Lane Detected", WINDOW_NORMAL);

    VideoCapture cap(inputPath);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Failed to open video file." << std::endl;
        return -1;
    }

    std::cout << "Processing video file..." << std::endl;

    Mat frame, gray, blurred, edges;
    while (cap.read(frame))
    {
        auto processingStart = std::chrono::high_resolution_clock::now();

        // Convert the frame to grayscale
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Apply Gaussian blur
        GaussianBlur(gray, blurred, Size(5, 5), 0);

        // Apply Canny edge detection
        double lowThreshold = 50;
        double highThreshold = 150;
        int apertureSize = 3;
        bool L2gradient = false;
        Canny(blurred, edges, lowThreshold, highThreshold, apertureSize, L2gradient);

        auto processingEnd = std::chrono::high_resolution_clock::now();
        double processingTime = std::chrono::duration_cast<std::chrono::milliseconds>(processingEnd - processingStart).count() / 100000.0;

        grayScaleTime += processingTime;
        gaussianBlurTime += processingTime;
        cannyEdgeDetectionTime += processingTime;

        // Display original and processed images
        imshow("Original", frame);
        imshow("Lane Detected", edges);

        if (waitKey(30) == 27) // Press ESC to exit
            break;
    }

    cap.release();
    std::cout << "=========================================================================" << std::endl;
    std::cout << "Grayscale conversion time: " << grayScaleTime << " seconds." << std::endl;
    std::cout << "Gaussian blur time: " << gaussianBlurTime << " seconds." << std::endl;
    std::cout << "Canny edge detection time: " << cannyEdgeDetectionTime << " seconds." << std::endl;

    auto totalTimeEnd = std::chrono::high_resolution_clock::now();
    double totalExecutionTime = std::chrono::duration_cast<std::chrono::milliseconds>(totalTimeEnd - totalTimeStart).count() / 100000.0;
    std::cout << "Total processing time: " << totalExecutionTime << " seconds." << std::endl;
    std::cout << "=========================================================================" << std::endl;

    waitKey(0);
    destroyAllWindows();

    return 0;
}
