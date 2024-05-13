#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

#define NTHREAD 2

using namespace cv;
using namespace std;

// Function to apply Canny edge detection
void applyCanny(const Mat& source, Mat& edges) {
    auto startTime = std::chrono::high_resolution_clock::now(); // Record the start time

    // Apply Canny edge detection algorithm to source and store the result in edges
    Canny(source, edges, 50, 150);

    auto endTime = std::chrono::high_resolution_clock::now(); // Record the end time
    auto executionTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000000.0; // Convert to seconds
    std::cout << "Canny Edge Detection completed in " << std::setprecision(7) << std::fixed << executionTime << " seconds." << std::endl;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Input file path
    string inputFilePath;
    if (argc < 2) {
        // Default input file path
        inputFilePath = "C:\\Users\\User\\source\\repos\\Assignment\\images\\image5.jpeg";
    }
    else {
        // Input file path provided as command-line argument
        inputFilePath = argv[1];
    }

    // Output file path
    string outputFilePath;
    if (inputFilePath.substr(inputFilePath.find_last_of(".") + 1) == "mp4") {
        // Video input
        outputFilePath = "processed_output_video1.mp4";
    }
    else {
        // Image input
        outputFilePath = "processed_output_image1.jpg";
    }

    auto totalTimeStart = std::chrono::high_resolution_clock::now(); // Record the start time for the entire process

    // Only let rank 0 process the video or image
    if (rank == 0) {
        // Check if the input file is an image or video
        bool isVideo = inputFilePath.substr(inputFilePath.find_last_of(".") + 1) == "mp4";

        if (isVideo) {
            // Open the input video file
            VideoCapture video(inputFilePath);

            if (!video.isOpened()) {
                cerr << "Error: Unable to open input video file!" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            // Get video properties
            int frameWidth = static_cast<int>(video.get(CAP_PROP_FRAME_WIDTH));
            int frameHeight = static_cast<int>(video.get(CAP_PROP_FRAME_HEIGHT));
            double fps = video.get(CAP_PROP_FPS);

            // Define codec and create VideoWriter object
            VideoWriter writer(outputFilePath, VideoWriter::fourcc('H', '2', '6', '4'), fps, Size(frameWidth, frameHeight));

            if (!writer.isOpened()) {
                cerr << "Error: Unable to open VideoWriter!" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            // Process frames from the video
            Mat frame, edges;
            while (video.read(frame)) {
                auto grayScaleStart = std::chrono::high_resolution_clock::now(); // Record the start time for grayscale conversion
                Mat gray;
                cvtColor(frame, gray, COLOR_BGR2GRAY);
                auto grayScaleEnd = std::chrono::high_resolution_clock::now(); // Record the end time for grayscale conversion
                auto grayScaleTime = std::chrono::duration_cast<std::chrono::microseconds>(grayScaleEnd - grayScaleStart).count() / 1000000.0; // Convert to seconds
                std::cout << "Grayscale conversion completed in " << std::setprecision(7) << std::fixed << grayScaleTime << " seconds." << std::endl;

                auto gaussianBlurStart = std::chrono::high_resolution_clock::now(); // Record the start time for Gaussian blur
                Mat blurred;
                GaussianBlur(gray, blurred, Size(5, 5), 0);
                auto gaussianBlurEnd = std::chrono::high_resolution_clock::now(); // Record the end time for Gaussian blur
                auto gaussianBlurTime = std::chrono::duration_cast<std::chrono::microseconds>(gaussianBlurEnd - gaussianBlurStart).count() / 1000000.0; // Convert to seconds
                std::cout << "Gaussian blur completed in " << std::setprecision(7) << std::fixed << gaussianBlurTime << " seconds." << std::endl;

                // Apply Canny edge detection
                applyCanny(blurred, edges);

                // Write the processed frame to the video file
                writer.write(edges);
            }

            // Release VideoWriter for the video file
            writer.release();
        }
        else {
            // Open the input image file
            Mat image = imread(inputFilePath);

            if (image.empty()) {
                cerr << "Error: Unable to open input image file!" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            auto grayScaleStart = std::chrono::high_resolution_clock::now(); // Record the start time for grayscale conversion
            Mat gray;
            cvtColor(image, gray, COLOR_BGR2GRAY);
            auto grayScaleEnd = std::chrono::high_resolution_clock::now(); // Record the end time for grayscale conversion
            auto grayScaleTime = std::chrono::duration_cast<std::chrono::microseconds>(grayScaleEnd - grayScaleStart).count() / 1000000.0; // Convert to seconds
            std::cout << "Grayscale conversion completed in " << std::setprecision(7) << std::fixed << grayScaleTime << " seconds." << std::endl;

            auto gaussianBlurStart = std::chrono::high_resolution_clock::now(); // Record the start time for Gaussian blur
            Mat blurred;
            GaussianBlur(gray, blurred, Size(5, 5), 0);
            auto gaussianBlurEnd = std::chrono::high_resolution_clock::now(); // Record the end time for Gaussian blur
            auto gaussianBlurTime = std::chrono::duration_cast<std::chrono::microseconds>(gaussianBlurEnd - gaussianBlurStart).count() / 1000000.0; // Convert to seconds
            std::cout << "Gaussian blur completed in " << std::setprecision(7) << std::fixed << gaussianBlurTime << " seconds." << std::endl;

            // Process the image using Canny edge detection
            Mat imageEdges;
            applyCanny(blurred, imageEdges);

            // Save the processed image to a file
            imwrite(outputFilePath, imageEdges);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD); // Wait for all processes to finish before calculating total time

    auto totalTimeEnd = std::chrono::high_resolution_clock::now(); // Record the end time for the entire process
    auto totalExecutionTime = std::chrono::duration_cast<std::chrono::microseconds>(totalTimeEnd - totalTimeStart).count() / 1000000.0; // Convert to seconds
    std::cout << "Total processing time: " << std::setprecision(7) << std::fixed << totalExecutionTime << " seconds." << std::endl;

    MPI_Finalize();
    return 0;
}
