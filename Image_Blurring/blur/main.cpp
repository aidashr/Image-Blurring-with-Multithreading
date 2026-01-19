#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <limits.h>

using namespace cv;
using namespace std;

float kernel[5][5] = {
    {1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25},
    {1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25},
    {1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25},
    {1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25},
    {1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25, 1.0 / 25}};

void applyBlur(const Mat &input, Mat &output, int startRow, int endRow)
{
    for (int y = startRow; y < endRow; ++y)
    {
        for (int x = 2; x < input.cols - 2; ++x)
        {
            for (int c = 0; c < input.channels(); ++c)
            {
                float sum = 0.0;
                for (int ky = -2; ky <= 2; ++ky)
                {
                    for (int kx = -2; kx <= 2; ++kx)
                    {
                        int px = x + kx;
                        int py = y + ky;
                        sum += input.at<Vec3b>(py, px)[c] * kernel[ky + 2][kx + 2];
                    }
                }
                output.at<Vec3b>(y, x)[c] = static_cast<uchar>(sum);
            }
        }
    }
}

void applyBlurRepeated(const Mat &input, Mat &output, int startRow, int endRow, int repetitions)
{
    Mat temp1 = input.clone();
    Mat temp2 = input.clone();

    for (int r = 0; r < repetitions; ++r)
    {
        applyBlur(temp1, temp2, startRow, endRow);
        std::swap(temp1, temp2);
    }

    output = temp1.clone();
}

void blurImageMultithreaded(const string &inputPath, const string &outputPath, int threadCount, int blurPasses)
{
    cout << "Trying to load: " << inputPath << endl;
    Mat input = imread(inputPath);
    if (input.empty())
    {
        char fullPath[PATH_MAX];
        realpath(inputPath.c_str(), fullPath);
        cerr << "Error loading image: " << fullPath << endl;
        return;
    }

    struct BlurTask
    {
        int startRow, endRow;
        Mat result;
    };

    int kernelMargin = 2;
    int rowsPerThread = input.rows / threadCount;
    vector<BlurTask> results(threadCount);
    vector<thread> threads;

    for (int i = 0; i < threadCount; ++i)
    {
        int startRow = i * rowsPerThread;
        int endRow = (i == threadCount - 1) ? input.rows : (i + 1) * rowsPerThread;

        int safeStart = max(0, startRow - kernelMargin);
        int safeEnd = min(input.rows, endRow + kernelMargin);

        threads.emplace_back([=, &input, &results]()
                             {
            Mat localOut = input.clone();
            applyBlurRepeated(input, localOut, safeStart, safeEnd, blurPasses);
            Mat region = localOut.rowRange(startRow, endRow).clone();

            BlurTask task{startRow, endRow, region};
            results[i] = std::move(task); });
    }

    for (auto &t : threads)
        t.join();

    Mat output = input.clone();
    for (const auto &task : results)
    {
        if (!task.result.empty())
        {
            task.result.copyTo(output.rowRange(task.startRow, task.endRow));
        }
    }

    imwrite(outputPath, output);
}

void blurImageSingleThreaded(const string &inputPath, const string &outputPath, int blurPasses)
{
    cout << "Trying to load: " << inputPath << endl;
    Mat input = imread(inputPath);
    if (input.empty())
    {
        char fullPath[PATH_MAX];
        realpath(inputPath.c_str(), fullPath);
        cerr << "Error loading image: " << fullPath << endl;
        return;
    }

    Mat output;
    applyBlurRepeated(input, output, 2, input.rows - 2, blurPasses);
    imwrite(outputPath, output);
}

int main()
{
    vector<string> inputImages = {
        "../images/image1.jpg",
        "../images/image2.jpg",
        "../images/image3.jpg",
        "../images/image4.jpg"};

    vector<string> outSingle = {
        "../output/single_blur1.jpg",
        "../output/single_blur2.jpg",
        "../output/single_blur3.jpg",
        "../output/single_blur4.jpg"};

    vector<string> outMulti = {
        "../output/multi_blur1.jpg",
        "../output/multi_blur2.jpg",
        "../output/multi_blur3.jpg",
        "../output/multi_blur4.jpg"};

    int blurPasses = 2;
    int threadsPerImage = 4;

    cout << "Running single-threaded blur...\n";
    auto start1 = chrono::high_resolution_clock::now();
    for (int i = 0; i < 4; ++i)
    {
        blurImageSingleThreaded(inputImages[i], outSingle[i], blurPasses);
    }
    auto end1 = chrono::high_resolution_clock::now();
    cout << "Single-threaded time: "
         << chrono::duration_cast<chrono::milliseconds>(end1 - start1).count()
         << " ms\n";

    cout << "Running multi-threaded blur...\n";
    auto start2 = chrono::high_resolution_clock::now();
    vector<thread> imageThreads;
    for (int i = 0; i < 4; ++i)
    {
        imageThreads.emplace_back(blurImageMultithreaded, inputImages[i], outMulti[i], threadsPerImage, blurPasses);
    }
    for (auto &t : imageThreads)
        t.join();
    auto end2 = chrono::high_resolution_clock::now();
    cout << "Multi-threaded time: "
         << chrono::duration_cast<chrono::milliseconds>(end2 - start2).count()
         << " ms\n";

    return 0;
}
