#include "image/patterns.hpp"
#include "image/constants.hpp"
#include "image/img_writer.hpp"

#include <cstdio>
#include <vector>
#include <random>
#include <chrono>

#include <opencv2/opencv.hpp>

int main(){
    Mat img(IMG_HEIGHT, IMG_WIDTH, CV_8UC3, Scalar(0, 0, 0));

    std::vector<uint8_t> raw_data(GRID_R * GRID_C + 1);
    
    uint64_t seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist_data(0, NUM_PATTERNS * NUM_COLORS - 1);

    std::vector<uint8_t> data(Config::PACKET_CAPACITY);
    for(auto& x : data) x = dist_data(rng);

    write_8bits_data(img, data);

    std::string imWindow = "code";
    imshow(imWindow, img);
    cv::waitKey();

    std::string filename = "encode.png";
    imwrite(filename, img);
}
