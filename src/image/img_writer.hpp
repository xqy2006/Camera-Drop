#pragma once

#include "painter.hpp"
#include "constants.hpp"
#include "util/config.hpp"
#include "util/BitConverter.hpp"

#include <new>
#include <vector>
#include <cstdint>

// 不会函数命名
void write_6bits_data(cv::Mat& img, uint8_t* data, size_t size){
    if(size != Config::UINTS_COUNT) return; // TODO: throw an exception

    Painter painter(img);

    int id = 0;
    for(uint8_t r = 0; r < GRID_SIZE; ++r){
        for(uint8_t c = 0; c < GRID_SIZE; ++c){
            if(Painter::is_reserved(r, c)) continue;
            painter.draw_tile(data[id++], r, c);
        }
    }
}

void write_6bits_data(cv::Mat& img, std::vector<uint8_t>& data){
    write_6bits_data(img, data.data(), data.size());
}

void write_8bits_data(cv::Mat& img, uint8_t* data, size_t size){
    size_t data_6bits_size = Config::UINTS_COUNT;
    uint8_t* data_6bits = new uint8_t[data_6bits_size];

    size_t res = BitConverter::convert_826(data, size, data_6bits, data_6bits_size);
    if(res != Config::UINTS_COUNT) return; // TODO: throw an exception
    write_6bits_data(img, data_6bits, data_6bits_size);
    delete[] data_6bits;
}

void write_8bits_data(cv::Mat& img, std::vector<uint8_t>& data){
    write_8bits_data(img, data.data(), data.size());
}