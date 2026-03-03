#pragma once
#include "util/config.hpp"
extern "C" {
    #include <correct.h>
}

#include <vector>
#include <memory>

class RSEncoder {
public:
    RSEncoder(){
        rs_ = correct_reed_solomon_create(
            correct_rs_primitive_polynomial_8_7_2_1_0,
            1, 1, Config::RS_PARITY_SIZE
        );
    }
    ~RSEncoder(){
        if(rs_) correct_reed_solomon_destroy(rs_);
    }

    // 编码一个数据块
    std::vector<uint8_t> encode(const std::vector<uint8_t>& data){
        if(data.size() > Config::RS_DATA_SIZE) return {};
        std::vector<uint8_t> encoded(Config::RS_BLOCK_SIZE);   
        ssize_t res = correct_reed_solomon_encode(rs_, data.data(), data.size(), encoded.data());
        return encoded;
    }
private:
    correct_reed_solomon* rs_;
};

class RSDecoder {
public:
    RSDecoder(){
        rs_ = correct_reed_solomon_create(
            correct_rs_primitive_polynomial_8_7_2_1_0,
            1, 1, Config::RS_PARITY_SIZE
        );
    }
    ~RSDecoder(){
        if(rs_) correct_reed_solomon_destroy(rs_);
    }

    // 解码一个数据块。解码失败返回空
    std::vector<uint8_t> decode(const std::vector<uint8_t>& encoded){
        if(encoded.size() != Config::RS_BLOCK_SIZE) return {};
        
        std::vector<uint8_t> decoded(Config::RS_DATA_SIZE);
        ssize_t res = correct_reed_solomon_decode(rs_, encoded.data(), Config::RS_BLOCK_SIZE, decoded.data());
        if(res < 0) return {};

        return decoded;
    }

private:
    correct_reed_solomon* rs_;
};