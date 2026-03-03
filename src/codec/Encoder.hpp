#pragma once

#include "util/config.hpp"
#include "util/DataPacket.hpp"
#include "util/file.hpp"
#include "reed_solomon.hpp"
#include "fountain_code.hpp"

class Encoder {
public:
    explicit Encoder(const std::string& filename, uint8_t encode_id = 0)
        : encode_id_(encode_id) {
    }
private:
    uint8_t encode_id_;
    std::vector<uint8_t> file_data_;
    std::unique_ptr<FountainEncoder> fountain_encoder_;
};