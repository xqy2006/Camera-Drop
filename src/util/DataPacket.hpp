#pragma once
#include "config.hpp"
#include <vector>
#include <cstring>
#include <cstdint>

struct FountainMetadata {
    uint32_t file_size;
    uint16_t block_id;
    uint8_t encode_id;

    void serialize(uint8_t* buffer) const {
        std::memcpy(buffer, &file_size, 4);
        std::memcpy(buffer + 4, &block_id, 2);
        buffer[6] = encode_id;
    }

    void deserialize(const uint8_t* buffer){
        std::memcpy(&file_size, buffer, 4);
        std::memcpy(&block_id, buffer + 4, 2);
        encode_id = buffer[6];
    }
};

class DataPacket {
public:
    DataPacket() : metadata_(), data_() {}

    void set_metadata(const FountainMetadata& meta){
        metadata_ = meta;
    }

    const FountainMetadata& metadata() const {
        return metadata_;
    }

    void set_data(const std::vector<uint8_t>& data){
        data_ = data;
    }

    const std::vector<uint8_t>& data() const {
        return data_;
    }

    // 序列化为字节流
    std::vector<uint8_t> serialize() const {
        std::vector<uint8_t> result(Config::FOUNTAIN_HEADER_SIZE + data_.size());
        metadata_.serialize(result.data());
        std::memcpy(result.data() + Config::FOUNTAIN_HEADER_SIZE, data_.data(), data_.size());
        return result;
    }

    // 从字节流反序列化
    bool deserialize(const uint8_t* buffer, size_t size){
        if(size < Config::FOUNTAIN_HEADER_SIZE) return false; // impossible
        metadata_.deserialize(buffer);
        // data_.resize(size - Config::FOUNTAIN_HEADER_SIZE);
        // std::memcpy(data_.data(), buffer + Config::FOUNTAIN_HEADER_SIZE, data_.size());
        size_t actual_size = size - Config::FOUNTAIN_HEADER_SIZE;
        actual_size = std::min(actual_size, (size_t)Config::FOUNTAIN_CHUNK_SIZE);
        data_.resize(actual_size);
        std::memcpy(data_.data(), buffer + Config::FOUNTAIN_HEADER_SIZE, actual_size);
        return true;
    }

private:
    FountainMetadata metadata_;
    std::vector<uint8_t> data_;
};
