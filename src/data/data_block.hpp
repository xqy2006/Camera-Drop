#pragma once
#include <vector>
#include <memory>
#include <cstdint>

class DataBlock {
public:
    DataBlock(size_t capacity = 0);
    DataBlock(const uint8_t* data, size_t size);
  
    void append(const uint8_t* data, size_t size);
    void clear();
    void resize(size_t new_size);

    uint8_t* data(){
        return buffer_.data();
    }
    const uint8_t* data() const {
        return buffer_.data();
    }

    size_t size() const {
        return buffer_.size();
    } 
    size_t capacity() const {
        return buffer_.capacity();
    }

    bool empty() const {
        return buffer_.empty();
    }

    void reserve(size_t capacity){
        buffer_.reserve(capacity);
    }

private:
    std::vector<uint8_t> buffer_;
};
