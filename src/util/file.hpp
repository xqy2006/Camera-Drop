#pragma once

#include "config.hpp"
#include <string>
#include <vector>
#include <cstdint>
#include <fstream>

class FileReader {
public:
    explicit FileReader(const std::string& filename) : file_(filename, std::ios::binary) {}
    ~FileReader(){
        if(file_.is_open()) file_.close();
    }

    bool is_open() const {
        return file_.is_open();
    }
    
    // 读取 size 大小的数据（先无视 MAX_FILE_SIZE 限制）
    std::vector<uint8_t> read(size_t size){
        std::vector<uint8_t> buffer(size);
        file_.read(reinterpret_cast<char*>(buffer.data()), size);
        size_t read_size = file_.gcount();
        buffer.resize(read_size);
        return buffer;
    }

    size_t file_size(){
        file_.seekg(0, std::ios::end);
        size_t size = file_.tellg();
        file_.seekg(0, std::ios::beg);
        return size;
    }

    // 全部读入
    std::vector<uint8_t> read_all(){
        size_t size = file_size();

        if(size > Config::MAX_FILE_SIZE){
            // TODO: Throw an error or output warning message?
            return {};
        }
        return read(size);
    }

    bool eof() const {
        return file_.eof();
    }

    void reset(){
        file_.clear();
        file_.seekg(0, std::ios::beg);
    }

private:
    std::ifstream file_;
};

class FileWriter {
public:
    explicit FileWriter(const std::string& filename) : file_(filename, std::ios::binary) {}     
    ~FileWriter(){
        if(file_.is_open()) file_.close();
    }

    bool is_open() const {
        return file_.is_open();
    }

    // 写入 size 大小的数据
    bool write(const uint8_t* data, size_t size){
        file_.write(reinterpret_cast<const char*>(data), size);
        return file_.good();
    }

    // 全部写入
    bool write(const std::vector<uint8_t>& data){
        return write(data.data(), data.size());
    }

private:
    std::ofstream file_;
};
