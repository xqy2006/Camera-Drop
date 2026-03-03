#pragma once

#include "util/config.hpp"

#include <zstd.h>
#include <memory>
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>

class ZstdCompressor {
public:
    ZstdCompressor(int compression_level = Config::COMPRESSION_LEVEL)
        : compression_level_(compression_level) {}

    // 压缩数据，文件名会添加到 skippable frame
    std::vector<uint8_t> compress(const uint8_t* data, size_t size, const std::string& filename = ""){
        std::vector<uint8_t> result;
        
        if(!filename.empty()) write_skippable_frame(result, filename);
        
        size_t bound = ZSTD_compressBound(size);
        std::vector<uint8_t> compressed(bound);
        
        size_t compressed_size = ZSTD_compress(
            compressed.data(),
            bound,
            data,
            size,
            compression_level_
        );

        if(ZSTD_isError(compressed_size)) return {};
        
        result.insert(result.end(), compressed.data(), compressed.data() + compressed_size);
        return result;
    }

private:
    int compression_level_;

    // 将 skippable frame 内容打包进 output
    void write_skippable_frame(std::vector<uint8_t>& output, const std::string& filename){
        const uint32_t magic = 0x184D2A50;
        const uint32_t frame_size = filename.size();
        
        output.resize(8 + frame_size);
        memcpy(output.data(), &magic, 4);
        memcpy(output.data() + 4, &frame_size, 4);
        memcpy(output.data() + 8, filename.data(), frame_size);
    }
};

class ZstdDecompressor {
public:
    struct Result {
        std::vector<uint8_t> data;
        std::string filename;
    };

    Result decompress(const uint8_t* compressed, size_t compressed_size){
        Result result;
        size_t offset = 0;
        if(compressed_size >= 8){
            uint32_t magic;
            memcpy(&magic, compressed, 4);
            if ((magic & 0xFFFFFFF0) == 0x184D2A50){
                uint32_t frame_size;
                memcpy(&frame_size, compressed + 4, 4);
                if (8 + frame_size <= compressed_size){
                    result.filename = std::string(
                        reinterpret_cast<const char*>(compressed + 8),
                        frame_size
                    );
                    offset = 8 + frame_size;
                }
            }
        }

        unsigned long long decompressed_size = ZSTD_getFrameContentSize(
            compressed + offset,
            compressed_size - offset
        );

        if(decompressed_size == ZSTD_CONTENTSIZE_ERROR || decompressed_size == ZSTD_CONTENTSIZE_UNKNOWN){
            // throw an exception?
            return result;
        }

        result.data.resize(decompressed_size);
        size_t actual_size = ZSTD_decompress(
            result.data.data(), decompressed_size,
            compressed + offset, compressed_size - offset
        );

        if(ZSTD_isError(actual_size)){
            result.data.clear();
        }

        return result;
    }
};

/*
namespace ZstdPack {

    // 压缩
    inline std::vector<uint8_t> _compress(const std::vector<uint8_t>& src, int level = 6){
        if (src.empty()) return {};
        size_t bound = ZSTD_compressBound(src.size());
        std::vector<uint8_t> res(bound);
        
        size_t cSize = ZSTD_compress(res.data(), res.size(), src.data(), src.size(), level);
        // hope this won't happen... :)
        if(ZSTD_isError(cSize)){
            throw std::runtime_error(std::string("ZSTD Compression failed: ") + ZSTD_getErrorName(cSize));
        }

        res.resize(cSize);
        return res;
    }

    // 流式解压
    inline std::vector<uint8_t> _decompress(const std::vector<uint8_t>& src){
        if(src.empty()) return {};
        // wtf what's this?
        std::unique_ptr<ZSTD_DCtx, decltype(&ZSTD_freeDCtx)> dctx(ZSTD_createDCtx(), ZSTD_freeDCtx);
        
        std::vector<uint8_t> res;
        ZSTD_inBuffer input = {src.data(), src.size(), 0};
   
        const size_t outBuffSize = ZSTD_DStreamOutSize();
        std::vector<uint8_t> outBuff(outBuffSize);
        
        while(input.pos < input.size){
            ZSTD_outBuffer output = {outBuff.data(), outBuff.size(), 0};
            size_t ret = ZSTD_decompressStream(dctx.get(), &output, &input);
            // hope this won't happen... :)
            if(ZSTD_isError(ret)){
                throw std::runtime_error(std::string("ZSTD Decompression failed: ") + ZSTD_getErrorName(ret));
            }
            res.insert(res.end(), outBuff.begin(), outBuff.begin() + output.pos);
        }
        return res;
    }

    // 写入 32 位小端整数
    inline void _write_le32(std::vector<uint8_t>& out, uint32_t val){
        out.push_back(val & 0xFF);
        out.push_back((val >> 8) & 0xFF);
        out.push_back((val >> 16) & 0xFF);
        out.push_back((val >> 24) & 0xFF);
    }
    // 读取 32 位小端整数
    inline uint32_t _read_le32(const uint8_t* ptr){
        return ptr[0] | (ptr[1] << 8) | (ptr[2] << 16) | (ptr[3] << 24);
    }

    // 打包。利用 Zstd 的 Skippable Frame，把文件名藏在文件头部
    inline std::vector<uint8_t> pack(const std::string& filename, const std::vector<uint8_t>& src, int level = 9){
        std::vector<uint8_t> res;
        // Magic Number of Skippable Frame
        _write_le32(res, 0x184D2A50);
    
        std::string payload = '\x23' + filename;
        _write_le32(res, static_cast<uint32_t>(payload.size()));   // 写入载荷长度
        res.insert(res.end(), payload.begin(), payload.end());     // 写入文件名载荷
        
        std::vector<uint8_t> comp_data = _compress(src, level);  
        res.insert(res.end(), comp_data.begin(), comp_data.end()); // 写入压缩数据
        return res;
    }
    // 解包，返回值表示是否成功
    inline bool unpack(const std::vector<uint8_t>& packed, std::string& filename, std::vector<uint8_t>& src){
        if(packed.size() < 8) return false;
        uint32_t magic = _read_le32(packed.data());
        size_t offset = 0;

        // 提取文件名
        if(magic == 0x184D2A50){
            uint32_t frame_size = _read_le32(packed.data() + 4);
            if(8 + frame_size > packed.size()) return false;
            if(frame_size > 0 and packed[8] == 0x23){
                filename = std::string(packed.begin() + 9, packed.begin() + 8 + frame_size);
            }
            offset = 8 + frame_size;
        }

        // 提取压缩数据并解压
        std::vector<uint8_t> comp_data(packed.begin() + offset, packed.end());
        src = _decompress(comp_data);
        return true;
    }
};*/