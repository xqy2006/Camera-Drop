#pragma once

#include "reed_solomon.hpp"
#include "fountain_code.hpp"
#include "util/config.hpp"
#include "util/DataPacket.hpp"
#include "util/file.hpp"
#include "util/errors.hpp"
#include "util/ZstdCompressor.hpp"

class Encoder {
public:
    
    Encoder(const std::string& filename, uint8_t encode_id = 0){
        FileReader reader(filename);
        if(!reader.is_open()){
            throw EncoderInitError("Failed to open file: " + filename);
        }
        if(reader.file_size() > Config::MAX_FILE_SIZE){
            throw EncoderInitError("File size exceeds limit: " + std::to_string(reader.file_size()));
        }
        
        std::vector<uint8_t> file_data = reader.read_all();
        if(reader.file_size() != 0 and file_data.empty()){
            throw EncoderInitError("Failed to read file: " + filename);
        }

        ZstdCompressor compressor(Config::COMPRESSION_LEVEL);
        data_ = compressor.compress(file_data.data(), file_data.size(), filename);
        if(data_.empty()){
            throw EncoderInitError("Failed to compress file: " + filename);
        }

        fountain_encoder_ = std::make_unique<FountainEncoder>(data_, encode_id_);
    }

    bool is_valid() const {
        return fountain_encoder_ && fountain_encoder_->is_valid();
    }

    // 生成下一个传输包
    std::vector<uint8_t> get_packet(){
        if(!is_valid()) return {};
        
        // 1. 生成 Fountain 编码块
        DataPacket fountain_packet = fountain_encoder_->encode_block();
        std::vector<uint8_t> fountain_data = fountain_packet.serialize();
        
        // 2. 对 Fountain 数据进行分块并 RS 编码
        std::vector<uint8_t> result;
        RSEncoder rs_encoder;
        
        size_t offset = 0;
        while(offset < fountain_data.size()){
            size_t chunk_size = std::min((size_t)Config::RS_DATA_SIZE, fountain_data.size() - offset);
            std::vector<uint8_t> chunk(
                fountain_data.begin() + offset,
                fountain_data.begin() + offset + chunk_size
            );
            std::vector<uint8_t> encoded = rs_encoder.encode(chunk);
            if(encoded.empty()) return {}; // TODO: Throw an exception?
            
            result.insert(result.end(), encoded.begin(), encoded.end());
            offset += chunk_size;
        }
        return result;
    }

    uint32_t packet_count_recommended() const {
        if(!is_valid()) return 0;
        return fountain_encoder_->blocks_recommended();
    }
    
private:
    uint8_t encode_id_;
    std::vector<uint8_t> data_;
    std::unique_ptr<FountainEncoder> fountain_encoder_;
};