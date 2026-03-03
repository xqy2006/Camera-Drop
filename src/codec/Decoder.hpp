#pragma once

#include "reed_solomon.hpp"
#include "fountain_code.hpp"
#include "util/config.hpp"
#include "util/DataPacket.hpp"
#include "util/file.hpp"
#include "util/errors.hpp"
#include "util/ZstdCompressor.hpp"

class Decoder {
public:
    Decoder() : fountain_decoder_(std::make_unique<FountainDecoder>()) {}

    bool process_packet(const std::vector<uint8_t>& packet_data){
        std::vector<uint8_t> decoded_data;
        RSDecoder rs_decoder;

        size_t offset = 0;
        while(offset < packet_data.size()){
            if(offset + Config::RS_BLOCK_SIZE > packet_data.size()){
                break;
            }

            std::vector<uint8_t> rs_block(
                packet_data.begin() + offset,
                packet_data.begin() + offset + Config::RS_BLOCK_SIZE
            );

            std::vector<uint8_t> decoded = rs_decoder.decode(rs_block);

            if(decoded.empty()) return false;  // 解包失败
            
            decoded_data.insert(
                decoded_data.end(), decoded.begin(), decoded.end()
            );
            offset += Config::RS_BLOCK_SIZE;
        }

        DataPacket packet;
        if(!packet.deserialize(decoded_data.data(), decoded_data.size())) return false;  // 反序列化失败
        return fountain_decoder_->add_block(packet);
    }

    bool is_complete() const {
        return fountain_decoder_->is_complete();
    }

    bool save_to_file(const std::string& filename){
        if(!is_complete()) return false;
        std::vector<uint8_t> data = fountain_decoder_->decode();
      
        ZstdDecompressor decompressor;
        auto decompressed = decompressor.decompress(data.data(), data.size());
        auto data = decompressed.first;
        //  if(data.empty()) return false;
    
        FileWriter writer(filename);
        if(!writer.is_open()) return false;
        return writer.write(data);
    }

private:
    std::unique_ptr<FountainDecoder> fountain_decoder_;
};