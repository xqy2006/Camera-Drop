// Author: Coast23
// Date: 2026-03-03
/*
 基本配置
 */

#pragma once
#include <cstdint>

class Config {
public:
    static const uint32_t BITS_PER_UNIT = 6; // 每个图案单元能编码的位数
    static constexpr uint32_t UNITS_PER_BYTE = 8 / BITS_PER_UNIT; // 每个字节能编码多少图案单元
    static const uint32_t PACKET_CAPACITY = 9300; // 数据包容量（字节）

    static const uint32_t RS_DATA_SIZE = 154;  // RS 数据字节数
    static const uint32_t RS_PARITY_SIZE = 32; // RS 校验字节数
    static constexpr uint32_t RS_BLOCK_SIZE = RS_DATA_SIZE + RS_PARITY_SIZE;

    static constexpr uint32_t FOUNTAIN_PAYLOAD_SIZE = // 有效载荷（不含 ECC）大小
                                PACKET_CAPACITY / RS_BLOCK_SIZE * RS_DATA_SIZE;
    static const uint32_t FOUNTAIN_HEADER_SIZE = 10;  // 帧头大小 file_size(4) + original_size(4) + block_id(2)
    static const uint32_t FOUNTAIN_CRC_SIZE = 4;      // 帧尾 CRC 大小
    static constexpr uint32_t FOUNTAIN_CHUNK_SIZE =   // 块大小
                                FOUNTAIN_PAYLOAD_SIZE - FOUNTAIN_HEADER_SIZE - FOUNTAIN_CRC_SIZE;

    static constexpr uint32_t MAX_FILE_SIZE = 200 * 1024 * 1024; // 限制文件大小不超过 200 MB
    
    static constexpr float REDUNDANCY_FACTOR = 2.0f; // 冗余系数

    static const int COMPRESSION_LEVEL = 9; // Zstd 压缩等级
};

