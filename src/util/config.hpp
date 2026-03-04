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
    
    static const uint32_t RS_DATA_SIZE = 223;  // RS 数据字节数
    static const uint32_t RS_PARITY_SIZE = 32; // RS 校验字节数
    static constexpr uint32_t RS_BLOCK_SIZE = RS_DATA_SIZE + RS_PARITY_SIZE;

    static const uint32_t FOUNTAIN_CHUNK_SIZE = 9300;
    static const uint32_t FOUNTAIN_HEADER_SIZE = 12;

    static constexpr uint32_t MAX_FILE_SIZE = 200 * 1024 * 1024;
    
    static constexpr float PRESET_LOSS_RATE = 0.25f; // 丢包率浅浅设为 25%
    static constexpr float REDUNDANCY_FACTOR = 1.5f; // 冗余系数

    static const int COMPRESSION_LEVEL = 9; // Zstd 压缩等级
};

