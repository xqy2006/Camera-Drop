#pragma once

#include "patterns.hpp"

#include <cstdint>
#include <opencv2/opencv.hpp>

constexpr int P_BITS = std::__lg(NUM_PATTERNS);
constexpr int C_BITS = std::__lg(NUM_COLORS);

const int INF = 1145141919;

#if defined(_MSC_VER)
    #include <intrin.h>
    #define popcount64 __popcnt64
#else
    #define popcount64 __builtin_popcountll
#endif

class Tile {
public:
    Tile(const uint8_t pattern_idx, const uint8_t color_idx) :
        pattern_idx_(pattern_idx), color_idx_(color_idx) {
            data_ = color_idx_ << P_BITS | pattern_idx_;
        }
    
    Tile(const uint8_t data) : data_(data){
        data_ &= 0x3f;
        pattern_idx_ = data_ & (NUM_PATTERNS - 1);
        color_idx_   = data_ >> P_BITS;
    }

    uint8_t get_pattern_idx() const {
        return pattern_idx_;
    }

    uint8_t get_color_idx() const {
        return color_idx_;
    }

    uint8_t get_data() const {
        return data_;
    }

    uint64_t get_pattern() const {
        return PATTERNS[pattern_idx_];
    }

    cv::Vec3b get_color() const {
        return get_color(color_idx_);
    }

    static uint64_t get_pattern(uint8_t pattern_idx){
        return PATTERNS[pattern_idx];
    }

    static cv::Vec3b get_color(uint8_t color_idx){
        switch(color_idx){
            case 0: return cv::Vec3b(0, 255, 255);   // Yellow (R+G)
            case 1: return cv::Vec3b(0, 255, 0);     // Green  (G)
            case 2: return cv::Vec3b(255, 255, 0);   // Cyan   (G+B)
            case 3: return cv::Vec3b(255, 0, 255);   // Magenta(R+B)
            default: return cv::Vec3b(255, 255, 255);
        }
    }

    static uint8_t match_pattern(uint64_t mask) {
        uint8_t dist = 65, best = -1;
        for(uint8_t i = 0; i < 16; ++i){
            if(popcount64(mask ^ PATTERNS[i]) < dist){
                dist = popcount64(mask ^ PATTERNS[i]);
                best = i;
            }
        }
        return best;
    }

    static uint8_t match_color(cv::Vec3b color, cv::Vec3b* palette){
        int dist = INF; uint8_t best = 0;
        for(uint8_t i = 0; i < 8; ++i){
            int db = color[0] - palette[i][0];
            int dg = color[1] - palette[i][1];
            int dr = color[2] - palette[i][2];
            int d = db*db + dg*dg + dr*dr;
            if(d < dist){
                dist = d;
                best = i;
            }
        }
        return best;
    }

private:
    uint8_t pattern_idx_;
    uint8_t color_idx_;
    uint8_t data_;      // 6 bits
};