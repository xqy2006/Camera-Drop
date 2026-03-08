#pragma once

#include "util/config.hpp"

// 图像常量
static const int IMG_WIDTH  = Config::IMG_WIDTH;
static const int IMG_HEIGHT = Config::IMG_HEIGHT;
static const int STRIDE     = Config::STRIDE;
static const int MARGIN     = Config::MARGIN;

static constexpr int GRID_R = (IMG_HEIGHT - MARGIN * 2) / STRIDE;
static constexpr int GRID_C = (IMG_WIDTH - MARGIN * 2) / STRIDE;

// Anchor 层级常量（TL/TR/BL 和 BR 共用相同尺寸，仅颜色不同）
const int ANCHOR_OUT_START = 2;
const int ANCHOR_L1_SIZE   = 56;   // 外层彩色
const int ANCHOR_L2_INSET  = 7;    // 黑色层 inset
const int ANCHOR_L2_SIZE   = 42;
const int ANCHOR_L3_INSET  = 14;   // 内彩色层 inset
const int ANCHOR_L3_SIZE   = 28;
const int ANCHOR_L4_INSET  = 21;   // 内黑色层 inset
const int ANCHOR_L4_SIZE   = 14;