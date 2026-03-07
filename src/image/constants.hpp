#pragma once

const int GRID_SIZE = 112;
const int STRIDE    = 9;
const int MARGIN    = 8;
const int IMG_SIZE  = 1024;

// Anchor 层级常量（TL/TR/BL 和 BR 共用相同尺寸，仅颜色不同）
const int ANCHOR_OUT_START = 2;
const int ANCHOR_L1_SIZE   = 56;   // 外层彩色
const int ANCHOR_L2_INSET  = 7;    // 黑色层 inset
const int ANCHOR_L2_SIZE   = 42;
const int ANCHOR_L3_INSET  = 14;   // 内彩色层 inset
const int ANCHOR_L3_SIZE   = 28;
const int ANCHOR_L4_INSET  = 21;   // 内黑色层 inset
const int ANCHOR_L4_SIZE   = 14;