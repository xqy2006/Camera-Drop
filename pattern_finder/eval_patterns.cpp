// eval_patterns.cpp
// 批量评测任意 pattern 组合的多档位缩放识别准确率。
//
// 协议（stdin → stdout，JSON）：
//   输入：{ "scales": [1.0, 0.9, ...], "weights": [1, 2, ...],
//           "seeds":  [0, 1, 2],
//           "combos": [ [mask0, mask1, ..., mask15], ... ] }
//     每个 mask 是 uint64，bit(r*8+c)=1 表示该像素为前景
//   输出：{ "scores": [score0, score1, ...] }
//     score ∈ [0,1]，加权 pattern 识别准确率
//
// 编译后由 select_patterns.py 通过 subprocess 调用，
// 每次调用传入一批组合，避免重复启动进程。

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <cassert>
#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>

// ── popcount 兼容层 ───────────────────────────────────────────────────────────
#ifdef _MSC_VER
#  include <intrin.h>
static inline int popcount64(uint64_t x) { return (int)__popcnt64(x); }
#else
static inline int popcount64(uint64_t x) { return __builtin_popcountll(x); }
#endif

// ── 与 config_acc_test.cpp 完全一致的布局常量 ─────────────────────────────
static const int GRID_SIZE = 112;
static const int STRIDE    = 9;
static const int MARGIN    = 8;
static const int IMG_SIZE  = 1024;
static const int NUM_COLORS = 4;

// ── 颜色表（BGR） ──────────────────────────────────────────────────────────
static const cv::Vec3b COLORS[4] = {
    {0, 255, 255},   // 0 Yellow
    {0, 255, 0},     // 1 Green
    {255, 255, 0},   // 2 Cyan
    {255, 0, 255},   // 3 Magenta
};

static inline bool is_reserved(int r, int c) {
    if (r > 105 && c > 105) return true;
    if (r == 0 && c >= 6 && c < 46) return true;
    return false;
}

static inline int match_color(const cv::Vec3b& px) {
    int best = 0, best_d = INT_MAX;
    for (int i = 0; i < NUM_COLORS; ++i) {
        int db = (int)px[0] - COLORS[i][0];
        int dg = (int)px[1] - COLORS[i][1];
        int dr = (int)px[2] - COLORS[i][2];
        int d  = db*db + dg*dg + dr*dr;
        if (d < best_d) { best_d = d; best = i; }
    }
    return best;
}

static inline int match_pattern(uint64_t tile_bits,
                                 const std::vector<uint64_t>& dict) {
    int best = 0, best_d = 65;
    for (int i = 0; i < (int)dict.size(); ++i) {
        int d = popcount64(tile_bits ^ dict[i]);
        if (d < best_d) { best_d = d; best = i; }
    }
    return best;
}

// ── 信道模拟（与 config_acc_test.cpp 完全一致） ───────────────────────────
static void stimulate_moire(cv::Mat& img) {
    for (int r = 0; r < IMG_SIZE; ++r) {
        for (int c = 0; c < IMG_SIZE; ++c) {
            float m = 0.85f + 0.20f * sinf(r * 0.45f + c * 0.35f);
            cv::Vec3b& px = img.at<cv::Vec3b>(r, c);
            px[0] = cv::saturate_cast<uchar>(px[0] * m);
            px[1] = cv::saturate_cast<uchar>(px[1] * m);
            px[2] = cv::saturate_cast<uchar>(px[2] * m);
        }
    }
}

static void stimulate_blur(cv::Mat& img) {
    cv::GaussianBlur(img, img, cv::Size(5, 5), 1.2);
}

static void stimulate_color_cast(cv::Mat& img) {
    for (int r = 0; r < IMG_SIZE; ++r) {
        for (int c = 0; c < IMG_SIZE; ++c) {
            cv::Vec3b& px = img.at<cv::Vec3b>(r, c);
            px[0] = cv::saturate_cast<uchar>(px[0] * 0.8 + 50);
            px[1] = cv::saturate_cast<uchar>(px[1] * 0.9 + 50);
            px[2] = cv::saturate_cast<uchar>(px[2] * 1.1 + 40);
        }
    }
}

static void stimulate_noise(cv::Mat& img, cv::RNG& rng) {
    cv::Mat noise(IMG_SIZE, IMG_SIZE, CV_8UC3);
    cv::randn(noise, cv::Scalar(0,0,0), cv::Scalar(15,15,15));
    cv::add(img, noise, img);
}

// ── 编码一帧 ──────────────────────────────────────────────────────────────
struct EncodeResult {
    cv::Mat img;
    std::vector<uint8_t> raw;   // raw[r*GRID_SIZE+c] = data byte
};

static EncodeResult encode_frame(const std::vector<uint64_t>& dict,
                                  unsigned rng_seed) {
    int N = (int)dict.size();
    int P_BITS = 0; { int tmp = N; while (tmp > 1) { tmp >>= 1; ++P_BITS; } }

    EncodeResult res;
    res.img  = cv::Mat(IMG_SIZE, IMG_SIZE, CV_8UC3, cv::Scalar(0,0,0));
    res.raw.assign(GRID_SIZE * GRID_SIZE, 0);

    cv::RNG rng(rng_seed);
    int total_vals = N * NUM_COLORS;

    for (int r = 0; r < GRID_SIZE; ++r) {
        for (int c = 0; c < GRID_SIZE; ++c) {
            if (is_reserved(r, c)) continue;
            uint8_t data = (uint8_t)(rng.uniform(0, total_vals));
            res.raw[r * GRID_SIZE + c] = data;
            int pat_idx   = data & (N - 1);
            int color_idx = data >> P_BITS;
            uint64_t mask = dict[pat_idx];
            int sx = MARGIN + c * STRIDE;
            int sy = MARGIN + r * STRIDE;
            for (int pr = 0; pr < 8; ++pr)
                for (int pc = 0; pc < 8; ++pc)
                    if ((mask >> (pr * 8 + pc)) & 1)
                        res.img.at<cv::Vec3b>(sy+pr, sx+pc) = COLORS[color_idx];
        }
    }
    return res;
}

// ── 解码一帧，返回 pattern 正确数 / 总格数 ────────────────────────────────
static std::pair<int,int> decode_frame(const cv::Mat& img,
                                        const std::vector<uint64_t>& dict,
                                        const std::vector<uint8_t>& raw) {
    int N = (int)dict.size();
    int P_BITS = 0; { int tmp = N; while (tmp > 1) { tmp >>= 1; ++P_BITS; } }

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    int correct = 0, total = 0;

    for (int r = 0; r < GRID_SIZE; ++r) {
        for (int c = 0; c < GRID_SIZE; ++c) {
            if (is_reserved(r, c)) continue;
            ++total;

            int sx = MARGIN + c * STRIDE, sy = MARGIN + r * STRIDE;
            cv::Rect roi(sx, sy, 8, 8);
            cv::Mat cell_gray = gray(roi);
            cv::Mat cell_bgr  = img(roi);

            cv::Mat bin;
            cv::threshold(cell_gray, bin, 0, 255,
                          cv::THRESH_BINARY | cv::THRESH_OTSU);

            uint64_t tile_bits = 0;
            for (int pr = 0; pr < 8; ++pr)
                for (int pc = 0; pc < 8; ++pc)
                    if (bin.at<uchar>(pr, pc) > 128)
                        tile_bits |= (1ULL << (pr*8+pc));

            int best_pat = match_pattern(tile_bits, dict);

            // 用解码到的 pattern mask 里的前景像素平均色确定颜色
            uint64_t best_mask = dict[best_pat];
            int sumB=0, sumG=0, sumR=0, vp=0;
            for (int pr = 0; pr < 8; ++pr)
                for (int pc = 0; pc < 8; ++pc)
                    if ((best_mask >> (pr*8+pc)) & 1) {
                        cv::Vec3b px = cell_bgr.at<cv::Vec3b>(pr, pc);
                        sumB += px[0]; sumG += px[1]; sumR += px[2]; ++vp;
                    }
            int best_col = 0;
            if (vp > 0) {
                cv::Vec3b avg(sumB/vp, sumG/vp, sumR/vp);
                best_col = match_color(avg);
            }

            uint8_t dec_byte = (best_col << P_BITS) | best_pat;
            uint8_t exp_byte = raw[r * GRID_SIZE + c];
            // 只统计 pattern 部分（颜色误差不纳入 pattern 筛选）
            if ((dec_byte & (N-1)) == (exp_byte & (N-1))) ++correct;
        }
    }
    return {correct, total};
}

// ── 对一组 dict（16 个 mask）做多档位完整评测，返回加权准确率 ─────────────
static double eval_one_combo(const std::vector<uint64_t>& dict,
                              const std::vector<double>& scales,
                              const std::vector<double>& weights,
                              const std::vector<unsigned>& rng_seeds) {
    double w_correct = 0.0, w_total = 0.0;

    for (unsigned seed : rng_seeds) {
        auto [enc_img, raw] = encode_frame(dict, seed);

        for (int si = 0; si < (int)scales.size(); ++si) {
            double scale = scales[si];
            double weight = weights[si];
            int px = std::max(2, (int)(IMG_SIZE * scale + 0.5));

            cv::Mat scaled, restored;
            cv::resize(enc_img, scaled,   cv::Size(px, px),          0, 0, cv::INTER_AREA);
            cv::resize(scaled,  restored, cv::Size(IMG_SIZE,IMG_SIZE),0, 0, cv::INTER_LINEAR);

            cv::RNG cv_rng(seed * 1000 + si);
            stimulate_moire(restored);
            stimulate_blur(restored);
            stimulate_color_cast(restored);
            stimulate_noise(restored, cv_rng);

            auto [correct, total] = decode_frame(restored, dict, raw);
            w_correct += weight * correct;
            w_total   += weight * total;
        }
    }
    return w_total > 0 ? w_correct / w_total : 0.0;
}

// ── 极简 JSON 解析（避免引入外部库） ─────────────────────────────────────
// 只处理我们自己生成的格式，不做通用解析

static std::string read_all_stdin() {
    std::ostringstream buf;
    buf << std::cin.rdbuf();
    return buf.str();
}

// 从 JSON 字符串里找 key 对应的数组内容（返回 '[' 到对应 ']' 之间的子串）
static std::string find_array(const std::string& json, const std::string& key) {
    std::string kstr = "\"" + key + "\"";
    auto pos = json.find(kstr);
    if (pos == std::string::npos) return "";
    pos = json.find('[', pos);
    if (pos == std::string::npos) return "";
    int depth = 0; size_t end = pos;
    for (; end < json.size(); ++end) {
        if (json[end] == '[') ++depth;
        else if (json[end] == ']') { --depth; if (depth == 0) break; }
    }
    return json.substr(pos, end - pos + 1);
}

static std::vector<double> parse_double_array(const std::string& arr) {
    std::vector<double> res;
    std::string s = arr;
    for (char& c : s) if (c == '[' || c == ']' || c == ',') c = ' ';
    std::istringstream ss(s);
    double v;
    while (ss >> v) res.push_back(v);
    return res;
}

static std::vector<unsigned> parse_uint_array(const std::string& arr) {
    std::vector<unsigned> res;
    std::string s = arr;
    for (char& c : s) if (c == '[' || c == ']' || c == ',') c = ' ';
    std::istringstream ss(s);
    unsigned v;
    while (ss >> v) res.push_back(v);
    return res;
}

// 解析 combos：[ [u64,u64,...], [u64,u64,...], ... ]
static std::vector<std::vector<uint64_t>> parse_combos(const std::string& arr) {
    std::vector<std::vector<uint64_t>> res;
    // 找每个内层数组
    size_t pos = 0;
    while (pos < arr.size()) {
        pos = arr.find('[', pos + 1);
        if (pos == std::string::npos) break;
        size_t end = arr.find(']', pos);
        if (end == std::string::npos) break;
        std::string inner = arr.substr(pos + 1, end - pos - 1);
        std::vector<uint64_t> combo;
        std::istringstream ss(inner);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            // trim
            while (!tok.empty() && (tok.front()==' '||tok.front()=='\n')) tok.erase(tok.begin());
            while (!tok.empty() && (tok.back()==' '||tok.back()=='\n')) tok.pop_back();
            if (!tok.empty()) combo.push_back(std::stoull(tok));
        }
        if (!combo.empty()) res.push_back(combo);
        pos = end + 1;
    }
    return res;
}

int main() {
    std::string input = read_all_stdin();

    // 解析 scales
    std::vector<double> scales  = parse_double_array(find_array(input, "scales"));
    std::vector<double> weights = parse_double_array(find_array(input, "weights"));
    std::vector<unsigned> seeds = parse_uint_array(find_array(input, "seeds"));
    std::vector<std::vector<uint64_t>> combos = parse_combos(find_array(input, "combos"));

    if (scales.empty())  { scales  = {1.0, 0.9, 0.8, 0.7, 0.6, 0.5}; }
    if (weights.empty()) { weights = {1,   2,   3,   4,   5,   6  }; }
    if (seeds.empty())   { seeds   = {0, 1, 2}; }

    // 输出
    printf("{\"scores\":[");
    for (int i = 0; i < (int)combos.size(); ++i) {
        double score = eval_one_combo(combos[i], scales, weights, seeds);
        printf("%.6f", score);
        if (i + 1 < (int)combos.size()) printf(",");
    }
    printf("]}\n");
    return 0;
}
