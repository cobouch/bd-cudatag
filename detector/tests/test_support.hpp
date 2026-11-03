// Synthetic tag rendering and tiny test helpers shared by the tests and bench.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "tag_families.hpp"

namespace bdcudatag::test {

struct Mat3 {
    double v[9];

    void project(double x, double y, double* out_x, double* out_y) const {
        const double w = v[6] * x + v[7] * y + v[8];
        *out_x = (v[0] * x + v[1] * y + v[2]) / w;
        *out_y = (v[3] * x + v[4] * y + v[5]) / w;
    }

    Mat3 inverse() const {
        const double a = v[0], b = v[1], c = v[2];
        const double d = v[3], e = v[4], f = v[5];
        const double g = v[6], h = v[7], i = v[8];
        const double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
        Mat3 out{};
        const double inv = 1.0 / det;
        out.v[0] = (e * i - f * h) * inv;
        out.v[1] = (c * h - b * i) * inv;
        out.v[2] = (b * f - c * e) * inv;
        out.v[3] = (f * g - d * i) * inv;
        out.v[4] = (a * i - c * g) * inv;
        out.v[5] = (c * d - a * f) * inv;
        out.v[6] = (d * h - e * g) * inv;
        out.v[7] = (b * g - a * h) * inv;
        out.v[8] = (a * e - b * d) * inv;
        return out;
    }
};

inline Mat3 make_pose(double cx, double cy, double size_px, double angle_rad,
                      double perspective = 0.0) {
    // Tag coordinates are the AprilTag convention: the black square spans
    // [-1, 1] x [-1, 1], x right, y down. The image pose places that square
    // with the requested centre, side length and rotation.
    const double c = std::cos(angle_rad);
    const double s = std::sin(angle_rad);
    const double h = 0.5 * size_px;
    Mat3 m{};
    m.v[0] = c * h;
    m.v[1] = -s * h;
    m.v[2] = cx;
    m.v[3] = s * h;
    m.v[4] = c * h;
    m.v[5] = cy;
    m.v[6] = perspective * c;
    m.v[7] = -perspective * s;
    m.v[8] = 1.0;
    return m;
}

inline int popcount(uint64_t v) {
    int count = 0;
    while (v) {
        v &= v - 1;
        ++count;
    }
    return count;
}

inline uint64_t rotate_code(uint64_t code, int bits) {
    const int n = bits * bits;
    uint64_t out = 0;
    for (int r = 0; r < bits; ++r) {
        for (int c = 0; c < bits; ++c) {
            const uint64_t bit = (code >> (n - 1 - (r * bits + c))) & 1ull;
            const int nr = c;
            const int nc = bits - 1 - r;
            out |= bit << (n - 1 - (nr * bits + nc));
        }
    }
    return out;
}

// Renders one tag family tag into a grayscale image. Data cells come from the
// row-major table. A 1-cell white quiet zone surrounds the black square; the
// image background is white. Supersampling keeps edges reasonably clean.
inline std::vector<uint8_t> render_tag(const detail::TagFamilyData& family, int tag_id,
                                       int width, int height, const Mat3& pose,
                                       int supersample = 2, uint64_t override_code = ~0ull,
                                       bool flip_one_bit = false) {
    std::vector<uint8_t> image(static_cast<size_t>(width) * height, 255);
    const int bits = family.bits_per_side;
    const int border = family.border_cells;
    const int dim = bits + 2 * border;
    const uint64_t code = override_code == ~0ull ? family.codes[tag_id] : override_code;
    const Mat3 inv = pose.inverse();

    const int ss = std::max(1, supersample);
    for (int py = 0; py < height; ++py) {
        for (int px = 0; px < width; ++px) {
            int accum = 0;
            for (int sy = 0; sy < ss; ++sy) {
                for (int sx = 0; sx < ss; ++sx) {
                    const double ix = px + (sx + 0.5) / ss;
                    const double iy = py + (sy + 0.5) / ss;
                    double tx, ty;
                    inv.project(ix, iy, &tx, &ty);
                    if (tx < -1.0 || tx > 1.0 || ty < -1.0 || ty > 1.0) {
                        accum += 255;
                        continue;
                    }
                    const int cx = std::min(dim - 1, std::max(0, static_cast<int>((tx + 1.0) * 0.5 * dim)));
                    const int cy = std::min(dim - 1, std::max(0, static_cast<int>((ty + 1.0) * 0.5 * dim)));
                    if (cx == 0 || cy == 0 || cx == dim - 1 || cy == dim - 1) {
                        accum += 0;  // black border
                        continue;
                    }
                    const int bx = cx - border;
                    const int by = cy - border;
                    const int pos = by * bits + bx;
                    uint64_t cell = (code >> (bits * bits - 1 - pos)) & 1ull;
                    if (flip_one_bit && pos == 0) cell ^= 1ull;
                    accum += cell ? 255 : 0;
                }
            }
            image[static_cast<size_t>(py) * width + px] =
                static_cast<uint8_t>(accum / (ss * ss));
        }
    }
    return image;
}

inline bool close(float a, float b, float tolerance) { return std::fabs(a - b) <= tolerance; }

inline void composite_image(std::vector<uint8_t>& dst, const std::vector<uint8_t>& src) {
    for (    size_t i = 0; i < dst.size() && i < src.size(); ++i) {
        dst[i] = std::min(dst[i], src[i]);
    }
}

inline int failures = 0;

#define CHECK(cond)                                                                      \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::fprintf(stderr, "  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);       \
            ++bdcudatag::test::failures;                                                 \
        }                                                                                \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                              \
    do {                                                                                   \
        const double va = (a);                                                             \
        const double vb = (b);                                                             \
        if (std::fabs(va - vb) > (tol)) {                                                  \
            std::fprintf(stderr, "  FAIL %s:%d: %s=%.4f vs %s=%.4f (tol %.3f)\n", __FILE__, \
                         __LINE__, #a, va, #b, vb, static_cast<double>(tol));              \
            ++bdcudatag::test::failures;                                                   \
        }                                                                                  \
    } while (0)

inline int finish(const char* name) {
    if (failures == 0) {
        std::printf("PASS %s\n", name);
        return 0;
    }
    std::printf("FAIL %s (%d failures)\n", name, failures);
    return 1;
}

}  // namespace bdcudatag::test
