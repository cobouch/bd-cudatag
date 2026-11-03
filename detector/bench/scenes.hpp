// Deterministic synthetic benchmark scenes with exact ground truth.
//
// A scene is a full-frame grayscale image plus the image-space corners of every
// tag it contains.  Tags are rendered from the same family tables the detector
// uses, so the ground truth is exact and no camera or external data is needed.
//
// The same scenes back the native `bdcudatag_bench` and `bdcudatag_scenes`,
// which exports them as PGM + JSON so that detectors not linked into this build
// (NAPA, ArUco, AprilTag) can be scored on identical pixels.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include "tag_families.hpp"
#include "test_support.hpp"

namespace bdcudatag::bench {

struct SceneTag {
    int id;
    // Maps the tag square [-1, 1]^2 to image pixels (AprilTag tag frame).
    test::Mat3 pose;
};

struct Scene {
    int width = 0;
    int height = 0;
    std::vector<SceneTag> tags;
    std::vector<uint8_t> image;
};

// Corners of the tag's outer black square, top-left, top-right, bottom-right,
// bottom-left in the tag's canonical orientation, in pixel-index coordinates.
//
// The renderer treats pixel i as covering [i, i + 1) and samples at i + 0.5, so
// a tag edge at continuous coordinate x appears in the image at index x - 0.5.
// Detectors report index coordinates, so the same half-pixel shift is applied
// here; without it every detector shows a constant ~0.7 px error and the metric
// measures the rendering convention instead of corner accuracy.
inline std::array<std::array<double, 2>, 4> project_square(const test::Mat3& pose) {
    static constexpr double kUnit[4][2] = {
        {-1.0, -1.0}, {1.0, -1.0}, {1.0, 1.0}, {-1.0, 1.0}};
    std::array<std::array<double, 2>, 4> out{};
    for (int i = 0; i < 4; ++i) {
        pose.project(kUnit[i][0], kUnit[i][1], &out[i][0], &out[i][1]);
        out[i][0] -= 0.5;
        out[i][1] -= 0.5;
    }
    return out;
}

inline std::array<std::array<double, 2>, 4> ground_truth_corners(const SceneTag& tag) {
    return project_square(tag.pose);
}

inline uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

inline double next_unit(uint32_t& state) {
    state = mix32(state + 0x9e3779b9u);
    return (state >> 8) * (1.0 / 16777216.0);
}

// Adds deterministic sensor-like noise (uniform in [-3, 3]) so timings and
// corner accuracy are not measured on perfectly noiseless images.
inline void add_noise(std::vector<uint8_t>& image, uint32_t seed) {
    uint32_t state = seed | 1u;
    for (uint8_t& pixel : image) {
        state = mix32(state);
        const int delta = static_cast<int>((state >> 28) & 7u) - 3;
        const int value = static_cast<int>(pixel) + delta;
        pixel = static_cast<uint8_t>(value < 0 ? 0 : (value > 255 ? 255 : value));
    }
}

// Renders one tag into `image`, touching only the pixels around it.  Rendering
// the whole frame once per tag would dominate scene generation at 1080p.
// `code` overrides the family codeword, which is how decoy tags are drawn.
inline void render_tag_into(std::vector<uint8_t>& image, int width, int height,
                            const detail::TagFamilyData& family, int id,
                            const test::Mat3& pose, uint64_t code = ~0ull) {
    const auto corners = project_square(pose);
    double min_x = corners[0][0], max_x = corners[0][0];
    double min_y = corners[0][1], max_y = corners[0][1];
    for (int i = 1; i < 4; ++i) {
        min_x = std::min(min_x, corners[i][0]);
        max_x = std::max(max_x, corners[i][0]);
        min_y = std::min(min_y, corners[i][1]);
        max_y = std::max(max_y, corners[i][1]);
    }

    constexpr int kMargin = 4;
    const int x0 = std::max(0, static_cast<int>(std::floor(min_x)) - kMargin);
    const int y0 = std::max(0, static_cast<int>(std::floor(min_y)) - kMargin);
    const int x1 = std::min(width - 1, static_cast<int>(std::ceil(max_x)) + kMargin);
    const int y1 = std::min(height - 1, static_cast<int>(std::ceil(max_y)) + kMargin);
    if (x1 < x0 || y1 < y0) return;

    const int region_w = x1 - x0 + 1;
    const int region_h = y1 - y0 + 1;
    test::Mat3 shifted = pose;
    shifted.v[2] -= x0;
    shifted.v[5] -= y0;
    const std::vector<uint8_t> patch =
        test::render_tag(family, id, region_w, region_h, shifted, 2, code);

    for (int y = 0; y < region_h; ++y) {
        uint8_t* dst = image.data() + static_cast<size_t>(y0 + y) * width + x0;
        const uint8_t* src = patch.data() + static_cast<size_t>(y) * region_w;
        for (int x = 0; x < region_w; ++x) dst[x] = std::min(dst[x], src[x]);
    }
}

// A random codeword that no rotation of any family codeword can be decoded to
// within `min_distance` bits.  Used to draw tag-shaped decoys whose true label
// is "not a tag", so a detector that decodes one has made a wrong id.
inline uint64_t random_non_tag_code(uint32_t& rng, const detail::TagFamilyData& family,
                                    int min_distance = 8) {
    const int cell_count = family.bits_per_side * family.bits_per_side;
    const uint64_t mask = cell_count >= 64 ? ~0ull : ((1ull << cell_count) - 1ull);
    for (;;) {
        const uint64_t code =
            ((static_cast<uint64_t>(mix32(rng)) << 32) ^ mix32(rng)) & mask;
        int best = 1000;
        for (int i = 0; i < family.code_count; ++i) {
            uint64_t rotated = family.codes[i];
            for (int k = 0; k < 4; ++k) {
                best = std::min(best, test::popcount(code ^ rotated));
                rotated = test::rotate_code(rotated, family.bits_per_side);
            }
        }
        if (best >= min_distance) return code;
    }
}

// Eight tag36h11 ids on a jittered 4x2 grid.  Sizes are relative to the image
// width, so the same scene description works at every resolution; the last tag
// is deliberately small to keep recall from saturating at 100%.
inline std::vector<Scene> make_scenes(int width, int height, int count = 32,
                                      uint32_t seed = 6324u) {
    static constexpr int kIds[8] = {0, 3, 7, 11, 17, 42, 101, 586};
    static constexpr double kScale[8] = {1.00, 0.85, 0.70, 1.12, 0.95, 0.60, 1.05, 0.30};
    static constexpr double kPi = 3.14159265358979323846;

    std::vector<Scene> scenes;
    scenes.reserve(static_cast<size_t>(count));
    const double base_size = width / 9.0;

    for (int s = 0; s < count; ++s) {
        uint32_t rng = seed + 0x9e3779b9u * static_cast<uint32_t>(s + 1);
        Scene scene;
        scene.width = width;
        scene.height = height;
        scene.image.assign(static_cast<size_t>(width) * height, 255);
        scene.tags.reserve(8);

        for (int t = 0; t < 8; ++t) {
            const int col = t % 4;
            const int row = t / 4;
            const double cx = (col + 0.5) / 4.0 * width +
                              (next_unit(rng) - 0.5) * 0.05 * width;
            const double cy = (row + 0.5) / 2.0 * height +
                              (next_unit(rng) - 0.5) * 0.06 * height;
            const double size = base_size * kScale[t] * (0.92 + 0.16 * next_unit(rng));
            const double angle = (next_unit(rng) - 0.5) * (0.8 * kPi / 2.0);
            const double perspective = (next_unit(rng) - 0.5) * 1.2e-3;

            const test::Mat3 pose = test::make_pose(cx, cy, size, angle, perspective);
            scene.tags.push_back(SceneTag{kIds[t], pose});
            render_tag_into(scene.image, width, height, detail::kTag36h11, kIds[t], pose);
        }
        // On a quarter of the scenes, two tag-shaped decoys with random data
        // sit between the rows.  They are not ground truth, so any detection
        // that decodes them counts as a wrong id.
        if (s % 8 == 3) {
            const double decoy_x[2] = {width * 0.30, width * 0.70};
            const double decoy_y[2] = {height * 0.50, height * 0.50};
            const double decoy_size[2] = {base_size * 0.50, base_size * 0.40};
            for (int k = 0; k < 2; ++k) {
                const test::Mat3 pose =
                    test::make_pose(decoy_x[k], decoy_y[k], decoy_size[k],
                                    next_unit(rng) * 0.6, 0.0);
                render_tag_into(scene.image, width, height, detail::kTag36h11, 0, pose,
                                random_non_tag_code(rng, detail::kTag36h11));
            }
        }

        add_noise(scene.image, seed + static_cast<uint32_t>(s));
        scenes.push_back(std::move(scene));
    }
    return scenes;
}

inline std::string join_path(const std::string& dir, const std::string& name) {
    return (std::filesystem::path(dir) / name).string();
}

inline bool write_pgm(const std::string& path, const Scene& scene) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    std::fprintf(f, "P5\n%d %d\n255\n", scene.width, scene.height);
    const bool ok = std::fwrite(scene.image.data(), 1, scene.image.size(), f) ==
                    scene.image.size();
    std::fclose(f);
    return ok;
}

// Writes `<dir>/scenes_<prefix>.json` plus one PGM per scene.  The manifest
// holds the resolution, the image file name and the ground-truth corners of
// every tag, which is what compare_detectors.py scores against.
inline bool write_scene_set(const std::vector<Scene>& scenes, const std::string& dir,
                            const std::string& prefix) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) return false;

    const std::string manifest_path = join_path(dir, "scenes_" + prefix + ".json");
    FILE* f = std::fopen(manifest_path.c_str(), "wb");
    if (!f) return false;

    std::fprintf(f, "{\n  \"family\": \"tag36h11\",\n  \"scenes\": [\n");
    for (size_t i = 0; i < scenes.size(); ++i) {
        const Scene& scene = scenes[i];
        char file_name[96];
        std::snprintf(file_name, sizeof file_name, "%s_%03zu.pgm", prefix.c_str(), i);
        if (!write_pgm(join_path(dir, file_name), scene)) {
            std::fclose(f);
            return false;
        }
        std::fprintf(f, "    {\"file\": \"%s\", \"width\": %d, \"height\": %d, \"tags\": [",
                     file_name, scene.width, scene.height);
        for (size_t t = 0; t < scene.tags.size(); ++t) {
            const auto corners = ground_truth_corners(scene.tags[t]);
            std::fprintf(f, "%s\n      {\"id\": %d, \"corners\": [", t ? "," : "",
                         scene.tags[t].id);
            for (int k = 0; k < 4; ++k) {
                std::fprintf(f, "%s[%.3f, %.3f]", k ? ", " : "", corners[k][0],
                             corners[k][1]);
            }
            std::fprintf(f, "]}");
        }
        std::fprintf(f, "\n    ]}%s\n", i + 1 < scenes.size() ? "," : "");
    }
    std::fprintf(f, "  ]\n}\n");
    std::fclose(f);
    return true;
}

}  // namespace bdcudatag::bench
