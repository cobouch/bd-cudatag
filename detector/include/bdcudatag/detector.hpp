// bdcudatag C++ API.
//
// Mirrors the C ABI but uses RAII and value types.  The detector is
// synchronous and thread-safe: calls on one instance are serialized internally,
// so it may be shared between threads (without running detections in parallel).
#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "bdcudatag/bdcudatag.h"

namespace bdcudatag {

enum class Family : int32_t {
    Tag36h11 = BD_FAMILY_TAG36H11,
    Tag16h5 = BD_FAMILY_TAG16H5,
};

struct DetectorConfig {
    int max_hamming = -1;          // -1: family default (36h11: 2, 16h5: 1)
    int min_cluster_pixels = 24;   // reject connected components smaller than this
    int min_tag_pixels = 8;        // minimum tag bounding-box side in pixels
    int max_detections = 64;       // hard cap, at most 256
    int max_candidates = 0;        // 0: automatic from image size
    bool refine_edges = true;      // subpixel edge refinement (small cost)
    bool use_cuda_graph = true;    // capture the host-frame pipeline as a CUDA graph
    float min_decision_margin = 3.0f;
    float decode_sharpening = 0.25f;
};

struct Detection {
    int id = -1;
    int hamming = 0;
    float decision_margin = 0.0f;
    // Top-left, top-right, bottom-right, bottom-left in the tag's canonical
    // orientation, in image pixel coordinates.
    std::array<std::array<float, 2>, 4> corners{};
    // Row-major homography mapping the tag's canonical square to image pixels.
    // Tag coordinates span [-1, 1] with x right and y down, so the first corner
    // is (-1, -1):  s * (px, py, 1)^T = homography * (x, y, 1)^T.  Normalised so
    // homography[8] == 1.
    std::array<float, 9> homography{};

    float center_x() const {
        return 0.25f * (corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]);
    }
    float center_y() const {
        return 0.25f * (corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]);
    }
};

struct Stats {
    uint64_t frames = 0;
    float last_gpu_ms = 0.0f;
    float average_gpu_ms = 0.0f;
    float last_total_ms = 0.0f;
    int runs = 0;
    int candidates = 0;
    int detections = 0;
    int quads = 0;
    int roi_rows = 0;
};

class BD_API Detector {
public:
    Detector(Family family, int width, int height, const DetectorConfig& config = {});
    ~Detector();

    Detector(Detector&& other) noexcept;
    Detector& operator=(Detector&& other) noexcept;
    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;

    // Writes at most out_capacity detections and returns the number written.
    int detect(const uint8_t* gray, int stride, Detection* out, int out_capacity);

    // Convenience overload; allocates the result vector.
    std::vector<Detection> detect(const uint8_t* gray, int stride);

    // Zero-copy input: gray_device is a CUDA device pointer.
    int detectDevice(const uint8_t* gray_device, int stride, Detection* out, int out_capacity);

    Stats stats() const;

    Family family() const { return family_; }
    int width() const { return width_; }
    int height() const { return height_; }

private:
    struct Impl;
    Impl* impl_ = nullptr;
    Family family_;
    int width_;
    int height_;
};

}  // namespace bdcudatag
