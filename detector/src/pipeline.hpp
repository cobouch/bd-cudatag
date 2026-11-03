// Internal interface between the portable host code (detector.cpp) and the
// CUDA translation unit (gpu_detector.cu).  No CUDA types escape this header.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "bdcudatag/bdcudatag.h"

namespace bdcudatag::detail {

struct GpuConfig {
    int family = 0;
    int width = 0;
    int height = 0;
    int max_hamming = -1;
    int min_cluster_pixels = 24;
    int min_tag_pixels = 8;
    int max_detections = 64;
    int max_candidates = 0;
    bool refine_edges = true;
    bool use_cuda_graph = true;
    float min_decision_margin = 3.0f;
    float decode_sharpening = 0.25f;
};

class GpuDetector {
public:
    // Returns nullptr on failure and fills *error (if non-null).
    static GpuDetector* create(const GpuConfig& config, std::string* error);
    static void destroy(GpuDetector* detector);

    GpuDetector(const GpuDetector&) = delete;
    GpuDetector& operator=(const GpuDetector&) = delete;

    // Returns the number of detections written, or -1 on failure (error filled).
    int detect_host(const uint8_t* gray, int stride, bd_detection* out, int out_capacity,
                    std::string* error);
    int detect_device(const uint8_t* gray_device, int stride, bd_detection* out,
                      int out_capacity, std::string* error);

    void get_stats(bd_stats* out) const;
    int max_detections() const;

private:
    GpuDetector();
    ~GpuDetector();

    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace bdcudatag::detail
