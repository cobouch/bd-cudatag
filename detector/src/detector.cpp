// Portable host layer: C ABI and the small C++ RAII wrapper.  All CUDA work
// lives in gpu_detector.cu.

#include "bdcudatag/bdcudatag.h"
#include "bdcudatag/detector.hpp"

#include "pipeline.hpp"
#include "tag_families.hpp"

#include <algorithm>
#include <cstring>
#include <exception>
#include <mutex>
#include <new>
#include <string>
#include <vector>

namespace {

thread_local std::string g_last_error;

void set_error(const std::string& message) { g_last_error = message; }

bool valid_family(int family) {
    return family >= 0 && family < bdcudatag::detail::kFamilyCount;
}

bool valid_dimensions(int width, int height) {
    return width >= 16 && width <= BD_MAX_WIDTH && height >= 16 && height <= BD_MAX_HEIGHT;
}

int family_default_hamming(int family) {
    // tag16h5's minimum distance is only 5; one corrected bit keeps false
    // positives under control while still tolerating a little blur. tag36h11
    // has enough distance to safely correct two bits, matching AprilTag.
    return family == BD_FAMILY_TAG36H11 ? 2 : 1;
}

void config_to_gpu(int family, int width, int height, const bd_config* config,
                   bdcudatag::detail::GpuConfig* out) {
    bd_config defaults;
    bd_config_init(&defaults);
    const bd_config* c = config ? config : &defaults;

    out->family = family;
    out->width = width;
    out->height = height;
    out->max_hamming = c->max_hamming;
    out->min_cluster_pixels = c->min_cluster_pixels;
    out->min_tag_pixels = c->min_tag_pixels;
    out->max_detections = c->max_detections;
    out->max_candidates = c->max_candidates;
    out->refine_edges = c->refine_edges != 0;
    out->use_cuda_graph = c->use_cuda_graph != 0;
    out->min_decision_margin = c->min_decision_margin;
    out->decode_sharpening = c->decode_sharpening;
}

}  // namespace

struct bd_detector {
    bdcudatag::detail::GpuDetector* gpu = nullptr;
    int family = 0;
    int width = 0;
    int height = 0;
    int max_detections = 0;
};

extern "C" {

uint32_t bd_api_version(void) { return BD_API_VERSION; }

const char* bd_version_string(void) { return "bdcudatag 0.1.0"; }

int32_t bd_family_count(void) { return static_cast<int32_t>(bdcudatag::detail::kFamilyCount); }

const char* bd_family_name(int32_t family) {
    if (!valid_family(family)) return nullptr;
    return bdcudatag::detail::kFamilies[family]->name;
}

int32_t bd_family_bits_per_side(int32_t family) {
    if (!valid_family(family)) return -1;
    return bdcudatag::detail::kFamilies[family]->bits_per_side;
}

int32_t bd_family_min_hamming(int32_t family) {
    if (!valid_family(family)) return -1;
    return bdcudatag::detail::kFamilies[family]->min_hamming;
}

int32_t bd_family_default_max_hamming(int32_t family) {
    if (!valid_family(family)) return -1;
    return family_default_hamming(family);
}

void bd_config_init(bd_config* config) {
    if (!config) return;
    std::memset(config, 0, sizeof(*config));
    config->struct_size = sizeof(bd_config);
    config->max_hamming = -1;
    config->min_cluster_pixels = 24;
    config->min_tag_pixels = 8;
    config->max_detections = 64;
    config->max_candidates = 0;
    config->refine_edges = 1;
    config->use_cuda_graph = 1;
    config->min_decision_margin = 3.0f;
    config->decode_sharpening = 0.25f;
}

bd_detector* bd_detector_create(int32_t family, int32_t width, int32_t height,
                                const bd_config* config, int32_t* status_out) {
    if (status_out) *status_out = BD_OK;
    if (!valid_family(family)) {
        set_error("unknown tag family");
        if (status_out) *status_out = BD_ERROR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (!valid_dimensions(width, height)) {
        set_error("image dimensions out of range (16..8192 x 16..16384)");
        if (status_out) *status_out = BD_ERROR_INVALID_ARGUMENT;
        return nullptr;
    }
    if (config && config->struct_size != 0 && config->struct_size < sizeof(bd_config)) {
        set_error("bd_config.struct_size is too small for this library version");
        if (status_out) *status_out = BD_ERROR_INVALID_ARGUMENT;
        return nullptr;
    }

    bdcudatag::detail::GpuConfig gpu_config;
    config_to_gpu(family, width, height, config, &gpu_config);

    std::string error;
    bdcudatag::detail::GpuDetector* gpu =
        bdcudatag::detail::GpuDetector::create(gpu_config, &error);
    if (!gpu) {
        set_error(error.empty() ? "failed to create CUDA detector" : error);
        if (status_out) *status_out = BD_ERROR_CUDA;
        return nullptr;
    }

    bd_detector* detector = new (std::nothrow) bd_detector();
    if (!detector) {
        bdcudatag::detail::GpuDetector::destroy(gpu);
        set_error("out of memory allocating detector handle");
        if (status_out) *status_out = BD_ERROR_OUT_OF_MEMORY;
        return nullptr;
    }
    detector->gpu = gpu;
    detector->family = family;
    detector->width = width;
    detector->height = height;
    detector->max_detections = gpu->max_detections();
    return detector;
}

void bd_detector_destroy(bd_detector* detector) {
    if (!detector) return;
    bdcudatag::detail::GpuDetector::destroy(detector->gpu);
    delete detector;
}

int32_t bd_detector_detect_host(bd_detector* detector, const uint8_t* gray, int32_t stride,
                                bd_detection* out, int32_t out_capacity) {
    if (!detector) {
        set_error("null detector");
        return BD_ERROR_INVALID_ARGUMENT;
    }
    if (!gray) {
        set_error("null image pointer");
        return BD_ERROR_INVALID_ARGUMENT;
    }
    if (stride < detector->width) {
        set_error("stride is smaller than the image width");
        return BD_ERROR_INVALID_ARGUMENT;
    }
    if (out_capacity < 0 || (out_capacity > 0 && !out)) {
        set_error("invalid output buffer");
        return BD_ERROR_INVALID_ARGUMENT;
    }

    std::string error;
    const int count = detector->gpu->detect_host(gray, stride, out, out_capacity, &error);
    if (count < 0) {
        set_error(error.empty() ? "CUDA detection failed" : error);
        return BD_ERROR_CUDA;
    }
    return count;
}

int32_t bd_detector_detect_device(bd_detector* detector, const uint8_t* gray_device,
                                  int32_t stride, bd_detection* out, int32_t out_capacity) {
    if (!detector) {
        set_error("null detector");
        return BD_ERROR_INVALID_ARGUMENT;
    }
    if (!gray_device) {
        set_error("null device image pointer");
        return BD_ERROR_INVALID_ARGUMENT;
    }
    if (stride < detector->width) {
        set_error("stride is smaller than the image width");
        return BD_ERROR_INVALID_ARGUMENT;
    }
    if (out_capacity < 0 || (out_capacity > 0 && !out)) {
        set_error("invalid output buffer");
        return BD_ERROR_INVALID_ARGUMENT;
    }

    std::string error;
    const int count = detector->gpu->detect_device(gray_device, stride, out, out_capacity, &error);
    if (count < 0) {
        set_error(error.empty() ? "CUDA detection failed" : error);
        return BD_ERROR_CUDA;
    }
    return count;
}

void bd_detector_get_stats(bd_detector* detector, bd_stats* out) {
    if (!detector || !out) return;
    detector->gpu->get_stats(out);
}

const char* bd_last_error(void) { return g_last_error.c_str(); }

}  // extern "C"

namespace bdcudatag {

struct Detector::Impl {
    bd_detector* handle = nullptr;
    int max_detections = 0;
    std::vector<bd_detection> scratch;
    // Serializes calls that share the scratch buffer; the native detector also
    // serializes its own state, so one instance is safe to share between threads.
    std::mutex mutex;
};

Detector::Detector(Family family, int width, int height, const DetectorConfig& config)
    : family_(family), width_(width), height_(height) {
    bd_config native;
    bd_config_init(&native);
    native.max_hamming = config.max_hamming;
    native.min_cluster_pixels = config.min_cluster_pixels;
    native.min_tag_pixels = config.min_tag_pixels;
    native.max_detections = config.max_detections;
    native.max_candidates = config.max_candidates;
    native.refine_edges = config.refine_edges ? 1 : 0;
    native.use_cuda_graph = config.use_cuda_graph ? 1 : 0;
    native.min_decision_margin = config.min_decision_margin;
    native.decode_sharpening = config.decode_sharpening;

    int32_t status = BD_OK;
    impl_ = new Impl();
    impl_->handle = bd_detector_create(static_cast<int32_t>(family), width, height, &native,
                                       &status);
    if (!impl_->handle) {
        const std::string message = bd_last_error();
        delete impl_;
        impl_ = nullptr;
        throw std::runtime_error(message.empty() ? "failed to create bdcudatag detector" : message);
    }
    // The native handle caps detections; mirror that cap for the scratch buffer.
    bd_config effective = native;
    if (effective.max_detections > 256) effective.max_detections = 256;
    if (effective.max_detections < 1) effective.max_detections = 1;
    impl_->max_detections = effective.max_detections;
    impl_->scratch.resize(static_cast<size_t>(impl_->max_detections));
}

Detector::~Detector() {
    if (!impl_) return;
    if (impl_->handle) bd_detector_destroy(impl_->handle);
    delete impl_;
}

Detector::Detector(Detector&& other) noexcept
    : impl_(other.impl_), family_(other.family_), width_(other.width_), height_(other.height_) {
    other.impl_ = nullptr;
}

Detector& Detector::operator=(Detector&& other) noexcept {
    if (this == &other) return *this;
    if (impl_) {
        if (impl_->handle) bd_detector_destroy(impl_->handle);
        delete impl_;
    }
    impl_ = other.impl_;
    family_ = other.family_;
    width_ = other.width_;
    height_ = other.height_;
    other.impl_ = nullptr;
    return *this;
}

int Detector::detect(const uint8_t* gray, int stride, Detection* out, int out_capacity) {
    if (!impl_) throw std::runtime_error("detector has been moved from");
    if (out_capacity < 0 || (out_capacity > 0 && !out)) {
        throw std::invalid_argument("invalid detection output buffer");
    }
    if (out_capacity > impl_->max_detections) out_capacity = impl_->max_detections;

    const std::lock_guard<std::mutex> lock(impl_->mutex);
    const int count = bd_detector_detect_host(impl_->handle, gray, stride, impl_->scratch.data(),
                                              out_capacity);
    if (count < 0) {
        throw std::runtime_error(bd_last_error());
    }
    const int written = std::min(count, out_capacity);
    for (int i = 0; i < written; ++i) {
        const bd_detection& src = impl_->scratch[static_cast<size_t>(i)];
        Detection& dst = out[i];
        dst.id = src.id;
        dst.hamming = src.hamming;
        dst.decision_margin = src.decision_margin;
        for (int k = 0; k < 4; ++k) {
            dst.corners[static_cast<size_t>(k)][0] = src.corners[k * 2 + 0];
            dst.corners[static_cast<size_t>(k)][1] = src.corners[k * 2 + 1];
        }
        for (int k = 0; k < 9; ++k) dst.homography[static_cast<size_t>(k)] = src.homography[k];
    }
    return written;
}

std::vector<Detection> Detector::detect(const uint8_t* gray, int stride) {
    if (!impl_) throw std::runtime_error("detector has been moved from");
    std::vector<Detection> result(static_cast<size_t>(impl_->max_detections));
    const int count = detect(gray, stride, result.data(), impl_->max_detections);
    result.resize(static_cast<size_t>(count));
    return result;
}

int Detector::detectDevice(const uint8_t* gray_device, int stride, Detection* out,
                           int out_capacity) {
    if (!impl_) throw std::runtime_error("detector has been moved from");
    if (out_capacity < 0 || (out_capacity > 0 && !out)) {
        throw std::invalid_argument("invalid detection output buffer");
    }
    if (out_capacity > impl_->max_detections) out_capacity = impl_->max_detections;

    const std::lock_guard<std::mutex> lock(impl_->mutex);
    const int count = bd_detector_detect_device(impl_->handle, gray_device, stride,
                                                impl_->scratch.data(), out_capacity);
    if (count < 0) {
        throw std::runtime_error(bd_last_error());
    }
    const int written = std::min(count, out_capacity);
    for (int i = 0; i < written; ++i) {
        const bd_detection& src = impl_->scratch[static_cast<size_t>(i)];
        Detection& dst = out[i];
        dst.id = src.id;
        dst.hamming = src.hamming;
        dst.decision_margin = src.decision_margin;
        for (int k = 0; k < 4; ++k) {
            dst.corners[static_cast<size_t>(k)][0] = src.corners[k * 2 + 0];
            dst.corners[static_cast<size_t>(k)][1] = src.corners[k * 2 + 1];
        }
        for (int k = 0; k < 9; ++k) dst.homography[static_cast<size_t>(k)] = src.homography[k];
    }
    return written;
}

Stats Detector::stats() const {
    if (!impl_) return {};
    const std::lock_guard<std::mutex> lock(impl_->mutex);
    bd_stats native;
    bd_detector_get_stats(impl_->handle, &native);
    Stats out;
    out.frames = native.frames;
    out.last_gpu_ms = native.last_gpu_ms;
    out.average_gpu_ms = native.average_gpu_ms;
    out.last_total_ms = native.last_total_ms;
    out.runs = native.runs;
    out.candidates = native.candidates;
    out.detections = native.detections;
    out.quads = native.quads;
    out.roi_rows = native.roi_rows;
    return out;
}

}  // namespace bdcudatag
