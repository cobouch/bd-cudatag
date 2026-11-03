// Throughput benchmark with synthetic tags. Build with -DCMAKE_BUILD_TYPE=Release.
//
// On a Jetson Orin Nano Super, build with -DCMAKE_CUDA_ARCHITECTURES=87 and run
// with the GPU clocks locked for stable numbers. The host path includes copying
// the frame into pinned staging; pass a device pointer via detectDevice for
// zero-copy camera pipelines.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "bdcudatag/detector.hpp"
#include "test_support.hpp"

using namespace bdcudatag;

int main(int argc, char** argv) {
    int width = 1280;
    int height = 720;
    int frames = 600;
    if (argc > 1) width = std::atoi(argv[1]);
    if (argc > 2) height = std::atoi(argv[2]);
    if (argc > 3) frames = std::atoi(argv[3]);
    if (width < 16 || height < 16 || frames < 1) {
        std::fprintf(stderr, "usage: %s [width] [height] [frames]\n", argv[0]);
        return 2;
    }

    const auto& family = bdcudatag::detail::kTag36h11;
    std::vector<uint8_t> image(static_cast<size_t>(width) * height, 255);

    struct Placement {
        int id;
        bdcudatag::test::Mat3 pose;
    };
    std::vector<Placement> placements;
    const int ids[8] = {0, 3, 7, 11, 17, 42, 101, 586};
    const double sizes[8] = {140, 110, 90, 160, 120, 100, 150, 130};
    const double xs[8] = {0.13, 0.36, 0.61, 0.86, 0.18, 0.42, 0.67, 0.90};
    const double ys[8] = {0.28, 0.24, 0.30, 0.26, 0.72, 0.76, 0.70, 0.74};
    const double angles[8] = {0.05, 0.5, -0.7, 0.2, 0.9, -0.2, 0.35, -0.55};
    for (int i = 0; i < 8; ++i) {
        placements.push_back({ids[i], bdcudatag::test::make_pose(xs[i] * width, ys[i] * height,
                                                                 sizes[i], angles[i])});
        bdcudatag::test::composite_image(
            image, bdcudatag::test::render_tag(family, ids[i], width, height,
                                               placements.back().pose));
    }

    DetectorConfig config;
    config.max_detections = 32;
    config.use_cuda_graph = std::getenv("BD_NO_GRAPH") == nullptr;
    Detector detector(Family::Tag36h11, width, height, config);
    std::vector<Detection> out(32);

    for (int i = 0; i < 50; ++i) {
        detector.detect(image.data(), width, out.data(), 32);
    }

    std::vector<double> times_ms;
    times_ms.reserve(frames);
    int last_count = 0;
    const auto total_start = std::chrono::steady_clock::now();
    for (int i = 0; i < frames; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        last_count = detector.detect(image.data(), width, out.data(), 32);
        const auto t1 = std::chrono::steady_clock::now();
        times_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    const auto total_end = std::chrono::steady_clock::now();
    const double total_s = std::chrono::duration<double>(total_end - total_start).count();

    std::vector<double> sorted = times_ms;
    std::sort(sorted.begin(), sorted.end());
    const double mean = total_s / frames * 1000.0;
    const double p50 = sorted[sorted.size() / 2];
    const double p99 = sorted[std::min(sorted.size() - 1, static_cast<size_t>(sorted.size() * 0.99))];
    const Stats stats = detector.stats();

    std::printf("bdcudatag benchmark\n");
    std::printf("  image      : %d x %d, %d tags\n", width, height, last_count);
    std::printf("  frames     : %d (+50 warm-up)\n", frames);
    std::printf("  throughput : %.1f FPS (%.3f ms/frame)\n", frames / total_s, mean);
    std::printf("  latency    : p50 %.3f ms, p99 %.3f ms\n", p50, p99);
    std::printf("  gpu        : last %.3f ms, avg %.3f ms\n", stats.last_gpu_ms,
                stats.average_gpu_ms);
    std::printf("  last frame : %d runs, %d candidates, %d quads, %d detections\n", stats.runs,
                stats.candidates, stats.quads, stats.detections);
    std::printf("  roi        : %d / %d rows searched\n", stats.roi_rows, height);
    return 0;
}
