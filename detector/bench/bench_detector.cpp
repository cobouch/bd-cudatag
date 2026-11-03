// Accuracy and latency benchmark for bdcudatag on deterministic synthetic
// scenes.  Build with -DCMAKE_BUILD_TYPE=Release.
//
// Per resolution it reports the median and p95 detection time, the share of
// ground-truth tags detected, the median corner error in pixels and the number
// of wrong tag ids.  Run with no arguments for the standard resolution sweep,
// or `bdcudatag_bench WIDTH HEIGHT [frames]` for a single resolution.  Pass
// `--markdown` for a pasteable table.
//
// On a Jetson Orin Nano Super, build with -DCMAKE_CUDA_ARCHITECTURES=87 and run
// with the GPU clocks locked for stable numbers.  The host path includes copying
// the frame into pinned staging; pass a device pointer via detectDevice for
// zero-copy camera pipelines.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "bdcudatag/detector.hpp"
#include "scenes.hpp"

using namespace bdcudatag;

namespace {

constexpr int kWarmupFrames = 20;
constexpr int kSceneSize = 8;  // tags per scene, fixed by bench::make_scenes

struct Accuracy {
    int tags = 0;
    int found = 0;
    int wrong_ids = 0;
    double corner_median_px = 0.0;
    bool has_corners = false;
};

double percentile(const std::vector<double>& sorted, double fraction) {
    if (sorted.empty()) return 0.0;
    size_t index = static_cast<size_t>(std::ceil(fraction * sorted.size()));
    index = index == 0 ? 0 : index - 1;
    return sorted[std::min(sorted.size() - 1, index)];
}

// A detection counts for a tag when the centre is within 0.75 tag sides, the
// same rule the cross-detector harness uses.  Without it a decoy that happened
// to decode as an id also present in the scene could pass as that tag.
bool matches_tag(const Detection& detection,
                 const std::array<std::array<double, 2>, 4>& truth) {
    double cx = 0.0, cy = 0.0, tx = 0.0, ty = 0.0, side = 0.0;
    for (int i = 0; i < 4; ++i) {
        cx += detection.corners[i][0];
        cy += detection.corners[i][1];
        tx += truth[i][0];
        ty += truth[i][1];
        const int j = (i + 1) % 4;
        side += std::hypot(truth[j][0] - truth[i][0], truth[j][1] - truth[i][1]);
    }
    return std::hypot(cx * 0.25 - tx * 0.25, cy * 0.25 - ty * 0.25) <=
           0.75 * (side * 0.25);
}

// Mean distance between the four reported corners and ground truth, matching
// corners as an unordered set.  AprilTag-based libraries do not all report the
// same starting corner or winding, so the metric is about geometric accuracy,
// not ordering.
double corner_error(const Detection& detection,
                    const std::array<std::array<double, 2>, 4>& truth) {
    int order[4] = {0, 1, 2, 3};
    double best = 1e9;
    do {
        double sum = 0.0;
        for (int i = 0; i < 4; ++i) {
            const int j = order[i];
            sum += std::hypot(detection.corners[i][0] - truth[j][0],
                              detection.corners[i][1] - truth[j][1]);
        }
        best = std::min(best, sum / 4.0);
    } while (std::next_permutation(order, order + 4));
    return best;
}

void run_resolution(int width, int height, int frames, int scene_count, bool markdown) {
    const auto scenes = bench::make_scenes(width, height, scene_count);

    DetectorConfig config;
    config.max_detections = 32;
    config.use_cuda_graph = std::getenv("BD_NO_GRAPH") == nullptr;
    Detector detector(Family::Tag36h11, width, height, config);
    std::vector<Detection> out(32);

    for (int i = 0; i < kWarmupFrames; ++i) {
        detector.detect(scenes[i % scenes.size()].image.data(), width, out.data(), 32);
    }

    std::vector<double> times_ms;
    times_ms.reserve(static_cast<size_t>(frames));
    for (int i = 0; i < frames; ++i) {
        const auto& scene = scenes[i % scenes.size()];
        const auto start = std::chrono::steady_clock::now();
        detector.detect(scene.image.data(), width, out.data(), 32);
        const auto end = std::chrono::steady_clock::now();
        times_ms.push_back(
            std::chrono::duration<double, std::milli>(end - start).count());
    }
    std::sort(times_ms.begin(), times_ms.end());
    const double median_ms = percentile(times_ms, 0.50);
    const double p95_ms = percentile(times_ms, 0.95);
    double mean_ms = 0.0;
    for (const double value : times_ms) mean_ms += value;
    if (!times_ms.empty()) mean_ms /= static_cast<double>(times_ms.size());

    // One accuracy pass over the unique scenes.
    Accuracy accuracy;
    std::vector<double> errors;
    for (const auto& scene : scenes) {
        const int count = detector.detect(scene.image.data(), width, out.data(), 32);
        for (const auto& tag : scene.tags) {
            const auto truth = bench::ground_truth_corners(tag);
            double best = 1e9;
            for (int i = 0; i < count; ++i) {
                if (out[i].id != tag.id || !matches_tag(out[i], truth)) continue;
                best = std::min(best, corner_error(out[i], truth));
            }
            ++accuracy.tags;
            if (best < 1e8) {
                ++accuracy.found;
                errors.push_back(best);
            }
        }
        const int tag_count = static_cast<int>(scene.tags.size());
        for (int i = 0; i < count; ++i) {
            bool known = false;
            for (int t = 0; t < tag_count; ++t) {
                if (scene.tags[t].id == out[i].id) known = true;
            }
            if (!known) ++accuracy.wrong_ids;
        }
    }
    std::sort(errors.begin(), errors.end());
    accuracy.corner_median_px = percentile(errors, 0.50);
    accuracy.has_corners = !errors.empty();

    const double recall = accuracy.tags == 0
                              ? 0.0
                              : 100.0 * accuracy.found / accuracy.tags;
    const Stats stats = detector.stats();

    if (!markdown) {
        std::printf("%dx%d: median %.3f ms, mean %.3f ms, p95 %.3f ms, "
                    "tags found %.1f%% (%d/%d), ",
                    width, height, median_ms, mean_ms, p95_ms, recall, accuracy.found,
                    accuracy.tags);
        if (accuracy.has_corners) {
            std::printf("corner %.2f px, ", accuracy.corner_median_px);
        } else {
            std::printf("corner n/a, ");
        }
        std::printf("wrong ids %d\n", accuracy.wrong_ids);
        std::printf("    gpu avg %.3f ms, roi %d/%d rows, last frame %d runs, %d "
                    "candidates, %d quads\n",
                    stats.average_gpu_ms, stats.roi_rows, height, stats.runs,
                    stats.candidates, stats.quads);
    }

    if (markdown) {
        char resolution[32];
        std::snprintf(resolution, sizeof resolution, "%dx%d", width, height);
        std::printf("| %s | %.2f | %.2f | %.1f%% | ", resolution, median_ms, p95_ms,
                    recall);
        if (accuracy.has_corners) {
            std::printf("%.2f | ", accuracy.corner_median_px);
        } else {
            std::printf("n/a | ");
        }
        std::printf("%d |\n", accuracy.wrong_ids);
    }
}

}  // namespace

int main(int argc, char** argv) {
    struct Resolution {
        int width;
        int height;
    };
    const Resolution sweep[] = {{640, 480}, {1280, 720}, {1280, 800}, {1920, 1080}};

    int width = 0;
    int height = 0;
    int frames = 200;
    int scene_count = 32;
    bool markdown = false;

    std::vector<int> positional;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--markdown") == 0) {
            markdown = true;
        } else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            frames = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--scenes") == 0 && i + 1 < argc) {
            scene_count = std::atoi(argv[++i]);
        } else if (argv[i][0] == '-') {
            std::fprintf(stderr, "unknown option: %s\n", argv[i]);
            return 2;
        } else {
            positional.push_back(std::atoi(argv[i]));
        }
    }
    if (positional.size() >= 2) {
        width = positional[0];
        height = positional[1];
    }
    if (positional.size() >= 3) frames = positional[2];

    if ((positional.size() >= 2 && (width < 16 || height < 16)) || frames < 1 ||
        scene_count < 1 || positional.size() == 1) {
        std::fprintf(stderr,
                     "usage: %s [width height [frames]] [--frames N] [--scenes N] "
                     "[--markdown]\n",
                     argv[0]);
        return 2;
    }

    if (!markdown) {
        std::printf("bdcudatag accuracy/latency benchmark\n");
        std::printf("  tag36h11, %d tags per scene, %d scenes, %d timed frames "
                    "(+%d warm-up)\n",
                    kSceneSize, scene_count, frames, kWarmupFrames);
    }

    if (markdown) {
        std::printf("| Resolution | Median (ms) | p95 (ms) | Tags found | Corner "
                    "error (px) | Wrong IDs |\n");
        std::printf("| --- | ---: | ---: | ---: | ---: | ---: |\n");
    }

    if (width > 0 && height > 0) {
        run_resolution(width, height, frames, scene_count, markdown);
    } else {
        for (const Resolution& r : sweep) {
            run_resolution(r.width, r.height, frames, scene_count, markdown);
        }
    }
    return 0;
}
