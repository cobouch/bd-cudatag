// Synthetic end-to-end tests. No camera or external data required: tags are
// rendered from the same family tables at known poses, then detected.
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <thread>
#include <vector>

#include "bdcudatag/detector.hpp"
#include "tag_families.hpp"
#include "test_support.hpp"

using namespace bdcudatag;
using namespace bdcudatag::test;

namespace {

constexpr int kW = 640;
constexpr int kH = 480;

const detail::TagFamilyData& family_of(Family family) {
    return *detail::kFamilies[static_cast<int>(family)];
}

std::vector<uint8_t> render_single(Family family, int id, int width, int height,
                                   const Mat3& pose) {
    return render_tag(family_of(family), id, width, height, pose);
}

void composite(std::vector<uint8_t>& dst, const std::vector<uint8_t>& src) {
    for (size_t i = 0; i < dst.size(); ++i) dst[i] = std::min(dst[i], src[i]);
}

DetectorConfig test_config() {
    DetectorConfig config;
    config.max_detections = 32;
    config.min_decision_margin = 2.0f;
    return config;
}

const std::array<std::array<double, 2>, 4> kUnitCorners = {{{-1.0, -1.0},
                                                             {1.0, -1.0},
                                                             {1.0, 1.0},
                                                             {-1.0, 1.0}}};

std::array<std::array<double, 2>, 4> ground_truth_corners(const Mat3& pose, double size_px) {
    // The pose is built for a square of side 2 in tag coordinates. Project the
    // unit-square corners directly.
    std::array<std::array<double, 2>, 4> out{};
    for (int i = 0; i < 4; ++i) {
        pose.project(kUnitCorners[i][0], kUnitCorners[i][1], &out[i][0], &out[i][1]);
    }
    (void)size_px;
    return out;
}

void check_corners(const Detection& detection, const Mat3& pose, double tolerance) {
    const auto gt = ground_truth_corners(pose, 0.0);
    for (int i = 0; i < 4; ++i) {
        CHECK_NEAR(detection.corners[i][0], gt[i][0], tolerance);
        CHECK_NEAR(detection.corners[i][1], gt[i][1], tolerance);
    }
}

void test_family_tables() {
    const int before = failures;
    const auto& fam36 = detail::kTag36h11;
    const auto& fam16 = detail::kTag16h5;
    CHECK(fam36.code_count == 587);
    CHECK(fam16.code_count == 30);
    CHECK(fam36.bits_per_side == 6);
    CHECK(fam16.bits_per_side == 4);
    CHECK(fam36.min_hamming == 11);
    CHECK(fam16.min_hamming == 5);

    const auto min_distance = [](const detail::TagFamilyData& fam) {
        int best = 1000;
        for (int i = 0; i < fam.code_count; ++i) {
            for (int j = 0; j < fam.code_count; ++j) {
                uint64_t rotated = fam.codes[j];
                for (int k = 0; k < 4; ++k) {
                    if (i != j || k != 0) {
                        best = std::min(best, popcount(fam.codes[i] ^ rotated));
                    }
                    rotated = rotate_code(rotated, fam.bits_per_side);
                }
            }
        }
        return best;
    };
    const int d36 = min_distance(fam36);
    const int d16 = min_distance(fam16);
    std::printf("family distances: tag36h11=%d tag16h5=%d\n", d36, d16);
    CHECK(d36 == 11);
    CHECK(d16 == 5);
    if (failures == before) std::printf("PASS family tables\n");
}

void test_centered_tag(Family family, int id, int size_px) {
    const int before = failures;
    const Mat3 pose = make_pose(kW * 0.5, kH * 0.5, size_px, 0.0);
    const auto image = render_single(family, id, kW, kH, pose);
    if (std::getenv("BD_DUMP")) {
        FILE* f = std::fopen("centered.pgm", "wb");
        if (f) {
            std::fprintf(f, "P5\n%d %d\n255\n", kW, kH);
            std::fwrite(image.data(), 1, image.size(), f);
            std::fclose(f);
        }
    }

    Detector detector(family, kW, kH, test_config());
    std::vector<Detection> out(32);
    const int count = detector.detect(image.data(), kW, out.data(), 32);
    const auto stats = detector.stats();
    std::printf("centered %s id %d size %d: count=%d runs=%d candidates=%d quads=%d detections=%d\n",
                family == Family::Tag36h11 ? "tag36h11" : "tag16h5", id, size_px, count,
                stats.runs, stats.candidates, stats.quads, stats.detections);
    CHECK(count == 1);
    if (count == 1) {
        CHECK(out[0].id == id);
        CHECK(out[0].hamming == 0);
        CHECK(out[0].decision_margin > 2.0f);
        check_corners(out[0], pose, 2.5);
        CHECK(stats.frames == 1);
        CHECK(stats.detections == 1);
        CHECK(stats.candidates >= 1);
    }
    std::printf("centered %s id %d size %d: count=%d\n",
                family == Family::Tag36h11 ? "tag36h11" : "tag16h5", id, size_px, count);
    if (failures == before) std::printf("PASS centered\n");
}

void test_rotations() {
    const int before = failures;
    const detail::TagFamilyData& fam = detail::kTag36h11;
    Detector detector(Family::Tag36h11, kW, kH, test_config());
    std::vector<Detection> out(32);
    for (int r = 0; r < 4; ++r) {
        const double angle = r * 1.5707963267948966;
        const Mat3 pose = make_pose(kW * 0.5, kH * 0.5, 170.0, angle);
        const auto image = render_tag(fam, 0, kW, kH, pose);
        const int count = detector.detect(image.data(), kW, out.data(), 32);
        CHECK(count == 1);
        if (count == 1) {
            CHECK(out[0].id == 0);
            check_corners(out[0], pose, 2.5);
        }
        std::printf("rotation %d: count=%d\n", r * 90, count);
    }
    if (failures == before) std::printf("PASS rotations\n");
}

void test_hamming() {
    const int before = failures;
    const detail::TagFamilyData& fam = detail::kTag36h11;
    const Mat3 pose = make_pose(kW * 0.5, kH * 0.5, 170.0, 0.15);
    const auto image = render_tag(fam, 0, kW, kH, pose, 2, ~0ull, true);

    {
        Detector detector(Family::Tag36h11, kW, kH, test_config());
        std::vector<Detection> out(32);
        const int count = detector.detect(image.data(), kW, out.data(), 32);
        CHECK(count == 1);
        if (count == 1) {
            CHECK(out[0].id == 0);
            CHECK(out[0].hamming == 1);
        }
        std::printf("single-bit error: count=%d hamming=%d\n", count,
                    count ? out[0].hamming : -1);
    }
    {
        DetectorConfig config = test_config();
        config.max_hamming = 0;
        Detector detector(Family::Tag36h11, kW, kH, config);
        std::vector<Detection> out(32);
        const int count = detector.detect(image.data(), kW, out.data(), 32);
        CHECK(count == 0);
        std::printf("single-bit error, hamming limit 0: count=%d\n", count);
    }
    if (failures == before) std::printf("PASS hamming\n");
}

void test_multiple() {
    const int before = failures;
    const int W = 1280;
    const int H = 720;
    const detail::TagFamilyData& fam = detail::kTag36h11;

    struct Case {
        int id;
        Mat3 pose;
    };
    std::vector<Case> cases;
    cases.push_back({3, make_pose(220, 180, 120, 0.2)});
    cases.push_back({17, make_pose(640, 180, 140, -0.35)});
    cases.push_back({101, make_pose(1050, 190, 110, 0.6)});
    cases.push_back({586, make_pose(420, 520, 150, 0.9)});
    cases.push_back({42, make_pose(950, 520, 100, -0.1)});

    std::vector<uint8_t> image(static_cast<size_t>(W) * H, 255);
    for (const auto& c : cases) {
        composite(image, render_tag(fam, c.id, W, H, c.pose));
    }

    DetectorConfig config = test_config();
    config.max_detections = 32;
    Detector detector(Family::Tag36h11, W, H, config);
    std::vector<Detection> out(32);
    const int count = detector.detect(image.data(), W, out.data(), 32);
    std::printf("multiple tags: count=%d\n", count);
    CHECK(count == static_cast<int>(cases.size()));

    std::set<int> found;
    for (int i = 0; i < count; ++i) found.insert(out[i].id);
    for (const auto& c : cases) {
        CHECK(found.count(c.id) == 1);
        for (int i = 0; i < count; ++i) {
            if (out[i].id == c.id) check_corners(out[i], c.pose, 3.0);
        }
    }
    if (failures == before) std::printf("PASS multiple\n");
}

void test_small_tag() {
    const int before = failures;
    const Mat3 pose = make_pose(kW * 0.5, kH * 0.5, 38.0, 0.25);
    const auto image = render_single(Family::Tag36h11, 7, kW, kH, pose);
    Detector detector(Family::Tag36h11, kW, kH, test_config());
    std::vector<Detection> out(32);
    const int count = detector.detect(image.data(), kW, out.data(), 32);
    std::printf("small tag (38 px): count=%d\n", count);
    CHECK(count == 1);
    if (count == 1) {
        CHECK(out[0].id == 7);
        check_corners(out[0], pose, 4.0);
    }
    if (failures == before) std::printf("PASS small tag\n");
}

void test_blank() {
    const int before = failures;
    std::vector<uint8_t> image(static_cast<size_t>(kW) * kH, 255);
    Detector detector(Family::Tag36h11, kW, kH, test_config());
    std::vector<Detection> out(32);
    const int count = detector.detect(image.data(), kW, out.data(), 32);
    CHECK(count == 0);
    std::vector<uint8_t> dark(static_cast<size_t>(kW) * kH, 0);
    CHECK(detector.detect(dark.data(), kW, out.data(), 32) == 0);
    std::printf("blank images: count=%d\n", count);
    if (failures == before) std::printf("PASS blank\n");
}

void test_device_path() {
    const int before = failures;
    const Mat3 pose = make_pose(kW * 0.5, kH * 0.5, 150.0, 0.1);
    const auto image = render_single(Family::Tag36h11, 11, kW, kH, pose);

    uint8_t* device = nullptr;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&device), image.size()) == cudaSuccess);
    if (!device) return;
    CHECK(cudaMemcpy(device, image.data(), image.size(), cudaMemcpyHostToDevice) ==
          cudaSuccess);

    Detector detector(Family::Tag36h11, kW, kH, test_config());
    std::vector<Detection> out(32);
    const int count = detector.detectDevice(device, kW, out.data(), 32);
    CHECK(count == 1);
    if (count == 1) {
        CHECK(out[0].id == 11);
        check_corners(out[0], pose, 2.5);
    }
    cudaFree(device);
    std::printf("device input: count=%d\n", count);
    if (failures == before) std::printf("PASS device input\n");
}

void test_homography() {
    const int before = failures;
    const Mat3 pose = make_pose(kW * 0.5, kH * 0.5, 170.0, 0.3, 0.00035);
    const auto image = render_single(Family::Tag36h11, 5, kW, kH, pose);

    Detector detector(Family::Tag36h11, kW, kH, test_config());
    std::vector<Detection> out(32);
    const int count = detector.detect(image.data(), kW, out.data(), 32);
    CHECK(count == 1);
    if (count == 1) {
        const Detection& d = out[0];
        const double tx[4] = {-1.0, 1.0, 1.0, -1.0};
        const double ty[4] = {-1.0, -1.0, 1.0, 1.0};
        for (int i = 0; i < 4; ++i) {
            const double w = d.homography[6] * tx[i] + d.homography[7] * ty[i] + d.homography[8];
            const double px =
                (d.homography[0] * tx[i] + d.homography[1] * ty[i] + d.homography[2]) / w;
            const double py =
                (d.homography[3] * tx[i] + d.homography[4] * ty[i] + d.homography[5]) / w;
            CHECK_NEAR(px, d.corners[i][0], 0.01);
            CHECK_NEAR(py, d.corners[i][1], 0.01);
        }
        CHECK_NEAR(d.homography[8], 1.0, 1e-5);

        double gx = 0.0, gy = 0.0;
        pose.project(0.0, 0.0, &gx, &gy);
        CHECK_NEAR(d.homography[2] / d.homography[8], gx, 3.0);
        CHECK_NEAR(d.homography[5] / d.homography[8], gy, 3.0);
    }
    std::printf("homography: count=%d\n", count);
    if (failures == before) std::printf("PASS homography\n");
}

void test_thread_safety() {
    const int before = failures;
    const Mat3 pose = make_pose(kW * 0.5, kH * 0.5, 170.0, 0.0);
    const auto image = render_single(Family::Tag36h11, 0, kW, kH, pose);

    Detector detector(Family::Tag36h11, kW, kH, test_config());
    std::atomic<int> mismatches{0};
    std::vector<std::thread> workers;
    for (int t = 0; t < 4; ++t) {
        workers.emplace_back([&detector, &image, &mismatches] {
            std::vector<Detection> out(32);
            for (int i = 0; i < 5; ++i) {
                const int count = detector.detect(image.data(), kW, out.data(), 32);
                if (count != 1 || out[0].id != 0) ++mismatches;
            }
        });
    }
    for (auto& worker : workers) worker.join();

    CHECK(mismatches.load() == 0);
    CHECK(detector.stats().frames == 20);
    std::printf("thread safety: frames=%llu mismatches=%d\n",
                static_cast<unsigned long long>(detector.stats().frames), mismatches.load());
    if (failures == before) std::printf("PASS thread safety\n");
}

void test_config_and_errors() {
    const int before = failures;
    CHECK(bd_family_count() == 2);
    CHECK(std::string(bd_family_name(0)) == "tag36h11");
    CHECK(std::string(bd_family_name(1)) == "tag16h5");
    CHECK(bd_family_default_max_hamming(0) == 2);
    CHECK(bd_family_default_max_hamming(1) == 1);

    int32_t status = BD_OK;
    bd_detector* bad = bd_detector_create(99, kW, kH, nullptr, &status);
    CHECK(bad == nullptr);
    CHECK(status == BD_ERROR_INVALID_ARGUMENT);
    bad = bd_detector_create(0, 0, kH, nullptr, &status);
    CHECK(bad == nullptr);
    CHECK(status == BD_ERROR_INVALID_ARGUMENT);

    bool threw = false;
    try {
        Detector detector(Family::Tag36h11, kW, kH, DetectorConfig{});
        std::vector<Detection> out(4);
        detector.detect(nullptr, kW, out.data(), 4);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
    if (failures == before) std::printf("PASS config/errors\n");
}

}  // namespace

int main() {
    std::printf("bdcudatag tests\n");
    test_family_tables();
    test_centered_tag(Family::Tag36h11, 0, 170);
    test_centered_tag(Family::Tag16h5, 0, 120);
    test_rotations();
    test_hamming();
    test_multiple();
    test_small_tag();
    test_blank();
    test_device_path();
    test_homography();
    test_thread_safety();
    test_config_and_errors();
    return finish("bdcudatag_tests");
}
