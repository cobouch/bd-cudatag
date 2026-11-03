// Exports the synthetic benchmark scenes (images + exact ground truth) so that
// detectors which are not linked into this build can be scored on identical
// frames.  This is the input to compare_detectors.py.
//
//   bdcudatag_scenes [output-dir] [scene-count]
//
// Writes scenes_<WxH>.json and one PGM per scene for every resolution.
#include <cstdio>
#include <cstdlib>
#include <string>

#include "scenes.hpp"

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "scenes";
    const int count = argc > 2 ? std::atoi(argv[2]) : 32;
    if (count < 1) {
        std::fprintf(stderr, "usage: %s [output-dir] [scene-count]\n", argv[0]);
        return 2;
    }

    struct Resolution {
        int width;
        int height;
    };
    const Resolution resolutions[] = {
        {640, 480}, {1280, 720}, {1280, 800}, {1920, 1080}};

    for (const Resolution& r : resolutions) {
        const auto scenes = bdcudatag::bench::make_scenes(r.width, r.height, count);
        char prefix[32];
        std::snprintf(prefix, sizeof prefix, "%dx%d", r.width, r.height);
        if (!bdcudatag::bench::write_scene_set(scenes, dir, prefix)) {
            std::fprintf(stderr, "failed to write scenes to %s\n", dir.c_str());
            return 1;
        }
        std::printf("wrote %zu scenes at %dx%d to %s\n", scenes.size(), r.width, r.height,
                    dir.c_str());
    }
    return 0;
}
