# bdcudatag

A CUDA AprilTag-family detector for FRC 6324, with a C++ core in `detector/` and
a Java FFM (Panama) wrapper in `java-api/`. It detects `tag36h11` and `tag16h5`
and deliberately does **not** solve pose: the output is the tag id, Hamming
distance, decision margin, four image-space corners and the 3x3 tag-to-image
homography. Pose estimation is left up to the consumer of the library.

The implementation is original. It is informed by the general structure of GPU
AprilTag detectors such as
[SpectrumJetson](https://https://github.com/Spectrum3847/SpectrumJetson) as well
as high-performance CPU detectors such as
[NAPA](https://github.com/zwatx/NAPA/), but shares no code with them. The
tag-family code tables are generated from the published AprilTag families and
validated bit-for-bit against the canonical tag images; see
`detector/third_party/NOTICE`.

## Pipeline

Each frame runs as one CUDA graph with pinned host staging:

1. **Coarse ROI.** The image is reduced to 8x8 tiles and each tile's min/max is
   measured. The threshold kernel pools those extremes over a 3x3 tile window
   and requires the pooled range to clear a per-frame noise floor (1.5x the
   lower quartile of tile spans), so sensor noise alone cannot mark tiles
   active. Active tile rows are closed vertically and dilated horizontally,
   producing the only rows that are binarized, run-length encoded and searched.
   Flat background rows cost almost nothing, and the number of searched rows is
   reported as `roiRows`.
2. **Binarization.** A tile with real contrast uses the midpoint of its pooled
   range. A tile with no contrast of its own falls back to the frame's Otsu
   split over the tile means, so a dim background stays white instead of merging
   with the tag border, while a thick black border stays black instead of
   splitting into a thin ring. Only rows inside the ROI are binarized; inactive
   rows are filled as background.
3. **Run-length connected components.** Black runs are extracted with one warp
   per 32-pixel chunk, joined at chunk boundaries, and merged across rows with a
   lock-free union-find.
4. **Candidate filtering.** Components are filtered by area, bounding-box size,
   border contact, aspect ratio and fill. Survivors are bucketed by area and
   flattened largest-first, so when a noisy or dense frame produces more
   candidates than the cap, the small specks and glyphs are dropped instead of
   whichever components the atomic counter happened to see first.
5. **Quad fitting.** Each candidate is one block. It collects the leftmost and
   rightmost component pixel of every row in its bounding box (in parallel),
   probes those points in 32 directions to recover the convex hull, and reduces
   the hull to four corners. Each of the four edges is then fitted to gradient
   peaks by one warp, with a parabolic subpixel peak and a second refinement
   pass. The first pass searches a full cell either way and, when the profile
   has two peaks, takes the inner one: a tag on a surface whose brightness sits
   between its black and white cells binarizes with a one-cell halo, and the
   hull then starts outside the black square. This replaced a serial Moore
   contour trace plus Douglas-Peucker simplification, which dominated frame time.
6. **Decoding.** The quad is sampled with a homography for all four rotations,
   matched against the family code table by popcount, and scored by decision
   margin. One warp decodes one candidate and eight warps share a block, so the
   grid is not padded to `max_candidates`. No decoding or corner work happens
   for frames with no candidates. The unrefined hull is decoded as well and
   used only when no rotation of the refined quad decodes, so edge refinement
   can never lose a tag it would otherwise have found.

Everything is allocated at construction time; `detect` does not allocate device
memory. A CUDA graph is captured on the first host-frame call and replayed
afterwards. `detectDevice` accepts a CUDA device pointer and skips the host to
device copy.

## Build

Requirements: CUDA 12.x, CMake 3.24+, a C++20 compiler. The default GPU
architecture is SM 8.7 (Jetson Orin Nano Super).

```bash
cmake -S detector -B detector/build -DCMAKE_BUILD_TYPE=Release
cmake --build detector/build -j
./detector/build/bdcudatag_tests
./detector/build/bdcudatag_bench 1280 800 600
./detector/build/bdcudatag_bench --markdown   # resolution sweep, accuracy + latency
./detector/build/bdcudatag_scenes detector/bench/scenes 32   # scenes for the comparison harness
```

On Windows with Visual Studio 2022:

```powershell
cmake -S detector -B detector/build -G "Visual Studio 17 2022" -A x64 `
      -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build detector/build --config Release
.\detector\build\Release\bdcudatag_tests.exe
```

On a Jetson, build on the device and lock the clocks for stable measurements:

```bash
cmake -S detector -B detector/build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build detector/build -j
sudo nvpmodel -m 0
sudo jetson_clocks
./detector/build/bdcudatag_bench 1280 720 2000
```

`BD_NO_GRAPH=1` runs the benchmark without CUDA graph capture, which is useful
when comparing driver overhead. `-DCMAKE_CUDA_ARCHITECTURES=86` targets desktop
Ampere for local testing; use `87` for the Orin Nano.

## Benchmarks

The benchmark compares bdcudatag with NAPA, OpenCV's ArUco and the original
AprilTag 3 library on identical frames. `detector/bench` renders deterministic
synthetic scenes from the family tables, so ground truth is exact and no camera
is needed: 32 scenes per resolution, eight `tag36h11` tags at known poses, and
on a quarter of the scenes two tag-shaped decoys filled with random data that
must not decode. Images use 2x2 supersampling plus deterministic noise of up to
3 DN.

Every detector runs single-threaded in one process, after 20 warm-up frames,
cycling through the scenes. Metrics:

- **Median / p95**: wall clock around one `detect` call, in milliseconds.
- **Tags found**: ground-truth tags matched to a detection with the same id and
  a centre within 0.75 tag sides, as a percentage.
- **Corner error**: median over matched tags of the mean distance between the
  four detected corners and ground truth, in pixels. Corners are matched as an
  unordered set because libraries disagree on the starting corner and winding.
- **Wrong IDs**: detections whose id belongs to no tag in the scene, including
  the decoys.

Measured on Windows 11 with an AMD Ryzen 9 7900X and an RTX 3060 (driver 610.88,
CUDA 12.9). All scenes use `tag36h11`. Configurations: bdcudatag with defaults
and the CUDA graph enabled; NAPA 0.5.1 with its defaults, plus `min_tag_px=16`
as a second row; ArUco from OpenCV 4.12.0 with `DICT_APRILTAG_36h11` and
`CORNER_REFINE_SUBPIX`; AprilTag 3 through pupil-apriltags 1.0.4.post11 at full
resolution (`quad_decimate=1.0`, `refine_edges=1`, one thread).

| Resolution | Detector | Median (ms) | p95 (ms) | Tags found | Corner error (px) | Wrong IDs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 640x480 | bdcudatag | 0.32 | 0.38 | 100.0% | 0.13 | 0 |
| 640x480 | NAPA | 0.25 | 0.28 | 99.6% | 0.10 | 0 |
| 640x480 | NAPA (min_tag_px=16) | 0.54 | 0.59 | 100.0% | 0.10 | 0 |
| 640x480 | ArUco (subpixel) | 1.17 | 1.42 | 99.6% | 0.30 | 0 |
| 640x480 | AprilTag 3 (full res) | 2.11 | 2.30 | 100.0% | 0.71 | 0 |
| 1280x720 | bdcudatag | 0.42 | 0.44 | 100.0% | 0.17 | 0 |
| 1280x720 | NAPA | 0.47 | 0.57 | 100.0% | 0.15 | 0 |
| 1280x720 | NAPA (min_tag_px=16) | 1.32 | 1.40 | 100.0% | 0.15 | 0 |
| 1280x720 | ArUco (subpixel) | 2.36 | 3.02 | 100.0% | 0.29 | 0 |
| 1280x720 | AprilTag 3 (full res) | 8.87 | 10.09 | 100.0% | 0.71 | 0 |
| 1280x800 | bdcudatag | 0.49 | 0.60 | 100.0% | 0.17 | 0 |
| 1280x800 | NAPA | 0.72 | 0.85 | 100.0% | 0.16 | 0 |
| 1280x800 | NAPA (min_tag_px=16) | 2.30 | 2.57 | 100.0% | 0.16 | 0 |
| 1280x800 | ArUco (subpixel) | 2.90 | 3.54 | 100.0% | 0.30 | 0 |
| 1280x800 | AprilTag 3 (full res) | 9.69 | 11.09 | 100.0% | 0.72 | 0 |
| 1920x1080 | bdcudatag | 0.70 | 0.75 | 100.0% | 0.23 | 0 |
| 1920x1080 | NAPA | 0.87 | 1.20 | 100.0% | 0.23 | 0 |
| 1920x1080 | NAPA (min_tag_px=16) | 3.03 | 4.56 | 100.0% | 0.23 | 0 |
| 1920x1080 | ArUco (subpixel) | 5.77 | 6.41 | 100.0% | 0.33 | 0 |
| 1920x1080 | AprilTag 3 (full res) | 18.47 | 19.89 | 100.0% | 0.72 | 0 |

bdcudatag is the fastest detector at 1280x720 (0.42 vs 0.47 ms for NAPA), at
1280x800 (0.49 vs 0.72 ms) and at 1920x1080 (0.70 vs 0.87 ms). Its corner error
matches NAPA's at 1080p (0.23 px) and is 0.01 px higher at 800p and 0.02 px
higher at 720p. At 640x480 NAPA is still slightly faster (0.25 vs 0.32 ms) and
more accurate (0.10 vs 0.13 px), but it misses one tag where bdcudatag finds
all 256. bdcudatag is 3.7-8.2x faster than ArUco and 6.6-26x faster than
full-resolution AprilTag. No detector decoded a decoy, so the wrong-id column is
zero throughout; bdcudatag finds every tag at every resolution.

### NAPA AprilTag Synthetic Benchmark

`detector/bench/datasets/napa-benchmark` holds the 4,500-frame NAPA synthetic
corpus: `main` (20 scenarios), `stress` (four harder ones) and `range` (a size
sweep), at 640x480, 1280x720 and 1920x1080, with float64 ground-truth corners
bounding the black square. `eval_napa.py` scores bdcudatag on it through the
same C ABI and the same match rule as the harness above:

```bash
cd detector/bench
.venv/Scripts/python eval_napa.py --group main --group stress --group range \
    --markdown napa_results.md
```

| Group | Resolution | Tags | Tags found | Corner (px) | Wrong IDs | Mean (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| main | 640x480 | 1811 | 99.6% | 0.04 | 1 | 0.34 |
| main | 1280x720 | 1888 | 99.7% | 0.05 | 0 | 0.55 |
| main | 1920x1080 | 1822 | 99.7% | 0.05 | 0 | 0.85 |
| stress | 640x480 | 1289 | 100.0% | 0.06 | 0 | 0.42 |
| stress | 1280x720 | 1642 | 100.0% | 0.06 | 0 | 0.69 |
| stress | 1920x1080 | 1646 | 100.0% | 0.07 | 0 | 1.02 |
| range | 640x480 | 900 | 100.0% | 0.05 | 0 | 0.32 |

Across the whole corpus bdcudatag finds 10,979 of 10,998 tags (99.8%) with a
0.05 px median corner error. Every stress and range tag is found, and the 19
remaining misses are 18 `low-contrast` tags plus one `steep-angle` tag at
640x480. The single wrong ID is `decoys-640x480-042`, the corpus's documented
two-bit collision with id 323; the same frame is NAPA's only wrong ID. Times are
end-to-end `detect` calls, so the 2 MB host-to-device copy dominates 1080p; in
the standard 32-scene sweep above the mean is 0.44 ms across the four
resolutions.

### Hard synthetic corpus

NAPA stops at moderately hard conditions, so `gen_synthetic.py` renders a second
corpus in the same layout for the conditions its `low-contrast` scenario only
hints at:
a lifted black level with a narrow dark quiet zone, deep underexposure, heavy
sensor noise, motion blur, a saturated highlight, and a quiet zone clipped by
the frame edge. Tags are rendered 4x supersampled from the OpenCV family table,
and ground-truth corners are the same homography the pixels were drawn through.

```bash
cd detector/bench
.venv/Scripts/python gen_synthetic.py --frames 30    # 780 frames, 13 scenarios
.venv/Scripts/python eval_napa.py --root datasets/synthetic-hard \
    --markdown synthetic_results.md
```

| Scenario | 640x480 | 1280x720 | Contents |
| --- | ---: | ---: | --- |
| tiny | 100.0% | 100.0% | 10-22 px tags, clean image |
| lowcontrast | 45.0% | 68.6% | black 96-118, cells 160-182, bg 242 |
| dark-noisy | 98.2% | 100.0% | background 28-48, cells 52-88, noise 5-9 |
| noise | 100.0% | 97.7% | noise sigma 15-22 on a textured background |
| underexposed | 100.0% | 100.0% | exposure 0.22-0.34 |
| blur | 80.7% | 100.0% | Gaussian sigma 1.1-2.2 on 18-60 px tags |
| motion | 68.9% | 96.5% | linear motion blur, 6-15 px kernels |
| jpeg | 100.0% | 98.5% | quality 8-20 |
| shadow | 100.0% | 100.0% | a shadow edge across the tag |
| glare | 98.1% | 92.7% | a saturated specular blob |
| edge | 100.0% | 100.0% | quiet zone clipped by the image border |
| dense | 99.8% | 99.8% | 12-20 tags of 22-44 px |
| mixed | 93.5% | 100.0% | dim, noisy, blurred and small together |

Overall 2,293 of 2,389 tags (96.0%) with a 0.27 px median corner error and no
wrong IDs. This corpus is deliberately harsher than NAPA — its low-contrast
scenario is still the open problem at 640x480, and it is the regression net for
further work there — so the numbers are lower by design and are not a
cross-detector leaderboard.

`detect` is synchronous and includes host-to-device staging and a device
synchronization, so these are end-to-end frame times rather than kernel times.
On Windows/WDDM part of each frame is driver submission overhead that does not
scale with resolution; on a Jetson (Linux/Tegra) the same call is cheaper, so
measure on-device before quoting FPS. The native benchmark also prints GPU time
and ROI rows for bdcudatag at each resolution.

### Reproducing

The native benchmark needs only the CUDA build:

```bash
cmake -S detector -B detector/build -DCMAKE_BUILD_TYPE=Release
cmake --build detector/build -j --target bdcudatag_bench bdcudatag_scenes
./detector/build/bdcudatag_bench --frames 200 --scenes 32 --markdown
```

The comparison harness runs all four detectors in one process. It needs a built
`bdcudatag` library plus optional Python packages; missing ones are skipped by
name, so the table can be regenerated one detector at a time:

```bash
./detector/build/bdcudatag_scenes detector/bench/scenes 32   # no GPU required
cd detector/bench
python -m venv --system-site-packages .venv
.venv/Scripts/python -m pip install -r requirements.txt      # .venv/bin/python on Linux
.venv/Scripts/python gen_synthetic.py --frames 30            # hard corpus, no GPU required
.venv/Scripts/python compare_detectors.py --scenes scenes --frames 200 \
    --markdown results.md
```

NAPA is GPL-3.0. It is imported only by the benchmark script for comparison;
the detector and the rest of the repository do not link against it.

## Java API

Requires Java 22+ (the project toolchain is Java 25). The jar is self-contained:
`./gradlew build` copies the native library from `detector/build/Release` into
the jar under `com/frc6324/bdcudatag/internal/native/<platform>/`, and the
loader extracts it to a temporary file on first use. Consumers need only the
jar; `detector/` does not have to exist on their machine.

```bash
cd java-api
./gradlew build                                        # bundles ../detector/build/Release
./gradlew build -Pbdcudatag.native.dir=/path/to/build  # or an explicit build directory
```

The resource platform tag comes from the machine that builds the jar (for
example `windows-x86_64`, `linux-aarch64`), so build the jar on each platform
you ship. If a jar was built without `detector/`, loading falls back to
`-Dbdcudatag.library.path=/path/to/library` and then `java.library.path`.

Loading the native library is a restricted operation, so start the JVM with
`--enable-native-access=com.frc6324.bdcudatag` when the jar is on the module
path (use `ALL-UNNAMED` when it is on the classpath). Java 25 only warns without
it, but the call will be rejected in a future release.

The JUnit suite auto-discovers a local build at
`../detector/build/Release/bdcudatag.dll` (or the Linux equivalent) and skips
the native tests if it is absent.

The jar is a JPMS module named `com.frc6324.bdcudatag`. It exports only
`com.frc6324.bdcudatag`; the FFM bindings live in
`com.frc6324.bdcudatag.internal.NativeLibrary` and are encapsulated, so the
public surface is the detector plus its configuration and result types.

```java
import com.frc6324.bdcudatag.BDTagDetector;
import com.frc6324.bdcudatag.TagDetection;
import com.frc6324.bdcudatag.TagFamily;

try (BDTagDetector detector = new BDTagDetector(TagFamily.Tag36h11, 1280, 720)) {
    for (TagDetection tag : detector.detect(frame)) { // byte[] grayscale
        System.out.printf("id=%d hamming=%d margin=%.1f center=(%.1f, %.1f)%n",
                tag.id(), tag.hamming(), tag.decisionMargin(),
                tag.centerX(), tag.centerY());
    }
}
```

There are two detection entry points, both returning `List<TagDetection>`:

- `detect(byte[] imageBytes)` copies a tightly packed frame into the detector. A
  buffer holding several frames works, since trailing bytes are ignored.
- `detect(MemorySegment imageData)` reads an off-heap frame in place. The row
  stride is inferred: exactly `width * height` bytes is tightly packed, while a
  larger segment whose size is a multiple of the height is read with stride
  `byteSize / height` (the usual padded-camera-buffer layout). Wrap raw
  addresses with `MemorySegment.ofAddress(ptr).reinterpret(len)` first.

`detectDevice(long cudaPointer, int stride)` remains available as the advanced
zero-copy path for frames that are already in device memory. Detectors are
synchronous and thread-safe: calls on one instance are serialized, so a single
detector may be shared between threads. `stats()` returns frame counts,
GPU/total timing, run/candidate/quad counts and ROI rows.

### Publishing to Maven Local

The module can be installed in the local Maven repository so sibling builds can
consume it as a normal dependency:

```bash
cd java-api
./gradlew publishToMavenLocal            # bundles ../detector/build/Release
./gradlew publishToMavenLocal -Pbdcudatag.native.dir=/path/to/build
```

That publishes `com.frc6324:bdcudatag:0.1.0` (binary jar with the bundled native
library, sources jar, javadoc jar, POM and Gradle module metadata) under
`~/.m2/repository`. In a consuming build:

```kotlin
repositories {
    mavenLocal()
}

dependencies {
    implementation("com.frc6324:bdcudatag:0.1.0")
}
```

The published jar contains the native library for the platform that built it,
so publish once per platform you target. The `--enable-native-access` flag from
above still applies to the consuming JVM.

## Camera test app

`viewer/` is a standalone Swing application for testing the detector against a
camera, video file or still image. It outlines each tag, labels its id, and
reports FPS, GPU time and ROI rows. See [`viewer/README.md`](viewer/README.md)
for requirements, options and platform notes.

```bash
cd viewer
./gradlew run
```

## C API

The ABI is in `detector/include/bdcudatag/bdcudatag.h`; the RAII C++ wrapper is
in `detector/include/bdcudatag/detector.hpp`.

```c
bd_config config;
bd_config_init(&config);
bd_detector* detector = bd_detector_create(BD_FAMILY_TAG36H11, 1280, 720,
                                           &config, NULL);
bd_detection detections[64];
int count = bd_detector_detect_host(detector, gray, 1280, detections, 64);
bd_detector_destroy(detector);
```

Corners are ordered top-left, top-right, bottom-right, bottom-left in the tag's
canonical orientation, matching AprilTag's convention. Each detection also
carries `homography`, a row-major 3x3 matrix mapping the tag's canonical square
to image pixels. With `(x, y)` in `[-1, 1]`, x right and y down, and the first
corner at `(-1, -1)`:

    s * (px, py, 1)^T = homography * (x, y, 1)^T

`homography[8]` is 1. It is derived from the refined corners, so it describes
only the tag plane; no intrinsics or pose are computed. In Java the same values
come from `TagDetection.homography()`, `projectX(x, y)` and `projectY(x, y)`.

Detectors are thread-safe: calls on one handle are serialized internally, so a
single detector may be shared between threads. Frames still run one at a time.

## Configuration

| Field                 | Default | Meaning                                                |
| --------------------- | ------- | ------------------------------------------------------ |
| `max_hamming`         | -1      | -1 uses family default (2 for tag36h11, 1 for tag16h5) |
| `min_cluster_pixels`  | 24      | reject smaller components                              |
| `min_tag_pixels`      | 8       | minimum tag bbox side                                  |
| `max_detections`      | 64      | hard cap, at most 256                                  |
| `max_candidates`      | 0       | 0 chooses an automatic cap from image size             |
| `refine_edges`        | true    | subpixel edge refinement                               |
| `use_cuda_graph`      | true    | capture/replay the pipeline                            |
| `min_decision_margin` | 3.0     | decode confidence floor                                |
| `decode_sharpening`   | 0.25    | cell sampling sharpening                               |

## Limitations

- Only `tag36h11` and `tag16h5`; one family per detector instance.
- No pose estimation or tag bundles; the homography is 2D only.
- The detector is fixed to the width and height passed to its constructor.
- Device-pointer detection requires the pointer to belong to the detector's CUDA
  context.
- Calls on one detector are serialized, so sharing it between threads does not
  increase throughput; use several detectors for parallel streams.

## Layout

```
detector/
  include/bdcudatag/   C and C++ public headers
  src/                 CUDA pipeline, host API, generated family tables
  tests/               synthetic end-to-end tests (no camera required)
  bench/               synthetic accuracy/latency benchmark, scene exporter and
                       cross-detector comparison harness
java-api/              Java FFM wrapper (JPMS module) and JUnit tests
viewer/                Swing camera test app (separate Gradle build)
```
