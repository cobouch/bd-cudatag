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

1. **Coarse ROI.** The image is reduced to 8x8 tiles. A tile is active when its
   min/max range exceeds a contrast threshold. Active tile rows are closed
   vertically and dilated horizontally, producing the only rows that are
   binarized, run-length encoded and searched. Flat background rows cost almost
   nothing, and the number of searched rows is reported as `roiRows`.
2. **Binarization.** A per-tile adaptive threshold is applied only inside the
   ROI; inactive rows are filled as background.
3. **Run-length connected components.** Black runs are extracted with one warp
   per 32-pixel chunk, joined at chunk boundaries, and merged across rows with a
   lock-free union-find.
4. **Candidate filtering.** Components are filtered by area, bounding-box size,
   border contact, aspect ratio and fill.
5. **Quad fitting.** The outer contour is traced with Moore-neighbour tracing,
   simplified to four dominant corners, and (optionally) refined against image
   gradients. Contour points are stored as packed 16-bit coordinates.
6. **Decoding.** The quad is sampled with a homography for all four rotations,
   matched against the family code table by popcount, and scored by decision
   margin. No decoding or corner work happens for frames with no candidates.

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
./detector/build/bdcudatag_bench 1280 720 600
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

## Results

Measured locally on an RTX 3060 (SM 8.6, desktop Linux/Windows driver, 1980 MHz
SM clock), 8 synthetic tags at 1280x720:

| Image     | Tags | Throughput | Latency p50 | ROI rows  |
| --------- | ---- | ---------- | ----------- | --------- |
| 640x480   | 5    | ~360 FPS   | 2.7 ms      | -         |
| 1280x720  | 8    | ~380 FPS   | 2.55 ms     | 552 / 720 |
| 1920x1080 | 8    | ~395 FPS   | 2.50 ms     | -         |

The 720p and 1080p numbers are nearly identical because the synchronous `detect`
call on Windows/WDDM has a fixed submission/synchronization floor of roughly 2.5
ms; the marginal GPU work for 1080p is small. On Jetson (Linux, Tegra driver)
this floor is much lower, so measure on-device before quoting FPS. The ROI still
pays off: only the rows containing contrasting content are searched, and a blank
frame has zero candidates.

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
  bench/               synthetic throughput benchmark
java-api/              Java FFM wrapper (JPMS module) and JUnit tests
viewer/                Swing camera test app (separate Gradle build)
```
