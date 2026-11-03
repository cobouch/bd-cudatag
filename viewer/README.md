# bdcudatag camera viewer

A small desktop application for exercising the CUDA detector with a real
camera. It captures frames with OpenCV, runs them through the Java API in
`../java-api/`, and draws the result over the live image:

- a green quad around every decoded tag,
- yellow dots on its four detected corners,
- a `#id` label at the tag centre,
- a status bar with the tag family, frame size, measured FPS, tag count, last
  GPU time and the number of image rows the ROI pass searched.

`Q` or `Esc` quits, `S` writes `snapshot-<timestamp>.png` next to the app.

## Requirements

- A CUDA-capable GPU and a built detector (see the [root README](../README.md)).
  The viewer never talks to CUDA directly; it uses the Java API, whose jar
  bundles the native library at build time.
- Java 25. Gradle provisions the toolchain if it is missing.
- A camera, video file or still image. OpenCV's platform capture backend is
  used (typically Media Foundation on Windows, V4L2 on Linux).

This is a separate Gradle build that includes `../java-api` as a composite
build, so it compiles against the local API and does not need
`publishToMavenLocal`.

## Build

```bash
cd viewer
./gradlew build
```

`./gradlew run` builds the library, bundles the native detector from
`../detector/build/Release`, and starts the app.

## Usage

```text
Usage: camera-viewer [options]

Input (at most one):
  --device N        camera index (default 0)
  --video PATH      read frames from a video file or capture pipeline
  --image PATH      run a single still image

Detection:
  --family NAME     tag36h11 (default) or tag16h5
  --width W         requested camera width (default 1280)
  --height H        requested camera height (default 720)

Runtime:
  --headless        no window; print detections to stdout
  --frames N        stop after N frames (default 0 = until EOF or quit)
  --help, -h        show this message
```

Examples:

```bash
./gradlew run                                             # first camera
./gradlew run --args="--device 2 --width 1920 --height 1080"
./gradlew run --args="--family tag16h5"
./gradlew run --args="--video /path/to/recording.mp4"
./gradlew run --args="--image /path/to/frame.png"
```

`--width` and `--height` are requests: the camera picks the closest supported
mode, and the detector is created for the size that actually arrives. If the
frame size changes during a run, the detector is recreated.

`--video` accepts anything OpenCV can open as a media source. `--image` accepts
any format OpenCV can decode and produces exactly one frame, which is the
simplest way to verify the detection pipeline without a camera.

## Headless use

`--headless` skips Swing and prints a line whenever the set of visible ids
changes, plus every 30th frame. This is useful over SSH or on a Jetson without a
desktop:

```bash
./gradlew run --args="--headless --frames 120"
./gradlew run --args="--headless --image frame.png"
```

A still image produces one frame and then exits. On a live camera, `--frames 1`
grabs a single frame. The process always prints a final summary with the total
frame count and average rate.

## Native library resolution

The Java API jar contains the native library for the platform that built it,
copied from `../detector/build/Release` (or `../detector/build`). If the
detector has not been built, loading it fails at startup; build it first, or
point the loader at an existing library:

```bash
./gradlew run -Dbdcudatag.library.path=/path/to/libbdcudatag.so
```

`run` forwards that system property to the application JVM and sets
`--enable-native-access=ALL-UNNAMED` for the FFM calls. `installDist` creates a
standalone tree whose start scripts pass the same flag:

```bash
./gradlew installDist
./build/install/bdcudatag-viewer/bin/bdcudatag-viewer --device 0
```

## Platform notes

- **Windows**: OpenCV opens the camera through its Windows backend. A camera
  already open in another application will fail to open; close other viewers
  first.
- **Jetson / Linux**: `/dev/videoN` devices are opened by index through V4L2.
  Membership in the `video` group is usually required. Build the native library
  on the Jetson itself so the jar contains the `linux-aarch64` copy.
- The `org.openpnp:opencv` dependency ships Windows x86_64 and Linux
  x86_64/ARMv8 natives, so the same app runs on a desktop and on a Jetson
  without installing OpenCV separately.
- Building the viewer does not require a camera; only `run` does.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `could not open camera N` | Try another `--device` index, close other camera apps, and check device permissions. Check the rest of the pipeline with `--image` or `--video`. |
| `no bdcudatag library for <platform>` | Build `../detector`, rerun, or pass `-Dbdcudatag.library.path=...`. |
| OpenCV fails to load its native library | The artifact covers Windows x86_64, Linux x86_64 and Linux ARMv8. Use a supported platform or provide a matching OpenCV build. |
| Image shows but no tags | Confirm `--family` matches the printed tags and that tags are at least 8 px across (`min_tag_pixels` default). |
| Low displayed FPS | The rate includes camera capture and Swing repaint. Compare it with the `gpu` time in the status bar to see where the time goes. |

## How it is put together

| Class | Role |
| --- | --- |
| `CameraViewer` | Argument handling, capture loop, FPS measurement, headless output |
| `FrameSource` | Opens a camera, video or still image and yields BGR `Mat` frames |
| `FrameProcessor` | Converts to 8-bit gray, owns the `BDTagDetector`, renders the overlay |
| `Overlay` | Draws quads, corner dots and `#id` labels with Java2D |
| `ViewerWindow` | Swing window, status bar, key bindings, scaled image display |

Capture and detection run on the main thread; Swing repaints on the EDT and
receives only finished `BufferedImage` frames. The detector API is thread-safe,
so that split is a convenience rather than a requirement.
