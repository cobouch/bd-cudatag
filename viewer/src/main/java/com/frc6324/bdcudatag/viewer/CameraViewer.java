package com.frc6324.bdcudatag.viewer;

import com.frc6324.bdcudatag.BDTagDetector;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;
import javax.imageio.ImageIO;
import nu.pattern.OpenCV;
import org.opencv.core.Mat;

/**
 * Live camera tester for the bdcudatag detector.
 *
 * <p>Reads frames from a camera, video file or still image, detects tags with
 * {@link BDTagDetector}, and draws each tag's outline and id onto the frame. A
 * Swing window shows the result unless {@code --headless} is given, in which
 * case a summary is printed per frame.
 */
public final class CameraViewer {

    private static final AtomicBoolean RUNNING = new AtomicBoolean(true);

    private CameraViewer() {}

    public static void main(String[] args) {
        final Options options;
        try {
            options = Options.parse(args);
        } catch (IllegalArgumentException error) {
            System.err.println("error: " + error.getMessage());
            System.err.println();
            System.err.print(Options.USAGE);
            System.exit(2);
            return;
        }
        if (options.help()) {
            System.out.print(Options.USAGE);
            return;
        }

        try {
            OpenCV.loadLocally();
            System.out.println("native: " + BDTagDetector.nativeVersion());
            try (FrameSource source = FrameSource.open(options)) {
                System.out.println("source: " + source.description());
                ViewerWindow window =
                        options.headless()
                                ? null
                                : new ViewerWindow("bdcudatag viewer | Q quit, S snapshot");
                try {
                    if (window != null) {
                        ViewerWindow gui = window;
                        gui.setOnQuit(
                                () -> {
                                    RUNNING.set(false);
                                    gui.close();
                                });
                        gui.setOnSnapshot(() -> saveSnapshot(gui));
                        gui.show();
                    }
                    run(options, source, window);
                } finally {
                    if (window != null) {
                        window.close();
                    }
                }
            }
        } catch (Throwable error) {
            System.err.println("error: " + error.getMessage());
            System.exit(1);
        }
    }

    private static void run(Options options, FrameSource source, ViewerWindow window)
            throws Exception {
        try (FrameProcessor processor = new FrameProcessor(options.family())) {
            Mat frame = new Mat();
            try {
                int frames = 0;
                long detections = 0;
                double fps = 0.0;
                long previous = 0L;
                long start = System.nanoTime();
                String previousIds = "";
                int consecutiveFailures = 0;

                while (RUNNING.get()
                        && (options.maxFrames() == 0 || frames < options.maxFrames())) {
                    if (!source.read(frame)) {
                        if (!source.live()) {
                            break;
                        }
                        if (++consecutiveFailures > 100) {
                            throw new IOException("the camera stopped delivering frames");
                        }
                        Thread.sleep(30L);
                        continue;
                    }
                    consecutiveFailures = 0;

                    long now = System.nanoTime();
                    if (previous != 0L) {
                        double instant = 1.0e9 / Math.max(1L, now - previous);
                        fps = fps == 0.0 ? instant : fps * 0.9 + instant * 0.1;
                    }
                    previous = now;

                    FrameProcessor.Result result = processor.process(frame);
                    ++frames;
                    detections += result.tags().size();

                    if (window != null) {
                        window.update(result.image(), status(options, result, fps));
                    } else {
                        String ids =
                                result.tags().stream()
                                        .map(tag -> "#" + tag.id())
                                        .sorted()
                                        .collect(Collectors.joining(" "));
                        if (frames == 1 || frames % 30 == 0 || !ids.equals(previousIds)) {
                            System.out.printf(
                                    Locale.ROOT,
                                    "frame %d: %s, %d tag%s%s%n",
                                    frames,
                                    fps > 0.0
                                            ? String.format(Locale.ROOT, "%.1f fps", fps)
                                            : "measuring rate",
                                    result.tags().size(),
                                    result.tags().size() == 1 ? "" : "s",
                                    ids.isEmpty() ? "" : " [" + ids + "]");
                        }
                        previousIds = ids;
                    }
                }

                double elapsed = (System.nanoTime() - start) / 1.0e9;
                System.out.printf(
                        Locale.ROOT,
                        "processed %d frame%s in %.2f s (%.1f fps), %d tag%s%n",
                        frames,
                        frames == 1 ? "" : "s",
                        elapsed,
                        elapsed > 0.0 ? frames / elapsed : 0.0,
                        detections,
                        detections == 1 ? "" : "s");
            } finally {
                frame.release();
            }
        }
    }

    private static String status(Options options, FrameProcessor.Result result, double fps) {
        int tags = result.tags().size();
        return String.format(
                Locale.ROOT,
                "%s  |  %d x %d  |  %.1f fps  |  %d tag%s  |  gpu %.2f ms  |  roi %d/%d rows",
                options.family().tagName(),
                result.width(),
                result.height(),
                fps,
                tags,
                tags == 1 ? "" : "s",
                result.stats().lastGpuMs(),
                result.stats().roiRows(),
                result.height());
    }

    private static void saveSnapshot(ViewerWindow window) {
        BufferedImage image = window.latestImage();
        if (image == null) {
            return;
        }
        Path path = Path.of("snapshot-" + System.currentTimeMillis() + ".png");
        try {
            ImageIO.write(image, "png", path.toFile());
            System.out.println("saved " + path.toAbsolutePath());
        } catch (IOException error) {
            System.err.println("could not save snapshot: " + error.getMessage());
        }
    }
}
