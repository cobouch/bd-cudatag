package com.frc6324.bdcudatag;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * End-to-end FFM smoke test. It is skipped when the native library has not been
 * built next to the Java project, so a plain {@code gradlew test} still works
 * on a machine without CUDA.
 */
class BDTagDetectorTest {

    private static final Path LIBRARY = locateLibrary();

    static {
        if (LIBRARY != null) {
            System.setProperty(BDTagDetector.LIBRARY_PATH_PROPERTY, LIBRARY.toString());
        }
    }

    private static Path locateLibrary() {
        String configured = System.getProperty(BDTagDetector.LIBRARY_PATH_PROPERTY);
        String mapped = System.mapLibraryName("bdcudatag");
        Path[] candidates = {
            configured == null ? null : Path.of(configured),
            Path.of("..", "detector", "build", "Release", mapped),
            Path.of("..", "detector", "build", mapped),
            Path.of("detector", "build", "Release", mapped)
        };
        for (Path candidate : candidates) {
            if (candidate != null && Files.isRegularFile(candidate)) {
                return candidate.toAbsolutePath();
            }
        }
        return null;
    }

    @BeforeAll
    static void requireNativeLibrary() {
        assumeTrue(LIBRARY != null, "native bdcudatag library not built");
    }

    @Test
    void blankFrameProducesNoDetections() {
        byte[] image = new byte[640 * 480];
        Arrays.fill(image, (byte) 255);
        try (BDTagDetector detector = new BDTagDetector(TagFamily.Tag36h11, 640, 480)) {
            assertEquals(0, detector.detect(image).size());
            DetectorStats stats = detector.stats();
            assertEquals(1, stats.frames());
            assertEquals(0, stats.detections());
            assertEquals(0, stats.roiRows());
        }
    }

    @Test
    void paddedSegmentIsReadWithInferredStride() {
        int width = 64;
        int height = 64;
        int stride = 128;
        try (Arena arena = Arena.ofConfined();
                BDTagDetector detector = new BDTagDetector(TagFamily.Tag16h5, width, height)) {
            MemorySegment padded = arena.allocate((long) stride * height, 64);
            padded.fill((byte) 255);
            List<TagDetection> detections = detector.detect(padded);
            assertEquals(0, detections.size());
            assertEquals(1, detector.stats().frames());
        }
    }

    @Test
    void unsizedSegmentIsRejected() {
        try (BDTagDetector detector = new BDTagDetector(TagFamily.Tag36h11, 64, 64)) {
            MemorySegment unsized = MemorySegment.ofAddress(0x1000L);
            assertThrows(IllegalArgumentException.class, () -> detector.detect(unsized));
        }
    }

    @Test
    void rejectsTooSmallImages() {
        byte[] image = new byte[10];
        try (BDTagDetector detector = new BDTagDetector(TagFamily.Tag16h5, 64, 64)) {
            assertThrows(IllegalArgumentException.class, () -> detector.detect(image));
        }
    }

    @Test
    void detectorCanBeSharedAcrossThreads() throws Exception {
        byte[] image = new byte[320 * 240];
        Arrays.fill(image, (byte) 255);
        try (BDTagDetector detector = new BDTagDetector(TagFamily.Tag36h11, 320, 240)) {
            ExecutorService pool = Executors.newFixedThreadPool(4);
            try {
                List<Future<Integer>> running = new ArrayList<>();
                for (int i = 0; i < 16; ++i) {
                    running.add(pool.submit(() -> detector.detect(image).size()));
                }
                for (Future<Integer> task : running) {
                    assertEquals(0, task.get(30, TimeUnit.SECONDS));
                }
            } finally {
                pool.shutdownNow();
            }
            assertEquals(16, detector.stats().frames());
        }
    }

    @Test
    void closeIsIdempotent() {
        BDTagDetector detector = new BDTagDetector(TagFamily.Tag36h11, 64, 64);
        detector.close();
        detector.close();
        assertThrows(IllegalStateException.class, detector::stats);
    }
}
