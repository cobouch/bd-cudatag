package com.frc6324.bdcudatag;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import java.util.List;
import java.util.Objects;

import com.frc6324.bdcudatag.internal.NativeLibrary;

/**
 * CUDA AprilTag-family detector backed by the native {@code bdcudatag} library.
 *
 * <p>A detector is fixed to one tag family and one image size, and owns all
 * native and device memory. Detection is synchronous. Instances are thread-safe:
 * calls are serialized on the detector, so one instance may be shared between
 * threads, at the cost of running one frame at a time. Close a detector to
 * release GPU memory.
 *
 * <p>The native library is loaded from the {@link #LIBRARY_PATH_PROPERTY} system
 * property when set, otherwise through {@link System#loadLibrary(String)} with
 * the name {@code bdcudatag}.
 *
 * <pre>{@code
 * try (BDTagDetector detector = new BDTagDetector(TagFamily.Tag36h11, 1280, 720)) {
 *     for (TagDetection tag : detector.detect(frame)) {
 *         System.out.printf("id=%d center=(%.1f, %.1f)%n", tag.id(), tag.centerX(), tag.centerY());
 *     }
 * }
 * }</pre>
 */
public final class BDTagDetector implements AutoCloseable {

    /** System property naming the full path to the native library. */
    public static final String LIBRARY_PATH_PROPERTY = NativeLibrary.PATH_PROPERTY;

    private final TagFamily family;
    private final int width;
    private final int height;
    private final int capacity;
    private final long pixelCount;
    private final Arena arena;
    private final MemorySegment input;
    private final MemorySegment output;
    private final MemorySegment statsBuffer;
    private MemorySegment handle;

    /**
     * Creates a detector with {@link DetectorConfig#DEFAULT}.
     *
     * @param family the single tag family to detect
     * @param width image width in pixels
     * @param height image height in pixels
     */
    public BDTagDetector(TagFamily family, int width, int height) {
        this(family, width, height, DetectorConfig.DEFAULT);
    }

    /**
     * Creates a detector.
     *
     * @param family the single tag family to detect
     * @param width image width in pixels
     * @param height image height in pixels
     * @param config tuning parameters
     * @throws DetectorException if the native detector cannot be created, for
     *     example when CUDA is unavailable
     */
    public BDTagDetector(TagFamily family, int width, int height, DetectorConfig config) {
        this.family = Objects.requireNonNull(family, "family");
        Objects.requireNonNull(config, "config");
        if (width < 16 || width > NativeLibrary.MAX_WIDTH) {
            throw new IllegalArgumentException(
                    "width must be in [16, " + NativeLibrary.MAX_WIDTH + "]");
        }
        if (height < 16 || height > NativeLibrary.MAX_HEIGHT) {
            throw new IllegalArgumentException(
                    "height must be in [16, " + NativeLibrary.MAX_HEIGHT + "]");
        }

        this.width = width;
        this.height = height;
        this.capacity = config.maxDetections();
        this.pixelCount = (long) width * height;
        this.arena = Arena.ofShared();
        this.input = arena.allocate(pixelCount, 64);
        this.output = arena.allocate((long) capacity * NativeLibrary.DETECTION_BYTES, 8);
        this.statsBuffer = arena.allocate(NativeLibrary.STATS_BYTES, 8);

        MemorySegment configSegment = arena.allocate(NativeLibrary.CONFIG_BYTES, 8);
        MemorySegment status = arena.allocate(JAVA_INT);
        try {
            NativeLibrary.writeConfig(configSegment, config, capacity);
            status.set(JAVA_INT, 0, 0);

            MemorySegment created =
                    NativeLibrary.create(family.nativeId(), width, height, configSegment, status);
            int createStatus = status.get(JAVA_INT, 0);
            if (created.equals(MemorySegment.NULL) || createStatus != 0) {
                throw new DetectorException(createStatus, NativeLibrary.lastError());
            }
            this.handle = created;
        } catch (Throwable error) {
            arena.close();
            throw error;
        }
    }

    /** The tag family this detector decodes. */
    public TagFamily family() {
        return family;
    }

    /** The image width this detector was built for. */
    public int width() {
        return width;
    }

    /** The image height this detector was built for. */
    public int height() {
        return height;
    }

    /** The maximum number of tags returned by one detection. */
    public int maxDetections() {
        return capacity;
    }

    /**
     * Detects tags in a tightly packed 8-bit grayscale image whose row stride
     * equals {@link #width()}.
     *
     * @param imageBytes at least {@code width * height} bytes; extra bytes are
     *     ignored, so a buffer holding several frames works as-is
     * @return the detected tags in native order, never {@code null}
     * @throws IllegalArgumentException if the array is too small
     * @throws IllegalStateException if the detector is closed
     * @throws DetectorException if the native call fails
     */
    public synchronized List<TagDetection> detect(byte[] imageBytes) {
        Objects.requireNonNull(imageBytes, "imageBytes");
        ensureOpen();
        if (imageBytes.length < pixelCount) {
            throw new IllegalArgumentException(
                    "imageBytes must contain at least " + pixelCount + " bytes, got "
                            + imageBytes.length);
        }
        input.copyFrom(MemorySegment.ofArray(imageBytes).asSlice(0, (int) pixelCount));
        return detectSegment(input, width, false);
    }

    /**
     * Detects tags in an off-heap 8-bit grayscale image, avoiding the copy into
     * the detector's own staging buffer.
     *
     * <p>The row stride is derived from the segment size. A segment of exactly
     * {@code width * height} bytes is read as tightly packed. A larger segment
     * whose size is a multiple of {@link #height()} is read as that many rows
     * with stride {@code byteSize / height}, which is how a padded camera buffer
     * is usually laid out. The segment must be sized, so wrap raw addresses with
     * {@link MemorySegment#reinterpret(long)}.
     *
     * @param imageData the image; it must stay alive and unmodified for the
     *     duration of the call
     * @return the detected tags in native order, never {@code null}
     * @throws IllegalArgumentException if the segment is unsized or too small
     * @throws IllegalStateException if the detector is closed
     * @throws DetectorException if the native call fails
     */
    public synchronized List<TagDetection> detect(MemorySegment imageData) {
        Objects.requireNonNull(imageData, "imageData");
        ensureOpen();

        final long size = imageData.byteSize();
        if (size == 0) {
            throw new IllegalArgumentException(
                    "imageData must be sized; use MemorySegment.ofAddress(ptr).reinterpret(len)");
        }
        if (size != Long.MAX_VALUE && size < pixelCount) {
            throw new IllegalArgumentException(
                    "imageData must contain at least " + pixelCount + " bytes, got " + size);
        }

        int stride = width;
        if (size != Long.MAX_VALUE && size != pixelCount && size % height == 0) {
            long inferred = size / height;
            if (inferred > width && inferred <= NativeLibrary.MAX_WIDTH) {
                stride = (int) inferred;
            }
        }
        long required = (long) (height - 1) * stride + width;
        if (size != Long.MAX_VALUE && size < required) {
            throw new IllegalArgumentException(
                    "imageData must contain at least " + required + " readable bytes, got " + size);
        }
        return detectSegment(imageData, stride, false);
    }

    /**
     * Detects tags in a CUDA device buffer without a host-to-device copy.
     *
     * <p>This is an advanced entry point for camera pipelines that already have
     * frames in device memory. The pointer must belong to the detector's CUDA
     * context and hold 8-bit grayscale pixels.
     *
     * @param deviceAddress a CUDA device pointer
     * @param stride bytes per row, at least {@link #width()}
     * @return the detected tags in native order, never {@code null}
     */
    public synchronized List<TagDetection> detectDevice(long deviceAddress, int stride) {
        ensureOpen();
        if (deviceAddress == 0) {
            throw new IllegalArgumentException("deviceAddress must be non-zero");
        }
        if (stride < width) {
            throw new IllegalArgumentException("stride must be >= width (" + width + ")");
        }
        long required = (long) (height - 1) * stride + width;
        MemorySegment device = MemorySegment.ofAddress(deviceAddress).reinterpret(required);
        return detectSegment(device, stride, true);
    }

    /** Counters from the most recent frame. */
    public synchronized DetectorStats stats() {
        ensureOpen();
        NativeLibrary.fetchStats(handle, statsBuffer);
        return NativeLibrary.decodeStats(statsBuffer);
    }

    @Override
    public synchronized void close() {
        if (handle != null) {
            MemorySegment toDestroy = handle;
            handle = null;
            try {
                NativeLibrary.destroy(toDestroy);
            } finally {
                arena.close();
            }
        }
    }

    /** Version string reported by the native library. */
    public static String nativeVersion() {
        return NativeLibrary.version();
    }

    private List<TagDetection> detectSegment(MemorySegment image, int stride, boolean device) {
        int count = device
                ? NativeLibrary.detectDevice(handle, image, stride, output, capacity)
                : NativeLibrary.detectHost(handle, image, stride, output, capacity);
        if (count < 0) {
            throw new DetectorException(count, NativeLibrary.lastError());
        }
        return List.of(NativeLibrary.decodeDetections(output, Math.min(count, capacity)));
    }

    private void ensureOpen() {
        if (handle == null) {
            throw new IllegalStateException("detector is closed");
        }
    }
}
