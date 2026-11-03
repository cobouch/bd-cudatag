package com.frc6324.bdcudatag.internal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import com.frc6324.bdcudatag.DetectorConfig;
import com.frc6324.bdcudatag.DetectorStats;
import com.frc6324.bdcudatag.TagDetection;
import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemoryLayout.PathElement;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Locale;

/**
 * Raw Java FFM bindings for the native {@code bdcudatag} library.
 *
 * <p>This class is the only place that knows the C ABI: struct layouts, symbol
 * names, method handles and the library-loading policy. It is public so the
 * rest of the library can use it, but its package is not exported by the
 * {@code com.frc6324.bdcudatag} module, so it is not part of the consumer API.
 *
 * <p>All methods are thin, synchronous wrappers around the C entry points. They
 * perform no argument validation beyond what the ABI requires; callers are
 * expected to validate sizes and state.
 */
public final class NativeLibrary {

    /** System property naming the full path to the native library. */
    public static final String PATH_PROPERTY = "bdcudatag.library.path";

    /** Name passed to {@link System#loadLibrary(String)} when no path is set. */
    public static final String LIBRARY_NAME = "bdcudatag";

    /** Classpath directory holding the platform-specific native libraries. */
    public static final String RESOURCE_ROOT = "com/frc6324/bdcudatag/internal/native/";

    public static final int CONFIG_BYTES = 40;
    public static final int DETECTION_BYTES = 84;
    public static final int STATS_BYTES = 40;
    public static final int MAX_WIDTH = 8192;
    public static final int MAX_HEIGHT = 16384;

    /** Mirrors {@code bd_config}. */
    public static final StructLayout CONFIG_LAYOUT = MemoryLayout.structLayout(
            JAVA_INT.withName("struct_size"),
            JAVA_INT.withName("max_hamming"),
            JAVA_INT.withName("min_cluster_pixels"),
            JAVA_INT.withName("min_tag_pixels"),
            JAVA_INT.withName("max_detections"),
            JAVA_INT.withName("max_candidates"),
            JAVA_INT.withName("refine_edges"),
            JAVA_INT.withName("use_cuda_graph"),
            JAVA_FLOAT.withName("min_decision_margin"),
            JAVA_FLOAT.withName("decode_sharpening"));

    /** Mirrors {@code bd_detection}. */
    public static final StructLayout DETECTION_LAYOUT = MemoryLayout.structLayout(
            JAVA_INT.withName("id"),
            JAVA_INT.withName("hamming"),
            JAVA_FLOAT.withName("decision_margin"),
            MemoryLayout.sequenceLayout(8, JAVA_FLOAT).withName("corners"),
            JAVA_INT.withName("reserved"),
            MemoryLayout.sequenceLayout(9, JAVA_FLOAT).withName("homography"));

    /** Mirrors {@code bd_stats}. */
    public static final StructLayout STATS_LAYOUT = MemoryLayout.structLayout(
            JAVA_LONG.withName("frames"),
            JAVA_FLOAT.withName("last_gpu_ms"),
            JAVA_FLOAT.withName("average_gpu_ms"),
            JAVA_FLOAT.withName("last_total_ms"),
            JAVA_INT.withName("runs"),
            JAVA_INT.withName("candidates"),
            JAVA_INT.withName("detections"),
            JAVA_INT.withName("quads"),
            JAVA_INT.withName("roi_rows"));

    private static final long DETECTION_ID = offset(DETECTION_LAYOUT, "id");
    private static final long DETECTION_HAMMING = offset(DETECTION_LAYOUT, "hamming");
    private static final long DETECTION_MARGIN = offset(DETECTION_LAYOUT, "decision_margin");
    private static final long DETECTION_CORNERS = offset(DETECTION_LAYOUT, "corners");
    private static final long DETECTION_HOMOGRAPHY = offset(DETECTION_LAYOUT, "homography");

    private static final MethodHandle CONFIG_INIT;
    private static final MethodHandle CREATE;
    private static final MethodHandle DESTROY;
    private static final MethodHandle DETECT_HOST;
    private static final MethodHandle DETECT_DEVICE;
    private static final MethodHandle GET_STATS;
    private static final MethodHandle LAST_ERROR;
    private static final MethodHandle VERSION;

    static {
        loadNativeLibrary();
        SymbolLookup lookup = SymbolLookup.loaderLookup();

        CONFIG_INIT = downcall(lookup, "bd_config_init", FunctionDescriptor.ofVoid(ADDRESS));
        CREATE = downcall(
                lookup,
                "bd_detector_create",
                FunctionDescriptor.of(ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS));
        DESTROY =
                downcall(lookup, "bd_detector_destroy", FunctionDescriptor.ofVoid(ADDRESS));
        DETECT_HOST = downcall(
                lookup,
                "bd_detector_detect_host",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT));
        DETECT_DEVICE = downcall(
                lookup,
                "bd_detector_detect_device",
                FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT));
        GET_STATS = downcall(
                lookup, "bd_detector_get_stats", FunctionDescriptor.ofVoid(ADDRESS, ADDRESS));
        LAST_ERROR = downcall(lookup, "bd_last_error", FunctionDescriptor.of(ADDRESS));
        VERSION = downcall(lookup, "bd_version_string", FunctionDescriptor.of(ADDRESS));

        if (CONFIG_LAYOUT.byteSize() != CONFIG_BYTES
                || DETECTION_LAYOUT.byteSize() != DETECTION_BYTES
                || STATS_LAYOUT.byteSize() != STATS_BYTES) {
            throw new ExceptionInInitializerError("bdcudatag ABI layout mismatch");
        }
    }

    private NativeLibrary() {}

    private static void loadNativeLibrary() {
        String explicitPath = System.getProperty(PATH_PROPERTY);
        if (explicitPath != null && !explicitPath.isBlank()) {
            System.load(Path.of(explicitPath).toAbsolutePath().toString());
            return;
        }

        Path bundled = extractBundledLibrary();
        if (bundled != null) {
            System.load(bundled.toAbsolutePath().toString());
            return;
        }

        try {
            System.loadLibrary(LIBRARY_NAME);
        } catch (UnsatisfiedLinkError error) {
            UnsatisfiedLinkError wrapped = new UnsatisfiedLinkError(
                    "Unable to load the "
                            + LIBRARY_NAME
                            + " native library. It is not bundled for "
                            + platformId()
                            + ", was not found on java.library.path, and -D"
                            + PATH_PROPERTY
                            + " is not set. Rebuild the jar with detector/ built, or pass -D"
                            + PATH_PROPERTY
                            + "=/path/to/"
                            + System.mapLibraryName(LIBRARY_NAME)
                            + ". Cause: "
                            + error.getMessage());
            wrapped.initCause(error);
            throw wrapped;
        }
    }

    /** Platform tag used in the bundled resource path, for example {@code linux-aarch64}. */
    public static String platformId() {
        String osName = System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
        String os;
        if (osName.contains("win")) {
            os = "windows";
        } else if (osName.contains("mac") || osName.contains("darwin")) {
            os = "macos";
        } else {
            os = "linux";
        }

        String rawArch = System.getProperty("os.arch", "").toLowerCase(Locale.ROOT);
        String arch = switch (rawArch) {
            case "amd64", "x86_64", "x64" -> "x86_64";
            case "aarch64", "arm64" -> "aarch64";
            default -> rawArch.replaceAll("[^a-z0-9_]", "");
        };
        return os + "-" + arch;
    }

    /**
     * Extracts the native library bundled in the jar to a temporary file.
     *
     * @return the extracted file, or {@code null} when this platform is not
     *     bundled
     */
    private static Path extractBundledLibrary() {
        String libraryName = System.mapLibraryName(LIBRARY_NAME);
        String resource = "/" + RESOURCE_ROOT + platformId() + "/" + libraryName;
        try (InputStream stream = NativeLibrary.class.getResourceAsStream(resource)) {
            if (stream == null) {
                return null;
            }
            Path directory = Files.createTempDirectory("bdcudatag-");
            directory.toFile().deleteOnExit();
            Path target = directory.resolve(libraryName);
            Files.copy(stream, target, StandardCopyOption.REPLACE_EXISTING);
            target.toFile().deleteOnExit();
            return target;
        } catch (IOException error) {
            throw new IllegalStateException(
                    "Unable to extract the bundled " + LIBRARY_NAME + " native library", error);
        }
    }

    /** Fills a native {@code bd_config} from the high-level configuration. */
    public static void writeConfig(
            MemorySegment config, DetectorConfig values, int maxDetections) {
        invokeVoid(CONFIG_INIT, config);
        config.set(JAVA_INT, offset(CONFIG_LAYOUT, "max_hamming"), values.maxHamming());
        config.set(JAVA_INT, offset(CONFIG_LAYOUT, "min_cluster_pixels"), values.minClusterPixels());
        config.set(JAVA_INT, offset(CONFIG_LAYOUT, "min_tag_pixels"), values.minTagPixels());
        config.set(JAVA_INT, offset(CONFIG_LAYOUT, "max_detections"), maxDetections);
        config.set(JAVA_INT, offset(CONFIG_LAYOUT, "max_candidates"), values.maxCandidates());
        config.set(JAVA_INT, offset(CONFIG_LAYOUT, "refine_edges"), values.refineEdges() ? 1 : 0);
        config.set(
                JAVA_INT, offset(CONFIG_LAYOUT, "use_cuda_graph"), values.useCudaGraph() ? 1 : 0);
        config.set(
                JAVA_FLOAT, offset(CONFIG_LAYOUT, "min_decision_margin"), values.minDecisionMargin());
        config.set(
                JAVA_FLOAT, offset(CONFIG_LAYOUT, "decode_sharpening"), values.decodeSharpening());
    }

    /**
     * Calls {@code bd_detector_create}. Returns {@link MemorySegment#NULL} on
     * failure and writes the status to {@code status}.
     */
    public static MemorySegment create(
            int family, int width, int height, MemorySegment config, MemorySegment status) {
        try {
            return (MemorySegment) CREATE.invokeExact(family, width, height, config, status);
        } catch (Throwable error) {
            throw new IllegalStateException("bd_detector_create failed", error);
        }
    }

    public static void destroy(MemorySegment detector) {
        invokeVoid(DESTROY, detector);
    }

    public static int detectHost(
            MemorySegment detector,
            MemorySegment grayscale,
            int stride,
            MemorySegment output,
            int capacity) {
        return detect(DETECT_HOST, detector, grayscale, stride, output, capacity);
    }

    public static int detectDevice(
            MemorySegment detector,
            MemorySegment grayscale,
            int stride,
            MemorySegment output,
            int capacity) {
        return detect(DETECT_DEVICE, detector, grayscale, stride, output, capacity);
    }

    /** Calls {@code bd_detector_get_stats}, writing a {@code bd_stats}. */
    public static void fetchStats(MemorySegment detector, MemorySegment stats) {
        invokeVoid(GET_STATS, detector, stats);
    }

    /** Converts a populated {@code bd_stats} buffer into a record. */
    public static DetectorStats decodeStats(MemorySegment stats) {
        return new DetectorStats(
                stats.get(JAVA_LONG, offset(STATS_LAYOUT, "frames")),
                stats.get(JAVA_FLOAT, offset(STATS_LAYOUT, "last_gpu_ms")),
                stats.get(JAVA_FLOAT, offset(STATS_LAYOUT, "average_gpu_ms")),
                stats.get(JAVA_FLOAT, offset(STATS_LAYOUT, "last_total_ms")),
                stats.get(JAVA_INT, offset(STATS_LAYOUT, "runs")),
                stats.get(JAVA_INT, offset(STATS_LAYOUT, "candidates")),
                stats.get(JAVA_INT, offset(STATS_LAYOUT, "detections")),
                stats.get(JAVA_INT, offset(STATS_LAYOUT, "quads")),
                stats.get(JAVA_INT, offset(STATS_LAYOUT, "roi_rows")));
    }

    /** Copies {@code count} {@code bd_detection} entries out of a native buffer. */
    public static TagDetection[] decodeDetections(MemorySegment buffer, int count) {
        TagDetection[] detections = new TagDetection[count];
        float[] corners = new float[8];
        float[] homography = new float[9];
        for (int i = 0; i < count; ++i) {
            long base = (long) i * DETECTION_BYTES;
            for (int corner = 0; corner < 8; ++corner) {
                corners[corner] = buffer.get(JAVA_FLOAT, base + DETECTION_CORNERS + corner * 4L);
            }
            for (int cell = 0; cell < 9; ++cell) {
                homography[cell] =
                        buffer.get(JAVA_FLOAT, base + DETECTION_HOMOGRAPHY + cell * 4L);
            }
            detections[i] = new TagDetection(
                    buffer.get(JAVA_INT, base + DETECTION_ID),
                    buffer.get(JAVA_INT, base + DETECTION_HAMMING),
                    buffer.get(JAVA_FLOAT, base + DETECTION_MARGIN),
                    corners,
                    homography);
        }
        return detections;
    }

    /** Message for the most recent failure on this thread. */
    public static String lastError() {
        MemorySegment text = callAddress(LAST_ERROR);
        return text.equals(MemorySegment.NULL) ? "" : text.reinterpret(Long.MAX_VALUE).getString(0);
    }

    /** Native library version string. */
    public static String version() {
        MemorySegment text = callAddress(VERSION);
        return text.equals(MemorySegment.NULL) ? "" : text.reinterpret(Long.MAX_VALUE).getString(0);
    }

    private static long offset(StructLayout layout, String field) {
        return layout.byteOffset(PathElement.groupElement(field));
    }

    private static int detect(
            MethodHandle handle,
            MemorySegment detector,
            MemorySegment grayscale,
            int stride,
            MemorySegment output,
            int capacity) {
        try {
            return (int) handle.invokeExact(detector, grayscale, stride, output, capacity);
        } catch (Throwable error) {
            throw new IllegalStateException("bd_detector_detect failed", error);
        }
    }

    private static MethodHandle downcall(
            SymbolLookup lookup, String name, FunctionDescriptor descriptor) {
        MemorySegment symbol = lookup.find(name)
                .orElseGet(() -> Linker.nativeLinker()
                        .defaultLookup()
                        .find(name)
                        .orElseThrow(() -> new UnsatisfiedLinkError(
                                "the " + LIBRARY_NAME + " library is missing symbol " + name)));
        return Linker.nativeLinker().downcallHandle(symbol, descriptor);
    }

    private static void invokeVoid(MethodHandle handle, MemorySegment... arguments) {
        try {
            switch (arguments.length) {
                case 1 -> handle.invokeExact(arguments[0]);
                case 2 -> handle.invokeExact(arguments[0], arguments[1]);
                default -> throw new IllegalArgumentException("unsupported arity");
            }
        } catch (Throwable error) {
            throw new IllegalStateException("native bdcudatag call failed", error);
        }
    }

    private static MemorySegment callAddress(MethodHandle handle) {
        try {
            return (MemorySegment) handle.invokeExact();
        } catch (Throwable error) {
            throw new IllegalStateException("native bdcudatag call failed", error);
        }
    }
}
