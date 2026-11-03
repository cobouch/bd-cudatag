package com.frc6324.bdcudatag;

/**
 * Detector tuning parameters. The defaults match the native C API and are a
 * good starting point for a 720p or 1080p camera.
 *
 * @param maxHamming maximum accepted Hamming distance, or {@code -1} for the
 *                   family default (2 for tag36h11, 1 for tag16h5)
 * @param minClusterPixels reject connected components smaller than this
 * @param minTagPixels minimum tag bounding-box side in pixels
 * @param maxDetections hard cap on returned detections, at most 256
 * @param maxCandidates candidate cap, or {@code 0} for an automatic value
 * @param refineEdges subpixel edge refinement (small cost, better corners)
 * @param useCudaGraph capture the pipeline as a CUDA graph
 * @param minDecisionMargin minimum decode decision margin
 * @param decodeSharpening sharpening applied while sampling tag cells
 */
public record DetectorConfig(
        int maxHamming,
        int minClusterPixels,
        int minTagPixels,
        int maxDetections,
        int maxCandidates,
        boolean refineEdges,
        boolean useCudaGraph,
        float minDecisionMargin,
        float decodeSharpening) {

    public static final DetectorConfig DEFAULT =
            new DetectorConfig(-1, 24, 8, 64, 0, true, true, 3.0f, 0.25f);

    public DetectorConfig {
        if (maxHamming < -1) {
            throw new IllegalArgumentException("maxHamming must be >= -1");
        }
        if (minClusterPixels < 1) {
            throw new IllegalArgumentException("minClusterPixels must be >= 1");
        }
        if (minTagPixels < 4) {
            throw new IllegalArgumentException("minTagPixels must be >= 4");
        }
        if (maxDetections < 1 || maxDetections > 256) {
            throw new IllegalArgumentException("maxDetections must be in [1, 256]");
        }
        if (maxCandidates < 0) {
            throw new IllegalArgumentException("maxCandidates must be >= 0");
        }
        if (!Float.isFinite(minDecisionMargin) || minDecisionMargin < 0.0f) {
            throw new IllegalArgumentException("minDecisionMargin must be finite and >= 0");
        }
        if (!Float.isFinite(decodeSharpening) || decodeSharpening < 0.0f) {
            throw new IllegalArgumentException("decodeSharpening must be finite and >= 0");
        }
    }
}
