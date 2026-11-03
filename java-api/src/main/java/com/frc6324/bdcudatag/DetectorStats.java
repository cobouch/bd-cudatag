package com.frc6324.bdcudatag;

/**
 * Counters from the most recent frame.
 *
 * @param frames total detect calls
 * @param lastGpuMs GPU time of the last frame
 * @param averageGpuMs average GPU time since construction
 * @param lastTotalMs wall time of the last detect call
 * @param runs black runs found in the active region of interest
 * @param candidates connected components that passed the tag-size filter
 * @param detections tags successfully decoded
 * @param quads candidate quads that were fitted
 * @param roiRows image rows searched after the coarse ROI pass
 */
public record DetectorStats(
        long frames,
        float lastGpuMs,
        float averageGpuMs,
        float lastTotalMs,
        int runs,
        int candidates,
        int detections,
        int quads,
        int roiRows) {}
