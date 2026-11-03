package com.frc6324.bdcudatag;

/**
 * One detected tag.
 *
 * <p>Corners are ordered top-left, top-right, bottom-right and bottom-left in
 * the tag's canonical orientation, in image pixel coordinates.
 *
 * <p>The homography maps the tag's canonical square to the image using
 * AprilTag's tag frame: {@code (x, y)} spans {@code [-1, 1]} on both axes, x
 * right and y down, so the first corner is at {@code (-1, -1)} and
 *
 * <pre>{@code
 * s * (px, py, 1)^T = homography * (x, y, 1)^T
 * }</pre>
 *
 * <p>The matrix is row-major and normalised so {@code homography[8] == 1}. It is
 * derived from the refined corners; no camera intrinsics or pose are involved.
 * {@link #projectX(float, float)} and {@link #projectY(float, float)} apply it.
 *
 * @param id the tag id within its family
 * @param hamming Hamming distance between the sampled code and the matched code
 * @param decisionMargin decode confidence; higher is better
 * @param corners eight floats, {@code x0,y0,x1,y1,x2,y2,x3,y3}
 * @param homography nine floats, the row-major tag-to-image homography
 */
public record TagDetection(
        int id, int hamming, float decisionMargin, float[] corners, float[] homography) {

    public TagDetection {
        if (corners == null || corners.length != 8) {
            throw new IllegalArgumentException("corners must contain 8 values");
        }
        if (homography == null || homography.length != 9) {
            throw new IllegalArgumentException("homography must contain 9 values");
        }
        corners = corners.clone();
        homography = homography.clone();
    }

    @Override
    public float[] corners() {
        return corners.clone();
    }

    @Override
    public float[] homography() {
        return homography.clone();
    }

    public float cornerX(int index) {
        if (index < 0 || index > 3) {
            throw new IndexOutOfBoundsException(index);
        }
        return corners[index * 2];
    }

    public float cornerY(int index) {
        if (index < 0 || index > 3) {
            throw new IndexOutOfBoundsException(index);
        }
        return corners[index * 2 + 1];
    }

    public float centerX() {
        return 0.25f * (corners[0] + corners[2] + corners[4] + corners[6]);
    }

    public float centerY() {
        return 0.25f * (corners[1] + corners[3] + corners[5] + corners[7]);
    }

    /** Projects a tag coordinate onto the image x axis. */
    public float projectX(float x, float y) {
        float w = homography[6] * x + homography[7] * y + homography[8];
        return (homography[0] * x + homography[1] * y + homography[2]) / w;
    }

    /** Projects a tag coordinate onto the image y axis. */
    public float projectY(float x, float y) {
        float w = homography[6] * x + homography[7] * y + homography[8];
        return (homography[3] * x + homography[4] * y + homography[5]) / w;
    }
}
