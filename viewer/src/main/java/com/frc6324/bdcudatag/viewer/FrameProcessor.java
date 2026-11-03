package com.frc6324.bdcudatag.viewer;

import com.frc6324.bdcudatag.BDTagDetector;
import com.frc6324.bdcudatag.DetectorStats;
import com.frc6324.bdcudatag.TagDetection;
import com.frc6324.bdcudatag.TagFamily;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.util.List;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/** Owns the detector and the conversion buffers for the current frame size. */
final class FrameProcessor implements AutoCloseable {

    record Result(
            BufferedImage image,
            List<TagDetection> tags,
            DetectorStats stats,
            int width,
            int height) {}

    private final TagFamily family;
    private final Mat gray = new Mat();
    private final Mat display = new Mat();
    private final Mat copy = new Mat();
    private BDTagDetector detector;
    private byte[] grayBytes = new byte[0];
    private int width = -1;
    private int height = -1;

    FrameProcessor(TagFamily family) {
        this.family = family;
    }

    Result process(Mat frame) {
        final int frameWidth = frame.cols();
        final int frameHeight = frame.rows();
        if (frameWidth <= 0 || frameHeight <= 0) {
            throw new IllegalStateException("empty frame");
        }
        if (frame.depth() != CvType.CV_8U) {
            throw new IllegalStateException("expected 8-bit frames, got depth " + frame.depth());
        }
        ensureDetector(frameWidth, frameHeight);

        Mat grayMat = toGray(frame);
        if (!grayMat.isContinuous()) {
            grayMat.copyTo(copy);
            grayMat = copy;
        }
        grayMat.get(0, 0, grayBytes);

        List<TagDetection> tags = detector.detect(grayBytes);
        DetectorStats stats = detector.stats();
        return new Result(render(frame, tags), tags, stats, frameWidth, frameHeight);
    }

    private void ensureDetector(int frameWidth, int frameHeight) {
        if (detector != null && frameWidth == width && frameHeight == height) {
            return;
        }
        closeDetector();
        detector = new BDTagDetector(family, frameWidth, frameHeight);
        width = frameWidth;
        height = frameHeight;
        grayBytes = new byte[frameWidth * frameHeight];
    }

    private Mat toGray(Mat frame) {
        return switch (frame.channels()) {
            case 1 -> frame;
            case 3 -> {
                Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);
                yield gray;
            }
            case 4 -> {
                Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGRA2GRAY);
                yield gray;
            }
            default ->
                throw new IllegalStateException(
                        "unsupported channel count " + frame.channels());
        };
    }

    private BufferedImage render(Mat frame, List<TagDetection> tags) {
        Mat bgr =
                switch (frame.channels()) {
                    case 3 -> frame;
                    case 4 -> {
                        Imgproc.cvtColor(frame, display, Imgproc.COLOR_BGRA2BGR);
                        yield display;
                    }
                    case 1 -> {
                        Imgproc.cvtColor(frame, display, Imgproc.COLOR_GRAY2BGR);
                        yield display;
                    }
                    default ->
                        throw new IllegalStateException(
                                "unsupported channel count " + frame.channels());
                };
        if (!bgr.isContinuous()) {
            bgr.copyTo(copy);
            bgr = copy;
        }

        BufferedImage image =
                new BufferedImage(bgr.cols(), bgr.rows(), BufferedImage.TYPE_3BYTE_BGR);
        byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        bgr.get(0, 0, pixels);
        Overlay.draw(image, tags);
        return image;
    }

    @Override
    public void close() {
        closeDetector();
        gray.release();
        display.release();
        copy.release();
    }

    private void closeDetector() {
        if (detector != null) {
            detector.close();
            detector = null;
        }
    }
}
