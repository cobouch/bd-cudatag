package com.frc6324.bdcudatag.viewer;

import java.io.IOException;
import java.util.Locale;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

/** A camera, video file or still image that produces BGR frames. */
final class FrameSource implements AutoCloseable {

    private final VideoCapture capture;
    private final Mat still;
    private final boolean live;
    private final String description;
    private boolean stillRead;

    private FrameSource(VideoCapture capture, Mat still, boolean live, String description) {
        this.capture = capture;
        this.still = still;
        this.live = live;
        this.description = description;
    }

    static FrameSource open(Options options) throws IOException {
        if (options.image() != null) {
            Mat image = Imgcodecs.imread(options.image().toString(), Imgcodecs.IMREAD_COLOR);
            if (image.empty()) {
                image.release();
                throw new IOException("could not read image " + options.image());
            }
            return new FrameSource(null, image, false, "image " + options.image().getFileName());
        }

        if (options.video() != null) {
            VideoCapture capture = new VideoCapture(options.video().toString());
            if (!capture.isOpened()) {
                capture.release();
                throw new IOException("could not open video " + options.video());
            }
            return new FrameSource(
                    capture, null, false, "video " + options.video().getFileName());
        }

        VideoCapture capture = new VideoCapture(options.device());
        if (!capture.isOpened()) {
            capture.release();
            throw new IOException(
                    "could not open camera "
                            + options.device()
                            + "; try another --device index or use --video/--image");
        }
        capture.set(Videoio.CAP_PROP_FRAME_WIDTH, options.width());
        capture.set(Videoio.CAP_PROP_FRAME_HEIGHT, options.height());
        String description =
                String.format(
                        Locale.ROOT,
                        "camera %d (%dx%d)",
                        options.device(),
                        (int) capture.get(Videoio.CAP_PROP_FRAME_WIDTH),
                        (int) capture.get(Videoio.CAP_PROP_FRAME_HEIGHT));
        return new FrameSource(capture, null, true, description);
    }

    /** Reads the next frame into {@code destination}, returning false at EOF or on failure. */
    boolean read(Mat destination) {
        if (still != null) {
            if (stillRead) {
                return false;
            }
            still.copyTo(destination);
            stillRead = true;
            return true;
        }
        return capture.read(destination);
    }

    /** Whether this source is a live device, which can recover from a failed read. */
    boolean live() {
        return live;
    }

    String description() {
        return description;
    }

    @Override
    public void close() {
        if (capture != null) {
            capture.release();
        }
        if (still != null) {
            still.release();
        }
    }
}
