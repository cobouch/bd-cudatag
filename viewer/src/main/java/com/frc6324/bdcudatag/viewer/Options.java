package com.frc6324.bdcudatag.viewer;

import com.frc6324.bdcudatag.TagFamily;
import java.nio.file.Path;
import java.util.Locale;

/** Parsed command line options for the camera viewer. */
record Options(
        boolean headless,
        Path video,
        Path image,
        int device,
        int width,
        int height,
        TagFamily family,
        int maxFrames,
        boolean help) {

    static final String USAGE =
            """
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

            In the window: Q or Esc quits, S saves a PNG snapshot.
            """;

    static Options parse(String[] args) {
        boolean headless = false;
        boolean help = false;
        Path video = null;
        Path image = null;
        int device = 0;
        int width = 1280;
        int height = 720;
        TagFamily family = TagFamily.Tag36h11;
        int maxFrames = 0;

        for (int i = 0; i < args.length; ++i) {
            String arg = args[i];
            switch (arg) {
                case "--device" -> device = integer(args, ++i, arg);
                case "--width" -> width = integer(args, ++i, arg);
                case "--height" -> height = integer(args, ++i, arg);
                case "--frames" -> maxFrames = integer(args, ++i, arg);
                case "--family" -> family = family(args, ++i);
                case "--video" -> video = Path.of(value(args, ++i, arg));
                case "--image" -> image = Path.of(value(args, ++i, arg));
                case "--headless" -> headless = true;
                case "--help", "-h" -> help = true;
                default -> throw new IllegalArgumentException("unknown option " + arg);
            }
        }

        if (video != null && image != null) {
            throw new IllegalArgumentException("--video and --image cannot be combined");
        }
        if (device < 0) {
            throw new IllegalArgumentException("--device must be >= 0");
        }
        if (width < 16 || height < 16) {
            throw new IllegalArgumentException("--width and --height must be >= 16");
        }
        if (maxFrames < 0) {
            throw new IllegalArgumentException("--frames must be >= 0");
        }
        return new Options(headless, video, image, device, width, height, family, maxFrames, help);
    }

    private static String value(String[] args, int index, String option) {
        if (index >= args.length) {
            throw new IllegalArgumentException(option + " requires a value");
        }
        return args[index];
    }

    private static int integer(String[] args, int index, String option) {
        String raw = value(args, index, option);
        try {
            return Integer.parseInt(raw);
        } catch (NumberFormatException error) {
            throw new IllegalArgumentException(option + " expects an integer, got " + raw);
        }
    }

    private static TagFamily family(String[] args, int index) {
        String raw = value(args, index, "--family");
        return switch (raw.toLowerCase(Locale.ROOT)) {
            case "tag36h11", "36h11" -> TagFamily.Tag36h11;
            case "tag16h5", "16h5" -> TagFamily.Tag16h5;
            default ->
                throw new IllegalArgumentException(
                        "--family expects tag36h11 or tag16h5, got " + raw);
        };
    }
}
