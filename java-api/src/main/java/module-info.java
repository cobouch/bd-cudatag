/**
 * CUDA AprilTag-family detector for FRC 6324.
 *
 * <p>Only {@code com.frc6324.bdcudatag} is exported. The FFM bindings live in
 * {@code com.frc6324.bdcudatag.internal} and are deliberately encapsulated, so
 * the public surface is just the detector, its configuration and its result
 * types.
 */
module com.frc6324.bdcudatag {
    exports com.frc6324.bdcudatag;
}
