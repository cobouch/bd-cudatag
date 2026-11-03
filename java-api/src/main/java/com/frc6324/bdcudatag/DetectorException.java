package com.frc6324.bdcudatag;

/** A failure reported by the native detector. The status is a negative bd_status. */
public final class DetectorException extends RuntimeException {
    private final int status;

    public DetectorException(int status, String message) {
        super(message == null || message.isBlank()
                ? "bdcudatag failed with status " + status
                : message + " (status " + status + ")");
        this.status = status;
    }

    public int status() {
        return status;
    }
}
