package com.frc6324.bdcudatag;

/** Tag families supported by the detector. A detector handles exactly one family. */
public enum TagFamily {
    Tag36h11(0, "tag36h11", 6, 11, 2),
    Tag16h5(1, "tag16h5", 4, 5, 1);

    private final int nativeId;
    private final String tagName;
    private final int bitsPerSide;
    private final int minHamming;
    private final int defaultMaxHamming;

    TagFamily(int nativeId, String tagName, int bitsPerSide, int minHamming, int defaultMaxHamming) {
        this.nativeId = nativeId;
        this.tagName = tagName;
        this.bitsPerSide = bitsPerSide;
        this.minHamming = minHamming;
        this.defaultMaxHamming = defaultMaxHamming;
    }

    public int nativeId() {
        return nativeId;
    }

    /** The canonical AprilTag family name, for example {@code tag36h11}. */
    public String tagName() {
        return tagName;
    }

    public int bitsPerSide() {
        return bitsPerSide;
    }

    /** Minimum Hamming distance between any two codes, including rotations. */
    public int minHamming() {
        return minHamming;
    }

    /** Hamming limit used when {@link DetectorConfig#maxHamming()} is {@code -1}. */
    public int defaultMaxHamming() {
        return defaultMaxHamming;
    }

    public static TagFamily fromNativeId(int nativeId) {
        for (TagFamily family : values()) {
            if (family.nativeId == nativeId) {
                return family;
            }
        }
        throw new IllegalArgumentException("Unknown native tag family id: " + nativeId);
    }
}
