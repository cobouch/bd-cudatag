package com.frc6324.bdcudatag;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class TagFamilyTest {

    @Test
    void familyMetadataMatchesAprilTag() {
        assertEquals(6, TagFamily.Tag36h11.bitsPerSide());
        assertEquals(11, TagFamily.Tag36h11.minHamming());
        assertEquals(2, TagFamily.Tag36h11.defaultMaxHamming());
        assertEquals(4, TagFamily.Tag16h5.bitsPerSide());
        assertEquals(5, TagFamily.Tag16h5.minHamming());
        assertEquals(1, TagFamily.Tag16h5.defaultMaxHamming());
        assertEquals("tag36h11", TagFamily.Tag36h11.tagName());
        assertEquals(TagFamily.Tag16h5, TagFamily.fromNativeId(1));
    }

    @Test
    void detectionDefensivelyCopiesCorners() {
        float[] corners = {1, 2, 3, 4, 5, 6, 7, 8};
        float[] homography = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        TagDetection detection = new TagDetection(42, 0, 12.5f, corners, homography);
        corners[0] = 99;
        homography[8] = 99;
        assertArrayEquals(new float[] {1, 2, 3, 4, 5, 6, 7, 8}, detection.corners());
        assertArrayEquals(new float[] {1, 0, 0, 0, 1, 0, 0, 0, 1}, detection.homography());
        assertEquals(1.0f, detection.cornerX(0));
        assertEquals(8.0f, detection.cornerY(3));
        assertEquals(4.0f, detection.centerX());
        assertEquals(5.0f, detection.centerY());
        assertEquals(3.0f, detection.projectX(3.0f, 4.0f));
        assertEquals(4.0f, detection.projectY(3.0f, 4.0f));
    }

    @Test
    void configValidationRejectsBadValues() {
        assertThrows(IllegalArgumentException.class,
                () -> new DetectorConfig(-2, 24, 8, 64, 0, true, true, 3.0f, 0.25f));
        assertThrows(IllegalArgumentException.class,
                () -> new DetectorConfig(-1, 24, 8, 0, 0, true, true, 3.0f, 0.25f));
        assertThrows(IllegalArgumentException.class,
                () -> new DetectorConfig(-1, 24, 8, 257, 0, true, true, 3.0f, 0.25f));
        assertThrows(IllegalArgumentException.class,
                () -> new DetectorConfig(-1, 24, 8, 64, -1, true, true, 3.0f, 0.25f));
    }
}
