package com.frc6324.bdcudatag.viewer;

import com.frc6324.bdcudatag.TagDetection;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.geom.Path2D;
import java.awt.image.BufferedImage;
import java.util.List;

/** Draws tag outlines, corners and ids directly into a frame. */
final class Overlay {

    private static final Color OUTLINE = new Color(64, 255, 96);
    private static final Color CORNER = new Color(255, 224, 64);
    private static final Color LABEL = new Color(0, 0, 0, 176);

    private Overlay() {}

    static void draw(BufferedImage image, List<TagDetection> tags) {
        if (tags.isEmpty()) {
            return;
        }
        Graphics2D graphics = image.createGraphics();
        try {
            graphics.setRenderingHint(
                    RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            graphics.setRenderingHint(
                    RenderingHints.KEY_TEXT_ANTIALIASING,
                    RenderingHints.VALUE_TEXT_ANTIALIAS_ON);

            final int width = image.getWidth();
            final int height = image.getHeight();
            graphics.setStroke(
                    new BasicStroke(
                            Math.max(2.0f, width / 480.0f),
                            BasicStroke.CAP_ROUND,
                            BasicStroke.JOIN_ROUND));

            final int dot = Math.max(3, Math.round(width / 260.0f));
            graphics.setFont(
                    new Font(
                            Font.MONOSPACED,
                            Font.BOLD,
                            Math.max(13, Math.round(width / 42.0f))));
            FontMetrics metrics = graphics.getFontMetrics();
            final int pad = Math.max(4, metrics.getHeight() / 4);

            for (TagDetection tag : tags) {
                Path2D.Float quad = new Path2D.Float();
                quad.moveTo(tag.cornerX(0), tag.cornerY(0));
                for (int i = 1; i < 4; ++i) {
                    quad.lineTo(tag.cornerX(i), tag.cornerY(i));
                }
                quad.closePath();
                graphics.setColor(OUTLINE);
                graphics.draw(quad);

                graphics.setColor(CORNER);
                for (int i = 0; i < 4; ++i) {
                    int x = Math.round(tag.cornerX(i));
                    int y = Math.round(tag.cornerY(i));
                    graphics.fillOval(x - dot, y - dot, dot * 2, dot * 2);
                }

                String label = "#" + tag.id();
                int textWidth = metrics.stringWidth(label);
                int boxWidth = textWidth + pad * 2;
                int boxHeight = metrics.getHeight() + pad;
                int boxX =
                        clamp(
                                Math.round(tag.centerX()) - boxWidth / 2,
                                0,
                                Math.max(0, width - boxWidth));
                int boxY =
                        clamp(
                                Math.round(tag.centerY()) - boxHeight / 2,
                                0,
                                Math.max(0, height - boxHeight));
                graphics.setColor(LABEL);
                graphics.fillRoundRect(boxX, boxY, boxWidth, boxHeight, pad * 2, pad * 2);
                graphics.setColor(Color.WHITE);
                graphics.drawString(label, boxX + pad, boxY + pad + metrics.getAscent());
            }
        } finally {
            graphics.dispose();
        }
    }

    private static int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }
}
