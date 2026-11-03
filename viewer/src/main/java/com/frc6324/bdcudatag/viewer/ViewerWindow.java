package com.frc6324.bdcudatag.viewer;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.ActionEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import javax.swing.AbstractAction;
import javax.swing.BorderFactory;
import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.KeyStroke;
import javax.swing.SwingUtilities;

/** Swing window that shows the newest frame and a one-line status. */
final class ViewerWindow implements AutoCloseable {

    private final JFrame frame;
    private final Canvas canvas = new Canvas();
    private final JLabel status = new JLabel(" starting...");
    private volatile Runnable onQuit = () -> {};
    private volatile Runnable onSnapshot = () -> {};
    private volatile BufferedImage latest;

    ViewerWindow(String title) {
        frame = new JFrame(title);
        frame.setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
        frame.addWindowListener(
                new WindowAdapter() {
                    @Override
                    public void windowClosing(WindowEvent event) {
                        onQuit.run();
                    }
                });

        status.setOpaque(true);
        status.setBackground(new Color(24, 24, 24));
        status.setForeground(new Color(220, 220, 220));
        status.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 13));
        status.setBorder(BorderFactory.createEmptyBorder(6, 10, 6, 10));

        frame.add(canvas, BorderLayout.CENTER);
        frame.add(status, BorderLayout.SOUTH);
        frame.getContentPane().setBackground(Color.BLACK);

        bind("Q", () -> onQuit.run());
        bind("ESCAPE", () -> onQuit.run());
        bind("S", () -> onSnapshot.run());

        frame.setMinimumSize(new Dimension(480, 360));
        frame.setPreferredSize(new Dimension(1280, 800));
        frame.pack();
        frame.setLocationRelativeTo(null);
    }

    void setOnQuit(Runnable action) {
        onQuit = action;
    }

    void setOnSnapshot(Runnable action) {
        onSnapshot = action;
    }

    void show() {
        SwingUtilities.invokeLater(() -> frame.setVisible(true));
    }

    void update(BufferedImage image, String text) {
        latest = image;
        SwingUtilities.invokeLater(
                () -> {
                    canvas.repaint();
                    status.setText(text);
                });
    }

    BufferedImage latestImage() {
        return latest;
    }

    @Override
    public void close() {
        SwingUtilities.invokeLater(frame::dispose);
    }

    private void bind(String key, Runnable action) {
        JComponent root = frame.getRootPane();
        root.getInputMap(JComponent.WHEN_IN_FOCUSED_WINDOW)
                .put(KeyStroke.getKeyStroke(key), key);
        root.getActionMap()
                .put(
                        key,
                        new AbstractAction() {
                            @Override
                            public void actionPerformed(ActionEvent event) {
                                action.run();
                            }
                        });
    }

    private final class Canvas extends JPanel {

        Canvas() {
            setBackground(Color.BLACK);
            setDoubleBuffered(true);
        }

        @Override
        protected void paintComponent(Graphics graphics) {
            super.paintComponent(graphics);
            BufferedImage image = latest;
            if (image == null) {
                return;
            }
            Graphics2D g = (Graphics2D) graphics.create();
            try {
                g.setRenderingHint(
                        RenderingHints.KEY_INTERPOLATION,
                        RenderingHints.VALUE_INTERPOLATION_BILINEAR);
                double scale =
                        Math.min(
                                getWidth() / (double) image.getWidth(),
                                getHeight() / (double) image.getHeight());
                int width = Math.max(1, (int) Math.round(image.getWidth() * scale));
                int height = Math.max(1, (int) Math.round(image.getHeight() * scale));
                int x = (getWidth() - width) / 2;
                int y = (getHeight() - height) / 2;
                g.drawImage(image, x, y, width, height, null);
            } finally {
                g.dispose();
            }
        }
    }
}
