package com.perceptron;

import javax.imageio.ImageIO;
import javax.swing.*;

import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class GridWindow extends JPanel {
    private static final int SIZE = 16;
    private static final Path SAMPLES_DIR = Path.of("samples");
    private final int[] buffer = new int[SIZE * SIZE];

    public GridWindow() {
        setPreferredSize(new Dimension(480, 480));
        addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                int cellW = getWidth() / SIZE;
                int cellH = getHeight() / SIZE;
                int col = e.getX() / cellW;
                int row = e.getY() / cellH;
                if (row >= 0 && row < SIZE && col >= 0 && col < SIZE) {
                    buffer[row * SIZE + col] ^= 1;
                    repaint();
                }
            }
        });
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        int cellW = getWidth() / SIZE;
        int cellH = getHeight() / SIZE;

        for (int row = 0; row < SIZE; row++) {
            for (int col = 0; col < SIZE; col++) {
                g.setColor(buffer[row * SIZE + col] == 1 ? Color.BLACK : Color.WHITE);
                g.fillRect(col * cellW, row * cellH, cellW, cellH);
                g.setColor(Color.GRAY);
                g.drawRect(col * cellW, row * cellH, cellW, cellH);
            }
        }
    }

    public int[] getBuffer() {
        return buffer;
    }

    private void save(boolean happy) throws IOException {
        java.nio.file.Files.createDirectories(SAMPLES_DIR);
        var ts = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss_SSS"));
        var path = SAMPLES_DIR.resolve((happy ? "happy" : "sad") + "_" + ts + ".png");

        var img = new BufferedImage(SIZE, SIZE, BufferedImage.TYPE_BYTE_BINARY);
        for (int i = 0; i < buffer.length; i++) {
            int x = i % SIZE;
            int y = i / SIZE;
            img.setRGB(x, y, buffer[i] == 1 ? Color.BLACK.getRGB() : Color.WHITE.getRGB());
        }
        ImageIO.write(img, "PNG", path.toFile());
    }

    public static void main(String[] args) throws IOException {
        var grid = new GridWindow();
        var model = new Network[]{Network.loadModel(App.MODEL_FILE)};

        var saveHappyItem = new JMenuItem("Save as happy");
        saveHappyItem.addActionListener(e -> {
            try {
                grid.save(true);
            } catch (IOException ex) {
                JOptionPane.showMessageDialog(grid, "Error: " + ex.getMessage());
            }
        });

        var saveSadItem = new JMenuItem("Save as sad");
        saveSadItem.addActionListener(e -> {
            try {
                grid.save(false);
            } catch (IOException ex) {
                JOptionPane.showMessageDialog(grid, "Error: " + ex.getMessage());
            }
        });

        var loadItem = new JMenuItem("Load");
        loadItem.addActionListener(e -> {
            var chooser = new JFileChooser(SAMPLES_DIR.toFile());
            if (chooser.showOpenDialog(grid) == JFileChooser.APPROVE_OPTION) {
                try {
                    var img = ImageIO.read(chooser.getSelectedFile());
                    if (img.getWidth() != SIZE || img.getHeight() != SIZE) {
                        JOptionPane.showMessageDialog(grid, "La imagen debe ser " + SIZE + "x" + SIZE);
                        return;
                    }
                    for (int y = 0; y < SIZE; y++) {
                        for (int x = 0; x < SIZE; x++) {
                            int rgb = img.getRGB(x, y);
                            int gray = (rgb >> 16) & 0xFF;
                            grid.buffer[y * SIZE + x] = gray < 128 ? 1 : 0;
                        }
                    }
                    grid.repaint();
                } catch (IOException ex) {
                    JOptionPane.showMessageDialog(grid, "Error: " + ex.getMessage());
                }
            }
        });

        var predictItem = new JMenuItem("Predict");
        predictItem.addActionListener(e -> {
            var inputs = new double[grid.buffer.length];
            for (int i = 0; i < grid.buffer.length; i++) {
                inputs[i] = grid.buffer[i];
            }
            var result = model[0].predict(inputs);
            var rounded = (int) Math.round(result);
            var label = rounded == 1 ? "happy :)" : "sad :(";
            JOptionPane.showMessageDialog(grid, "Resultado: " + label + "\nPredict: "+result,
                    "Prediccion", JOptionPane.INFORMATION_MESSAGE);
        });

        var trainItem = new JMenuItem("Train");
        trainItem.addActionListener(e -> {
            try {
                var data = App.loadSamples();
                var inputCount = data[0].inputs().length;
                var learningRate = 0.5;
                var net = new Network(inputCount, learningRate, 16, 4, 1);
                net.train(data, 10000);
                net.saveModel(App.MODEL_FILE);
                model[0] = Network.loadModel(App.MODEL_FILE);
                JOptionPane.showMessageDialog(grid, String.format("Entrenamiento completo\nPresicion: %.0f%%", net.accuracy(data) * 100),
                    "Entrenamiento", JOptionPane.INFORMATION_MESSAGE);
            } catch (IOException e1) {
                e1.printStackTrace();
            }
        });

        var fileMenu = new JMenu("File");
        fileMenu.add(loadItem);
        fileMenu.add(saveHappyItem);
        fileMenu.add(saveSadItem);
        
        var IAMenu = new JMenu("IA");
        IAMenu.add(predictItem);
        IAMenu.add(trainItem);

        var menuBar = new JMenuBar();
        menuBar.add(fileMenu);
        menuBar.add(IAMenu);

        var frame = new JFrame(String.format("Grid %dx%d", SIZE, SIZE));
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setJMenuBar(menuBar);
        frame.setLayout(new BorderLayout());
        frame.add(grid, BorderLayout.CENTER);
        frame.pack();
        frame.setVisible(true);
    }
}
