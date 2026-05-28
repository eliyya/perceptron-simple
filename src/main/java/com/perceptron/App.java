package com.perceptron;

import javax.imageio.ImageIO;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;

import com.perceptron.Neuron.Data;

public class App {

    public static final String MODEL_FILE = "model.json";
    private static final Path SAMPLES_DIR = Path.of("samples");

    static final int INPUT_SIZE = 16 * 16;

    static final Data[] AND_DATA = {
            new Data(new double[] { 0, 0 }, 0),
            new Data(new double[] { 0, 1 }, 0),
            new Data(new double[] { 1, 0 }, 0),
            new Data(new double[] { 1, 1 }, 1),
    };
    static final Data[] OR_DATA = {
            new Data(new double[] { 0, 0 }, 0),
            new Data(new double[] { 0, 1 }, 1),
            new Data(new double[] { 1, 0 }, 1),
            new Data(new double[] { 1, 1 }, 1),
    };
    static final Data[] XOR_DATA = {
            new Data(new double[] { 0, 0 }, 0),
            new Data(new double[] { 0, 1 }, 1),
            new Data(new double[] { 1, 0 }, 1),
            new Data(new double[] { 1, 1 }, 0),
    };

    public static Data[] loadSamples() throws IOException {
        var list = new ArrayList<Data>();
        if (!Files.isDirectory(SAMPLES_DIR)) return list.toArray(Data[]::new);

        try (var stream = Files.newDirectoryStream(SAMPLES_DIR, "*.png")) {
            for (var file : stream) {
                var img = ImageIO.read(file.toFile());
                if (img == null) continue;

                var inputs = new double[INPUT_SIZE];
                for (int y = 0; y < 16; y++) {
                    for (int x = 0; x < 16; x++) {
                        var gray = (img.getRGB(x, y) >> 16) & 0xFF;
                        inputs[y * 16 + x] = gray < 128 ? 1.0 : 0.0;
                    }
                }
                var name = file.getFileName().toString().toLowerCase();
                var target = name.startsWith("happy") ? 1.0 : 0.0;
                list.add(new Data(inputs, target));
            }
        }
        return list.toArray(Data[]::new);
    }

    public static void main(String[] args) throws IOException {
        var data = loadSamples();
        IO.println(""+data.length);
        if (data.length == 0) {
            System.out.println("No hay muestras en " + SAMPLES_DIR + ". Usando XOR_DATA.");
            data = XOR_DATA;
        }

        var inputCount = data[0].inputs().length;
        var learningRate = 0.5;

        var net = new Network(inputCount, learningRate, 16, 1);
        System.out.println("Seed usada: " + ConsoleColor.YELLOW + net.seed() + ConsoleColor.RESET);
        System.out.println("Muestras: " + ConsoleColor.YELLOW + data.length + ConsoleColor.RESET);
        System.out.println("Antes de entrenar:");
        printPredictions(net, data);

        net.train(data, 1000);
        System.out.println("Despues de entrenar:");
        printPredictions(net, data);

        net.saveModel(MODEL_FILE);
        System.out.println("Modelo guardado en: " + ConsoleColor.YELLOW + MODEL_FILE + ConsoleColor.RESET);

        var loaded = Network.loadModel(MODEL_FILE);
        System.out.println("Predicciones con modelo cargado:");
        printPredictions(loaded, data);
        IO.print(String.format("Precision:" + ConsoleColor.YELLOW + " %.0f%%\n" + ConsoleColor.RESET,
                loaded.accuracy(data) * 100));
    }

    private static void printPredictions(Network p, Data[] data) {
        for (var i = 0; i < data.length; i++) {
            var pred = p.predict(data[i].inputs());
            var rounded = (int) Math.round(pred);
            var ok = rounded == (int) data[i].target();
            String line = String.format("  muestra %d -> %.4f (esperado: %d)%s",
                    i, pred, (int) data[i].target(),
                    ok ? "" : " <-- MAL");
            if (ok) {
                IO.print(ConsoleColor.GREEN + line + ConsoleColor.RESET + "\n");
            } else {
                IO.print(ConsoleColor.RED_BOLD + line + ConsoleColor.RESET + "\n");
            }
        }
    }
}
