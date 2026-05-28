package com.perceptron;

import java.io.IOException;
import com.perceptron.Neuron.Data;

public class App {

    public static final String MODEL_FILE = "model.json";
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

    public static void main(String[] args) throws IOException {
        var data = XOR_DATA;
        var inputCount = 2;
        var learningRate = 0.1;

        var layer = new Layer(inputCount, 2, learningRate);
        var output = new Layer(inputCount, 1, learningRate, layer.seed());
        System.out.println("Seed usada: " + ConsoleColor.YELLOW + output.seed() + ConsoleColor.RESET);
        System.out.println("Antes de entrenar:");
        printPredictions(output, data);

        // output.train(data, 10);
        System.out.println("Despues de entrenar:");
        printPredictions(output, data);
        // IO.print(String.format("Precision:" + ConsoleColor.YELLOW + " %.0f%%\n" + ConsoleColor.RESET,
        //         output.accuracy(data) * 100));

        // output.saveModel(MODEL_FILE);
        System.out.println("Modelo guardado en: " + ConsoleColor.YELLOW + MODEL_FILE + ConsoleColor.RESET);

        var loaded = Neuron.loadModel(MODEL_FILE);
        System.out.println("Modelo cargado desde: " + ConsoleColor.YELLOW + MODEL_FILE + ConsoleColor.RESET);
        System.out.println("Predicciones con modelo cargado:");
        // printPredictions(loaded, data);
        IO.print(String.format("Precision:" + ConsoleColor.YELLOW + " %.0f%%\n" + ConsoleColor.RESET,
                loaded.accuracy(data) * 100));
    }

    private static void printPredictions(Layer p, Data[] data) {
        for (var i = 0; i < data.length; i++) {
            var pred = p.predict(data[i].inputs());
            var ok = (int) pred[0] == data[i].target();
            String line = String.format(
                    "  %d , %d -> %d (esperado: %d)",
                    (int) data[i].inputs()[0],
                    (int) data[i].inputs()[1],
                    pred, data[i].target());
            if (ok) {
                IO.print(ConsoleColor.GREEN + line + ConsoleColor.RESET + "\n");
            } else {
                IO.print(ConsoleColor.RED_BOLD + line + " <-- MAL" + ConsoleColor.RESET + "\n");
            }
        }
    }
}
