package com.perceptron;

import java.io.IOException;
import com.perceptron.Perceptron.Data;

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

    public static void main(String[] args) throws IOException {
        var data = OR_DATA;

        var p = new Perceptron(2, 0.1);
        System.out.println("Seed usada: " + ConsoleColor.YELLOW + p.seed() + ConsoleColor.RESET);
        System.out.println("Antes de entrenar:");
        printPredictions(p, data);

        p.train(data, 10);
        System.out.println("Despues de entrenar:");
        printPredictions(p, data);
        IO.print(String.format("Precision:" + ConsoleColor.YELLOW + " %.0f%%\n" + ConsoleColor.RESET,
                p.accuracy(data) * 100));

        p.saveModel(MODEL_FILE);
        System.out.println("Modelo guardado en: " + ConsoleColor.YELLOW + MODEL_FILE + ConsoleColor.RESET);

        var loaded = Perceptron.loadModel(MODEL_FILE);
        System.out.println("Modelo cargado desde: " + ConsoleColor.YELLOW + MODEL_FILE + ConsoleColor.RESET);
        System.out.println("Predicciones con modelo cargado:");
        printPredictions(loaded, data);
        IO.print(String.format("Precision:" + ConsoleColor.YELLOW + " %.0f%%\n" + ConsoleColor.RESET,
                loaded.accuracy(data) * 100));
    }

    private static void printPredictions(Perceptron p, Data[] data) {
        for (var i = 0; i < data.length; i++) {
            var pred = p.predict(data[i].inputs());
            var ok = pred == data[i].target();
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
