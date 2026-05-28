package com.perceptron;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import com.google.gson.Gson;
import com.perceptron.Perceptron.Data;

public class App {

    public static final String MODEL_FILE = "model.json";
    public static final String DATA_FILE = "data.json";

    public static Data[] loadData(String filePath) throws IOException {
        var gson = new Gson();
        var json = Files.readString(Path.of(filePath));
        return gson.fromJson(json, Data[].class);
    }

    public static void saveData(String filePath, Data[] data) throws IOException {
        var gson = new Gson();
        Files.writeString(Path.of(filePath), gson.toJson(data));
    }

    public static void main(String[] args) throws IOException {
        var data = loadData(DATA_FILE);

        var p = new Perceptron(data[0].inputs().length, 0.1);
        System.out.println("Seed usada: " + ConsoleColor.YELLOW + p.seed() + ConsoleColor.RESET);
        System.out.println("Antes de entrenar:");
        printPredictions(p, data);

        p.train(data, 20);
        System.out.println("Despues de entrenar:");
        printPredictions(p, data);
        IO.print(String.format("Precision:" + ConsoleColor.YELLOW + " %.0f%%\n" + ConsoleColor.RESET,
                p.accuracy(data) * 100));

        p.saveModel(MODEL_FILE);
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
