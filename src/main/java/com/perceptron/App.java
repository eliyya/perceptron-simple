package com.perceptron;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import com.google.gson.Gson;
import com.perceptron.Neuron.Data;

public class App {

    public static final String MODEL_FILE = "model.json";
    public static final String DATA_FILE = "data.json";

    public static Data[] loadData(String filePath) throws IOException {
        var gson = new Gson();
        var json = Files.readString(Path.of(filePath));
        return gson.fromJson(json, Data[].class);
    }

    public static void main(String[] args) throws IOException {
        var data = loadData(DATA_FILE);
        var inputCount = data[0].inputs().length;
        var learningRate = 0.5;
 
        var net = new Network(inputCount, learningRate, 2, 1);
        System.out.println("Seed usada: " + ConsoleColor.YELLOW + net.seed() + ConsoleColor.RESET);
        System.out.println("Antes de entrenar:");
        printPredictions(net, data);

        net.train(data, 10000);
        System.out.println("Despues de entrenar:");
        printPredictions(net, data);

        net.saveModel(MODEL_FILE);
        System.out.println("Modelo guardado en: " + ConsoleColor.YELLOW + MODEL_FILE + ConsoleColor.RESET);

        var loaded = Network.loadModel(MODEL_FILE);
        System.out.println("Modelo cargado desde: " + ConsoleColor.YELLOW + MODEL_FILE + ConsoleColor.RESET);
        System.out.println("Predicciones con modelo cargado:");
        printPredictions(loaded, data);
        IO.print(String.format("Precision:" + ConsoleColor.YELLOW + " %.0f%%\n" + ConsoleColor.RESET,
                loaded.accuracy(data) * 100));
    }

    private static void printPredictions(Network p, Data[] data) {
        for (var i = 0; i < data.length; i++) {
            var pred = p.predict(data[i].inputs());
            var rounded = (int) Math.round(pred);
            var ok = rounded == ((int) data[i].target());
            String line = String.format(
                    "  %d , %d -> %.4f (esperado: %d)",
                    (int) data[i].inputs()[0],
                    (int) data[i].inputs()[1],
                    pred,
                    (int) data[i].target());
            if (ok) {
                IO.print(ConsoleColor.GREEN + line + ConsoleColor.RESET + "\n");
            } else {
                IO.print(ConsoleColor.RED_BOLD + line + " <-- MAL" + ConsoleColor.RESET + "\n");
            }
        }
    }
}
