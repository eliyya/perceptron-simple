package com.perceptron;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

public class Neuron {

    private final double[] weights;
    private double bias;
    private final double learningRate;
    private long seed;

    public Neuron(int numInputs, double learningRate) {
        this(numInputs, learningRate, null);
    }

    public Neuron(int numInputs, double learningRate, Long seed) {
        this.learningRate = learningRate;
        this.weights = new double[numInputs];
        this.seed = (seed != null)
                ? seed
                : System.currentTimeMillis();
        var rng = new Random(this.seed);
        for (var i = 0; i < numInputs; i++) {
            weights[i] = rng.nextDouble() * 2 - 1;
        }
        bias = rng.nextDouble() * 2 - 1;
    }

    private Neuron(double[] weights, double bias, double learningRate) {
        this.weights = weights.clone();
        this.bias = bias;
        this.learningRate = learningRate;
    }

    public long seed() {
        return this.seed;
    }

    public double predict(double[] inputs) {
        var sum = bias;
        for (var i = 0; i < weights.length; i++) {
            sum += weights[i] * inputs[i];
        }
        return sum;
    }

    public void train(Data[] data, int epochs) {
        for (var ep = 0; ep < epochs; ep++) {
            for (var s = 0; s < data.length; s++) {
                var prediction = predict(data[s].inputs());
                var error = data[s].target - prediction;
                if (error != 0) {
                    for (var i = 0; i < weights.length; i++) {
                        weights[i] += learningRate * error * data[s].inputs()[i];
                    }
                    bias += learningRate * error;
                }
            }
        }
    }

    public double accuracy(Data[] data) {
        var correct = 0;
        for (var i = 0; i < data.length; i++) {
            if (predict(data[i].inputs) == data[i].target)
                correct++;
        }
        return (double) correct / data.length;
    }

    public void saveModel(String filePath) throws IOException {
        var data = new ModelData(weights, bias, learningRate);
        var gson = new GsonBuilder().setPrettyPrinting().create();
        Files.writeString(Path.of(filePath), gson.toJson(data));
    }

    public static Neuron loadModel(String filePath) throws IOException {
        var gson = new Gson();
        var data = gson.fromJson(Files.readString(Path.of(filePath)), ModelData.class);
        return new Neuron(data.weights, data.bias, data.learningRate);
    }

    private record ModelData(double[] weights, double bias, double learningRate) {
    }

    public record Data(double[] inputs, int target) {
    }
}
