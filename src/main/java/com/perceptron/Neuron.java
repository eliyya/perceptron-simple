package com.perceptron;

import com.google.gson.Gson;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

public class Neuron {

    private final double[] weights;
    private double bias;
    private final double learningRate;
    private long seed;
    private double previousOutput = 0;

    public Neuron(int numInputs, double learningRate, long seed) {
        this.learningRate = learningRate;
        this.weights = new double[numInputs];
        this.seed = seed;
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
        previousOutput = sum;
        return sum;
    }

    public ModelData model() {
        return new ModelData(weights, bias, learningRate);
    }

    public static Neuron loadModel(String filePath) throws IOException {
        var gson = new Gson();
        var data = gson.fromJson(Files.readString(Path.of(filePath)), ModelData.class);
        return new Neuron(data.weights, data.bias, data.learningRate);
    }

    public void adjust(double delta) {
        for (int i = 0; i < weights.length; i++) {
            weights[i] += learningRate * delta * previousOutput;
        }
        bias += learningRate * delta;
    }

    public record ModelData(double[] weights, double bias, double learningRate) {
    }
    public record Data(double[] inputs, double target) {
    }
}
