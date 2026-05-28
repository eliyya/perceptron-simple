package com.perceptron;

import java.util.Random;

public class Neuron {

    double[] weights;
    double bias;
    private final double learningRate;
    private final long seed;
    private double[] lastInputs;
    private double lastOutput;

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

    public Neuron(double[] weights, double bias, double learningRate, long seed) {
        this.weights = weights.clone();
        this.bias = bias;
        this.seed = seed;
        this.learningRate = learningRate;
    }

    public long seed() {
        return this.seed;
    }

    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    public double predict(double[] inputs) {
        lastInputs = inputs.clone();
        var sum = bias;
        for (var i = 0; i < weights.length; i++) {
            sum += weights[i] * inputs[i];
        }
        lastOutput = sigmoid(sum);
        return lastOutput;
    }

    public double lastOutput() {
        return lastOutput;
    }

    public double sigmoidDeriv() {
        return lastOutput * (1.0 - lastOutput);
    }

    public void train(double delta) {
        for (int i = 0; i < weights.length; i++) {
            weights[i] += learningRate * delta * lastInputs[i];
        }
        bias += learningRate * delta;
    }

    public NeuronModelData model() {
        return new NeuronModelData(weights, bias);
    }

    public record NeuronModelData(double[] weights, double bias) {
    }
    public record Data(double[] inputs, double target) {
    }
}
