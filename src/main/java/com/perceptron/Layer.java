package com.perceptron;

import com.perceptron.Neuron.Data;

public class Layer {
    Neuron[] neurons;
    private long seed;

    public Layer(int inputCount, int neuronCount, double learningRate) {
        this(inputCount, neuronCount, learningRate, null);
    }

    public Layer(int inputCount, int neuronCount, double learningRate, Long seed) {
        this.neurons = new Neuron[neuronCount];
        this.seed = (seed != null)
                ? seed
                : System.currentTimeMillis();

        for (int i = 0; i < neuronCount; i++) {
            this.neurons[i] = new Neuron(inputCount, learningRate);
        }
    }

    public long seed() {
        return this.seed;
    }

    public double[] predict(double[] inputs) {
        double[] outputs = new double[this.neurons.length];

        for (int i = 0; i < this.neurons.length; i++) {
            outputs[i] = this.neurons[i].predict(inputs);
        }

        return outputs;
    }
}
