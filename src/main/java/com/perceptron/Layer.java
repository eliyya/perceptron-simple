package com.perceptron;

import com.perceptron.Neuron.NeuronModelData;

public class Layer {
    private final Neuron[] neurons;
    private final long seed;
    private double[] lastInputs;

    public Layer(LayerModelData data, double learningRate, long seed) {
        this.seed = seed;
        var length = data.neurons.length;
        this.neurons = new Neuron[length];
        for (int i = 0; i < length; i++) {
            var neuron = data.neurons[i];
            this.neurons[i] = new Neuron(neuron.weights(), neuron.bias(), learningRate, seed);
        }
    }

    public Layer(int inputCount, int neuronCount, double learningRate, long seed) {
        this.neurons = new Neuron[neuronCount];
        this.seed = seed;

        for (int i = 0; i < neuronCount; i++) {
            this.neurons[i] = new Neuron(inputCount, learningRate, seed + i + 1);
        }
    }

    public long seed() {
        return this.seed;
    }

    public LayerModelData model() {
        var data = new NeuronModelData[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            data[i] = neurons[i].model();
        }
        
        return new LayerModelData(data);
    }

    public double[] forward(double[] inputs) {
        lastInputs = inputs.clone();
        var outputs = new double[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            outputs[i] = neurons[i].predict(inputs);
        }
        return outputs;
    }

    public double[] backward(double[] deltas) {
        var prevDeltas = new double[lastInputs.length];
        for (int j = 0; j < lastInputs.length; j++) {
            double sum = 0;
            for (int i = 0; i < neurons.length; i++) {
                sum += deltas[i] * neurons[i].weights[j];
            }
            prevDeltas[j] = sum;
        }
        for (int i = 0; i < neurons.length; i++) {
            neurons[i].train(deltas[i]);
        }
        return prevDeltas;
    }

    public record LayerModelData(NeuronModelData[] neurons) {
    }
}
