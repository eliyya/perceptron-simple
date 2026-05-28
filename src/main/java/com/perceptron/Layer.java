package com.perceptron;

import com.perceptron.Neuron.ModelData;

public class Layer {
    Neuron[] neurons;
    private long seed;

    public Layer(int inputCount, int neuronCount, double learningRate, long seed) {
        this.neurons = new Neuron[neuronCount];
        this.seed = seed;

        for (int i = 0; i < neuronCount; i++) {
            this.neurons[i] = new Neuron(inputCount, learningRate, seed);
        }
    }

    public long seed() {
        return this.seed;
    }

    public ModelData[] model() {
        var data = new ModelData[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            data[i] = neurons[i].model();            
        }
        return data;
    }

    public double[] predict(double[] inputs) {
        double[] outputs = new double[this.neurons.length];

        for (int i = 0; i < this.neurons.length; i++) {
            outputs[i] = this.neurons[i].predict(inputs);
        }

        return outputs;
    }

    public void adjust(double delta) {
        for (int i = 0; i < neurons.length; i++) {
            neurons[i].adjust(delta);
        }
    }
}
