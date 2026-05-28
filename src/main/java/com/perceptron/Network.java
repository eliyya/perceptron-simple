package com.perceptron;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import com.google.gson.GsonBuilder;
import com.perceptron.Neuron.ModelData;
import com.perceptron.Neuron.Data;

public class Network {
    private final Layer[] layers;
    private long seed;

    public Network(int inputCount, double learningRate, int... layerSizes) {
        this(inputCount, learningRate, null, layerSizes);
    }

    public Network(int inputCount, double learningRate, Long seed, int... layerSizes) {
        this.layers = new Layer[layerSizes.length];
        this.seed = (seed != null)
                ? seed
                : System.currentTimeMillis();

        int currentInputs = inputCount;

        for (int i = 0; i < layerSizes.length; i++) {
            this.layers[i] = new Layer(
                    currentInputs,
                    layerSizes[i],
                    learningRate,
                    this.seed
                );

            currentInputs = layerSizes[i];
        }
    }

    public long seed() {
        return seed;
    }

    public double[] predict(double[] inputs) {
        double[] outputs = inputs;

        for (Layer layer : this.layers) {
            outputs = layer.predict(outputs);
        }

        return outputs;
    }

    public double[][] forward(double[] inputs) {
        double[][] outputs = new double[this.layers.length + 1][];

        outputs[0] = inputs;

        for (int i = 0; i < this.layers.length; i++) {
            outputs[i + 1] = this.layers[i].predict(outputs[i]);
        }

        return outputs;
    }

    public void train(Data[] data, int epochs) {
        for (var ep = 0; ep < epochs; ep++) {
            for (var s = 0; s < data.length; s++) {
                var outputs = forward(data[s].inputs());
                var prediction = outputs[outputs.length - 1][0];
                var error = data[s].target() - prediction;
                var delta = error * prediction * (1 - prediction);
                if (delta != 0) {
                    layers[layers.length - 1].adjust(delta);
                }
            }
        }
    }

    public void saveModel(String filePath) throws IOException {
        ModelData[][] model = new ModelData[this.layers.length][];

        for (int i = 0; i < this.layers.length; i++) {
            model[i] = this.layers[i].model();
        }

        var gson = new GsonBuilder().setPrettyPrinting().create();
        Files.writeString(Path.of(filePath), gson.toJson(model));
    }

    public double accuracy(Data[] data) {
        var correct = 0;
        for (var i = 0; i < data.length; i++) {
            if (predict(data[i].inputs())[0] == data[i].target())
                correct++;
        }
        return (double) correct / data.length;
    }
}
