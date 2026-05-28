package com.perceptron;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import com.google.gson.GsonBuilder;
import com.perceptron.Layer.LayerModelData;
import com.perceptron.Neuron.Data;

public class Network {
    private final Layer[] layers;
    private final long seed;
    private final double learningRate;

    public Network(NetworkModelData data) {
        this.learningRate = data.learningRate;
        this.seed = data.seed;
        this.layers = new Layer[data.layers.length];
        for (int i = 0; i < layers.length; i++) {
            var layer = data.layers[i];
            this.layers[i] = new Layer(layer, data.learningRate, data.seed);
        }
    }

    public Network(int inputCount, double learningRate, int... layerSizes) {
        this(inputCount, learningRate, null, layerSizes);
    }

    public Network(int inputCount, double learningRate, Long seed, int... layerSizes) {
        this.layers = new Layer[layerSizes.length];
        this.learningRate = learningRate;
        this.seed = (seed != null)
                ? seed
                : System.currentTimeMillis();

        int currentInputs = inputCount;

        var layerSeed = this.seed;
        for (int i = 0; i < layerSizes.length; i++) {
            this.layers[i] = new Layer(
                    currentInputs,
                    layerSizes[i],
                    learningRate,
                    layerSeed);

            currentInputs = layerSizes[i];
            layerSeed += 100;
        }
    }

    public long seed() {
        return seed;
    }

    public double predict(double[] inputs) {
        var outputs = inputs;
        for (Layer layer : this.layers) {
            outputs = layer.forward(outputs);
        }
        return outputs[0];
    }

    public void train(Data[] data, int epochs) {
        for (var ep = 0; ep < epochs; ep++) {
            for (var s = 0; s < data.length; s++) {
                var layerCount = layers.length;

                var layerOutputs = new double[layerCount][];
                var inputs = data[s].inputs();
                for (int l = 0; l < layerCount; l++) {
                    layerOutputs[l] = layers[l].forward(inputs);
                    inputs = layerOutputs[l];
                }

                var target = data[s].target();
                var output = layerOutputs[layerCount - 1][0];
                var error = target - output;
                var deltaOut = error * output * (1.0 - output);
                var deltas = new double[] { deltaOut };

                for (int l = layerCount - 1; l >= 0; l--) {
                    deltas = layers[l].backward(deltas);
                    if (l > 0) {
                        for (int j = 0; j < deltas.length; j++) {
                            var out = layerOutputs[l - 1][j];
                            deltas[j] *= out * (1.0 - out);
                        }
                    }
                }
            }
        }
    }

    public void saveModel(String filePath) throws IOException {
        LayerModelData[] model = new LayerModelData[this.layers.length];

        for (int i = 0; i < this.layers.length; i++) {
            model[i] = this.layers[i].model();
        }

        var gson = new GsonBuilder().setPrettyPrinting().create();
        Files.writeString(Path.of(filePath),
                gson.toJson(new NetworkModelData(learningRate, seed, model)));
    }

    public static Network loadModel(String filePath) throws IOException {
        var gson = new GsonBuilder().create();
        var json = Files.readString(Path.of(filePath));
        var networkData = gson.fromJson(json, NetworkModelData.class);
        return new Network(networkData);
    }

    public double accuracy(Data[] data) {
        var correct = 0;
        for (var i = 0; i < data.length; i++) {
            var pred = predict(data[i].inputs());
            if ((int) Math.round(pred) == (int) Math.round(data[i].target()))
                correct++;
        }
        return (double) correct / data.length;
    }

    @FunctionalInterface
    public interface DoubleArrayFunction<T> {
        T apply(double[] values);
    }

    public record NetworkModelData(double learningRate, long seed, LayerModelData[] layers) {
    }
}
