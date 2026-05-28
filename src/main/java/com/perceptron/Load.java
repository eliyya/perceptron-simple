package com.perceptron;

import java.io.IOException;

public class Load {
    public static void main(String[] args) throws IOException {
        var model = Neuron.loadModel(App.MODEL_FILE);
        IO.println("Modelo cargado desde: " + App.MODEL_FILE);
        IO.println("Escribe 'q' para terminar.\n");

        while (true) {
            var input = IO.readln("x1,x2: ");
            if (input.equalsIgnoreCase("q")) break;

            var parts = input.replaceAll(" ", "").split(",");
            if (parts.length < 2) {
                IO.println("Ingresa dos numeros separados por espacio.");
                continue;
            }

            try {
                var x1 = Double.parseDouble(parts[0]);
                var x2 = Double.parseDouble(parts[1]);
                var pred = model.predict(new double[]{x1, x2});

                var color = pred == 1 ? ConsoleColor.GREEN : ConsoleColor.RED;
                IO.print("  -> " + color + pred + ConsoleColor.RESET + "\n");
            } catch (NumberFormatException e) {
                IO.println("Numeros invalidos.");
            }
        }
    }
}
