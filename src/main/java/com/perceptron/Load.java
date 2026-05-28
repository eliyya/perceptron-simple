package com.perceptron;

import java.io.IOException;

public class Load {
    public static void main(String[] args) throws IOException {
        var model = Network.loadModel(App.MODEL_FILE);
        IO.println("Modelo cargado desde: " + App.MODEL_FILE);
        IO.println("Escribe 'q' para terminar.\n");

        while (true) {
            var input = IO.readln("x1 x2: ");
            if (input.equalsIgnoreCase("q")) break;

            var parts = input.split("\\s+");
            if (parts.length < 2) {
                IO.println("Ingresa dos numeros separados por espacio.");
                continue;
            }

            try {
                var x1 = Double.parseDouble(parts[0]);
                var x2 = Double.parseDouble(parts[1]);
                var pred = model.predict(new double[]{x1, x2})[0];
                var rounded = (int) Math.round(pred);

                var color = rounded == 1 ? ConsoleColor.GREEN : ConsoleColor.RED;
                IO.print("  -> " + color + rounded + ConsoleColor.RESET + "\n");
            } catch (NumberFormatException e) {
                IO.println("Numeros invalidos.");
            }
        }
    }
}
