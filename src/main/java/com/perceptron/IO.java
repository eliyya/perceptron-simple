package com.perceptron;

import java.util.Scanner;

public class IO {
    private static final Scanner scanner = new Scanner(System.in);

    public static void print(String msg) {
        System.out.print(msg);
    }

    public static void println(String msg) {
        System.out.println(msg);
    }

    public static String readln(String prompt) {
        System.out.print(prompt);
        return scanner.nextLine().trim();
    }
}
