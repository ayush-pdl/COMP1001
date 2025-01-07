#!/bin/bash


INPUT_PATH="/home/ayush-paudel/Documents/input_images/"
BLUR_OUTPUT_PATH="/home/ayush-paudel/Documents/output_images/"
EDGE_OUTPUT_PATH="/home/ayush-paudel/Documents/output_images/"

# Compile the C program
gcc -o image_processor q3b.c -lm


# Run the program with the specified paths
./image_processor "$INPUT_PATH" "$BLUR_OUTPUT_PATH" "$EDGE_OUTPUT_PATH"

