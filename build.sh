#!/bin/bash

# Set the directory paths
SRC_DIR="src"
INCLUDE_DIR="include"
EXAMPLES_DIR="examples"
BUILD_DIR="build"

# Create build directory if it doesn't exist
mkdir -p "$BUILD_DIR"

# Compile all source files to object files
g++ -c "$SRC_DIR/NeuralNetwork.cpp" -o "$BUILD_DIR/NeuralNetwork.o" -I"$INCLUDE_DIR"
g++ -c "$SRC_DIR/Layer.cpp" -o "$BUILD_DIR/Layer.o" -I"$INCLUDE_DIR"
g++ -c "$SRC_DIR/Neuron.cpp" -o "$BUILD_DIR/Neuron.o" -I"$INCLUDE_DIR"
g++ -c "$SRC_DIR/ActivationFunctions.cpp" -o "$BUILD_DIR/ActivationFunctions.o" -I"$INCLUDE_DIR"
g++ -c "$SRC_DIR/UtilityFunctions.cpp" -o "$BUILD_DIR/UtilityFunctions.o" -I"$INCLUDE_DIR"
g++ -c "$EXAMPLES_DIR/xor_example.cpp" -o "$BUILD_DIR/xor_example.o" -I"$INCLUDE_DIR"

# Link all object files together to create the executable
g++ "$BUILD_DIR/NeuralNetwork.o" "$BUILD_DIR/Layer.o" "$BUILD_DIR/Neuron.o" "$BUILD_DIR/ActivationFunctions.o" "$BUILD_DIR/UtilityFunctions.o" "$BUILD_DIR/xor_example.o" -o "$BUILD_DIR/xor_example"

# Optionally, remove object files after linking
rm -rf "$BUILD_DIR"/*.o
