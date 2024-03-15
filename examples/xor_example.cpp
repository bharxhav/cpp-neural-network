#include <iostream>
#include "../include/NeuralNetwork.h"

int main() {
    // Define the XOR input and corresponding labels
    std::vector<std::vector<double> > inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double> > labels = {{0}, {1}, {1}, {0}};

    // Define the neural network architecture
    std::vector<int> layerSizes = {2, 2, 1}; // Input layer: 2 neurons, 1 hidden layer with 2 neurons, output layer: 1 neuron

    // Create the neural network
    NeuralNetwork neuralNetwork(layerSizes);

    // Train the neural network using backpropagation
    size_t numEpochs = 10000;
    double learningRate = 0.1;
    for (size_t epoch = 0; epoch < numEpochs; ++epoch) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            // Forward pass
            std::vector<double> output = neuralNetwork.feedForward(inputs[i]);

            // Backpropagation
            neuralNetwork.backpropagate(labels[i], learningRate);
        }
        std::cout << "Epoch " << epoch << " completed" << std::endl;
    }

    // Test the neural network on XOR inputs
    std::cout << "XOR predictions:" << std::endl;
    for (size_t i = 0; i < inputs.size(); ++i) {
        std::vector<double> output = neuralNetwork.feedForward(inputs[i]);
        std::cout << inputs[i][0] << " XOR " << inputs[i][1] << " = " << output[0] << std::endl;
    }

    return 0;
}
