#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Layer.h"

class NeuralNetwork
{
public:
    NeuralNetwork(const std::vector<int> &layerSizes);
    void setWeights(const std::vector<std::vector<Mat> > &weights);
    std::vector<double> feedForward(const std::vector<double> &inputs);
    void backpropagate(const std::vector<double> &targets, double learningRate);

private:
    std::vector<Layer> layers;
    std::vector<Mat> layerInputs;
    std::vector<Mat> layerOutputs;
    std::vector<Mat> layerDeltas;
};

#endif
