#ifndef LAYER_H
#define LAYER_H

#include "Neuron.h"

class Layer
{
public:
    Layer(int numNeurons, int numInputsPerNeuron);
    void setWeights(const std::vector<Mat> &weights);
    std::vector<double> feedForward(const Mat &inputs);
    std::vector<double> getOutputs() const;

private:
    std::vector<Neuron> neurons;
};

#endif
