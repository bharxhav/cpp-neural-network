#include "Neuron.h"

Neuron::Neuron(int numInputs)
{
    weights = Mat(1, numInputs);
    weights.randomize();
    output = 0.0;
}

void Neuron::setWeights(const Mat &weights)
{
    this->weights = weights;
}

double Neuron::feedForward(const Mat &inputs)
{
    Mat inputMat = transpose(inputs);
    Mat weightedSum = dotProduct(weights, inputMat);
    output = sigmoid(weightedSum).data[0][0]; // Assuming single output neuron
    return output;
}

double Neuron::getOutput() const
{
    return output;
}
