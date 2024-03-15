#include "Layer.h"

Layer::Layer(int numNeurons, int numInputsPerNeuron)
{
    neurons.resize(numNeurons, Neuron(numInputsPerNeuron));
}

void Layer::setWeights(const std::vector<Mat> &weights)
{
    for (int i = 0; i < neurons.size(); ++i)
    {
        neurons[i].setWeights(weights[i]);
    }
}

std::vector<double> Layer::feedForward(const Mat &inputs)
{
    std::vector<double> outputs;
    for (const Neuron &neuron : neurons)
    {
        outputs.push_back(neuron.feedForward(inputs));
    }
    return outputs;
}

std::vector<double> Layer::getOutputs() const
{
    std::vector<double> outputs;
    for (const Neuron &neuron : neurons)
    {
        outputs.push_back(neuron.getOutput());
    }
    return outputs;
}
