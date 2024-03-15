#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(const std::vector<int> &layerSizes)
{
    for (int i = 0; i < layerSizes.size(); ++i)
    {
        if (i < layerSizes.size() - 1)
        {
            layers.push_back(Layer(layerSizes[i + 1], layerSizes[i]));
        }
    }
}

void NeuralNetwork::setWeights(const std::vector<std::vector<Mat> > &weights)
{
    for (int i = 0; i < layers.size(); ++i)
    {
        layers[i].setWeights(weights[i]);
    }
}

std::vector<double> NeuralNetwork::feedForward(const std::vector<double> &inputs)
{
    layerInputs.clear();
    layerOutputs.clear();
    layerInputs.push_back(Mat(inputs.size(), 1));
    for (int i = 0; i < inputs.size(); ++i)
    {
        layerInputs[0].data[i][0] = inputs[i];
    }
    layerOutputs.push_back(layerInputs[0]);
    for (int i = 0; i < layers.size(); ++i)
    {
        layerInputs.push_back(dotProduct(layers[i].getOutputs(), layerOutputs.back()));
        layerOutputs.push_back(sigmoid(layerInputs.back()));
    }
    return layerOutputs.back().data[0]; // Assuming single output neuron
}

void NeuralNetwork::backpropagate(const std::vector<double> &targets, double learningRate)
{
    // Calculate output layer delta
    Mat outputError(targets.size(), 1);
    for (int i = 0; i < targets.size(); ++i)
    {
        outputError.data[i][0] = layerOutputs.back().data[i][0] - targets[i];
    }
    Mat outputDelta = hadamardProduct(outputError, hadamardProduct(layerOutputs.back(), transpose(1 - layerOutputs.back())));

    // Update weights and biases for output layer
    Mat deltaWeightsOutput = dotProduct(outputDelta, transpose(layerOutputs[layerOutputs.size() - 2]));
    layers[layers.size() - 1].setWeights({deltaWeightsOutput});
    Mat deltaBiasesOutput = outputDelta;
    // Update biases for output layer - assuming each neuron has its own bias
    for (int i = 0; i < layers[layers.size() - 1].getOutputs().size(); ++i)
    {
        layers[layers.size() - 1].neurons[i].setBias(layers[layers.size() - 1].neurons[i].getBias() - learningRate * deltaBiasesOutput.data[i][0]);
    }

    // Calculate hidden layers delta and update weights and biases
    for (int i = layers.size() - 2; i >= 0; --i)
    {
        Mat hiddenError = dotProduct(transpose(layers[i + 1].neurons[0].getWeights()), outputDelta);
        Mat hiddenDelta = hadamardProduct(hiddenError, hadamardProduct(layerOutputs[i], transpose(1 - layerOutputs[i])));
        Mat deltaWeightsHidden = dotProduct(hiddenDelta, transpose(layerOutputs[i - 1]));
        layers[i].setWeights({deltaWeightsHidden});
        Mat deltaBiasesHidden = hiddenDelta;
        // Update biases for hidden layer - assuming each neuron has its own bias
        for (int j = 0; j < layers[i].getOutputs().size(); ++j)
        {
            layers[i].neurons[j].setBias(layers[i].neurons[j].getBias() - learningRate * deltaBiasesHidden.data[j][0]);
        }
        outputDelta = hiddenDelta;
    }
}
