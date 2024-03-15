#ifndef NEURON_H
#define NEURON_H

#include "UtilityFunctions.h"

class Neuron
{
public:
    Neuron(int numInputs);
    void setWeights(const Mat &weights);
    double feedForward(const Mat &inputs);
    double getOutput() const;

private:
    Mat weights;
    double output;
};

#endif
