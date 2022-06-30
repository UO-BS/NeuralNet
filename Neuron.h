#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include "Layer.h"

class Layer;

class Neuron
{
private:
    double sigmoid(double input);
public:
    Neuron();
    Neuron(const Layer* prevLayer);
    Neuron(const Neuron& orig);
    ~Neuron();

    void reassignPreviousLayer(const Layer* newLayer);

    void update();

    bool isInputNeuron;
    const Layer* previousLayer;
    std::vector<double> inboundWeights;
    double neuronValue;
};

#endif