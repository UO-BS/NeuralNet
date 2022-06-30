#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include "Neuron.h"

class Neuron;

class Layer
{
private:

    //Helper sigmoid function
    double sigmoid(double input);
public:
    Layer() = delete;
    Layer(std::vector<Neuron> neurons);
    Layer(const Layer* previousLayer, int layerSize);
    ~Layer();

    std::vector<Neuron> containedNeurons;

    void reassignNeuronsPreviousLayer(const Layer* previousLayer);

    void updateNeurons();

    int size() const;

};

#endif