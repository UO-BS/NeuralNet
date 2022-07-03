#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include "Neuron.h"

class Neuron;

class Layer
{
private:

    //Helper sigmoid function
    double sigmoidPrime(double input) const;
    double sigmoid(double input) const;

public:
    Layer() = delete;
    Layer(int layerSize); //For input layer
    Layer(const Layer& previousLayer, int layerSize);
    Layer(std::vector<Neuron> neurons);
    ~Layer();

    std::vector<Neuron> containedNeurons;

    void reassignNeuronsPreviousLayer(const Layer& previousLayer);

    void updateNeurons(const Layer& previousLayer);

    int size() const;
    void printToConsole() const;

    double findCostOfPrevNeuronForLayer(const Layer& previousLayer, int neuronIndex, std::vector<double> desiredValues) const;

    void adjustContainedNeuronWeights(const Layer& previousLayer, std::vector<double> desiredValues);
};

#endif