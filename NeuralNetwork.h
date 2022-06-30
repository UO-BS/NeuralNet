#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "Layer.h"
#include <vector>

class NeuralNetwork
{
private:

    Layer inputLayer;
    std::vector<Layer> hiddenLayers;
    Layer outputLayer;

public:

    NeuralNetwork() = delete;
    NeuralNetwork(int inputLayerSize, int outputLayerSize);
    ~NeuralNetwork();

    void addHiddenLayer(int layerSize);

    void setInputNeurons(const std::vector<double>& values);
    void update();
    std::vector<double> getOutputValues();
};

#endif