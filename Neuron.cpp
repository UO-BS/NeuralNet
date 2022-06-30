#include "Neuron.h"
#include "Layer.h"
#include <math.h>

Neuron::Neuron() : isInputNeuron{true}, previousLayer{nullptr}, inboundWeights(0, 0), neuronValue{0}{}
Neuron::Neuron(const Layer* prevLayer) : isInputNeuron{false}, previousLayer{prevLayer}, inboundWeights(previousLayer->size(), 0), neuronValue{0.5}{}
Neuron::Neuron(const Neuron& orig) : isInputNeuron{orig.isInputNeuron}, previousLayer{orig.previousLayer}, inboundWeights{orig.inboundWeights}, neuronValue{orig.neuronValue}{}
Neuron::~Neuron(){}

void Neuron::reassignPreviousLayer(const Layer* newLayer)
{
    previousLayer = newLayer;
}

void Neuron::update()
{
    if (!isInputNeuron) {
        double temp{};
        temp =0;
        for (int i=0;i<previousLayer->size();i++) {
            temp += (*previousLayer).containedNeurons[i].neuronValue*inboundWeights[i]; //No bias yet
        }
        neuronValue = sigmoid(temp);
    }
}

double Neuron::sigmoid(double input)
{
    return (1.0 / (1.0 + exp(-input)));
}