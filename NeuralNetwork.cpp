#include "NeuralNetwork.h"
#include <math.h>

NeuralNetwork::NeuralNetwork(int inputLayerSize, int outputLayerSize) : inputLayer{std::vector<Neuron> (inputLayerSize)}, outputLayer{&inputLayer, outputLayerSize}{}
NeuralNetwork::~NeuralNetwork()
{

}

void NeuralNetwork::addHiddenLayer(int layerSize)
{
    if (hiddenLayers.size() ==0) {
        Layer newLayer = Layer(&inputLayer,layerSize);
        hiddenLayers.push_back(newLayer);
        outputLayer.reassignNeuronsPreviousLayer(&newLayer);
    } else {
        Layer newLayer = Layer(&hiddenLayers[hiddenLayers.size()-1],layerSize);
        hiddenLayers.push_back(newLayer);
        outputLayer.reassignNeuronsPreviousLayer(&newLayer);
    }
}

void NeuralNetwork::setInputNeurons(const std::vector<double>& values){
    if (values.size() != inputLayer.size()) {
        return; //THIS SHOULD RETURN AN ERROR
    }
    int index=0;
    for (Neuron node : inputLayer.containedNeurons) {
        node.neuronValue = values[index++];
    }
}

void NeuralNetwork::update()
{
    inputLayer.updateNeurons();
    for (int i=0;i<hiddenLayers.size();i++) {
        hiddenLayers[i].updateNeurons();
    }
    outputLayer.updateNeurons();
}

std::vector<double> NeuralNetwork::getOutputValues(){
    std::vector<double> values;
    values.reserve(outputLayer.size());
    for (Neuron eachNeuron : outputLayer.containedNeurons) {
        values.push_back(eachNeuron.neuronValue);
    }
    return values;
}

/*

To be added for backpropagation:

double NeuralNetwork::sigmoidPrime(double input)
{
    double sigmoidTemp = sigmoid(input);
    return (sigmoidTemp*(1-sigmoidTemp));
}

*/

//derivative of Cost Function of 1 output with respect to 1 of the Weights
//2(desiredvalue - currentNeuronValue) * sigmoid'(lastNeuronValueWeightedUnsigmoid) * lastNeuronValue

//derivative of Cost Function of 1 output with respect to 1 of the Neurons (previous node)
//2(desiredvalue - currentNeuronValue) * sigmoid'(lastNeuronValueUnsigmoid) * WeightLinkingTheNodes

//to backpropagate: find derivative of cost function with respect to each previous node
//Only Difference is now you sum the deriv for all neurons that are ahead of this neuron.
//SUM FOR ALL OUTPUT NODES: 2(desiredvalue - currentNeuronValue) * sigmoid'(lastNeuronValueUnsigmoid) * WeightLinkingTheNodes