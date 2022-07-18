#include "NeuralNetwork.h"
#include <math.h>
#include <iostream>

NeuralNetwork::NeuralNetwork(int inputLayerSize, int outputLayerSize) : inputLayer{inputLayerSize}, outputLayer{inputLayer, outputLayerSize}{}
NeuralNetwork::~NeuralNetwork()
{

}

void NeuralNetwork::addHiddenLayer(int layerSize)
{
    if (hiddenLayers.size() ==0) {
        Layer newLayer = Layer(inputLayer,layerSize);
        hiddenLayers.push_back(newLayer);
        outputLayer.reassignNeuronsPreviousLayer(newLayer);
    } else {
        Layer newLayer = Layer(hiddenLayers[hiddenLayers.size()-1],layerSize);
        hiddenLayers.push_back(newLayer);
        outputLayer.reassignNeuronsPreviousLayer(newLayer);
    }
    //SHOULD ADD UPDATE HERE!!!
}

void NeuralNetwork::setInputNeurons(const std::vector<double>& values){
    if (values.size() != inputLayer.size()) {
        return; //THIS SHOULD RETURN AN ERROR
    }
    for (int i=0;i<inputLayer.size();i++) {
        inputLayer.containedNeurons[i].neuronValue = values[i];
    }    

}

void NeuralNetwork::update()
{
    if (hiddenLayers.size()>=1) {
        hiddenLayers[0].updateNeurons(inputLayer);
        for (int i=1;i<hiddenLayers.size();i++) {
            hiddenLayers[i].updateNeurons(hiddenLayers[i-1]);
        }
        outputLayer.updateNeurons(hiddenLayers[hiddenLayers.size()-1]);
    } else {
        outputLayer.updateNeurons(inputLayer);
    }
}

std::vector<double> NeuralNetwork::getOutputValues(){
    std::vector<double> values;
    values.reserve(outputLayer.size());
    for (int i=0;i<outputLayer.size();i++) {
        values.push_back(outputLayer.containedNeurons[i].neuronValue);
    }
    return values;
}

void NeuralNetwork::printToConsole() const
{
    std::cout << "Neural Network: \n";
    inputLayer.printToConsole();
    for (int i=0;i<hiddenLayers.size();i++) {
        hiddenLayers[i].printToConsole();
    }
    outputLayer.printToConsole();
}

void NeuralNetwork::train(std::vector<double> desiredValues)
{
    //TEMP CHANGES
    std::vector<std::vector<double>> derivativeOfCostRespectNeuron(hiddenLayers.size()+1);
    for (int i=0;i<hiddenLayers.size();i++) {
        derivativeOfCostRespectNeuron[i].resize(hiddenLayers[i].size());
    }
    derivativeOfCostRespectNeuron[hiddenLayers.size()].resize(outputLayer.size());

    //Finds cost of outputLayer
    for (int i=0;i<outputLayer.size();i++) {
        derivativeOfCostRespectNeuron[hiddenLayers.size()][i] = 2*(desiredValues[i] - outputLayer.containedNeurons[i].neuronValue);
    }

    //Finds cost of hidden layers
    for (int i=hiddenLayers.size()-1;i>=0;i--) {
        for (int k=0;k<hiddenLayers[i].size();k++) {
            derivativeOfCostRespectNeuron[i][k] = outputLayer.findCostOfPrevNeuronForLayer(hiddenLayers[i],k,derivativeOfCostRespectNeuron[i+1]);
        }
    }

    //adjusting weights
    for (int i=0;i<hiddenLayers.size();i++) {
        if (i==0) {
            hiddenLayers[i].adjustContainedNeuronWeights(inputLayer, derivativeOfCostRespectNeuron[i]);
        } else {
            hiddenLayers[i].adjustContainedNeuronWeights(hiddenLayers[i-1], derivativeOfCostRespectNeuron[i]);
        }
    }
    if (hiddenLayers.size() == 0) {
        outputLayer.adjustContainedNeuronWeights(inputLayer, derivativeOfCostRespectNeuron[0]);
    } else {
        outputLayer.adjustContainedNeuronWeights(hiddenLayers[hiddenLayers.size()-1], derivativeOfCostRespectNeuron[derivativeOfCostRespectNeuron.size()-1]);
    }
    
    update();
    return;
}