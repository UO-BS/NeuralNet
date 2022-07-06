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
    //POTENTIAL PROBLEMS: update() or no update() between weight changes?
    update();

    std::vector<double> derivativeOfCostRespectOutputNeurons(outputLayer.size());
    for (int i=0;i<outputLayer.size();i++) {
        derivativeOfCostRespectOutputNeurons[i] = 2*(desiredValues[i] - outputLayer.containedNeurons[i].neuronValue);
    }

    //Adjust outputLayer weights
    if (hiddenLayers.size()==0) {
        outputLayer.adjustContainedNeuronWeights(inputLayer, derivativeOfCostRespectOutputNeurons);
        update();
        return;    //If there are no hidden Layers, then we are only able to change the outputlayer weights
    }
    outputLayer.adjustContainedNeuronWeights(hiddenLayers[hiddenLayers.size()-1], derivativeOfCostRespectOutputNeurons);

    //Now working on last Hidden layer weights
    std::vector<double> derivativeOfCostRespectNeuron(hiddenLayers[hiddenLayers.size()-1].size());
    for (int i=0;i<hiddenLayers[hiddenLayers.size()-1].size();i++) {
        derivativeOfCostRespectNeuron[i] = outputLayer.findCostOfPrevNeuronForLayer(hiddenLayers[hiddenLayers.size()-1],i,derivativeOfCostRespectOutputNeurons);
    }
    if (hiddenLayers.size()-1 == 0) {
        hiddenLayers[hiddenLayers.size()-1].adjustContainedNeuronWeights(inputLayer,derivativeOfCostRespectNeuron);
        update();
        return;
    } else {
        hiddenLayers[hiddenLayers.size()-1].adjustContainedNeuronWeights(hiddenLayers[hiddenLayers.size()-2],derivativeOfCostRespectNeuron);
    }

    //Now Working on each hidden layer (other than the last)
    std::vector<double> prevDerivativeOfCostRespectNeuron = derivativeOfCostRespectNeuron;
    for (int i=hiddenLayers.size()-1;i>=0;i--) {
        derivativeOfCostRespectNeuron.resize(hiddenLayers[i-1].size());
        for (int j=0;j<hiddenLayers[i-1].size();j++) {
            derivativeOfCostRespectNeuron[j] = hiddenLayers[i].findCostOfPrevNeuronForLayer(hiddenLayers[i-1],j,prevDerivativeOfCostRespectNeuron);
        }
        if (hiddenLayers.size()==i+1) {
            hiddenLayers[i-1].adjustContainedNeuronWeights(inputLayer,derivativeOfCostRespectNeuron);
            update();
            return;
        } else {
            hiddenLayers[i-1].adjustContainedNeuronWeights(hiddenLayers[i-2],derivativeOfCostRespectNeuron);
        }
        std::vector<double> prevDerivativeOfCostRespectNeuron = derivativeOfCostRespectNeuron;
    }
}