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

    //Adjust outputLayer weights
    if (hiddenLayers.size()==0) {
        outputLayer.adjustContainedNeuronWeights(inputLayer, desiredValues);
        update();
        return;    //If there are no hidden Layers, then we are only able to change the outputlayer weights
    }
    outputLayer.adjustContainedNeuronWeights(hiddenLayers[hiddenLayers.size()-1], desiredValues);

    //Now working on last Hidden layer weights
    std::vector<double> neededNeuronChanges(hiddenLayers[hiddenLayers.size()-1].size()); //Holds the desired Values of this last hidden layer
    for (int i=0;i<hiddenLayers[hiddenLayers.size()-1].size();i++) {
        neededNeuronChanges[i] = hiddenLayers[hiddenLayers.size()-1].containedNeurons[i].neuronValue + outputLayer.findCostOfPrevNeuronForLayer(hiddenLayers[hiddenLayers.size()-1], i, desiredValues);
    }
    if (hiddenLayers.size()==1) {
        hiddenLayers[hiddenLayers.size()-1].adjustContainedNeuronWeights(inputLayer, neededNeuronChanges);
        update();
        return;
    } else {
        hiddenLayers[hiddenLayers.size()-1].adjustContainedNeuronWeights(hiddenLayers[hiddenLayers.size()-2], neededNeuronChanges);
    }
    
    //Now Working on each hidden layer (other than the last)
    for (int i=hiddenLayers.size()-1;i>=0;i--) {
        std::vector<double> neededNeuronChanges(hiddenLayers[i-1].size());
        for (int j=0;j<hiddenLayers[i-1].size();j++) {
            neededNeuronChanges[j] = hiddenLayers[i-1].containedNeurons[j].neuronValue + hiddenLayers[i].findCostOfPrevNeuronForLayer(hiddenLayers[i-1],j,desiredValues);
        }
        if (hiddenLayers.size()==i+1) {
            hiddenLayers[i-1].adjustContainedNeuronWeights(inputLayer,neededNeuronChanges);
            update();
            return;
        } else {
            hiddenLayers[i-1].adjustContainedNeuronWeights(hiddenLayers[i-2],neededNeuronChanges);
        }
    }
}