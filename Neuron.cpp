#include "Neuron.h"
#include "Layer.h"
#include <math.h>
#include <iostream>

std::random_device Neuron::rd;
std::seed_seq Neuron::seed{rd(),rd(),rd()};
std::mt19937 Neuron::mt{seed};
std::uniform_real_distribution<double> Neuron::randomGenerator(-1.0,1.0);

Neuron::Neuron() : isInputNeuron{true}, inboundWeights(0, 0), neuronValue{0}{}
Neuron::Neuron(int numberOfWeights) : isInputNeuron{(numberOfWeights==0)?true : false}, inboundWeights(numberOfWeights + (numberOfWeights==0)?0 : 1), neuronValue{0.5}{
    for (int i=0;i<inboundWeights.size();i++) {
        inboundWeights[i] = randomGenerator(mt);
    }
}
Neuron::Neuron(const Neuron& orig) : isInputNeuron{orig.isInputNeuron}, inboundWeights{orig.inboundWeights}, neuronValue{orig.neuronValue}{}
Neuron::~Neuron(){}

void Neuron::reinitializeWeights(int weightCount)
{
    inboundWeights = std::vector<double> (weightCount+ ((weightCount==0)?0 : 1));
    if (weightCount>0) {
        isInputNeuron=false;
    }
    for (int i=0;i<inboundWeights.size();i++) {
        inboundWeights[i] = randomGenerator(mt);
    }
}

void Neuron::update(const Layer& previousLayer)
{
    if (!isInputNeuron) {
        double temp{};
        temp =0;
        for (int i=0;i<previousLayer.size();i++) {
            temp += previousLayer.containedNeurons[i].neuronValue*inboundWeights[i]; //No bias yet
        }
        temp += inboundWeights[inboundWeights.size()-1]*bias;
        neuronValue = sigmoid(temp);
    }
}

double Neuron::sigmoid(double input) const
{
    return (1.0 / (1.0 + exp(-input)));
}

void Neuron::printToConsole() const
{
    if (isInputNeuron) {
        std::cout << "Input Neuron Value: " << neuronValue << "\n";
        return;
    }
    std::cout << "Neuron Value: " << neuronValue << "\n";
    std::cout << "Weights : ";
    for (int i=0;i<inboundWeights.size()-1;i++) {
        std::cout << inboundWeights[i] << " ";
    }
    std::cout << "\nBias (1.0) Weight: " << inboundWeights[inboundWeights.size()-1];
    std::cout << "\n";
    
}

double Neuron::findCostOfWeight(const Layer& previousLayer, int weightIndex, double desiredValue) const
{
    //derivative of Cost Function of 1 output with respect to 1 of the Weights
    //2(desiredvalue - currentNeuronValue) * sigmoid'(lastNeuronValueWeightedUnsigmoid) * lastNeuronValue
    double dCostdSigmoidedValue = 2*(desiredValue - neuronValue);
    
    double weightedValue{};
    for (int i=0;i<previousLayer.size();i++) {
        weightedValue += previousLayer.containedNeurons[i].neuronValue*inboundWeights[i];
    }
    weightedValue += bias*inboundWeights[inboundWeights.size()-1];
    double dSigmoiddLastWeightedValue = -sigmoidPrime(weightedValue);
    
    double dLastWeightedValuedLastValue{};
    if (weightIndex == inboundWeights.size()-1) {
        dLastWeightedValuedLastValue = previousLayer.containedNeurons[weightIndex].neuronValue;
    } else {
        dLastWeightedValuedLastValue = bias;
    }
    return  dCostdSigmoidedValue * dSigmoiddLastWeightedValue * dLastWeightedValuedLastValue;
}
double Neuron::findCostOfPrevNeuron(const Layer& previousLayer, int neuronIndex, double desiredValue) const
{
    //derivative of Cost Function of 1 output with respect to 1 of the Neurons (previous node)
    //2(desiredvalue - currentNeuronValue) * sigmoid'(lastNeuronValueWeightedUnsigmoid) * WeightLinkingTheNodes
    double dCostdSigmoidedValue = 2*(desiredValue - neuronValue);
    double weightedValue{};
    for (int i=0;i<previousLayer.size();i++) {
        weightedValue += previousLayer.containedNeurons[i].neuronValue*inboundWeights[i];
    }
    double dSigmoiddLastWeightedValue = -sigmoidPrime(weightedValue);
    double dLastWeightedValuedLastWeight = inboundWeights[neuronIndex];
    return  dCostdSigmoidedValue * dSigmoiddLastWeightedValue * dLastWeightedValuedLastWeight;
}

double Neuron::sigmoidPrime(double input) const
{
    double sigmoidTemp = sigmoid(input);
    return (sigmoidTemp*(1-sigmoidTemp));
}

void Neuron::adjustInboundWeights(const Layer& previousLayer, double desiredNeuronValue) {
    std::vector<double> neededWeightChanges(inboundWeights.size());
    //Changing a weight will change the cost function for other weights. The calculation must be seperated from the changes
    
    for (int j=0;j<inboundWeights.size();j++) {
        neededWeightChanges[j] = findCostOfWeight(previousLayer, j,desiredNeuronValue)*1.0; 
    }
    
    for (int j=0;j<inboundWeights.size();j++) {
        inboundWeights[j] -= neededWeightChanges[j]; 
    }

}

double Neuron::findError(double desiredValue) const 
{
    double temp = desiredValue - neuronValue;
    return (temp) * (temp);
}