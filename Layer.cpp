#include "Layer.h"
#include "Neuron.h"
#include <iostream>
#include <math.h>

Layer::Layer(int layerSize) : containedNeurons(layerSize){}

Layer::Layer(const Layer& previousLayer, int layerSize) : containedNeurons(layerSize)
{
    //using an allocator with std::vector calls copy constructor during initilization which does not randomize neuron weights
    for (int i=0;i<containedNeurons.size();i++) {
        containedNeurons[i].reinitializeWeights(previousLayer.size());
    }
}

Layer::Layer(std::vector<Neuron> neurons) : containedNeurons{neurons}{}

Layer::~Layer()
{

}

void Layer::updateNeurons(const Layer& previousLayer)
{
    for (int i=0;i<containedNeurons.size();i++){
        containedNeurons[i].update(previousLayer); 
    }
}

void Layer::reassignNeuronsPreviousLayer(const Layer& previousLayer)
{
    for (int i=0;i<size();i++) {
        containedNeurons[i].reinitializeWeights(previousLayer.size());
    }
}

int Layer::size() const {return containedNeurons.size();}

void Layer::printToConsole() const
{
    std::cout << "Layer:\n";
    for (int i=0;i<containedNeurons.size();i++) {
        containedNeurons[i].printToConsole();
    }
}


double Layer::findCostOfPrevNeuronForLayer(const Layer& previousLayer, int neuronIndex, std::vector<double> desiredValues) const
{
    //to backpropagate: find derivative of cost function with respect to each previous node
    //Only Difference is now you sum the deriv for all neurons that are ahead of this neuron.
    //SUM FOR ALL OUTPUT NODES: 2(desiredvalue - currentNeuronValue) * sigmoid'(lastNeuronValueUnsigmoid) * WeightLinkingTheNodes
    double totalCost{0.0};
    for (int i=0;i<containedNeurons.size();i++) {
        totalCost += containedNeurons[i].findCostOfPrevNeuron(previousLayer, neuronIndex , desiredValues[i]);
    }
    return totalCost;
}

void Layer::adjustContainedNeuronWeights(const Layer& previousLayer, std::vector<double> desiredValues) {
    for (int i=0;i<containedNeurons.size();i++) {
        containedNeurons[i].adjustInboundWeights(previousLayer, desiredValues[i]);
    }
}