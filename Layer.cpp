#include "Layer.h"
#include "Neuron.h"
#include <math.h>

Layer::Layer(const Layer* previousLayer, int layerSize) : containedNeurons(layerSize, Neuron{previousLayer}){}

Layer::Layer(std::vector<Neuron> neurons) : containedNeurons{neurons}{}

Layer::~Layer()
{

}

void Layer::updateNeurons()
{
    for (int i=0;i<containedNeurons.size();i++){
        containedNeurons[i].update();
    }
}

void Layer::reassignNeuronsPreviousLayer(const Layer* previousLayer)
{
    for (int i=0;i<size();i++) {
        containedNeurons[i].reassignPreviousLayer(previousLayer);
    }
}

int Layer::size() const {return containedNeurons.size();}
