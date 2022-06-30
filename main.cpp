#include "NeuralNetwork.h"
#include <iostream>

//This is for testing purposes only
int main() {

    NeuralNetwork newNet{10, 1};
    newNet.update();
    std::vector<double> output = newNet.getOutputValues();
    for (double value : output) {
        std::cout << value;
    }

    newNet.addHiddenLayer(5);
    newNet.update();
    output = newNet.getOutputValues();
    for (double value : output) {
        std::cout << value;
    }

    return 0;
}