#include "NeuralNetwork.h"
#include <iostream>
#include <random>
#include <vector>

//This is for testing purposes only
//MAJOR PROBLEMS: CHANGING -20>20 to -200>200 and changing anything to one sided data ex:1>20
//Potential insight: When a hidden layer is added, it sets 1 neuron to 1 and the other to 0, then mixes. Ignores inputs
int main() {

    NeuralNetwork newNet{2, 1};
    newNet.addHiddenLayer(2);
    newNet.update();
    std::cout << "----------------------------------------------------------------------------";
    newNet.printToConsole();

    //Testing y=2x 
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(1,20);
    for (int i=0;i<50000;i++) {
        double x = distribution(generator);
        double y = distribution(generator);
        std::vector<double> temp{x,y};
        newNet.setInputNeurons(temp);
        newNet.update();
        newNet.train(std::vector<double> (1,(x*2 > y)?1.0:0.0));
        newNet.update();
        if (newNet.outputLayer.containedNeurons[0].findError((x*2 > y)?1.0:0.0) > 0.5) {
            std::cout << newNet.outputLayer.containedNeurons[0].findError((x*2 > y)?1.0:0.0) << " " << i <<"\n";
        }
        
    }

    std::vector<double> temp{5,9};
    newNet.setInputNeurons(temp);

    newNet.update();
    std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
    newNet.printToConsole();

    temp[1] = 11;
    newNet.setInputNeurons(temp);

    newNet.update();
    std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
    newNet.printToConsole();

    std::cout << "Press Enter to exit the program ";
    std::cin.clear(); 
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get(); 

    return 0;
}