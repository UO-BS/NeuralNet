#include "NeuralNetwork.h"
#include <iostream>
#include <random>
#include <vector>

//This is for testing purposes only
//Potential insight: When a hidden layer is added, the inputs dont have much of an effect anymore.
int main() {

    //Perceptron Testing
    NeuralNetwork newNet{2, 1};
    newNet.addHiddenLayer(1);
    newNet.update();
    std::cout << "----------------------------------------------------------------------------";
    newNet.printToConsole();

    //Testing
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1,1);
    for (int i=0;i<300000;i++) {
        double x = distribution(generator);
        double y = distribution(generator);
        std::vector<double> temp{x,y};
        newNet.setInputNeurons(temp);
        newNet.update();
        newNet.train(std::vector<double> (1,(x*2 > y)?1.0:-1.0));
        newNet.update();
        //if (newNet.outputLayer.containedNeurons[0].findError((x*2 > y)?1.0:-1.0) > 0.5) {
            //std::cout << newNet.outputLayer.containedNeurons[0].findError((x*2 > y)?1.0:-1.0) << " " << i <<"\n";
        //}
    }

    std::cout << newNet.outputLayer.containedNeurons[0].inboundWeights[0]/newNet.outputLayer.containedNeurons[0].inboundWeights[1];

    std::vector<double> temp{0.3,0.5};
    newNet.setInputNeurons(temp);

    newNet.update();
    std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
    newNet.printToConsole();

    temp[1] = 0.7;
    newNet.setInputNeurons(temp);

    newNet.update();
    std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
    newNet.printToConsole();

    /*
    //Xor gate testing
    NeuralNetwork xorNet{2,1};
    xorNet.addHiddenLayer(2);
    xorNet.update();
    xorNet.printToConsole();

    //Testing
    int failCounter=0;

    std::mt19937 generator(std::random_device{}());
    std::uniform_int_distribution<> distribution(0,1);
    for (int i=0;i<10000;i++) {
        double x = distribution(generator);
        double y = distribution(generator);
        std::vector<double> temp{x,y};
        xorNet.setInputNeurons(temp);
        xorNet.update();
        if (xorNet.outputLayer.containedNeurons[0].findError((!!x != !!y)?1.0:-1.0) >= 0.3) {
            failCounter++;
        }
        xorNet.train(std::vector<double> (1,(!!x != !!y)?1.0:-1.0));
        xorNet.update();
    }
    std::cout<< failCounter;
    std::cout << (!!1 != !!0) << (!!0 != !!1) << (!!0 != !!0) << (!!1 != !!1);

    std::vector<double> temp{1.0,0.0};
    xorNet.setInputNeurons(temp);
    xorNet.update();
    std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
    xorNet.printToConsole();

    temp[1] = 1.0;
    temp[0] = 0.0;
    xorNet.setInputNeurons(temp);
    xorNet.update();
    std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
    xorNet.printToConsole();

    temp[1] = 0.0;
    temp[0] = 0.0;
    xorNet.setInputNeurons(temp);
    xorNet.update();
    std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
    xorNet.printToConsole();

    temp[1] = 1.0;
    temp[0] = 1.0;
    xorNet.setInputNeurons(temp);
    xorNet.update();
    std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
    xorNet.printToConsole();
    */
    

    std::cout << "\nPress Enter to exit the program ";
    std::cin.clear(); 
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get(); 

    return 0;
}