#include "NeuralNetwork.h"
#include <iostream>
#include <random>
#include <vector>

//This is for testing purposes only

int main() {
    
    //Perceptron Testing ----------------------------------------------------------------------------------------------
    NeuralNetwork newNet{2, 1,};
    newNet.addHiddenLayer(3);
    newNet.update();
    std::cout << "--------------------------------------------------------------------";
    newNet.printToConsole();

    //Testing
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1,1);
    for (int i=0;i<10000;i++) {

        int batchSize =10;
        std::vector<std::vector<double>> inputs;
        std::vector<std::vector<double>> outputs;
        for (int batch=0;batch<batchSize;batch++) {
            double x = distribution(generator);
            double y = distribution(generator);
            inputs.push_back({x,y});
            outputs.push_back({(3*x > y)?1.0:-1.0});
        }
        newNet.trainFromInputSet(inputs,outputs);
        //std::cout << newNet.averageErrorOnSet(inputs,outputs) << "\n";

    }

    std::vector<double> temp{0.2,0.7};
    newNet.setInputNeurons(temp);

    newNet.update();
    std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
    newNet.printToConsole();

    temp[1] = 0.50;
    newNet.setInputNeurons(temp);

    newNet.update();
    std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
    newNet.printToConsole();
    

    /*
    //Gate testing ------------------------------------------------------------------------------------------------------
    NeuralNetwork xorNet{2,1};
    xorNet.addHiddenLayer(2);
    xorNet.printToConsole();
        
    std::mt19937 generator(std::random_device{}());
    std::uniform_int_distribution<> distribution(0,1);
    for (int i=0;i<50000;i++) {
        int batchSize =10;
        std::vector<std::vector<double>> inputs;
        std::vector<std::vector<double>> outputs;
        for (int batch=0;batch<batchSize;batch++) {
            double x = distribution(generator);
            double y = distribution(generator);
            inputs.push_back({x,y});
            outputs.push_back({(!!x != !!y)?1.0:-1.0});
        }
        xorNet.trainFromInputSet(inputs,outputs);
        std::cout << xorNet.averageErrorOnSet(inputs,outputs) << "\n";
    }
    

    std::vector<double> temp{1.0,0.0};
    xorNet.setInputNeurons(temp);
    xorNet.update();
    std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
    xorNet.printToConsole();

    temp[1] = 1.0;
    temp[0] = 0.0;
    xorNet.setInputNeurons(temp);
    xorNet.update();
    std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
    xorNet.printToConsole();

    temp[1] = 0.0;
    temp[0] = 0.0;
    xorNet.setInputNeurons(temp);
    xorNet.update();
    std::cout << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
    xorNet.printToConsole();

    temp[1] = 1.0;
    temp[0] = 1.0;
    xorNet.setInputNeurons(temp);
    xorNet.update();
    std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
    xorNet.printToConsole();
    */
    

    std::cout << "\nPress Enter to exit the program ";
    std::cin.clear(); 
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::cin.get(); 

    return 0;
}