#include "CTRNN.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <cmath>
double sigmoid(double x){
  return 1/(1 + exp(-x));
}

CTRNN::CTRNN(int size, double WR, double BR, double TR, double TA)
{
    this->size = size;
    weights.resize(size, size);
    outputs.resize(1, size);
    biases.resize(1, size);
    taus.resize(1, size);
    inputs.resize(1, size);
    weightcenters.resize(size, size);
    params.resize(1, size*size+2*size);
    params = Eigen::MatrixXd::Zero(1, size*size+2*size);
    weights = Eigen::MatrixXd::Zero(size, size);
    outputs = Eigen::MatrixXd::Zero(1, size);
    biases = Eigen::MatrixXd::Zero(1, size);
    taus= Eigen::MatrixXd::Zero(1, size);
    invTaus= Eigen::MatrixXd::Zero(1, size);
    inputs = Eigen::MatrixXd::Zero(1, size);
    weightcenters = Eigen::MatrixXd::Zero(size, size);
    this->WR = WR;
    this->BR= WR;
    this->TR= TR;
    this->TA= TA;
}

CTRNN::CTRNN()
{
    this->WR = 16.0;
    this->BR= 16.0;
    this->TR= 5.0;
    this->TA= 6.0;
}
void CTRNN::setSize(int size){this->size = size;

    weights.resize(size, size);
    outputs.resize(1, size);
    biases.resize(1, size);
    biascenters.resize(1, size);
    taus.resize(1, size);
    params.resize(1, size*size+2*size);
    inputs.resize(1, size);
    voltages.resize(1, size);
    weightcenters.resize(size, size);
    weights = Eigen::MatrixXd::Zero(size, size);
    outputs = Eigen::MatrixXd::Zero(1, size);
    biases = Eigen::MatrixXd::Zero(1, size);
    biascenters = Eigen::MatrixXd::Zero(1, size);
    taus= Eigen::MatrixXd::Zero(1, size);
    inputs = Eigen::MatrixXd::Zero(1, size);
    voltages = Eigen::MatrixXd::Zero(1,size);

    weightcenters = Eigen::MatrixXd::Zero(size, size);
}
void CTRNN::reset(){
    weights = Eigen::MatrixXd::Zero(size, size);
    outputs = Eigen::MatrixXd::Zero(1, size);
    biases = Eigen::MatrixXd::Zero(1, size);
    biascenters = Eigen::MatrixXd::Zero(1, size);
    taus= Eigen::MatrixXd::Zero(1, size);
    inputs = Eigen::MatrixXd::Zero(1, size);
    weightcenters = Eigen::MatrixXd::Zero(size, size);

}

void CTRNN::randomizeParameters(){
    weights = Eigen::MatrixXd::Random(size, size)*WR;
    weightcenters = weights;
    biases= Eigen::MatrixXd::Random(1, size)*WR;
    biascenters= biases;
    taus=Eigen::MatrixXd::Random(1, size)*TR+ Eigen::MatrixXd::Ones(1, size)*TA;
    invTaus = taus.cwiseInverse();

}
void CTRNN::setVoltages(Eigen::MatrixXd voltages){
    voltages = voltages;
}
void CTRNN::setTaus(Eigen::MatrixXd taus){
    taus = taus;
    invTaus = taus.cwiseInverse();

}
void CTRNN::setWeightCenters(Eigen::MatrixXd weightCenters){
    weightcenters = weightCenters;

}
void CTRNN::setInputs(Eigen::MatrixXd &inputs){
    this->inputs = inputs;

}

void CTRNN::initializeState(Eigen::MatrixXd v){
    voltages = v;
    invTaus = taus.cwiseInverse();
    for(int i=0; i<size;i++){
       outputs(0, i)  = sigmoid(voltages(0, i)+biases(0, i));
    }
}
void CTRNN::recoverParameters(){
    for(int i=0; i<size; i++){
        for(int j=0; j<size; j++){
            params(0, i*size+j)  = weightcenters(i, j)/WR; //return weights to original encoding [-1, 1]
        }
       params(0, size*size+i) = biascenters(0, i)/BR;
       params(0, size * size + size+i) = (taus(0, i)-TA)/TR;
    }
}

void CTRNN::setGenome(Eigen::MatrixXd genome) {
    if (genome.size()!=this->size*this->size+2*this->size){
        throw "Matrix is of wrong size!";
}
    for(int i=0;i<this->size;i++){
        for(int j=0; j<this->size; j++){
            weights(i, j) = genome(0, i*size+j)*WR;}
        biases(0, i) = genome(0, size*size+i)*BR;
        taus(0, i) = genome(0, size*size+size+i)*TR+TA;
}
    weightcenters = weights;
    biascenters = biases;
    invTaus = taus.cwiseInverse();
}

void CTRNN::EulerStep(double stepsize){
    Eigen::MatrixXd netInput(1, size);
    netInput = Eigen::MatrixXd::Zero(1, size);
    netInput = inputs + (weightcenters.transpose()*outputs.transpose()).transpose();

    voltages +=Eigen::MatrixXd::Constant(1, size, stepsize).cwiseProduct(invTaus.cwiseProduct((-voltages+ netInput)));
    for(int i =0; i<size; i++){
       outputs(0, i) = sigmoid(voltages(0,i)+biases(0, i));
}

}

void CTRNN::print(){
    std::cout<<"Weights:\n"<<weights<<std::endl;
    std::cout<<"WeightCenters:\n"<<weightcenters<<std::endl;
    std::cout<<"Biases:\n"<<biases<<std::endl;
    std::cout<<"BiasCenters:\n"<<biascenters<<std::endl;
    std::cout<<"Taus:\n"<<taus<<std::endl;
    std::cout<<"invTaus:\n"<<invTaus<<std::endl;

}
