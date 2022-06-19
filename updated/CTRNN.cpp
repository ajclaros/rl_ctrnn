#include "CTRNN.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <cmath>


CTRNN::CTRNN(int size, double WR, double BR, double TR, double TA)
{
    this->size = size;
    weights.resize(size, size);
    outputs.resize(size, 1);
    biases.resize(size, 1);
    taus.resize(size, 1);
    inputs.resize(size, 1);
    weightcenters.resize(size, size);
    params.resize(size*size+2*size, 1);
    params = Eigen::MatrixXd::Zero(size*size+2*size,1);
    weights = Eigen::MatrixXd::Zero(size, size);
    outputs = Eigen::MatrixXd::Zero(size, 1);
    biases = Eigen::MatrixXd::Zero(size, 1);
    taus= Eigen::MatrixXd::Zero(size, 1);
    invTaus= Eigen::MatrixXd::Zero(size, 1);
    inputs = Eigen::MatrixXd::Zero(size, 1);
    weightcenters = Eigen::MatrixXd::Zero(size, size);
    this->WR = WR;
    this->BR= WR;
    this->TR= TR;
    this->TA= TA;
}
void CTRNN::setSize(int size){this->size = size;

    weights.resize(size, size);
    outputs.resize(size, 1);
    biases.resize(size, 1);
    biascenters.resize(size, 1);
    taus.resize(size, 1);
    params.resize(size*size+2*size, 1);
    inputs.resize(size, 1);
    weightcenters.resize(size, size);
    weights = Eigen::MatrixXd::Zero(size, size);
    outputs = Eigen::MatrixXd::Zero(size, 1);
    biases = Eigen::MatrixXd::Zero(size, 1);
    biascenters = Eigen::MatrixXd::Zero(size, 1);
    taus= Eigen::MatrixXd::Zero(size, 1);
    inputs = Eigen::MatrixXd::Zero(size, 1);
    weightcenters = Eigen::MatrixXd::Zero(size, size);
}
void CTRNN::reset(){
    weights = Eigen::MatrixXd::Zero(size, size);
    outputs = Eigen::MatrixXd::Zero(size, 1);
    biases = Eigen::MatrixXd::Zero(size, 1);
    biascenters = Eigen::MatrixXd::Zero(size, 1);
    taus= Eigen::MatrixXd::Zero(size, 1);
    inputs = Eigen::MatrixXd::Zero(size, 1);
    weightcenters = Eigen::MatrixXd::Zero(size, size);

}

void CTRNN::randomizeParameters(){
    weights = Eigen::MatrixXd::Random(size, size)*WR;
    weightcenters = weights;
    biases= Eigen::MatrixXd::Random(size, 1)*WR;
    biascenters= biases;
    taus=Eigen::MatrixXd::Random(size, 1)*TR+ Eigen::MatrixXd::Ones(size,1)*TA;
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
void CTRNN::setInputs(Eigen::MatrixXd inputs){
    inputs = inputs;

}

void CTRNN::initializeState(Eigen::MatrixXd v){
    voltages = v;
    invTaus = taus.cwiseInverse();
    for(int i=0; i<size;i++){
       outputs(i, 0)  = sigmoid(voltages(i)+biases(0));


    }
}
void CTRNN::recoverParameters(){
    for(int i=0; i<size; i++){
        for(int j=0; j<size; j++){
            params(i*size+j,0)  = weightcenters(i, j)/WR; //return weights to original encoding [-1, 1]
        }
       params(size*size+i,0) = biascenters(i,0)/BR;
       params(size * size + size+i,0) = (taus(i,0)-TA)/TR;
    }
}

void CTRNN::setGenome(Eigen::MatrixXd genome) {
    if (genome.size()!=this->size*this->size+2*this->size){
        throw "Matrix is of wrong size!";


}
    for(int i=0;i<this->size;i++){
        for(int j=0; j<this->size; j++){
            weights(i, j) = genome(i*size+j, 0)*WR;}
        biases(i, 0) = genome(size*size+i, 0)*BR;
        taus(i, 0) = genome(size*size+size+i,0)*TR+TA;
}
    weightcenters = weights;
    biascenters = biases;
    invTaus = taus.cwiseInverse();
}

void CTRNN::print(){
    std::cout<<"Weights:\n"<<weights<<std::endl;
    std::cout<<"WeightCenters:\n"<<weightcenters<<std::endl;
    std::cout<<"Biases:\n"<<biases<<std::endl;
    std::cout<<"BiasCenters:\n"<<biascenters<<std::endl;
    std::cout<<"Taus:\n"<<taus<<std::endl;
    std::cout<<"invTaus:\n"<<invTaus<<std::endl;

}
