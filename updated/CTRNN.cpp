#include "CTRNN.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <cmath>
using Eigen::MatrixXd;


CTRNN::CTRNN(int size, double WR, double BR, double TR, double TA)
{
    this->size = size;
    weights.resize(size, size);
    outputs.resize(size, 1);
    biases.resize(size, 1);
    taus.resize(size, 1);
    inputs.resize(size, 1);
    weightcenters.resize(size, size);
    weights = MatrixXd::Zero(size, size);
    outputs = MatrixXd::Zero(size, 1);
    biases = MatrixXd::Zero(size, 1);
    taus= MatrixXd::Zero(size, 1);
    invTaus= MatrixXd::Zero(size, 1);
    inputs = MatrixXd::Zero(size, 1);
    weightcenters = MatrixXd::Zero(size, size);
    this->WR = WR;
    this->BR= WR;
    this->TR= TR;
    this->TA= TA;
}
void CTRNN::setSize(int size){this->size = size;

    weights.resize(size, size);
    outputs.resize(size, 1);
    biases.resize(size, 1);
    taus.resize(size, 1);
    inputs.resize(size, 1);
    weightcenters.resize(size, size);
    weights = MatrixXd::Zero(size, size);
    outputs = MatrixXd::Zero(size, 1);
    biases = MatrixXd::Zero(size, 1);
    taus= MatrixXd::Zero(size, 1);
    inputs = MatrixXd::Zero(size, 1);
    weightcenters = MatrixXd::Zero(size, size);
}
void CTRNN::reset(){
    weights = MatrixXd::Zero(size, size);
    outputs = MatrixXd::Zero(size, 1);
    biases = MatrixXd::Zero(size, 1);
    taus= MatrixXd::Zero(size, 1);
    inputs = MatrixXd::Zero(size, 1);
    weightcenters = MatrixXd::Zero(size, size);

}

void CTRNN::randomizeParameters(){
    weights = MatrixXd::Random(size, size)*WR;
    biases= MatrixXd::Random(size, 1)*WR;
    taus=MatrixXd::Random(size, 1)*TR+ MatrixXd::Ones(size,1)*TA;
    invTaus = taus.cwiseInverse();

}
void CTRNN::setVoltages(MatrixXd voltages){
    voltages = voltages;
}
void CTRNN::setTaus(MatrixXd taus){
    taus = taus;
    invTaus = taus.cwiseInverse();

}
void CTRNN::setWeightCenters(MatrixXd weightCenters){
    weightcenters = weightCenters;

}
void CTRNN::setInputs(MatrixXd inputs){
    inputs = inputs;

}

void CTRNN::initializeState(MatrixXd v){
    voltages = v;
    invTaus = taus.cwiseInverse();
    for(int i=0; i<size;i++){
       outputs(i, 0)  = sigmoid(voltages(i)+biases(0));

}

}
