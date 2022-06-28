#include "RLCTRNN.h"
#include "CTRNN.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <time.h>
#include <iostream>
#include <eigen3/Eigen/Dense>

const long double PI =  3.141592653589793;

RLCTRNN::RLCTRNN(int size, int windowSize, double initFlux, double fluxConvRate, double maxFluxAmp,
                 int fluxPeriodMin, int fluxPeriodMax, bool gaussianMode,
                 bool indepBiasFlux, double initBiasFlux =-1, double biasFluxConvRate=-1, double biasMaxFluxAmp=-1, int biasFluxPeriodMin=-1, int biasFluxPeriodMax=-1){

    CTRNN(size, WR=16, BR=16, TR=5, TA=6);
    weights.resize(1,size);
    weights = weightcenters;
    biases.resize(1,size);
    biases = biascenters;
    runningaverage.setSize(windowSize, 0);
    this->currentFlux= initFlux;
    this->initFlux = initFlux;
    this->fluxConvRate = fluxConvRate;
    this->maxFluxAmp = maxFluxAmp;
    this->fluxPeriodMin = fluxPeriodMin;
    this->fluxPeriodMax = fluxPeriodMax;
    this->gaussianMode = gaussianMode;
    this->indepBiasFlux= indepBiasFlux;

    if(indepBiasFlux){
        this->initBiasFlux = initBiasFlux;
        this->biasFluxConvRate = biasFluxConvRate;
        this->biasCurrentFlux= initBiasFlux;
        this->biasMaxFluxAmp = biasMaxFluxAmp;
        this->biasFluxPeriodMin = biasFluxPeriodMin;
        this->biasFluxPeriodMax = biasFluxPeriodMax;
    }
    else{
        this->initBiasFlux = initFlux;
        this->biasFluxConvRate = fluxConvRate;
        this->biasCurrentFlux =initFlux;
        this->biasFluxPeriodMin = fluxPeriodMin;
        this->biasFluxPeriodMax = fluxPeriodMax;
    }
    innerFluxPeriods.resize(size,size);
    innerFluxPeriods = Eigen::MatrixXd::Zero(size, size);
    innerFluxMoments.resize(size, size);
    innerFluxMoments = Eigen::MatrixXd::Zero(size, size);

    biasInnerFluxPeriods.resize(1, size);
    biasInnerFluxPeriods = Eigen::MatrixXd::Zero(1, size);
    biasInnerFluxMoments.resize(1, size);
    biasInnerFluxMoments = Eigen::MatrixXd::Zero(1, size);
}

RLCTRNN::RLCTRNN(int size, int windowSize, double initFluxAmp, double maxFluxAmp,
                 int fluxPeriodMin, int fluxPeriodMax, bool gaussianMode){
    CTRNN(size, WR=16, BR=16, TR=5, TA=6);
    weights.resize(size,size);
    weights = weightcenters;
    biases.resize(1,size);
    biases = biascenters;
    flux.resize(size, size);
    innerFluxPeriods.resize(size, size);
    innerFluxPeriods = Eigen::MatrixXd::Zero(size, size);
    innerFluxMoments.resize(size, size);
    innerFluxMoments = Eigen::MatrixXd::Zero(size,size);
    runningaverage.setSize(windowSize, 0);
    this->currentFlux= initFlux;
    this->biasCurrentFlux= initFlux;
    this->maxFluxAmp= maxFluxAmp;
    this->fluxPeriodMin = fluxPeriodMin;
    this->fluxPeriodMax = fluxPeriodMax;
    this->gaussianMode = gaussianMode;
    this->indepBiasFlux= false;
    this->initBiasFlux = initFlux;
}
void RLCTRNN::reset(){
    currentFlux = initFlux;
    innerFluxMoments = Eigen::MatrixXd::Zero(size, size);
    innerFluxPeriods= Eigen::MatrixXd::Zero(size, size);
    innerFluxPeriods.resize(1, size);
    innerFluxPeriods = Eigen::MatrixXd::Zero(1, size);
    innerFluxMoments.resize(1, size);
    innerFluxMoments = Eigen::MatrixXd::Zero(1, size);
    CTRNN::reset();
}
void RLCTRNN::initializeState(Eigen::MatrixXd v){
    currentFlux = initFlux;
    if(indepBiasFlux){
        biasCurrentFlux= initBiasFlux;
}
    else{
        biasCurrentFlux= initFlux;
}
    CTRNN::initializeState(v);
}

void RLCTRNN::randomizeParameters(std::default_random_engine seed){
    if(gaussianMode){
        double center = (fluxPeriodMax+fluxPeriodMin)/2;
        double standDev = (fluxPeriodMax-fluxPeriodMin)/4;
        std::normal_distribution<double> normal(center, standDev);

        for(int i=0;i<weights.rows(); i++){
            for(int j=0; j<weights.cols(); j++){
                innerFluxPeriods(i,j) = normal(seed);
}
}
        roundnPlacesMatrix(innerFluxPeriods, 3);
        clip(innerFluxPeriods, fluxPeriodMin, fluxPeriodMax);
        for(int j=0; j<biasInnerFluxPeriods.cols();j++){
            biasInnerFluxPeriods(0,j) = normal(seed);
}
        roundnPlacesMatrix(biasInnerFluxMoments, 3);
        clip(biasInnerFluxPeriods, biasFluxPeriodMin, biasFluxPeriodMax);
}
    else{
        std::uniform_real_distribution<double> uniform(fluxPeriodMin, fluxPeriodMax);
        std::uniform_real_distribution<double> biasUniform(fluxPeriodMin, fluxPeriodMax);
        for(int i=0; i<innerFluxPeriods.rows();i++){
            for(int j=0; j<innerFluxPeriods.cols();j++){
                innerFluxPeriods(i,j) = uniform(seed);
}
}
        roundnPlacesMatrix(innerFluxPeriods, 3);
        clip(innerFluxPeriods, fluxPeriodMin, fluxPeriodMax);
        for(int j=0; j<biasInnerFluxPeriods.cols(); j++){
            biasInnerFluxPeriods(0,j) = biasUniform(seed);
}
        roundnPlacesMatrix(biasInnerFluxPeriods, 3);
        clip(biasInnerFluxPeriods, biasFluxPeriodMin, biasFluxPeriodMax);
}
    CTRNN::randomizeParameters();
    
}
void RLCTRNN::updateWeightsandFluxWithReward(double reward){
    currentFlux -= fluxConvRate * reward;
    currentFlux = min((max(currentFlux, 0)), maxFluxAmp);
    biasCurrentFlux -= biasFluxConvRate * reward;
    biasCurrentFlux= min((max(biasCurrentFlux, 0)), biasMaxFluxAmp);
    Eigen::MatrixXd innerFluxCenterDisplacements(size,size);
    for(int i=0; i<innerFluxCenterDisplacements.rows(); i++){
        for(int j=0; j<innerFluxCenterDisplacements.cols(); j++){
            innerFluxCenterDisplacements(i,j) = currentFlux*sin(innerFluxMoments(i,j)/innerFluxPeriods(i,j)*2*PI);
}
}
    Eigen::MatrixXd biasInnerFluxCenterDisplacements(size,size);
    for(int j=0; j<biasInnerFluxCenterDisplacements.cols();j++ ){
        biasInnerFluxCenterDisplacements(0, j) = biasCurrentFlux*sin(biasInnerFluxMoments(0,j)/biasInnerFluxPeriods(0,j)*2*PI);
}
    weights = weightcenters + innerFluxCenterDisplacements;
    weightcenters = weightcenters + learnrate*reward*innerFluxCenterDisplacements;
    clip(weightcenters, -WR, WR);
    biases = biascenters + biasInnerFluxCenterDisplacements;
    biascenters = biascenters + learnrate*reward*biasInnerFluxCenterDisplacements;
}

void RLCTRNN::calcInnerWeightsWithFlux(){
   Eigen::MatrixXd temp(size, size);
   for(int i=0; i<temp.rows(); i++){
       for(int j=0; j<temp.cols(); j++){
           temp(i,j) = weightcenters(i,j)+ currentFlux*sin(innerFluxMoments(i,j)*2*PI);
}
}
   std::cout<<temp.transpose()<<std::endl;
}