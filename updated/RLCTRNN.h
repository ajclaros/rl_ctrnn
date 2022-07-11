#include "CTRNN.h"
#include "auxilary.h"
#include "queue.h"
#include <cmath>
#include <random>
#include <time.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#pragma once

class RLCTRNN:public CTRNN{

    public:
    RLCTRNN(int size, int windowSize, double initFlux, double fluxConvRate, double maxFluxAmp,
          int fluxPeriodMin, int fluxPeriodMax, bool gaussianMode,
          bool indepBiasFlux, double initBiasFlux, double biasFluxConvRate, double biasMaxFluxAmp, int biasFluxPeriodMin, int biasFluxPeriodMax);
    RLCTRNN(int size, int windowSize, double initFluxAmp, double maxFluxAmp,
          int fluxPeriodMin, int fluxPeriodMax, bool gaussianMode);
    void reset();
    queue<double> runningaverage; // = queue<double>(4000, 0);
    bool gaussianMode, indepBiasFlux;
    Eigen::MatrixXd weights, biases;
    Eigen::MatrixXd flux, innerFluxPeriods, innerFluxMoments, biasInnerFluxPeriods, biasInnerFluxMoments;
    int fluxPeriodMin, fluxPeriodMax, biasFluxPeriodMin, biasFluxPeriodMax;

    double learnrate, pastperf, reward, maxFluxAmp, biasMaxFluxAmp, initFlux, initBiasFlux, currentFlux, biasCurrentFlux,
        fluxConvRate, biasFluxConvRate;
    void initializeState(Eigen::MatrixXd v);
    void randomizeParameters(std::default_random_engine seed);
    void setLearnRate(double value){learnrate = value;}
    void updateWeightsandFluxWithReward(double reward);
   // Eigen::VectorXd distance;
    Eigen::MatrixXd calcInnerWeightsWithFlux();
    Eigen::MatrixXd calcBiasWithFlux();
    void step(std::default_random_engine, double stepsize);
    void performanceFunction();
    void rewardFunction();
};
