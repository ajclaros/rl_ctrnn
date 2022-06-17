#include <eigen3/Eigen/Dense>
#include <iostream>
#include "queue.h"
#include <cmath>
#pragma once
const double Pi = 3.1415926;
// The sigmoid function
#ifdef FAST_SIGMOID
const int SigTabSize = 400;
const double SigTabRange = 15.0;
double fastsigmoid(double x);
#endif
inline double sigma(double x)
{
  return 1/(1 + exp(-x));
}
inline double sigmoid(double x)
{
#ifndef FAST_SIGMOID
  return sigma(x);
#else
  return fastsigmoid(x);
#endif
}
// The inverse sigmoid function
inline double InverseSigmoid(double y)
{
  return log(y/(1-y));
}
// The CTRNN class declaration
class CTRNN {
    public:
        int size;
        // The constructor
        CTRNN(int size, double WR=16.0, double BR=16.0, double TR=5.0, double TA=6.0);
        void setSize(int size);
        void reset();
        void randomizeParameters();
        void setVoltages(Eigen::MatrixXd voltages);
        void setTaus(Eigen::MatrixXd taus);
        void setInputs(Eigen::MatrixXd inputs);
        void setWeightCenters(Eigen::MatrixXd weightCenters);
        void initializeState(Eigen::MatrixXd v);
        // The destructor
        queue<double> runningaverage = queue<double>(4000, 0);
        // // Accessors
        double weightRange;
        double biasRange;
        double timeRange[2] = {0.0,0.0};
        Eigen::MatrixXd weights, voltages, outputs, biases, gains, taus, invTaus, inputs; //, Rtaus;
        //Matrix<double, Eigen::Dynamic TempStates,TempOutputs,k1,k2,k3,k4;
        Eigen::MatrixXd weightcenters;
        Eigen::MatrixXd biascenters;
        Eigen::MatrixXd flux ;
        Eigen::MatrixXd period,periodcount;
        Eigen::MatrixXd params;
        double WR, BR, TR, TA;
        double amp,meanPeriod,stdPeriod, convergence, initamp;
        int fluxPeriodMin, fluxPeriodMax;
        double learnrate;
        int fluxsize;
        double pastperf, reward;
        double max_flux_amp = 40.0;
        double ampGain;
        void recoverParameters();
        int getCircuitSize(void) {return size;};
        //double getNeuronOutput(int i) {return outputs[i];};
        //double getNeuronBias(int i) {return biases[i];};
        //void setNeuronBias(int i, double value) {biases[i] = value; biascenters[i] = value;};
        //double getNeuronTimeConstant(int i) {return taus[i];};
        //void setNeuronTimeConstant(int i, double value) {taus[i] = value; invTaus[i] = 1/value;};
        //void setWeight(int from, int to, double value) {weights[from][to] = value; weightcenters[from][to] = value;};
        //double getWeight(int from, int to) {return weights[from][to];}; //connection from N_i to N_j
        void setLearnRate(double value){learnrate=value;};
        //void Flux(RandomState &rs);
        void Learn(double performance);
        void EulerStep(double stepsize);
        //double getNeuronState(int i) {return states[i];};
        //double &getNeuronStateReference(int i) {return states[i];};
        //void setNeuronState(int i, double value){states[i] = value;outputs[i] = sigmoid(gains[i]*(states[i] + biases[i]));};
        //double &getNeuronOutputReference(int i) {return outputs[i];};
        //double getNeuronGain(int i) {return gains[i];};
        //void setNeuronGain(int i, double value) {gains[i] = value;};
        //double getNeuronExternalInput(int i) {return externalinputs[i];};
        //double &getNeuronExternalInputReference(int i) {return externalinputs[i];};
        //void setNeuronExternalInput(int i, double value) {externalinputs[i] = value;};
        //void setPeriod(double meanvalue, double stdvalue){meanPeriod=meanvalue;stdPeriod=stdvalue;};
        //void RandomizeCircuitState(double lb, double ub);
        //void RandomizeCircuitState(double lb, double ub, RandomState &rs);
        //void RandomizeCircuitOutput(double lb, double ub);
        //void RandomizeCircuitOutput(double lb, double ub, RandomState &rs);
        // Matrix<double, Eigen::Dynamic TempStates,TempOutputs,k1,k2,k3,k4;
        // // NEW FOR RL
};
