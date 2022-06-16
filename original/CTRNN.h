// ***********************************************************
// A class for continuous-time recurrent neural networks
//
// RDB
//  8/94 Created
//  12/98 Optimized integration
//  1/08 Added table-based fast sigmoid w/ linear interpolation
// ************************************************************

// Uncomment the following line for table-based fast sigmoid w/ linear interpolation
//#define FAST_SIGMOID

#include "VectorMatrix.h"
#include "random.h"
#include <iostream>
#include <math.h>
#include "queue.h"
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
        // The constructor
        CTRNN(int newsize = 0);
        // The destructor
        ~CTRNN();

                queue<double> runningaverage = queue<double>(4000, 0);
        void recoverParameters();
        // Accessors
        int CircuitSize(void) {return size;};
        double weightRange;
        double biasRange;
        double timeRange[2] = {0.0,0.0};
        void SetCircuitSize(int newsize);

        double NeuronState(int i) {return states[i];};
        double &NeuronStateReference(int i) {return states[i];};
        void SetNeuronState(int i, double value)
            {states[i] = value;outputs[i] = sigmoid(gains[i]*(states[i] + biases[i]));};
        double NeuronOutput(int i) {return outputs[i];};
        double &NeuronOutputReference(int i) {return outputs[i];};
        void SetNeuronOutput(int i, double value)
            {outputs[i] = value; states[i] = InverseSigmoid(value)/gains[i] - biases[i];};
        double NeuronBias(int i) {return biases[i];};
        void SetNeuronBias(int i, double value) {biases[i] = value; biascenters[i] = value;};
        double NeuronGain(int i) {return gains[i];};
        void SetNeuronGain(int i, double value) {gains[i] = value;};
        double NeuronTimeConstant(int i) {return taus[i];};
        void SetNeuronTimeConstant(int i, double value) {taus[i] = value;Rtaus[i] = 1/value;};
        double NeuronExternalInput(int i) {return externalinputs[i];};
        double &NeuronExternalInputReference(int i) {return externalinputs[i];};
        void SetNeuronExternalInput(int i, double value) {externalinputs[i] = value;};
        double ConnectionWeight(int from, int to) {return weights[from][to];};
        void SetConnectionWeight(int from, int to, double value) {weights[from][to] = value; weightcenters[from][to] = value;};


        void LesionNeuron(int n)
        {
            for (int i = 1; i<= size; i++) {
                SetConnectionWeight(i,n,0);
                SetConnectionWeight(n,i,0);
            }
        }
        void SetCenterCrossing(void);
        void SetAmp(double gain){ampGain=gain;};
        void SetLearnRate(double value){learnrate=value;};
        void SetPeriod(double meanvalue, double stdvalue){meanPeriod=meanvalue;stdPeriod=stdvalue;};

        // Input and output
        friend ostream& operator<<(ostream& os, CTRNN& c);
        friend istream& operator>>(istream& is, CTRNN& c);

        // Control
        void RandomizeCircuitState(double lb, double ub);
        void RandomizeCircuitState(double lb, double ub, RandomState &rs);
        void RandomizeCircuitOutput(double lb, double ub);
        void RandomizeCircuitOutput(double lb, double ub, RandomState &rs);
        void Flux(RandomState &rs);
        void Learn(double performance);
        void EulerStep(double stepsize);
        void RK4Step(double stepsize);

        int size;

        TVector<double> states, outputs, biases, gains, taus, Rtaus, externalinputs, parameters;
        TMatrix<double> weights;
        TVector<double> TempStates,TempOutputs,k1,k2,k3,k4;
        // NEW FOR RL
        TMatrix<double> weightcenters;
        TVector<double> biascenters;
        TVector<double> flux;
        TVector<int> period,periodcount;
        double amp,meanPeriod,stdPeriod, convergence, initamp;
        int fluxPeriodMin, fluxPeriodMax;
        double learnrate;
        int fluxsize;
        double max_flux_amp = 40.0;
        double ampGain;
        double pastperf, reward;
        void PrintParameters();
};
