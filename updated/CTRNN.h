#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cmath>
#include "auxilary.h"
#pragma once
// The CTRNN class declaration
class CTRNN {
    public:
        int size;
        // The constructor
        CTRNN(int size, double WR=16.0, double BR=16.0, double TR=5.0, double TA=6.0);
        CTRNN();
        void setGenome(const Eigen::MatrixXd genome);
        void setSize(int size);
        virtual void reset();
        void randomizeParameters();
        void setVoltages(const Eigen::MatrixXd voltages);
        void setTaus(Eigen::MatrixXd taus);
        void setInputs(Eigen::MatrixXd &inputs);
        void setWeightCenters(Eigen::MatrixXd weightCenters);
        virtual void initializeState(Eigen::MatrixXd v);
        void print();
        double weightRange;
        double biasRange;
        double timeRange[2] = {0.0,0.0};
        Eigen::MatrixXd voltages, outputs, gains, taus, invTaus, inputs; //, Rtaus;
        Eigen::MatrixXd weightcenters, biascenters;
        Eigen::MatrixXd params;
        double WR, BR, TR, TA;
        void recoverParameters();
        int getCircuitSize(void) {return size;};
        //void setLearnRate(double value){learnrate=value;};
        //void Learn(double performance);
        void EulerStep(double stepsize);
};
