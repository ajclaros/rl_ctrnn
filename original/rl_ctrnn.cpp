#include "rl_ctrnn.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdarg>
#include "CTRNN.h"

#include <eigen3/Eigen/Dense>
using namespace std;

RLCTRNN::RLCTRNN(int size=2, double weightBounds=16.0, double biasBounds= 16.0, double tConstMin = 1.0, double tConstMax=1.0,
                double initFluxAmp = 1.0, double maxFluxAmp = 10.0, double minFluxPeriod = 2.0, double maxFluxPeriod = 10.0, double convRateFlux = 0.1,
                double learnRate = 1.0, bool gaussianMode = false, bool squareOscilationMode = false,
                double initBiasFluxAmp = 0.0, double maxFluxBiasAmp = 0.0, double minFluxPeriodBias = 0.0, double maxFluxPeriodBias = 0.0,
                double convRateFluxBias = 0.1, double weightRangeMap = 16.0, double biasRangeMap = 16.0, double tConstRangeMap = 5.0, double tConstAdd = 6.0)
{



}

RLCTRNN::~RLCTRNN(){
    SetCircuitSize(0);

}
