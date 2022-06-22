#include <iostream>
//#include "queue.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include "CTRNN.h"
#include "microbial.h"
#include <time.h>
#include <stdlib.h>
#include "LeggedAgent.h"
#include <random>
using namespace std;
double duration = 220.0;
int s = 2;
double stepsize = 0.1;


double fitnessFunction(Eigen::MatrixXd &genome){

    LeggedAgent legged(s);
    legged.nervousSystem.setGenome(genome);
    Eigen::MatrixXd z(1,s);
    z = Eigen::MatrixXd::Zero(1,s);

    Eigen::MatrixXd o(1,s);
    o = Eigen::MatrixXd::Ones(1,s);
    legged.nervousSystem.initializeState(z);
    Eigen::VectorXd time = Eigen::VectorXd::LinSpaced(duration*10, 0, duration-0.1);
    Eigen::MatrixXd inputz(1, s);
    for(int i =0; i<time.size(); i++){
        for(int i=0; i<s;i++){inputz(0,i) = legged.getAngleFeedback();}
        legged.nervousSystem.setInputs(inputz);
        legged.nervousSystem.EulerStep(stepsize);
        legged.step1(stepsize, legged.nervousSystem.outputs);


}
    return legged.cx/duration;
}

int main(int argc, const char* argv[]){


    default_random_engine seed(time(NULL));
    //
    Eigen::MatrixXd genome(1, s*s+2*s);
    //this genome should have fitness of 0.627687
    genome << 0.99388489,  -0.19977217,   0.80557307,  0.66176187, -0.41946752,  0.00756486, -0.72451768, -0.50670193;
    //cout<<fitnessFunction(genome);

    Microbial m(10, s*s+2*s, 0.5, 0.1, 2, 20);
    m.setFitnessFunction(fitnessFunction);
    double mean = 0.0;
    double stddev  = 1.0;
    m.run(seed);
    m.fitStats();
    cout<<m.bestHistory<<endl;

    return 0;

}
