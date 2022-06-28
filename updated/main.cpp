#include <iostream>
#include <eigen3/Eigen/Dense>
#include "CTRNN.h"
#include "RLCTRNN.h"
#include "microbial.h"
#include <time.h>
#include <stdlib.h>
#include "LeggedAgent.h"
#include "auxilary.h"
#include <random>
using namespace std;
double duration = 100.0;
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
    //
    //genome << 0.794092, -0.532777, -0.851293,  0.471561,-0.486715,       -1,        -1,  -0.9212929;

    //genome << 0.0289362, 0.0205082, -0.987006,   0.97889, -0.982338,  0.557834,  0.375119, -0.192675;
    //genome << 0.99388489,  -0.19977217,   0.80557307,  0.66176187, -0.41946752,  0.00756486, -0.72451768, -0.50670193;
    //cout<<fitnessFunction(genome);
    int popSize=100;
    int genesize = s*s+2*s;
    double recombProb = 0.5;
    double mutateProb = 0.01;
    int demesize = 2;
    int numgenerations = 100;
    RLCTRNN a(s, 4000, 2.75, 40.0,
              200, 400, false);
    a.randomizeParameters(seed);
    a.calcInnerWeightsWithFlux();
    //Eigen::MatrixXd values(3,3);
    //values = Eigen::MatrixXd::Random(3,3);
    //cout<<"Values: \n"<<values<<endl;
    //roundnPlacesMatrix(values, 3);
    //cout<<"Values roundnPlaces:\n"<<values<<endl;
    //clip(values, 0.3, 0.6);
    //cout<<"Values clip (0.4, 0.6):\n"<<values<<endl;



    //Microbial m(popSize, genesize, recombProb, mutateProb, demesize, numgenerations);
    //m.setFitnessFunction(fitnessFunction);
    //m.run(seed);
    //genome = m.population.row(maxOfAxis(0, m.fitness));
    //std::cout<<genome<<std::endl;
    //std::cout<<"|"<<fitnessFunction(genome);
    return 0;

}
