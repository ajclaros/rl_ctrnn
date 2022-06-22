#include <iostream>
//#include "queue.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include "CTRNN.h"
#include "microbial.h"
#include <time.h>
#include <stdlib.h>
#include "LeggedAgent.h"
using namespace std;
int duration = 2.0;
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
    Eigen::VectorXd time = Eigen::VectorXd::LinSpaced(duration*10, 0, duration-1);
    Eigen::MatrixXd inputz(1, s);
    for(int i =0; i<time.size(); i++){
        for(int i=0; i<s;i++){inputz(0,i) = legged.getAngleFeedback();}
        legged.nervousSystem.setInputs(inputz);
        legged.nervousSystem.EulerStep(stepsize);
        //cout<<legged.nervousSystem.outputs<<endl;
        //legged.step1(stepsize, legged.nervousSystem.outputs);

}
    return legged.cx/duration;
}

int main(int argc, const char* argv[]){
    srand(20);

    //srand(time(NULL));
    //Microbial m(100, size*size*2*size, 0.5, 0.5, 2, 20);
    //
    Eigen::MatrixXd genome(1, s*s+2*s);

    genome << 0.99388489,  -0.19977217,   0.80557307,  0.66176187, -0.41946752,  0.00756486, -0.72451768, -0.50670193;
    cout<<fitnessFunction(genome);
    return 0;

}
