#include <iostream>
//#include "queue.h"
#include <eigen3/Eigen/Dense>
#include "CTRNN.h"
#include <time.h>
#include <stdlib.h>
using namespace std;

//Matrix<double, Eigen::Dynamic, Eigen::Dynamic> a;
int maxOfAxis(int axis, Eigen::MatrixXd m){ //
    int val;
    if(axis==0){
        for(int i=0; i<m.rows(); i++)
            if(m(i,0)==m.maxCoeff()) val = i;
}
    else{
        for(int j=0; j<m.cols(); j++){
            if(m(0,j)==m.maxCoeff()) val = j;

}
}
    return val;
}

int main(int argc, const char* argv[]){
    srand(20);
    //srand(time(NULL));
    int size = 2;
    CTRNN network(size);
    Eigen::MatrixXd genome(1, size*size+size*2);// = Eigen::MatrixXd::Random(1, size*size+size*2);
    genome<< 1.        , -0.17390272,  0.27844626,  0.66946256, -0.30683733, 0.05880393, -1.        , -0.31919337;

    network.setGenome(genome);
    network.initializeState(Eigen::MatrixXd::Zero(1,size));
    network.EulerStep(0.1);




//    network.recoverParameters();
//    network.initializeState(Eigen::MatrixXd::Zero(1,size));
    return 0;

}
