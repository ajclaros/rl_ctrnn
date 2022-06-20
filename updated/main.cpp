#include <iostream>
//#include "queue.h"
#include <eigen3/Eigen/Dense>
#include "CTRNN.h"
#include "microbial.h"
#include <time.h>
#include <stdlib.h>
#include "LeggedAgent.h"
using namespace std;

// //Matrix<double, Eigen::Dynamic, Eigen::Dynamic> a;
// int maxOfAxis(int axis, Eigen::MatrixXd m){ //
//     int val;
//     if(axis==0){
//         for(int i=0; i<m.rows(); i++)
//             if(m(i,0)==m.maxCoeff()) val = i;
// }
//     else{
//         for(int j=0; j<m.cols(); j++){
//             if(m(0,j)==m.maxCoeff()) val = j;

// }
// }
//     return val;
// }
double fitnessFunction(Eigen::MatrixXd &genome){


    return 0.5;

}

int main(int argc, const char* argv[]){
    srand(20);
    //srand(time(NULL));
    int size = 2;
    Microbial m(100, size*size*2*size, 0.5, 0.5, 2, 20);
    m.setFitnessFunction(fitnessFunction);
    return 0;

}
