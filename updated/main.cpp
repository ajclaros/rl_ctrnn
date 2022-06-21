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
    LeggedAgent legged(2);
    legged.nervousSystem.setGenome(genome);
    Eigen::VectorXd time = Eigen::VectorXd::LinSpaced(2200 , 0, 220);
    for(int i =0; i<time.size(); i++){
        1+1;
        
        
}
    return 0.5;

}

int main(int argc, const char* argv[]){
    srand(20);
    //srand(time(NULL));
    int size = 2;
    //Microbial m(100, size*size*2*size, 0.5, 0.5, 2, 20);
    //
    Eigen::MatrixXd genome(1, size*size+2*size);

    genome << 0.12029618, 0.03063469, 0.07051985, 0.09383968, 0.0928963 ,
       0.12052319, 0.        , 0.        ;
    Eigen::VectorXd time = Eigen::VectorXd::LinSpaced(100, 0, 9.9);
    cout<<time<<endl;
    return 0;

}
