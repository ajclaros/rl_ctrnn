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
    //srand(20);
    srand(time(NULL));
    CTRNN network(3);
    network.randomizeParameters();
    cout<<network.taus.transpose()<<endl;
    cout<<network.taus<<endl;
    cout<<endl;
    cout<<"Max of rows:"<<endl;
    cout<<maxOfAxis(1, network.taus.transpose())<<endl;
    //cout<<network.taus<<endl<<endl;
    //a.setSize(4);
    //cout<<a.size<<endl<<endl;
    return 0;

}
