
#include <iostream>
//#include "queue.h"
#include <eigen3/Eigen/Dense>
#include "CTRNN.h"
#include <time.h>
#include <stdlib.h>
using namespace std;

//Matrix<double, Eigen::Dynamic, Eigen::Dynamic> a;
int main(int argc, const char* argv[]){
    srand(20);
    //srand(time(NULL));
    CTRNN network(2);
    network.randomizeParameters();
    cout<<network.taus<<endl;
    cout<<network.invTaus<<endl;

    //cout<<network.taus<<endl<<endl;
    //a.setSize(4);
    //cout<<a.size<<endl<<endl;
    return 0;

}
