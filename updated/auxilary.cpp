#include "auxilary.h"
#include <eigen3/Eigen/Dense>
int maxOfAxis(int axis, Eigen::MatrixXd &m){
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

double sigmoid(double x){
  return 1/(1 + exp(-x));
}

void clip(Eigen::MatrixXd &m, double low, double high){
    for(int i=0; i<m.rows();i++){
        for(int j=0; j<m.cols(); j++){
            if(m(i,j)<low){*(&m(i,j)) = low;}
            if(m(i,j)>high){*(&m(i,j)) = high;}
}
}
}
void roundnPlaces(double &value, const uint32_t &to)
{
    uint32_t places = 1, whole = *(&value);
    for(uint32_t i = 0; i < to; i++) places *= 10;
    value -= whole; //leave decimals
    value *= places;  //0.1234 -> 123.4
    value  = round(value);//123.4 -> 123
    value /= places;  //123 -> .123
    value += whole; //bring the whole value back
}
void roundnPlacesMatrix(Eigen::MatrixXd &m, const uint32_t &to){
    for(int i=0; i<m.rows(); i++){
        for(int j=0; j<m.cols(); j++){
            roundnPlaces(m(i,j), to);
}
}
    
}
