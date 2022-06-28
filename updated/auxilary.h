#include <eigen3/Eigen/Dense>
#include <cmath>
int maxOfAxis(int axis, Eigen::MatrixXd &m);
double sigmoid(double x);
void clip(Eigen::MatrixXd &m, double low, double high);
void roundnPlaces(double &value, const uint32_t &to);
void roundnPlacesMatrix(Eigen::MatrixXd &m , const uint32_t &to);
