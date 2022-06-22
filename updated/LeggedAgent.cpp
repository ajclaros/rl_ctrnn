// ***********************
// Methods for LeggedAgent
//
// RDB 2/16/96
// ***********************

#include "LeggedAgent.h"
#include <eigen3/Eigen/Dense>
#include <cmath>

const long double Pi =  3.141592653589793;
// Constants
const int    LegLength = 15;
const double MaxLegForce = 0.05;
const double ForwardAngleLimit = Pi/6.0;
const double BackwardAngleLimit = -Pi/6.0;
const double MaxVelocity = 6.0;
const double MaxTorque = 0.5;
const double MaxOmega = 1.0;

LeggedAgent::LeggedAgent(int size){
	cx= 0.0;
	cy = 0.0;
	vx = 0.0;
	footState = 0.0;
	angle = ForwardAngleLimit;
	omega = 0.0;
	backwardForce = 0.0;
	forwardForce= 0.0;
	jointX = cx;
	jointY = cy + 12.5;
	footX = jointX + LegLength + sin(angle);
	footY = jointY + LegLength + sin(angle);
	nervousSystem.setSize(size);
}
Eigen::MatrixXd LeggedAgent::state(){
	Eigen::MatrixXd s(1,3);
	s << angle, omega, footState;
	return s;
}
double LeggedAgent::getAngleFeedback(){
	return angle*5.0/ForwardAngleLimit;
}
void LeggedAgent::step1(double stepsize, Eigen::MatrixXd u){
	double force = 0.0;
	if(u(0,0)>0.5){
		footState = 1;
		omega = 0.0;
		forwardForce = 2*(u(0,0) - 0.5) * MaxLegForce;
		backwardForce = 0.0;
}
	else{
		footState = 0;
		forwardForce = 0.0;
        backwardForce = 2 * (0.5 - u(0, 0)) * MaxLegForce;
}
	double f = forwardForce - backwardForce;
	if(footState == 1.0){
		if((angle>=BackwardAngleLimit && angle <=ForwardAngleLimit)||
		   (angle < BackwardAngleLimit && f<0) ||
		   (angle >ForwardAngleLimit and f >0)){
			force = f;

}
}
	//std::cout<<footState<<std::endl;
	vx += stepsize*force;
	if(vx < -MaxVelocity){vx = -MaxVelocity;}
	if(vx >MaxVelocity){vx = MaxVelocity;}
	cx += stepsize * vx;
	jointX+= stepsize* vx;
	if(footState==1){
		double tempAngle = atan2(footX - jointX, footY - jointY);
		omega = (tempAngle - angle)/stepsize;
		angle = tempAngle;
}
	else{
		vx = 0.0;
		omega += stepsize*MaxTorque*(backwardForce - forwardForce);
		if(omega < -MaxOmega){
			omega = -MaxOmega;

}
		angle+=stepsize * omega;
		if(omega > MaxOmega){
			omega = MaxOmega;
}
		angle+= angle + stepsize*omega;
		if(angle < BackwardAngleLimit){
			angle = BackwardAngleLimit;
			omega = 0;
}
		if(angle > ForwardAngleLimit){
			angle = ForwardAngleLimit;
			omega = 0;
}
		footX = jointX + LegLength + sin(angle);
		footY = jointY + LegLength + cos(angle);
}
	if(cx - footX > 20.0){
		vx = 0.0;
}

}
