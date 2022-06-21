// *************************
// A class for legged agents
//
// RDB 2/16/96
// *************************
#pragma once
#include "CTRNN.h"
// Global constants
//const double Pi = 3.1415926;
// The Leg class declaration
// The LeggedAgent class declaration
class LeggedAgent {
	public:
		int footState;
		double cx, cy, vx,  angle, omega, forwardForce, backwardForce,
			jointX, jointY, footX, footY;
		CTRNN nervousSystem;//(double WR=16.0, double BR= 16.0, double TR=5.0, double TA = 6.0);
		LeggedAgent(int size);
		Eigen::MatrixXd state();
		double getAngleFeedback();
		void step1(double stepsize, Eigen::MatrixXd u);
		void step2(double stepsize, Eigen::MatrixXd u);
		void step3(double stepsize, Eigen::MatrixXd u);

};
