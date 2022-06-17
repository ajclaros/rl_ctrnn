#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cmath>

class Microbial{
    public:
        double (*fitnessFunction)(Eigen::MatrixXd &v);
        int popSize, genesize;
        double recombProb, mutateProb;
        int demeSize, numGenerations, tournaments;
        Eigen::MatrixXd population, fitness, avgHistory, bestHistory;
        int currentGen= 0;
        Microbial(int popSize, int geneSize, double recombProb, double mutateProb, int demeSize, int numGenerations);
        void run();
        void setFitnessFunction(double (*evalFn)(Eigen::MatrixXd &v)){fitnessFunction= evalFn;}
        void fitStats();
};
