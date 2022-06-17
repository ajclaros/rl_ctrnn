#include "microbial.h"
#include <cmath>
#include <eigen3/Eigen/Dense>

Microbial::Microbial(int popSize, int geneSize, double recombProb, double mutateProb, int demeSize, int numGenerations){
    this->popSize = popSize;
    this->genesize = geneSize;
    this->recombProb = recombProb;
    this->mutateProb = mutateProb;
    this->numGenerations = numGenerations;
    this->demeSize = int(demeSize/2);
    tournaments = numGenerations*popSize;
    population.resize(popSize, geneSize);
    population = Eigen::MatrixXd::Random(popsize, genesize);
    fitness.resize(popsize, genesize);
    fitness = Eigen::MatrixXd::Zero(popsize, 1);
    avgHistory = Eigen::MatrixXd::Zero(numGenerations, 1);
    bestHistory = Eigen::MatrixXd::Zero(numGenerations, 1);

}

void CTRNN::fitStats(){
    1+1;




}


void CTRNN::run(){
    for(int i=0; i<popSize; i++){
        fitness(i) = fitnessFunction(population(i,0));
}
    for(int g=0; g<generations;g++){



}
}
