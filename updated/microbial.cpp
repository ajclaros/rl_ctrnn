#include <cmath>
#include "microbial.h"
#include <eigen3/Eigen/Dense>
#include "CTRNN.h"

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
Microbial::Microbial(int popSize, int geneSize, double recombProb, double mutateProb, int demeSize, int numGenerations){
    this->popSize = popSize;
    this->genesize = geneSize;
    this->recombProb = recombProb;
    this->mutateProb = mutateProb;
    this->numGenerations = numGenerations;
    this->demeSize = int(demeSize/2);
    tournaments = numGenerations*popSize;
    population.resize(popSize, geneSize);
    population = Eigen::MatrixXd::Random(popSize, genesize);
    fitness.resize(popSize, genesize);
    fitness = Eigen::MatrixXd::Zero(popSize, 1);
    avgHistory = Eigen::MatrixXd::Zero(numGenerations, 1);
    bestHistory = Eigen::MatrixXd::Zero(numGenerations, 1);
}

void Microbial::fitStats(){
    int bestInd = maxOfAxis(0, population);
    bestFitness = maxOfAxis(0, fitness);
    avgFitness = fitness.mean();
    avgHistory(currentGen, 0) = avgFitness;
    bestHistory(currentGen, 0) = bestFitness;
}


//void Microbial::run(){
//    for(int i=0; i<popSize; i++){
//        fitness(i) = fitnessFunction(population(i,0));
//}
//    for(int g=0; g<generations;g++){
//        1+1;
//}
//}
