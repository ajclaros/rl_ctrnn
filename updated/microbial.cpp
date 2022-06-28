#include <cmath>
#include "microbial.h"
#include <eigen3/Eigen/Dense>
#include "CTRNN.h"
#include <random>
#include <time.h>
#include "auxilary.h"

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
    fitness.resize(popSize, 1);
    fitness = Eigen::MatrixXd::Zero(popSize, 1);
    avgHistory = Eigen::MatrixXd::Zero(numGenerations, 1);
    bestHistory = Eigen::MatrixXd::Zero(numGenerations, 2);

}

void Microbial::fitStats(){
    bestFitness = maxOfAxis(0, fitness);
    avgFitness = fitness.mean();
    avgHistory(currentGen, 0) = avgFitness;
    bestHistory(currentGen, 0) = bestFitness;
    bestHistory(currentGen, 1) = fitness.maxCoeff();// fitness(bestFitness,0);
}


void Microbial::run(std::default_random_engine seed){
    int winner, loser, a, b;
    std::uniform_real_distribution<double> uniform(0, 1);
    std::normal_distribution<double> normal(0, mutateProb);
    for(int i=0; i<popSize; i++){
        tempGenome= population.row(i);
        fitness(i,0 ) = fitnessFunction(tempGenome);
}
    std::cout<<fitness<<std::endl;
    for(int g=0; g<numGenerations;g++){
        std::cout<<"Generation: "<<g<<std::endl;
        currentGen = g;
        fitStats();
        //Generate matrix of size = (popsize, 2), where each row are two individuals that will be compared
        Eigen::VectorXd individuals= Eigen::VectorXd::LinSpaced(popSize, 0, popSize-1);
        std::random_shuffle(individuals.begin(), individuals.end());
        Eigen::MatrixXd indivMat  = individuals;
        indivMat.resize(popSize/2,2);
        //std::cout<<indivMat<<std::endl;
        for(int i=0; i<popSize/2; i++){
            a = indivMat(i,0);
            b = indivMat(i,1);
            if(fitness(a,0)>fitness(b,0)){
                winner = a;
                loser = b;
}
            else{
                winner = b;
                loser = a;
}
            for(int l=0;l<genesize; l++){
                if(uniform(seed)< recombProb){
                   population(loser, l) = population(winner, l);}
}
            for(int l=0; l<genesize; l++){
                population(loser, l) += normal(seed);
                if(population(loser, l)>1.0) population(loser, l) = 1.0;
                if(population(loser, l)<-1.0) population(loser, l) = -1.0;
}
            tempGenome= population.row(loser);
            fitness(loser,0) = fitnessFunction(tempGenome);
}

}

}
