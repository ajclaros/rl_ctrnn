#include "TSearch.h"
#include "LeggedAgent.h"
#include "CTRNN.h"
#include "VectorMatrix.h"
#include "random.h"

//#define EVOLVE
//#define PRINTOFILE

// Task params
const double StepSize = 0.1; //0.05
const double RunDuration = 2000; //1*2200.0;
const int RunDurSteps = int(RunDuration/StepSize);
const double MaxPerf = 0.627;

// EA params
const int POPSIZE = 96;
const int GENS = 100;
const double MUTVAR = 0.1;
const double CROSSPROB = 0.5;
const double EXPECTED = 1.1;
const double ELITISM = 0.1;

// Nervous system params
const int N = 2;
const double WR = 16.0;
const double BR = 16.0;
const double TMIN = 1.0;
const double TMAX = 11.0 ;

// NEW PARAMS FOR RL
const double PERFWINDOW = 400;  // In units of time
const int PERFWINSTEPS = int(PERFWINDOW/StepSize);
const int PERFAVGWINSTEPS = 10;
const int MeanPeriod = 8000; // in time steps
const int StdPeriod = 8000000; // in time steps
const double AmpGain = 1.0;
const double LearnRate = 14.0;
const int LearnReps = 100;

int	VectSize = N*N + 2*N;

// ------------------------------------
// Genotype-Phenotype Mapping Functions
// ------------------------------------
void GenPhenMapping(TVector<double> &gen, TVector<double> &phen)
{
	int k = 1;
	// Time-constants
	for (int i = 1; i <= N; i++) {
		phen(k) = MapSearchParameter(gen(k), TMIN, TMAX);
		k++;
	}
	// Bias
	for (int i = 1; i <= N; i++) {
		phen(k) = MapSearchParameter(gen(k), -BR, BR);
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			phen(k) = MapSearchParameter(gen(k), -WR, WR);
			k++;
		}
	}
}

// ------------------------------------
// Fitness function
// ------------------------------------
double FitnessFunction(TVector<double> &genotype, RandomState &rs)
{
	// Map genootype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agent
	LeggedAgent Insect;

	// Instantiate the nervous system
	Insect.NervousSystem.SetCircuitSize(N);
	int k = 1;
	// Time-constants
	for (int i = 1; i <= N; i++) {
		Insect.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Bias
	for (int i = 1; i <= N; i++) {
		Insect.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Insect.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}

	Insect.Reset(0, 0, 0);

	// Run the agent
	for (double time = 0; time < 2200; time += StepSize) {
		Insect.Step1RPG(StepSize, 0, rs, 0);
	}
	// Finished
	return (Insect.cx/RunDuration);
}

double TracesWithoutLearning(TVector<double> &genotype, RandomState &rs)
{
	ofstream file;
	file.open("traceswol.dat");
	// Map genootype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agent
	LeggedAgent Insect;

	// Instantiate the nervous system
	Insect.NervousSystem.SetCircuitSize(N);
	int k = 1;
	// Time-constants
	for (int i = 1; i <= N; i++) {
		Insect.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Bias
	for (int i = 1; i <= N; i++) {
		Insect.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Insect.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}

	Insect.Reset(0, 0, 0);

	// Set RL PARAMS
	Insect.SetLearnParams(0.0,0.0,MeanPeriod,StdPeriod);

	// Run the agent
	// Step 1. As you run the agent, keep track of the performance in time using an average window
	// Save the history of positions over time
	double currentperf=0.0;
	TVector<double> posHist;
	posHist.SetBounds(1,RunDurSteps);
	posHist.FillContents(0.0);
	// Run initial transient
	double time = 0.0;
	for (int i = 1; i <= PERFWINSTEPS; i++) {
		Insect.Step1RPG(StepSize, 0, rs, 0);
		posHist(i) = Insect.cx;
		if (i % 100 == 0)
		{
			file << time << " " << posHist(i) << " " << currentperf << " " << Insect.NervousSystem.amp << " ";
			for (int x = 1; x <= N; x++) {
				for (int y = 1; y <= N; y++) {
					file << Insect.NervousSystem.weights[x][y] << " ";
				}
			}
			file << Insect.NervousSystem.biases << " " << Insect.NervousSystem.outputs << " ";
			for (int x = 1; x <= N; x++) {
				for (int y = 1; y <= N; y++) {
					file << Insect.NervousSystem.weightcenters[x][y] << " ";
				}
			}
			file << endl;
		}
		time += StepSize;
	}
	currentperf = ((Insect.cx - posHist(1)))/StepSize;
	//currentperf = currentperf < 0.0 ? 0.0 : currentperf;
	// Run with reward
	for (int i = PERFWINSTEPS+1; i <= RunDurSteps; i++) {
		Insect.Step1RPG(StepSize, 0, rs, 0);
		posHist(i) = Insect.cx;
		currentperf = ((Insect.cx - posHist(i-1))/StepSize);
		//currentperf = currentperf < 0.0 ? 0.0 : currentperf;
		if (i % 100 == 0)
		{
			file << time << " " << posHist(i) << " " << currentperf << " " << Insect.NervousSystem.amp << " ";
			for (int x = 1; x <= N; x++) {
				for (int y = 1; y <= N; y++) {
					file << Insect.NervousSystem.weights[x][y] << " ";
				}
			}
			file << Insect.NervousSystem.biases << " " << Insect.NervousSystem.outputs << " ";
			for (int x = 1; x <= N; x++) {
				for (int y = 1; y <= N; y++) {
					file << Insect.NervousSystem.weightcenters[x][y] << " ";
				}
			}
			file << endl;
		}
		time += StepSize;
	}
	// Finished
	file.close();
	return Insect.cx/RunDuration; // Insect.cx/RunDuration;
}

// ------------------------------------
// Fitness function
// ------------------------------------
double FitnessFunctionWithLearning(TVector<double> &genotype, RandomState &rs, double ampgain, double learnrate)
{

	// Map genootype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);
	double fitness = 0.0;
	// Create the agent
	LeggedAgent Insect;
	TVector<double> posHist;
	posHist.SetBounds(1,RunDurSteps);
	posHist.FillContents(0.0);
	//
	TVector<double> perfHist;
	perfHist.SetBounds(1,RunDurSteps);
	perfHist.FillContents(0.0);
	// Instantiate the nervous system
	Insect.NervousSystem.SetCircuitSize(N);
	for (int r = 1; r <= LearnReps; r++)
	{
		int k = 1;
		// Time-constants
		for (int i = 1; i <= N; i++) {
			Insect.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
			k++;
		}
		// Bias
		for (int i = 1; i <= N; i++) {
			Insect.NervousSystem.SetNeuronBias(i,phenotype(k));
			k++;
		}
		// Weights
		for (int i = 1; i <= N; i++) {
			for (int j = 1; j <= N; j++) {
				Insect.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
				k++;
			}
		}

		Insect.Reset(0, 0, 0);
		// Set RL PARAMS
		Insect.SetLearnParams(ampgain,learnrate,MeanPeriod,StdPeriod);
		// Run the agent
		// Step 1. As you run the agent, keep track of the performance in time using an average window
		// Save the history of positions over time
		double currentperf=0.0,avgperf=0.0;
		// Run initial transient
		double time = 0.0;
		for (int i = 1; i <= PERFWINSTEPS; i++) {

			Insect.Step1RPG(StepSize, 0, rs, 0);
			posHist(i) = Insect.cx;
			Insect.NervousSystem.runningaverage.push((posHist(i)-posHist(i-1))/StepSize);
			perfHist(i) = currentperf;
		}
		// Run with reward
		for (int i = PERFWINSTEPS+1; i <= RunDurSteps; i++) {
			avgperf = 0.0;
			//cout<<currentperf<<endl;
			for (int j = 1; j <= PERFAVGWINSTEPS; j++)
			{
				avgperf += perfHist(i-j);
			}
			avgperf /= PERFAVGWINSTEPS;

			Insect.Step1RPG(StepSize, currentperf - Insect.NervousSystem.runningaverage.avg(), rs, 0);
			posHist(i) = Insect.cx;
			currentperf= (posHist(i)-posHist(i-1))/StepSize;
			Insect.NervousSystem.runningaverage.push(currentperf);
			perfHist(i) = currentperf;
		}
	}
	Insect.NervousSystem.recoverParameters();
	return FitnessFunction(Insect.NervousSystem.parameters, rs);
}

double TracesWithLearning(TVector<double> &genotype, RandomState &rs)
{
	// Map genootype to phenotype
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);
	GenPhenMapping(genotype, phenotype);

	// Create the agent
	LeggedAgent Insect;
	Insect.NervousSystem.weightRange = WR;
	Insect.NervousSystem.biasRange = BR;
	Insect.NervousSystem.timeRange[0] = TMIN;
	Insect.NervousSystem.timeRange[1] = TMAX;
	// Instantiate the nervous system
	Insect.NervousSystem.SetCircuitSize(N);
	int k = 1;
	// Time-constants
	for (int i = 1; i <= N; i++) {
		Insect.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Bias
	for (int i = 1; i <= N; i++) {
		Insect.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Insect.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}


	Insect.Reset(0, 0, 0);

	// Set RL PARAMS
	//
	Insect.SetLearnParams(AmpGain,LearnRate,MeanPeriod,StdPeriod);
	Insect.NervousSystem.recoverParameters();
	// Run the agent
	// Step 1. As you run the agent, keep track of the performance in time using an average window
	// Save the history of positions over time
	double currentperf=0.0,avgperf=0.0,pastperf=0.0;
	TVector<double> posHist;
	posHist.SetBounds(1,RunDurSteps);
	posHist.FillContents(0.0);

	TVector<double> perfHist;
	perfHist.SetBounds(1,RunDurSteps);
	perfHist.FillContents(0.0);

	// Run initial transient
	double time = 0.0;
	for (int i = 1; i <= PERFWINSTEPS; i++) {
		posHist(i) = Insect.cx;
		//cout<<(Insect.cx-posHist[i-1])/StepSize<<endl;
		currentperf = ((Insect.cx - posHist[i-1])/(StepSize));
		perfHist(i) = currentperf;
		Insect.Step1RPG(StepSize, currentperf, rs, 0);
		Insect.NervousSystem.runningaverage.push(currentperf);
		time+=StepSize;
	}
	// Run with reward
	for (int i = PERFWINSTEPS+1; i <= RunDurSteps; i++) {
		avgperf /= PERFAVGWINSTEPS;
		posHist(i) = Insect.cx;
		currentperf = (posHist(i)- posHist(i-1))/StepSize;
		perfHist(i) = currentperf;
		Insect.Step1RPG(StepSize, currentperf, rs, 1);
		//cout<<Insect.NervousSystem.weightcenters[1][1]<<" ";
		time += StepSize;
	}
	Insect.NervousSystem.recoverParameters();
	return FitnessFunction(Insect.NervousSystem.parameters, rs);
}

// ------------------------------------
// Display functions
// ------------------------------------
void EvolutionaryRunDisplay(int Generation, double BestPerf, double AvgPerf, double PerfVar)
{
	cout << Generation << " " << BestPerf << " " << AvgPerf << " " << PerfVar << endl;
}

void ResultsDisplay(TSearch &s)
{

	TVector<double> bestVector;

	ofstream BestIndividualFile;
	TVector<double> phenotype;
	phenotype.SetBounds(1, VectSize);

	// Save the genotype of the best individual
	bestVector = s.BestIndividual();
	RandomState rs;
	// cout << TracesWi(bestVector,rs)  << endl;

	BestIndividualFile.open("best.gen.dat");
	BestIndividualFile << bestVector << endl;
	BestIndividualFile.close();

	// Also show the best individual in the Circuit Model form
	BestIndividualFile.open("best.ns.dat");
	GenPhenMapping(bestVector, phenotype);
	LeggedAgent Insect;
	// Instantiate the nervous system
	Insect.NervousSystem.SetCircuitSize(N);
	int k = 1;
	// Time-constants
	for (int i = 1; i <= N; i++) {
		Insect.NervousSystem.SetNeuronTimeConstant(i,phenotype(k));
		k++;
	}
	// Bias
	for (int i = 1; i <= N; i++) {
		Insect.NervousSystem.SetNeuronBias(i,phenotype(k));
		k++;
	}
	// Weights
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			Insect.NervousSystem.SetConnectionWeight(i,j,phenotype(k));
			k++;
		}
	}
	BestIndividualFile << Insect.NervousSystem;
	BestIndividualFile.close();
}

// ------------------------------------
// The main program
// ------------------------------------
#ifdef EVOLVE
int main (int argc, const char* argv[]) {

	cout<<"EVOLVE"<< endl;
	long randomseed = static_cast<long>(time(NULL));
	if (argc == 2)
	randomseed += atoi(argv[1]);
	TSearch s(VectSize);

	#ifdef PRINTOFILE
	ofstream file;
	file.open("evol.dat");
	cout.rdbuf(file.rdbuf());
	#endif

	// Configure the search
	s.SetRandomSeed(randomseed);
	s.SetSearchResultsDisplayFunction(ResultsDisplay);
	s.SetPopulationStatisticsDisplayFunction(EvolutionaryRunDisplay);
	s.SetSelectionMode(RANK_BASED);
	s.SetReproductionMode(GENETIC_ALGORITHM);
	s.SetPopulationSize(POPSIZE);
	s.SetMaxGenerations(GENS);
	s.SetCrossoverProbability(CROSSPROB);
	s.SetCrossoverMode(UNIFORM);
	s.SetMutationVariance(MUTVAR);
	s.SetMaxExpectedOffspring(EXPECTED);
	s.SetElitistFraction(ELITISM);
	s.SetSearchConstraint(1);
	s.SetReEvaluationFlag(1);

	// Run Stage 1
	s.SetEvaluationFunction(FitnessFunction);
	s.ExecuteSearch();

	return 0;
}
#else
int main (int argc, const char* argv[]) {
	cout<<"LEARNING"<<endl;
	ifstream genefile;
	ofstream outfile;
	outfile.open("learnrate_PW400.dat");
	genefile.open("best.gen.dat");
	TVector<double> genotype(1, VectSize);
	genefile >> genotype;
	long randomseed = static_cast<long>(time(NULL));
	RandomState rs;
	rs.SetRandomSeed(randomseed);
	double baseline,fit;
	TVector<double> randomvect(1, VectSize);

	baseline = FitnessFunction(genotype,rs);
	RandomUnitVector(randomvect);
	TVector<double> perturbed(1, VectSize);
	perturbed.FillContents(0);
	for(int i=1; i<VectSize; i++){
		perturbed[i]= genotype[i]+randomvect[i]*3;
	}
	//cout<<FitnessFunction(genotype, rs)<<endl;
	//cout<<perturbed<<"\n"<<endl;

	//rs.SetRandomSeed(randomseed);
	cout<<FitnessFunction(perturbed, rs)<<endl;
	cout<<"Learning"<<endl;
	cout<<TracesWithLearning(genotype, rs)<<endl;
	//cout<<TracesWithLearning(perturbed, rs);
	return 0;
}
#endif
