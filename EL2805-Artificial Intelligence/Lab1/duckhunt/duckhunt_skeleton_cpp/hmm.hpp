#ifndef _HMM_H_
#define _HMM_H_

#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#define DELTA 10e-50
#define MAXITER 50  
const double LOGZERO = std::numeric_limits<double>::quiet_NaN();

class HMM 
{

    public:
        HMM(){};
        HMM(unsigned int N, unsigned int K);
        HMM(const HMM &model);
        HMM(std::vector<std::vector<double> > _A, std::vector<std::vector<double> > _B, std::vector<double> _pi);

        // Get funcs
        unsigned int getNStates() const {return N;}
        unsigned int getNObservations() const {return K;}
        std::vector<std::vector<double> > getA() const {return A;}
        std::vector<std::vector<double> > getB() const {return B;}
        std::vector<double> getPi() const {return pi;}
	

        // intermediate functions
        std::vector<double> getNextObsProbDist(std::vector<int> sequence);

       //important functions of the HMM
        void estimateModel(std::vector<int> sequence); // akka train
        double getObsSeqProb(std::vector<int> O); // probability of observation sequence
        std::vector<double> estimateDistribution(std::vector<int> observations); // estimate distribution of next state after the observation seq
        std::vector<double> probabilityDistributionOfNextObs(std::vector<double> stateDistribution); // probability distribution of possible next emission
        unsigned int getNextMostProbObs(std::vector<int> sequence, double &maxProb); // returns next mos probable emission

    private:
        unsigned int N; // nr of states
        unsigned int K; // nr of observations
        std::vector<std::vector<double> > A;  // Transition matrix 
        std::vector<std::vector<double> > B;  // Emission matrix 
        std::vector<double> pi;  // Initial state distribution 

        std::vector<std::vector<double> > alpha;  // Alpha value matrix of the HMM
        std::vector<std::vector<double> > beta;   // Beta value matrix of the HMM
        std::vector<std::vector<double> > gamma;  // Gamma value matrix of the HMM
        std::vector<std::vector<std::vector<double> > > diGamma;  // Di-Gamma value matrix of the HMM
        std::vector<double> c;  // Constant value vector for the scaling (Stamp)

        /* Hidden functionalities of the HMM */
        void alphaPass(std::vector<int> sequence);
        void betaPass(std::vector<int> sequence);
        void computeGammaAndDiGamma(std::vector<int> sequence);
};



// extra functions
std::vector<double> normalize(std::vector<double> x);





















#endif